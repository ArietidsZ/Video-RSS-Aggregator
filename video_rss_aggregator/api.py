from __future__ import annotations

import platform
import shutil
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from adapter_gui import render_setup_page
from core_config import Config
from service_media import runtime_dependency_report
from video_rss_aggregator.bootstrap import AppRuntime, build_runtime
from video_rss_aggregator.domain.outcomes import Failure
from video_rss_aggregator.setup_view_models import (
    build_diagnostics_view,
    build_runtime_view,
)


class IngestRequest(BaseModel):
    feed_url: str
    process: bool = False
    max_items: int | None = None


class ProcessRequest(BaseModel):
    source_url: str
    title: str | None = None


def create_app(
    runtime: AppRuntime | None = None,
    config: Config | None = None,
) -> FastAPI:
    resolved_config = (
        runtime.config if runtime is not None else config or Config.from_env()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = (
            runtime if runtime is not None else await build_runtime(resolved_config)
        )
        try:
            yield
        finally:
            close_result = app.state.runtime.close()
            if close_result is not None:
                await close_result

    app = FastAPI(title="Video RSS Aggregator", version="0.1.0", lifespan=lifespan)
    if runtime is not None:
        app.state.runtime = runtime
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).resolve().parent / "static"),
        name="static",
    )

    def _runtime(request: Request) -> AppRuntime:
        return request.app.state.runtime

    def _check_auth(
        authorization: str | None = Header(None), x_api_key: str | None = Header(None)
    ) -> None:
        if resolved_config.api_key is None:
            return
        token = None
        if authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        if token is None:
            token = x_api_key
        if token != resolved_config.api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/", response_class=HTMLResponse)
    async def setup_home(request: Request) -> str:
        return render_setup_page(
            resolved_config,
            root_path=request.scope.get("root_path", ""),
        )

    @app.get("/setup/config")
    async def setup_config() -> dict[str, object]:
        return resolved_config.as_setup_payload()

    @app.get("/setup/diagnostics")
    async def setup_diagnostics(request: Request) -> dict[str, object]:
        media_tools = runtime_dependency_report()
        yt_dlp_cmd = shutil.which("yt-dlp")
        ytdlp = {
            "command": yt_dlp_cmd,
            "module_available": find_spec("yt_dlp") is not None,
        }
        ytdlp["available"] = bool(ytdlp["command"] or ytdlp["module_available"])

        ollama: dict[str, object] = {
            "base_url": resolved_config.ollama_base_url,
            "reachable": False,
            "version": None,
            "models_found": 0,
            "error": None,
        }
        try:
            runtime_status = await _runtime(
                request
            ).use_cases.get_runtime_status.execute()
            runtime_details = runtime_status
            if isinstance(runtime_details, dict):
                ollama["reachable"] = True
                ollama["version"] = runtime_details.get("ollama_version")
                local_models = runtime_details.get("local_models", {})
                ollama["models_found"] = (
                    len(local_models) if isinstance(local_models, dict) else 0
                )
        except Exception as exc:
            ollama["error"] = str(exc)

        ffmpeg_ok = bool(media_tools["ffmpeg"].get("available"))
        ffprobe_ok = bool(media_tools["ffprobe"].get("available"))
        ytdlp_ok = bool(ytdlp["available"])
        ollama_ok = bool(ollama["reachable"])

        diagnostics_payload = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "python_version": sys.version.split()[0],
                "python_executable": sys.executable,
            },
            "dependencies": {
                "ffmpeg": media_tools["ffmpeg"],
                "ffprobe": media_tools["ffprobe"],
                "yt_dlp": ytdlp,
                "ollama": ollama,
            },
            "ready": ffmpeg_ok and ffprobe_ok and ytdlp_ok and ollama_ok,
        }
        diagnostics_payload["setup_view"] = build_diagnostics_view(diagnostics_payload)
        return diagnostics_payload

    @app.post("/setup/bootstrap")
    async def setup_bootstrap(
        request: Request, _=Depends(_check_auth)
    ) -> dict[str, object]:
        report = await _runtime(request).use_cases.bootstrap_runtime.execute()
        models_prepared = report.get("models_prepared")
        if models_prepared is None:
            models_prepared = report["models"]

        runtime_payload = report.get("runtime")
        if isinstance(runtime_payload, dict):
            runtime_response = runtime_payload
        else:
            runtime_response = await _runtime(
                request
            ).use_cases.get_runtime_status.execute()

        return {
            "models_prepared": models_prepared,
            "runtime": runtime_response,
        }

    @app.post("/ingest")
    async def ingest(
        req: IngestRequest, request: Request, _=Depends(_check_auth)
    ) -> dict[str, object]:
        report = await _runtime(request).use_cases.ingest_feed.execute(
            req.feed_url,
            req.process,
            req.max_items,
        )
        return {
            "feed_title": report.feed_title,
            "item_count": report.item_count,
            "processed_count": report.processed_count,
        }

    @app.post("/process")
    async def process(
        req: ProcessRequest, request: Request, _=Depends(_check_auth)
    ) -> dict[str, object]:
        outcome = await _runtime(request).use_cases.process_source.execute(
            req.source_url,
            req.title,
        )
        if isinstance(outcome, Failure):
            raise HTTPException(status_code=502, detail=outcome.reason)
        return {
            "source_url": outcome.media.source_url,
            "title": outcome.media.title,
            "transcript_chars": outcome.summary.transcript_chars,
            "frame_count": outcome.summary.frame_count,
            "summary": {
                "summary": outcome.summary.summary,
                "key_points": list(outcome.summary.key_points),
                "visual_highlights": list(outcome.summary.visual_highlights),
                "model_used": outcome.summary.model_used,
                "vram_mb": outcome.summary.vram_mb,
                "error": outcome.summary.error,
            },
        }

    @app.get("/rss")
    async def rss_feed(
        request: Request, limit: int = Query(20, ge=1, le=200)
    ) -> Response:
        xml = await _runtime(request).use_cases.render_rss_feed.execute(limit)
        return Response(content=xml, media_type="application/rss+xml")

    @app.get("/runtime")
    async def runtime_status(
        request: Request, _=Depends(_check_auth)
    ) -> dict[str, object]:
        runtime_payload = await _runtime(request).use_cases.get_runtime_status.execute()
        return {
            **runtime_payload,
            "setup_view": build_runtime_view(runtime_payload),
        }

    return app
