from __future__ import annotations

import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.util import find_spec

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from adapter_gui import render_setup_page
from core_config import Config
from service_media import runtime_dependency_report
from service_pipeline import Pipeline


class IngestRequest(BaseModel):
    feed_url: str
    process: bool = False
    max_items: int | None = None


class ProcessRequest(BaseModel):
    source_url: str
    title: str | None = None


def create_app(pipeline: Pipeline, config: Config) -> FastAPI:
    app = FastAPI(title="Video RSS Aggregator", version="0.1.0")

    def _check_auth(
        authorization: str | None = Header(None), x_api_key: str | None = Header(None)
    ):
        if config.api_key is None:
            return
        token = None
        if authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        if token is None:
            token = x_api_key
        if token != config.api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/", response_class=HTMLResponse)
    async def setup_home():
        return render_setup_page(config)

    @app.get("/setup/config")
    async def setup_config():
        return {
            "bind_address": f"{config.bind_host}:{config.bind_port}",
            "storage_dir": config.storage_dir,
            "database_path": config.database_path,
            "ollama_base_url": config.ollama_base_url,
            "model_priority": list(config.model_priority),
            "vram_budget_mb": config.vram_budget_mb,
            "model_selection_reserve_mb": config.model_selection_reserve_mb,
            "max_frames": config.max_frames,
            "frame_scene_detection": config.frame_scene_detection,
            "frame_scene_threshold": config.frame_scene_threshold,
            "frame_scene_min_frames": config.frame_scene_min_frames,
            "api_key_required": config.api_key is not None,
            "quick_commands": {
                "bootstrap": "python -m vra bootstrap",
                "status": "python -m vra status",
                "serve": "python -m vra serve --bind 127.0.0.1:8080",
            },
        }

    @app.get("/setup/diagnostics")
    async def setup_diagnostics():
        media_tools = runtime_dependency_report()
        yt_dlp_cmd = shutil.which("yt-dlp")
        ytdlp = {
            "command": yt_dlp_cmd,
            "module_available": find_spec("yt_dlp") is not None,
        }
        ytdlp["available"] = bool(ytdlp["command"] or ytdlp["module_available"])

        ollama: dict[str, object] = {
            "base_url": config.ollama_base_url,
            "reachable": False,
            "version": None,
            "models_found": 0,
            "error": None,
        }
        try:
            runtime = await pipeline.runtime_status()
            ollama["reachable"] = True
            ollama["version"] = runtime.get("ollama_version")
            local_models = runtime.get("local_models", {})
            ollama["models_found"] = len(local_models)
        except Exception as exc:
            ollama["error"] = str(exc)

        ffmpeg_ok = bool(media_tools["ffmpeg"].get("available"))
        ffprobe_ok = bool(media_tools["ffprobe"].get("available"))
        ytdlp_ok = bool(ytdlp["available"])
        ollama_ok = bool(ollama["reachable"])

        return {
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

    @app.post("/setup/bootstrap")
    async def setup_bootstrap(_=Depends(_check_auth)):
        return await pipeline.bootstrap_models()

    @app.post("/ingest")
    async def ingest(req: IngestRequest, _=Depends(_check_auth)):
        report = await pipeline.ingest_feed(req.feed_url, req.process, req.max_items)
        return {
            "feed_title": report.feed_title,
            "item_count": report.item_count,
            "processed_count": report.processed_count,
        }

    @app.post("/process")
    async def process(req: ProcessRequest, _=Depends(_check_auth)):
        report = await pipeline.process_source(req.source_url, req.title)
        return {
            "source_url": report.source_url,
            "title": report.title,
            "transcript_chars": report.transcript_chars,
            "frame_count": report.frame_count,
            "summary": {
                "summary": report.summary.summary,
                "key_points": report.summary.key_points,
                "visual_highlights": report.summary.visual_highlights,
                "model_used": report.summary.model_used,
                "vram_mb": report.summary.vram_mb,
                "error": report.summary.error,
            },
        }

    @app.get("/rss")
    async def rss_feed(limit: int = Query(20, ge=1, le=200)):
        xml = await pipeline.rss_feed(limit)
        return Response(content=xml, media_type="application/rss+xml")

    @app.get("/runtime")
    async def runtime(_=Depends(_check_auth)):
        return await pipeline.runtime_status()

    return app
