"""FastAPI application. OWNER: agent-D.

create_app(runtime) per docs/v2/SPEC.md API table. `runtime` is ANY object
carrying the AppRuntime facade attributes documented in vra/runtime.py
(config, store, queue, ingester, processor, scheduler, summarizer, engine,
start, close, stats) — this module never imports vra.runtime.

Design:
- Lifespan owns the runtime: `await runtime.start()` on boot, `close()` on
  shutdown (scheduler + queue workers live inside the runtime).
- Auth (Bearer or X-API-Key vs config.api_key, 401 {"detail":"unauthorized"})
  guards the mutating POST/DELETE routes; all GET routes are open.
- SSRF guard on every user-supplied fetchable URL: http/https only, resolved
  addresses must not be private/loopback/link-local/reserved unless
  config.allow_private_net. Local paths only when config.allow_local_path.
- /setup/diagnostics ports the v1 setup-studio behavior onto the engine
  posture: ffmpeg/ffprobe/yt-dlp binaries + engine model + ASR model checks.
- Static assets (setup studio) are mounted from vra/static; extracted frames
  are served at /media/{video_id}/{idx}.jpg strictly inside config.data_dir.
- Request bodies are capped at 1 MB; all errors are {"detail": str} except
  /setup/bootstrap failures which surface {"ok": false, "error": str} (502)
  per the no-fallback bootstrap contract.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import ipaddress
import logging
import re
import shutil
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import fastapi
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vra import __version__
from vra.api.gui import (
    build_diagnostics_view,
    build_runtime_view,
    render_setup_page,
    setup_payload,
)
from vra.config import AsrMode, Config
from vra.identity import source_identity
from vra.rss import RssWriter
from vra.types import JobKind, JobState

log = logging.getLogger(__name__)

_MAX_BODY_BYTES = 1_000_000
_VIDEO_ID_RE = re.compile(r"^[0-9a-f-]{36}$")
_FRAME_IDX_RE = re.compile(r"^\d+$")
_RESOLVE_TIMEOUT_S = 5.0


class _RuntimeLike(Protocol):
    """Structural type for the runtime facade consumed here."""

    config: Config

    def stats(self) -> dict[str, Any]: ...


class FeedBody(BaseModel):
    """POST /feeds body."""

    url: str
    poll_minutes: int | None = None


class IngestBody(BaseModel):
    """POST /ingest body."""

    url: str


class ProcessBody(BaseModel):
    """POST /process body (exactly one of source_url / video_id)."""

    source_url: str | None = None
    video_id: str | None = None
    force: bool = False


def canonical_for_source(source: str) -> str:
    """Cheap canonical id for a source URL/path before platform resolution."""
    return source_identity(source)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def _record_dict(record: Any) -> dict[str, Any]:
    """Dataclass -> JSON dict (enums -> values, datetimes -> ISO strings)."""
    out: dict[str, Any] = {}
    for key, value in asdict(record).items():
        if hasattr(value, "value"):  # StrEnum -> its str value
            value = value.value
        out[key] = _iso(value) if hasattr(value, "isoformat") else value
    return out


def _binaries_report() -> dict[str, dict[str, Any]]:
    """ffmpeg/ffprobe availability report (vra.platform.binaries, agent-A)."""
    from vra.platform.binaries import dependency_report

    report = dependency_report()
    return {"ffmpeg": report["ffmpeg"], "ffprobe": report["ffprobe"]}


def _yt_dlp_state() -> dict[str, Any]:
    command = shutil.which("yt-dlp")
    module_available = find_spec("yt_dlp") is not None
    return {
        "command": command,
        "module_available": module_available,
        "available": bool(command or module_available),
    }


async def _resolved_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    loop = asyncio.get_running_loop()
    infos = await asyncio.wait_for(
        loop.getaddrinfo(host, None, type=socket.SOCK_STREAM), timeout=_RESOLVE_TIMEOUT_S
    )
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _is_forbidden_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


async def ensure_public_url(url: str, *, allow_private: bool) -> None:
    """Reject non-http(s) schemes and private-network targets (SSRF guard).

    Raises HTTPException(400) when the URL must not be fetched server-side.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="url must be http(s) with a hostname")
    if allow_private:
        return
    try:
        addresses = await _resolved_addresses(parsed.hostname)
    except (socket.gaierror, OSError, TimeoutError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot resolve host {parsed.hostname!r}: {exc}") from exc
    for addr in addresses:
        if _is_forbidden_address(addr):
            raise HTTPException(
                status_code=400,
                detail=f"host {parsed.hostname!r} resolves to a non-public address ({addr})",
            )


async def validate_source_url(url: str, config: Config) -> None:
    """Policy for /process sources: public URL check or explicit local-path opt-in."""
    if config.allow_url(url, local_paths_enabled=config.allow_local_path):
        if urlparse(url).scheme in {"http", "https"}:
            await ensure_public_url(url, allow_private=config.allow_private_net)
        return
    raise HTTPException(status_code=400, detail="source must be an http(s) URL")


def _job_dict(job: Any) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": job.kind.value if hasattr(job.kind, "value") else str(job.kind),
        "state": job.state.value if hasattr(job.state, "value") else str(job.state),
        "payload": dict(job.payload),
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "run_at": _iso(job.run_at),
        "error": job.error,
        "video_id": job.video_id,
        "created_at": _iso(job.created_at),
        "updated_at": _iso(job.updated_at),
    }


def create_app(runtime: _RuntimeLike) -> FastAPI:
    """Build the v2 API around an already-constructed runtime facade."""
    config: Config = runtime.config

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = fastapi.FastAPI(
        title="Video RSS Aggregator",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next: Any) -> Response:
        content_length = request.headers.get("content-length", "")
        if content_length.isdigit() and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
        return await call_next(request)

    def require_auth(
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None),
    ) -> None:
        if config.api_key is None:
            return
        token: str | None = None
        if authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        if token is None:
            token = x_api_key
        if token != config.api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    static_ref = importlib.resources.files("vra").joinpath("static")
    with importlib.resources.as_file(static_ref) as static_dir:
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ---------------------------------------------------------------- health

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # ----------------------------------------------------------- setup studio

    @app.get("/", response_class=HTMLResponse)
    async def setup_home(request: Request) -> Response:
        html = await asyncio.to_thread(
            render_setup_page, config, request.scope.get("root_path", "")
        )
        return HTMLResponse(html)

    @app.get("/setup/config")
    async def setup_config() -> dict[str, Any]:
        return setup_payload(config)

    @app.get("/setup/diagnostics")
    async def setup_diagnostics() -> dict[str, Any]:
        binaries_report = await asyncio.to_thread(_binaries_report)
        ytdlp = _yt_dlp_state()

        engine: dict[str, Any] = {
            "base_url": config.engine_base_url,
            "model": config.model,
            "reachable": False,
            "models": [],
            "model_present": False,
            "error": None,
        }
        asr: dict[str, Any] = {
            "base_url": config.effective_asr_base_url,
            "model": config.asr_model,
            "enabled": config.asr is AsrMode.AUTO,
            "reachable": False,
            "models": [],
            "model_present": False,
            "error": None,
        }
        try:
            status = await runtime.summarizer.runtime_status()
            engine_block = status.get("engine") or {}
            asr_block = status.get("asr") or {}
            engine.update(engine_block)
            asr.update(asr_block)
            engine.setdefault("reachable", False)
            asr.setdefault("model_present", False)
        except Exception as exc:
            log.warning("engine/ASR status check failed: %s", exc)
            engine["error"] = str(exc)
            asr["error"] = str(exc)

        ffmpeg_ok = bool(binaries_report.get("ffmpeg", {}).get("available"))
        ffprobe_ok = bool(binaries_report.get("ffprobe", {}).get("available"))
        ytdlp_ok = bool(ytdlp["available"])
        engine_ok = bool(engine["model_present"])
        asr_ok = asr["model_present"] or config.asr is AsrMode.OFF

        report = {
            "platform": {
                "system": sys.platform,
                "python_version": sys.version.split()[0],
                "python_executable": sys.executable,
            },
            "dependencies": {
                "ffmpeg": binaries_report.get("ffmpeg"),
                "ffprobe": binaries_report.get("ffprobe"),
                "yt_dlp": ytdlp,
                "engine": engine,
                "asr": asr,
            },
            "ready": ffmpeg_ok and ffprobe_ok and ytdlp_ok and engine_ok and asr_ok,
        }
        report["setup_view"] = build_diagnostics_view(report, asr_enabled=asr["enabled"])
        report["runtime_view"] = build_runtime_view(engine, asr)
        return report

    @app.post("/setup/bootstrap", dependencies=[Depends(require_auth)])
    async def setup_bootstrap() -> Response:
        try:
            prepared = await runtime.summarizer.prepare_models()
        except Exception as exc:
            log.warning("bootstrap failed: %s", exc)
            return JSONResponse(status_code=502, content={"ok": False, "error": str(exc)})
        return JSONResponse(content={"ok": True, "models_prepared": list(prepared)})

    # ------------------------------------------------------------------ feeds

    @app.post("/feeds", status_code=201, dependencies=[Depends(require_auth)])
    async def add_feed(body: FeedBody) -> dict[str, Any]:
        await ensure_public_url(body.url, allow_private=config.allow_private_net)
        record = await runtime.store.upsert_feed(
            body.url, None, body.poll_minutes or config.poll_minutes
        )
        job = await runtime.queue.enqueue(JobKind.INGEST, {"url": body.url})
        return {**_record_dict(record), "job_id": job.id}

    @app.get("/feeds")
    async def list_feeds() -> list[dict[str, Any]]:
        return [_record_dict(record) for record in await runtime.store.list_feeds()]

    @app.delete("/feeds/{feed_id}", status_code=204, dependencies=[Depends(require_auth)])
    async def remove_feed(feed_id: str) -> Response:
        await runtime.store.delete_feed(feed_id)
        return Response(status_code=204)

    # ------------------------------------------------------- ingest / process

    @app.post("/ingest", dependencies=[Depends(require_auth)])
    async def ingest_now(body: IngestBody) -> dict[str, Any]:
        await ensure_public_url(body.url, allow_private=config.allow_private_net)
        job = await runtime.queue.enqueue(JobKind.INGEST, {"url": body.url})
        return {"job_id": job.id}

    @app.post("/process", dependencies=[Depends(require_auth)])
    async def process_source(body: ProcessBody) -> dict[str, Any]:
        if body.video_id:
            video_id = body.video_id
        elif body.source_url:
            body.source_url = body.source_url.strip()
            await validate_source_url(body.source_url, config)
            canonical = canonical_for_source(body.source_url)
            record = await runtime.store.get_video_by_canonical(canonical)
            if record is None:
                record, _created = await runtime.store.upsert_video_ref(
                    canonical_id=canonical,
                    source_url=body.source_url,
                    feed_id=None,
                    guid=None,
                    title=None,
                    published_at=None,
                )
            video_id = record.id
        else:
            raise HTTPException(status_code=400, detail="source_url or video_id required")

        job = await runtime.queue.enqueue(
            JobKind.PROCESS,
            {"video_id": video_id, **({"force": True} if body.force else {})},
            video_id=video_id,
        )
        return {"job_id": job.id, "video_id": video_id}

    # ------------------------------------------------------------- jobs/videos

    @app.get("/jobs")
    async def list_jobs(
        state: str | None = Query(None),
        kind: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        state_filter: JobState | None = None
        if state:
            try:
                state_filter = JobState(state)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"unknown state {state!r}") from exc
        kind_filter: JobKind | None = None
        if kind:
            try:
                kind_filter = JobKind(kind)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"unknown kind {kind!r}") from exc
        jobs = await runtime.queue.list(
            state=state_filter, kind=kind_filter, limit=limit
        )
        return [_job_dict(job) for job in jobs]

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = await runtime.queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_dict(job)

    @app.get("/videos")
    async def list_videos(
        limit: int = Query(50, ge=1, le=200),
        feed_id: str | None = Query(None),
    ) -> list[dict[str, Any]]:
        items = await runtime.store.latest_publications(limit)
        if feed_id is not None:
            # latest_publications() has no feed filter (frozen contract); filter
            # in-memory when the store includes feed_id on publication rows.
            items = [i for i in items if i.get("feed_id") == feed_id]
        return items

    @app.get("/videos/{video_id}")
    async def get_video(video_id: str) -> dict[str, Any]:
        detail = await runtime.store.video_detail(video_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="video not found")
        return detail

    # ------------------------------------------------------------------ output

    @app.get("/rss")
    async def rss_feed(limit: int = Query(20, ge=1, le=200)) -> Response:
        items = await runtime.store.latest_publications(limit)
        xml = RssWriter(config).render(items)
        return Response(content=xml, media_type="application/rss+xml")

    @app.get("/media/{video_id}/{filename}")
    async def media_frame(video_id: str, filename: str) -> FileResponse:
        if not _VIDEO_ID_RE.fullmatch(video_id):
            raise HTTPException(status_code=404, detail="video not found")
        match = _FRAME_IDX_RE.fullmatch(filename.removesuffix(".jpg"))
        if match is None or not filename.endswith(".jpg"):
            raise HTTPException(status_code=404, detail="frame not found")
        idx = int(match.group(0))

        detail = await runtime.store.video_detail(video_id)
        frames = (detail or {}).get("frames") or []
        if idx >= len(frames):
            raise HTTPException(status_code=404, detail="frame not found")
        raw_path = str(frames[idx].get("path", ""))

        base = Path(config.data_dir).resolve()
        path = Path(raw_path)
        if not path.is_absolute():
            path = base / path
        resolved = path.resolve()
        if not resolved.is_relative_to(base):
            log.warning("blocked frame path outside data dir: %s", raw_path)
            raise HTTPException(status_code=404, detail="frame not found")
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(resolved, media_type="image/jpeg")

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        counts = await runtime.store.counts()
        return {**counts, "model": config.model, "runtime": runtime.stats()}

    return app
