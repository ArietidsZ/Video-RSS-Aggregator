"""Composition root (lead-owned). Binds concrete implementations to contracts.

AppRuntime facade consumed by vra/api/app.py and vra/cli.py:
    config, store, queue, ingester, processor, scheduler, summarizer
    start() / close() / stats()

No-fallback engine posture: exactly ONE inference path — an
OpenAI-compatible server (vLLM / SGLang / oMLX / llama-server / frontier
APIs) reached via EngineClient; one ASR path — /v1/audio/transcriptions
hosting Qwen3-ASR (separate base URL allowed). All quantization and VRAM
management are server-side; the app owns routing, budgets, quality gates.

Worker model:
- `cpu_workers` identical workers pull jobs from the durable queue. CPU/IO
  stages (yt-dlp download, ffmpeg decode) run concurrently across workers.
- Inference concurrency is managed by the serving engine (continuous
  batching); the app does not gate it further.
- On boot, running jobs stale for > 30 minutes are requeued.
- Normal worker cancellation records a retry (or terminal failure at the
  existing attempt limit) before SQLite closes. This is not a heartbeat or
  recovery guarantee for forced termination, repeated cancellation or DB errors.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from vra.config import Config
from vra.contracts import Job, JobKind
from vra.inference.engine import EngineClient
from vra.inference.summarize import VlmSummarizer
from vra.platform.asr import OpenAiAsrTranscriber
from vra.platform.media import FfmpegFrameExtractor
from vra.platform.ytdlp import YtDlpAcquirer
from vra.service.ingester import FeedIngester, HttpFeedFetcher
from vra.service.processor import Processor
from vra.service.scheduler import PollScheduler
from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.store.repos import SqlVideoStore
from vra.types import VideoStatus

log = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0


async def _run_job(job: Job, coro: Any, queue: SqlJobQueue, store: SqlVideoStore) -> None:
    try:
        status = await coro
        if job.kind is JobKind.PROCESS and status == "failed":
            detail = await _video_error(store, str(job.payload.get("video_id", "")))
            await queue.fail(job.id, detail or "processing failed", retry=True)
        else:
            await queue.complete(job.id)
    except asyncio.CancelledError:
        # Normal shutdown waits for workers before closing SQLite. Persist the
        # interruption here so a quick restart cannot strand a fresh running
        # job behind the startup-only 30-minute stale threshold.
        detail = "job cancelled during execution"
        if job.kind is JobKind.PROCESS:
            video_id = str(job.payload.get("video_id", ""))
            try:
                rec = await store.get_video(video_id)
                if rec is not None and rec.status in (
                    VideoStatus.ACQUIRING, VideoStatus.SUMMARIZING,
                ):
                    await store.set_video_status(video_id, VideoStatus.FAILED, error=detail)
            except Exception:
                log.exception("could not mark interrupted video %s as failed", video_id)
        await queue.fail(job.id, detail, retry=True)
        raise
    except Exception as exc:
        log.warning("job %s failed: %s", job.id, exc)
        await queue.fail(job.id, str(exc), retry=True)


async def _video_error(store: SqlVideoStore, video_id: str) -> str | None:
    try:
        rec = await store.get_video(video_id)
        return rec.error if rec else None
    except Exception:
        return None


async def _ingest_worker(
    queue: SqlJobQueue,
    store: SqlVideoStore,
    ingester: FeedIngester,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            job = await queue.claim((JobKind.INGEST,))
            if job is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
                continue
            await _run_job(
                job,
                ingester.ingest(str(job.payload.get("url", ""))),
                queue,
                store,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # queue/db level failure — back off
            log.error("ingest worker error: %s", exc)
            await asyncio.sleep(_POLL_INTERVAL_S * 5)


async def _process_worker(
    queue: SqlJobQueue,
    store: SqlVideoStore,
    processor: Processor,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            job = await queue.claim((JobKind.PROCESS,))
            if job is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
                continue
            await _run_job(
                job,
                processor.process(
                    str(job.payload.get("video_id", "")),
                    force=bool(job.payload.get("force", False)),
                ),
                queue,
                store,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("process worker error: %s", exc)
            await asyncio.sleep(_POLL_INTERVAL_S * 5)


@dataclass(frozen=True)
class AppRuntime:
    config: Config
    store: SqlVideoStore
    queue: SqlJobQueue
    ingester: FeedIngester
    processor: Processor
    scheduler: PollScheduler
    summarizer: VlmSummarizer
    engine: EngineClient
    _feed_http: httpx.AsyncClient | None = field(default=None, repr=False, compare=False)
    _transcriber: OpenAiAsrTranscriber | None = field(default=None, repr=False, compare=False)
    _state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    async def start(self) -> None:
        if self._state.get("started"):
            return
        self._state["started"] = True
        stop = asyncio.Event()
        self._state["stop"] = stop
        tasks: list[asyncio.Task[Any]] = []
        stale = await self.queue.requeue_stale(older_than_s=1800.0)
        if stale:
            log.info("requeued %d stale jobs", stale)
        tasks.append(
            asyncio.create_task(
                _ingest_worker(self.queue, self.store, self.ingester, stop),
                name="vra-ingest-worker",
            )
        )
        for i in range(self.config.cpu_workers):
            tasks.append(
                asyncio.create_task(
                    _process_worker(self.queue, self.store, self.processor, stop),
                    name=f"vra-process-worker-{i}",
                )
            )
        self._state["tasks"] = tasks
        self.scheduler.start()

    async def close(self) -> None:
        stop: asyncio.Event | None = self._state.get("stop")
        if stop is not None:
            stop.set()
        tasks: list[asyncio.Task[Any]] = self._state.get("tasks", [])
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._state["tasks"] = []
        self._state.pop("stop", None)
        self._state["started"] = False
        # These clients are created by build_runtime, not owned by the
        # summarizer. Unwind every owned resource even if another close fails.
        async with contextlib.AsyncExitStack() as cleanup:
            cleanup.push_async_callback(self.queue.close)
            if self._feed_http is not None:
                cleanup.push_async_callback(self._feed_http.aclose)
            if self._transcriber is not None:
                cleanup.push_async_callback(self._transcriber.close)
            cleanup.push_async_callback(self.summarizer.close)
            await self.scheduler.stop()

    def stats(self) -> dict[str, Any]:
        return {
            "workers": self.config.cpu_workers,
            "started": bool(self._state.get("started")),
        }


async def build_runtime(config: Config | None = None) -> AppRuntime:
    resolved = config or Config.from_env()
    database = await Database.connect(resolved.database_path)
    await database.migrate()

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=300.0, write=300.0, pool=60.0),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )

    store = SqlVideoStore(database)
    queue = SqlJobQueue(database)
    engine = EngineClient(
        resolved.engine_base_url,
        resolved.engine_api_key,
        timeout_s=float(resolved.timeout_infer_s),
    )
    asr_probe = (
        engine
        if resolved.effective_asr_base_url == resolved.engine_base_url
        else EngineClient(resolved.effective_asr_base_url, resolved.engine_api_key)
    )
    summarizer = VlmSummarizer(resolved, engine, asr_client=asr_probe)
    acquirer = YtDlpAcquirer(resolved)
    transcriber = OpenAiAsrTranscriber(resolved)
    extractor = FfmpegFrameExtractor(resolved)
    processor = Processor(store, acquirer, transcriber, extractor, summarizer, resolved)

    fetcher = HttpFeedFetcher(http, allow_private=resolved.allow_private_net)
    ingester = FeedIngester(fetcher, store, queue, resolved)
    scheduler = PollScheduler(ingester, store, queue, resolved)

    return AppRuntime(
        config=resolved,
        store=store,
        queue=queue,
        ingester=ingester,
        processor=processor,
        scheduler=scheduler,
        summarizer=summarizer,
        engine=engine,
        _feed_http=http,
        _transcriber=transcriber,
    )
