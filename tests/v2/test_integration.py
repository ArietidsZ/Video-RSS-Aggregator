"""Integration tests (lead-owned): real store/queue/services + fakes for
acquisition/ASR/frames/summarizer, driven through the queue the way the
runtime workers drive it. Hermetic: no network, no engine, no ffmpeg.

Activates fully once the store+service slice lands; skips at module level
while vra.store/vra.service are still stubs."""

from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

try:
    from vra.config import AsrMode, Config
    from vra.contracts import JobKind
    from vra.runtime import AppRuntime, _run_job
    from vra.service.ingester import FeedIngester
    from vra.service.processor import Processor
    from vra.store.db import Database
    from vra.store.queue import SqlJobQueue
    from vra.store.repos import SqlVideoStore
    from vra.types import (
        Frame,
        FrameSet,
        JobState,
        MediaAsset,
        Summary,
        Transcript,
        VideoMeta,
    )
except ImportError:  # slice not merged yet
    pytest.skip("store+service slice not merged", allow_module_level=True)


def _cfg(tmp_path, **overrides):
    base = Config.from_env(env={})
    return dc.replace(
        base, data_dir=str(tmp_path), database_path=str(tmp_path / "t.db"), **overrides
    )


@dataclass
class FakeFetcher:
    entries: list[dict[str, object]]
    not_modified: bool = False

    async def fetch(self, url, *, etag=None, last_modified=None):
        if self.not_modified:
            return [], None, None, None, True
        return self.entries, "FakeFeed", "https://example.com", "etag-1", False


@dataclass
class FakeAcquirer:
    downloads: int = 0

    async def resolve(self, source: str) -> VideoMeta:
        return VideoMeta(
            canonical_id="fake:video1", title="T", description="D",
            duration_s=12.0, webpage_url=source,
        )

    async def subtitles(self, source: str, canonical_id: str) -> Transcript | None:
        return Transcript(text="hello world. " * 20, source="subtitle", lang="en")

    async def download(self, source: str, canonical_id: str) -> str:
        self.downloads += 1
        return "/tmp/nonexistent.mp4"


@dataclass
class NoSubAcquirer:
    downloads: int = 0

    async def resolve(self, source):
        return VideoMeta(canonical_id="fake:v2", title="T", duration_s=5.0)

    async def subtitles(self, source, canonical_id):
        return None

    async def download(self, source, canonical_id):
        self.downloads += 1
        return "/tmp/x.mp4"


@dataclass
class FakeTranscriber:
    available_flag: bool = False

    async def available(self) -> bool:
        return self.available_flag

    async def transcribe(self, media_path: str) -> Transcript:
        return Transcript(text="asr text", source="asr")


@dataclass
class FakeExtractor:
    async def extract(self, media_path, *, canonical_id, duration_s, max_frames):
        return FrameSet(
            frames=(
                Frame(path="/tmp/f0.jpg", ts_s=0.0, scene_score=0.5, dhash="ab" * 8),
                Frame(path="/tmp/f1.jpg", ts_s=1.0, scene_score=0.1, dhash="cd" * 8),
            )
        )


@dataclass
class FakeSummarizer:
    calls: int = 0
    fail: bool = False

    async def summarize(self, asset: MediaAsset) -> Summary:
        self.calls += 1
        if self.fail:
            raise RuntimeError("engine down")
        return Summary(
            summary="A" * 60, key_points=("p1", "p2"), visual_highlights=(),
            model="qwen3.5-9b",
        )

    async def runtime_status(self):
        return {}

    async def prepare_models(self):
        return ["qwen3.5-9b"]

    async def close(self):
        return None


@dataclass
class FakeScheduler:
    started: bool = False

    def start(self):
        self.started = True

    async def stop(self):
        self.started = False


async def _make_runtime(tmp_path, fetcher, acquirer, transcriber, extractor,
                        summarizer, **cfg_overrides):
    cfg = _cfg(tmp_path, **cfg_overrides)
    db = await Database.connect(cfg.database_path)
    await db.migrate()
    store = SqlVideoStore(db)
    queue = SqlJobQueue(db)
    runtime = AppRuntime(
        config=cfg,
        store=store,
        queue=queue,
        ingester=FeedIngester(fetcher, store, queue, cfg),
        processor=Processor(store, acquirer, transcriber, extractor, summarizer, cfg),
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        summarizer=summarizer,  # type: ignore[arg-type]
    )
    return runtime


async def _pump(runtime: AppRuntime) -> None:
    """Run queued jobs to completion exactly the way the workers would."""
    while True:
        job = await runtime.queue.claim((JobKind.INGEST, JobKind.PROCESS))
        if job is None:
            return
        if job.kind is JobKind.INGEST:
            coro = runtime.ingester.ingest(str(job.payload.get("url", "")))
        else:
            coro = runtime.processor.process(
                str(job.payload.get("video_id", "")),
                force=bool(job.payload.get("force", False)),
            )
        await _run_job(job, coro, runtime.queue, runtime.store)


_ENTRY = {"source_url": "https://example.com/watch?v=1", "title": "E1",
          "guid": "https://example.com/watch?v=1", "published_at": datetime.now(UTC)}


async def test_end_to_end_ingest_process_idempotent(tmp_path):
    acquirer = FakeAcquirer()
    summarizer = FakeSummarizer()
    runtime = await _make_runtime(
        tmp_path, FakeFetcher(entries=[_ENTRY, {**_ENTRY, "source_url": "",
                                                "guid": None}]),
        acquirer, FakeTranscriber(), FakeExtractor(), summarizer,
    )
    try:
        await runtime.queue.enqueue(
            JobKind.INGEST, {"url": "https://example.com/feed.xml"})
        await _pump(runtime)

        feeds = await runtime.store.list_feeds()
        assert len(feeds) == 1 and feeds[0].title == "FakeFeed"
        counts = await runtime.store.counts()
        assert counts["videos_by_status"].get("pending", 0) == 1  # blank skipped
        assert counts["jobs_by_state"].get("queued", 0) == 1

        await _pump(runtime)

        counts = await runtime.store.counts()
        assert counts["videos_by_status"].get("done", 0) == 1
        assert counts["summaries"] == 1
        assert summarizer.calls == 1
        assert acquirer.downloads == 1
        pubs = await runtime.store.latest_publications(10)
        assert len(pubs) == 1 and pubs[0]["summary"] == "A" * 60

        # idempotency: re-ingest never reprocesses a done video
        before = summarizer.calls
        await runtime.queue.enqueue(
            JobKind.INGEST, {"url": "https://example.com/feed.xml"})
        await _pump(runtime)
        assert summarizer.calls == before
    finally:
        await runtime.queue.close()


async def test_failure_marks_failed_and_requeues_with_backoff(tmp_path):
    summarizer = FakeSummarizer(fail=True)
    runtime = await _make_runtime(
        tmp_path,
        FakeFetcher(entries=[{**_ENTRY, "source_url": "https://e.com/v2",
                              "guid": "u2", "title": "E2"}]),
        FakeAcquirer(), FakeTranscriber(), FakeExtractor(), summarizer,
    )
    try:
        await runtime.queue.enqueue(
            JobKind.INGEST, {"url": "https://example.com/feed.xml"})
        await _pump(runtime)  # ingest
        await _pump(runtime)  # process attempt 1 fails

        counts = await runtime.store.counts()
        assert counts["videos_by_status"].get("failed", 0) == 1
        queued = await runtime.queue.list(state=JobState.QUEUED, limit=5)
        assert len(queued) == 1 and queued[0].attempts == 1
        assert queued[0].run_at is not None  # backoff scheduled
        assert not await runtime.queue.list(state=JobState.FAILED, limit=5)
    finally:
        await runtime.queue.close()


async def test_asr_used_only_when_no_subtitles(tmp_path):
    acquirer = NoSubAcquirer()
    transcriber = FakeTranscriber(available_flag=True)
    summarizer = FakeSummarizer()
    runtime = await _make_runtime(
        tmp_path,
        FakeFetcher(entries=[{"source_url": "https://e.com/v", "title": "E",
                              "guid": "g", "published_at": None}]),
        acquirer, transcriber, FakeExtractor(), summarizer,
    )
    try:
        await runtime.queue.enqueue(JobKind.INGEST, {"url": "https://e.com/f.xml"})
        await _pump(runtime)
        await _pump(runtime)
        assert acquirer.downloads == 1  # one download serves ASR + frames
        counts = await runtime.store.counts()
        assert counts["videos_by_status"].get("done", 0) == 1
    finally:
        await runtime.queue.close()


async def test_asr_off_still_completes_without_transcript(tmp_path):
    acquirer = NoSubAcquirer()
    summarizer = FakeSummarizer()
    runtime = await _make_runtime(
        tmp_path,
        FakeFetcher(entries=[{"source_url": "https://e.com/v3", "title": "E",
                              "guid": "g3", "published_at": None}]),
        acquirer, FakeTranscriber(), FakeExtractor(), summarizer,
        asr=AsrMode.OFF,
    )
    try:
        await runtime.queue.enqueue(JobKind.INGEST, {"url": "https://e.com/f.xml"})
        await _pump(runtime)
        await _pump(runtime)
        # max_frames > 0 => download still required for frames; transcript empty
        assert acquirer.downloads == 1
        counts = await runtime.store.counts()
        assert counts["videos_by_status"].get("done", 0) == 1
    finally:
        await runtime.queue.close()
