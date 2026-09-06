"""Shared in-memory fakes for v2 service-layer tests (no test functions).

These implement the frozen contracts (contracts.FeedFetcher, VideoAcquirer,
Transcriber, FrameExtractor, Summarizer, IngestService) with canned behavior
so ingester/processor/scheduler tests run hermetically.
"""

from __future__ import annotations

from typing import Any

from vra.types import (
    FeedRecord,
    Frame,
    FrameSet,
    IngestResult,
    MediaAsset,
    Summary,
    Transcript,
    VideoMeta,
)


class FakeFetcher:
    """contracts.FeedFetcher fake: canned entries, optional 304 or error."""

    def __init__(
        self,
        entries: list[dict[str, Any]] | None = None,
        *,
        feed_title: str | None = "Feed Title",
        site_url: str | None = "https://example.com",
        etag: str | None = '"etag-1"',
        not_modified: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.entries = entries if entries is not None else []
        self.feed_title = feed_title
        self.site_url = site_url
        self.etag = etag
        self.not_modified = not_modified
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def fetch(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, str | None, str | None, bool]:
        self.calls.append(
            {"url": url, "etag": etag, "last_modified": last_modified}
        )
        if self.error is not None:
            raise self.error
        if self.not_modified:
            return [], None, None, None, True
        return (
            [dict(entry) for entry in self.entries],
            self.feed_title,
            self.site_url,
            self.etag,
            False,
        )


class FakeAcquirer:
    """contracts.VideoAcquirer fake with call recording."""

    def __init__(
        self,
        *,
        meta: VideoMeta | None = None,
        transcript: Transcript | None = None,
        resolve_error: Exception | None = None,
    ) -> None:
        self.meta = meta
        self.transcript = transcript
        self.resolve_error = resolve_error
        self.resolves: list[str] = []
        self.subtitle_calls: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.media_paths: list[str] = []

    async def resolve(self, source: str) -> VideoMeta:
        self.resolves.append(source)
        if self.resolve_error is not None:
            raise self.resolve_error
        if self.meta is not None:
            return self.meta
        return VideoMeta(
            canonical_id=f"cid:{source}",
            title="Meta Title",
            description="A description",
            duration_s=123.5,
            webpage_url=source,
            chapters=((0.0, "intro"), (12.0, "body")),
        )

    async def subtitles(self, source: str, canonical_id: str) -> Transcript | None:
        self.subtitle_calls.append((source, canonical_id))
        return self.transcript

    async def download(self, source: str, canonical_id: str) -> str:
        self.downloads.append((source, canonical_id))
        path = f"/media/{canonical_id}.mp4"
        self.media_paths.append(path)
        return path


class FakeTranscriber:
    """contracts.Transcriber fake."""

    def __init__(self, *, is_available: bool = True, text: str = "asr words") -> None:
        self.is_available = is_available
        self.text = text
        self.calls: list[str] = []

    async def available(self) -> bool:
        return self.is_available

    async def transcribe(self, media_path: str) -> Transcript:
        self.calls.append(media_path)
        return Transcript(text=self.text, source="asr", lang="en")


class FakeExtractor:
    """contracts.FrameExtractor fake with deterministic frames."""

    def __init__(self, *, count: int = 2, error: Exception | None = None) -> None:
        self.count = count
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def extract(
        self,
        media_path: str,
        *,
        canonical_id: str,
        duration_s: float | None,
        max_frames: int,
    ) -> FrameSet:
        self.calls.append(
            {
                "media_path": media_path,
                "canonical_id": canonical_id,
                "duration_s": duration_s,
                "max_frames": max_frames,
            }
        )
        if self.error is not None:
            raise self.error
        frames = tuple(
            Frame(
                path=f"/frames/{canonical_id}-{i}.jpg",
                ts_s=float(i) * 10.0,
                scene_score=0.5,
                dhash=f"{i:016x}",
            )
            for i in range(min(self.count, max(0, max_frames)))
        )
        return FrameSet(frames)


class FakeSummarizer:
    """contracts.Summarizer fake recording the assets it is given."""
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.assets: list[MediaAsset] = []

    async def summarize(self, asset: MediaAsset) -> Summary:
        self.assets.append(asset)
        if self.error is not None:
            raise self.error
        return Summary(
            summary=f"Summary of {asset.meta.canonical_id}",
            key_points=("point one", "point two"),
            visual_highlights=("frame 0",),
            model="fake-vlm",
            think="off",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=10,
            vram_mb=1024.0,
            transcript_chars=len(asset.transcript.text),
            frame_count=len(asset.frames.frames),
        )

    async def runtime_status(self) -> dict[str, Any]:
        return {"model": "fake-vlm", "ready": True}

    async def prepare_models(self) -> list[str]:
        return []


class FakeIngester:
    """contracts.IngestService fake for scheduler tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    async def ingest(self, url: str, *, max_items: int | None = None) -> IngestResult:
        self.calls.append((url, max_items))
        return IngestResult(
            feed=FeedRecord(id="fake", url=url),
            new_videos=0,
            enqueued=0,
            skipped_done=0,
        )


# --------------------------------------------------------------------------
# contract conformance (runtime_checkable protocols check method presence)
# --------------------------------------------------------------------------

def test_implementations_satisfy_frozen_protocols() -> None:
    """Real and fake implementations expose every protocol method."""
    from vra.config import Config
    from vra.contracts import (
        FeedFetcher,
        FeedScheduler,
        FrameExtractor,
        IngestService,
        JobQueue,
        SourceProcessor,
        Summarizer,
        Transcriber,
        VideoAcquirer,
        VideoStore,
    )
    from vra.service.ingester import FeedIngester, HttpFeedFetcher
    from vra.service.processor import Processor
    from vra.service.scheduler import PollScheduler
    from vra.store.queue import SqlJobQueue
    from vra.store.repos import SqlVideoStore

    config = Config()
    fetcher = FakeFetcher()
    acquirer = FakeAcquirer()
    transcriber = FakeTranscriber()
    extractor = FakeExtractor()
    summarizer = FakeSummarizer()
    ingester = FakeIngester()

    assert isinstance(fetcher, FeedFetcher)
    assert isinstance(acquirer, VideoAcquirer)
    assert isinstance(transcriber, Transcriber)
    assert isinstance(extractor, FrameExtractor)
    assert isinstance(summarizer, Summarizer)
    assert isinstance(ingester, IngestService)

    # Constructors only hold their dependencies, so plain sentinels suffice
    # for structural protocol checks.
    unused = object()
    store = SqlVideoStore(unused)
    queue = SqlJobQueue(unused)
    processor = Processor(store, acquirer, transcriber, extractor, summarizer, config)
    service = FeedIngester(fetcher, store, queue, config)
    scheduler = PollScheduler(ingester, store, queue, config)
    http_fetcher = HttpFeedFetcher(unused)

    assert isinstance(store, VideoStore)
    assert isinstance(queue, JobQueue)
    assert isinstance(processor, SourceProcessor)
    assert isinstance(service, IngestService)
    assert isinstance(scheduler, FeedScheduler)
    assert isinstance(http_fetcher, FeedFetcher)
