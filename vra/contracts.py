"""Behavior contracts wiring v2 services together. FROZEN signatures.

Implementations live in vra/platform, vra/inference, vra/store,
vra/service. The composition root (vra/runtime.py, lead-owned) binds
concrete classes to these Protocols.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vra.types import (
    FeedRecord,
    FrameSet,
    IngestResult,
    Job,
    JobKind,
    JobState,
    MediaAsset,
    Summary,
    Transcript,
    VideoMeta,
    VideoRecord,
)


@runtime_checkable
class FeedFetcher(Protocol):
    """Fetch + parse an RSS/Atom feed over HTTP with conditional GET."""

    async def fetch(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, str | None, str | None, bool]:
        """Return (entries, feed_title, site_url, new_etag, not_modified).

        Each entry dict has keys: source_url, title, guid, published_at
        (datetime|None). not_modified=True means server returned 304 and
        entries is empty. Raises httpx.HTTPError on network failure.
        """
        ...


@runtime_checkable
class VideoAcquirer(Protocol):
    """Resolve a URL/local path into platform metadata + subtitles."""

    async def resolve(self, source: str) -> VideoMeta: ...

    async def subtitles(self, source: str, canonical_id: str) -> Transcript | None: ...

    async def download(self, source: str, canonical_id: str) -> str: ...


@runtime_checkable
class Transcriber(Protocol):
    """Optional ASR fallback for videos without subtitles."""

    async def available(self) -> bool: ...

    async def transcribe(self, media_path: str) -> Transcript: ...


@runtime_checkable
class FrameExtractor(Protocol):
    """Extract a deduplicated, quality-ranked frame set (single decode pass)."""

    async def extract(
        self,
        media_path: str,
        *,
        canonical_id: str,
        duration_s: float | None,
        max_frames: int,
    ) -> FrameSet: ...


@runtime_checkable
class Summarizer(Protocol):
    """Summarize an acquired asset with the routed local VLM."""

    async def summarize(self, asset: MediaAsset) -> Summary: ...

    async def runtime_status(self) -> dict[str, Any]: ...

    async def prepare_models(self) -> list[str]: ...


@runtime_checkable
class JobQueue(Protocol):
    """Durable asyncio job queue backed by the jobs table."""

    async def enqueue(
        self,
        kind: JobKind,
        payload: dict[str, Any],
        *,
        video_id: str | None = None,
        run_at: Any = None,
        dedup_key: str | None = None,
    ) -> Job: ...

    async def claim(self, kinds: tuple[JobKind, ...]) -> Job | None: ...

    async def complete(self, job_id: str) -> None: ...

    async def fail(self, job_id: str, error: str, *, retry: bool) -> None: ...

    async def requeue_stale(self, older_than_s: float) -> int: ...

    async def list(
        self,
        *,
        state: JobState | None = None,
        kind: JobKind | None = None,
        limit: int = 50,
    ) -> list[Job]: ...

    async def get(self, job_id: str) -> Job | None: ...


@runtime_checkable
class VideoStore(Protocol):
    """Persistence for feeds/videos/transcripts/frames/summaries."""

    async def upsert_feed(self, url: str, title: str | None, poll_minutes: int) -> FeedRecord: ...

    async def save_feed_state(
        self,
        feed_id: str,
        *,
        title: str | None = None,
        site_url: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        error: str | None = None,
    ) -> None: ...

    async def get_feed_by_url(self, url: str) -> FeedRecord | None: ...

    async def list_feeds(self) -> list[FeedRecord]: ...

    async def delete_feed(self, feed_id: str) -> None: ...

    async def upsert_video_ref(
        self,
        *,
        canonical_id: str,
        source_url: str,
        feed_id: str | None,
        guid: str | None,
        title: str | None,
        published_at: Any,
    ) -> tuple[VideoRecord, bool]:
        """Return (record, created)."""
        ...

    async def set_video_status(
        self,
        video_id: str,
        status: Any,
        *,
        error: str | None = None,
    ) -> None: ...

    async def save_meta(self, video_id: str, meta: VideoMeta, media_path: str | None) -> None: ...

    async def save_transcript(self, video_id: str, transcript: Transcript) -> None: ...

    async def save_frames(self, video_id: str, frames: FrameSet) -> None: ...

    async def save_summary(self, video_id: str, summary: Summary) -> None: ...

    async def get_video(self, video_id: str) -> VideoRecord | None: ...

    async def get_video_by_canonical(self, canonical_id: str) -> VideoRecord | None: ...

    async def latest_publications(self, limit: int) -> list[dict[str, Any]]: ...

    async def video_detail(self, video_id: str) -> dict[str, Any] | None: ...

    async def counts(self) -> dict[str, Any]: ...


@runtime_checkable
class SourceProcessor(Protocol):
    """Full per-video pipeline: acquire → transcript → frames → summary."""

    async def process(self, video_id: str, *, force: bool = False) -> str:
        """Run pipeline for video_id, persist results, return final status."""
        ...


@runtime_checkable
class IngestService(Protocol):
    """Poll a feed and enqueue process jobs for new/failed videos."""

    async def ingest(self, url: str, *, max_items: int | None = None) -> IngestResult: ...


@runtime_checkable
class FeedScheduler(Protocol):
    """Background loop ingesting every feed on its poll interval."""

    def start(self) -> None: ...

    async def stop(self) -> None: ...


__all__ = [
    "FeedFetcher",
    "FeedScheduler",
    "FrameExtractor",
    "IngestService",
    "JobQueue",
    "SourceProcessor",
    "Summarizer",
    "Transcriber",
    "VideoAcquirer",
    "VideoStore",
    "FeedRecord",
    "FrameSet",
    "IngestResult",
    "Job",
    "JobKind",
    "JobState",
    "MediaAsset",
    "Summary",
    "Transcript",
    "VideoMeta",
    "VideoRecord",
]
