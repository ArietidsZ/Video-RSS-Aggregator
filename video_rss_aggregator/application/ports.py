from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from video_rss_aggregator.domain.models import PreparedMedia, SummaryResult
from video_rss_aggregator.domain.outcomes import ProcessOutcome
from video_rss_aggregator.domain.publication import PublicationRecord


class RuntimeInspector(Protocol):
    async def status(self) -> dict[str, object]: ...

    async def bootstrap(self) -> list[str]: ...


class MediaPreparationService(Protocol):
    async def prepare(self, source_url: str, title: str | None) -> PreparedMedia: ...


@dataclass(frozen=True)
class FetchedFeedEntry:
    source_url: str | None
    title: str | None = None
    guid: str | None = None
    published_at: datetime | None = None


@dataclass(frozen=True)
class FetchedFeed:
    entries: tuple[FetchedFeedEntry, ...] = field(default_factory=tuple)
    title: str | None = None
    site_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))


class FeedSource(Protocol):
    async def fetch(
        self, feed_url: str, max_items: int | None = None
    ) -> FetchedFeed: ...


class FeedRepository(Protocol):
    async def save(self, feed_url: str, feed: FetchedFeed) -> None: ...


class FeedVideoRepository(Protocol):
    async def save_feed_item(self, feed_url: str, entry: FetchedFeedEntry) -> None: ...


class SourceProcessor(Protocol):
    async def execute(self, source_url: str, title: str | None) -> ProcessOutcome: ...


class Summarizer(Protocol):
    async def summarize(self, prepared_media: PreparedMedia) -> SummaryResult: ...


class VideoRepository(Protocol):
    async def save(self, media: PreparedMedia) -> str: ...


class SummaryRepository(Protocol):
    async def save(self, video_id: str, summary: SummaryResult) -> None: ...


class PublicationRepository(Protocol):
    async def latest_publications(self, limit: int) -> Sequence[PublicationRecord]: ...


class PublicationRenderer(Protocol):
    async def render(self, publications: Sequence[PublicationRecord]) -> str: ...
