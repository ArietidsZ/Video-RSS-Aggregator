from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from video_rss_aggregator.application.ports import FetchedFeed, FetchedFeedEntry
from video_rss_aggregator.domain.models import PreparedMedia
from video_rss_aggregator.domain.models import SummaryResult
from video_rss_aggregator.domain.publication import PublicationRecord
from video_rss_aggregator.storage import Database


@dataclass(frozen=True)
class SQLiteFeedRepository:
    database: Database

    async def save(self, feed_url: str, feed: FetchedFeed) -> None:
        await self.database.upsert_feed(feed_url, feed.title)


@dataclass(frozen=True)
class SQLiteFeedVideoRepository:
    database: Database

    async def save_feed_item(self, feed_url: str, entry: FetchedFeedEntry) -> None:
        if entry.source_url is None:
            raise ValueError("source_url is required")
        feed_id = await self.database.upsert_feed(feed_url, None)
        await self.database.upsert_video(
            feed_id=feed_id,
            guid=entry.guid,
            title=entry.title,
            source_url=entry.source_url,
            published_at=entry.published_at,
        )


@dataclass(frozen=True)
class SQLiteVideoRepository:
    database: Database

    async def save(self, media: PreparedMedia) -> str:
        video_id = await self.database.upsert_video(
            feed_id=None,
            guid=None,
            title=media.title,
            source_url=media.source_url,
            published_at=None,
            media_path=media.media_path,
        )
        if media.transcript:
            await self.database.insert_transcript(video_id, media.transcript)
        return video_id


@dataclass(frozen=True)
class SQLiteSummaryRepository:
    database: Database

    async def save(self, video_id: str, summary: SummaryResult) -> None:
        await self.database.insert_summary(video_id, summary)


@dataclass(frozen=True)
class SQLitePublicationRepository:
    database: Database

    async def latest_publications(self, limit: int) -> Sequence[PublicationRecord]:
        return await self.database.latest_publications(limit)
