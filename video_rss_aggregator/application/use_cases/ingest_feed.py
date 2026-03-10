from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from video_rss_aggregator.application.ports import (
    FetchedFeed,
    FetchedFeedEntry,
    FeedRepository,
    FeedSource,
    FeedVideoRepository,
    SourceProcessor,
)


@dataclass(frozen=True)
class IngestReport:
    feed_title: str | None
    item_count: int
    processed_count: int


@dataclass(frozen=True)
class IngestFeed:
    feed_source: FeedSource
    feeds: FeedRepository
    videos: FeedVideoRepository
    process_source: SourceProcessor

    async def execute(
        self, feed_url: str, process: bool = False, max_items: int | None = None
    ) -> IngestReport:
        fetched_feed = await self.feed_source.fetch(feed_url, max_items=max_items)
        normalized_entries: list[FetchedFeedEntry] = []

        for entry in fetched_feed.entries:
            if entry.source_url is None:
                continue

            source_url = entry.source_url.strip()
            if not source_url:
                continue

            normalized_entries.append(
                FetchedFeedEntry(
                    source_url=source_url,
                    title=entry.title,
                    guid=entry.guid,
                )
            )

        valid_entries = tuple(normalized_entries)
        normalized_feed = FetchedFeed(
            title=fetched_feed.title,
            site_url=fetched_feed.site_url,
            entries=valid_entries,
        )

        processed_count = 0

        await self.feeds.save(feed_url, normalized_feed)

        for entry in valid_entries:
            await self.videos.save_feed_item(feed_url, entry)

            if process:
                await self.process_source.execute(
                    cast(str, entry.source_url), entry.title
                )
                processed_count += 1

        return IngestReport(
            feed_title=normalized_feed.title,
            item_count=len(valid_entries),
            processed_count=processed_count,
        )
