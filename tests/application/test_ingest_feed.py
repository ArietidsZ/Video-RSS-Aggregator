from datetime import datetime, timezone

import pytest

from video_rss_aggregator.application.ports import FetchedFeed, FetchedFeedEntry
from video_rss_aggregator.application.use_cases.ingest_feed import IngestFeed
from video_rss_aggregator.domain.outcomes import Failure


class FakeFeedSource:
    async def fetch(self, feed_url: str, max_items: int | None = None):
        entries = (
            FetchedFeedEntry(source_url="https://example.com/1", title="One", guid="1"),
            FetchedFeedEntry(source_url="https://example.com/2", title="Two", guid="2"),
        )
        return FetchedFeed(
            title="Example Feed",
            site_url="https://example.com",
            entries=entries[:max_items] if max_items is not None else entries,
        )


class FakeFeedRepository:
    def __init__(self) -> None:
        self.saved: list[tuple[str, FetchedFeed]] = []

    async def save(self, feed_url: str, feed: FetchedFeed) -> None:
        self.saved.append((feed_url, feed))


class FakeVideoRepository:
    def __init__(self) -> None:
        self.saved: list[tuple[str, FetchedFeedEntry]] = []

    async def save_feed_item(self, feed_url: str, entry: FetchedFeedEntry) -> None:
        self.saved.append((feed_url, entry))


class FakeProcessSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.results: dict[str, object] = {}

    async def execute(self, source_url: str, title: str | None):
        self.calls.append((source_url, title))
        return self.results.get(source_url)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_ingest_feed_tracks_processed_items() -> None:
    feeds = FakeFeedRepository()
    videos = FakeVideoRepository()
    process_source = FakeProcessSource()
    use_case = IngestFeed(
        feed_source=FakeFeedSource(),
        feeds=feeds,
        videos=videos,
        process_source=process_source,
    )

    report = await use_case.execute(
        "https://example.com/feed.xml", process=True, max_items=1
    )

    assert report.item_count == 1
    assert report.processed_count == 1
    assert report.feed_title == "Example Feed"
    assert feeds.saved == [
        (
            "https://example.com/feed.xml",
            FetchedFeed(
                title="Example Feed",
                site_url="https://example.com",
                entries=(
                    FetchedFeedEntry(
                        source_url="https://example.com/1", title="One", guid="1"
                    ),
                ),
            ),
        )
    ]
    assert videos.saved == [
        (
            "https://example.com/feed.xml",
            FetchedFeedEntry(source_url="https://example.com/1", title="One", guid="1"),
        )
    ]
    assert process_source.calls == [("https://example.com/1", "One")]


class FakeFeedSourceWithInvalidEntries:
    async def fetch(self, feed_url: str, max_items: int | None = None):
        return FetchedFeed(
            title=None,
            site_url=None,
            entries=(
                FetchedFeedEntry(source_url="", title="Blank", guid="blank"),
                FetchedFeedEntry(
                    source_url="https://example.com/valid", title=None, guid="ok"
                ),
                FetchedFeedEntry(source_url="   ", title="Whitespace", guid="space"),
            ),
        )


@pytest.mark.anyio
async def test_ingest_feed_skips_entries_without_source_url() -> None:
    feeds = FakeFeedRepository()
    videos = FakeVideoRepository()
    process_source = FakeProcessSource()
    use_case = IngestFeed(
        feed_source=FakeFeedSourceWithInvalidEntries(),
        feeds=feeds,
        videos=videos,
        process_source=process_source,
    )

    report = await use_case.execute("https://example.com/feed.xml", process=True)

    assert report.item_count == 1
    assert report.processed_count == 1
    assert report.feed_title is None
    assert feeds.saved == [
        (
            "https://example.com/feed.xml",
            FetchedFeed(
                title=None,
                site_url=None,
                entries=(
                    FetchedFeedEntry(
                        source_url="https://example.com/valid", title=None, guid="ok"
                    ),
                ),
            ),
        )
    ]
    assert videos.saved == [
        (
            "https://example.com/feed.xml",
            FetchedFeedEntry(
                source_url="https://example.com/valid", title=None, guid="ok"
            ),
        )
    ]
    assert process_source.calls == [("https://example.com/valid", None)]


@pytest.mark.anyio
async def test_ingest_feed_counts_only_non_failure_results_as_processed() -> None:
    feeds = FakeFeedRepository()
    videos = FakeVideoRepository()
    process_source = FakeProcessSource()
    process_source.results = {
        "https://example.com/2": Failure(
            source_url="https://example.com/2", reason="download failed"
        )
    }
    use_case = IngestFeed(
        feed_source=FakeFeedSource(),
        feeds=feeds,
        videos=videos,
        process_source=process_source,
    )

    report = await use_case.execute(
        "https://example.com/feed.xml", process=True, max_items=2
    )

    assert report.item_count == 2
    assert report.processed_count == 1


class FakeFeedSourceWithPublishedEntries:
    async def fetch(self, feed_url: str, max_items: int | None = None):
        return FetchedFeed(
            title="Published Feed",
            site_url="https://example.com",
            entries=(
                FetchedFeedEntry(
                    source_url="https://example.com/published",
                    title="Published item",
                    guid="published-guid",
                    published_at=datetime(2024, 1, 2, 3, 4, tzinfo=timezone.utc),
                ),
            ),
        )


@pytest.mark.anyio
async def test_ingest_feed_preserves_publication_timestamps() -> None:
    feeds = FakeFeedRepository()
    videos = FakeVideoRepository()
    process_source = FakeProcessSource()
    use_case = IngestFeed(
        feed_source=FakeFeedSourceWithPublishedEntries(),
        feeds=feeds,
        videos=videos,
        process_source=process_source,
    )

    await use_case.execute("https://example.com/feed.xml", process=False)

    saved_feed = feeds.saved[0][1]
    saved_entry = videos.saved[0][1]

    assert saved_feed.entries[0].published_at == datetime(
        2024, 1, 2, 3, 4, tzinfo=timezone.utc
    )
    assert saved_entry.published_at == datetime(2024, 1, 2, 3, 4, tzinfo=timezone.utc)
