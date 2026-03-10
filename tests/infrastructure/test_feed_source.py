from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from video_rss_aggregator.application.ports import FetchedFeed, FetchedFeedEntry
from video_rss_aggregator.infrastructure import feed_source as feed_source_module
from video_rss_aggregator.infrastructure.feed_source import HttpFeedSource


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeAsyncClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    async def get(self, url: str) -> FakeResponse:
        self.calls.append(url)
        return self.response


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_http_feed_source_fetches_and_maps_entries() -> None:
    client = FakeAsyncClient(
        FakeResponse(
            """
            <rss version="2.0">
              <channel>
                <title>Example feed</title>
                <link>https://example.com</link>
                <item>
                  <title>First</title>
                  <guid>first-guid</guid>
                  <pubDate>Tue, 02 Jan 2024 03:04:05 GMT</pubDate>
                  <enclosure url="https://cdn.example.com/video.mp4" type="video/mp4" />
                </item>
                <item>
                  <title>Second</title>
                  <guid>second-guid</guid>
                  <link>https://example.com/watch?v=2</link>
                </item>
              </channel>
            </rss>
            """
        )
    )
    adapter = HttpFeedSource(client)

    feed = await adapter.fetch("https://example.com/feed.xml", max_items=1)

    assert client.calls == ["https://example.com/feed.xml"]
    assert feed == FetchedFeed(
        title="Example feed",
        site_url="https://example.com",
        entries=(
            FetchedFeedEntry(
                source_url="https://cdn.example.com/video.mp4",
                title="First",
                guid="first-guid",
                published_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        ),
    )


@pytest.mark.anyio
async def test_http_feed_source_falls_back_to_updated_timestamp() -> None:
    client = FakeAsyncClient(
        FakeResponse(
            """
            <feed xmlns="http://www.w3.org/2005/Atom">
              <title>Example atom feed</title>
              <link href="https://example.com" />
              <entry>
                <title>First</title>
                <id>first-guid</id>
                <updated>2024-01-02T03:04:05Z</updated>
                <link href="https://example.com/watch?v=1" />
              </entry>
            </feed>
            """
        )
    )
    adapter = HttpFeedSource(client)

    feed = await adapter.fetch("https://example.com/feed.xml")

    assert feed == FetchedFeed(
        title="Example atom feed",
        site_url="https://example.com",
        entries=(
            FetchedFeedEntry(
                source_url="https://example.com/watch?v=1",
                title="First",
                guid="first-guid",
                published_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        ),
    )


@pytest.mark.anyio
async def test_http_feed_source_ignores_malformed_parsed_dates(monkeypatch) -> None:
    client = FakeAsyncClient(FakeResponse("ignored"))
    adapter = HttpFeedSource(client)

    def fake_parse(_text: str) -> SimpleNamespace:
        return SimpleNamespace(
            feed={"title": "Example feed", "link": "https://example.com"},
            entries=[
                {
                    "title": "First",
                    "id": "first-guid",
                    "link": "https://example.com/watch?v=1",
                    "published_parsed": (2024, 1),
                }
            ],
        )

    monkeypatch.setattr(feed_source_module.feedparser, "parse", fake_parse)

    feed = await adapter.fetch("https://example.com/feed.xml")

    assert feed == FetchedFeed(
        title="Example feed",
        site_url="https://example.com",
        entries=(
            FetchedFeedEntry(
                source_url="https://example.com/watch?v=1",
                title="First",
                guid="first-guid",
                published_at=None,
            ),
        ),
    )
