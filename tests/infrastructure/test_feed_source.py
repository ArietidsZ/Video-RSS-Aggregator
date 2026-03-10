import pytest

from video_rss_aggregator.application.ports import FetchedFeed, FetchedFeedEntry
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
            ),
        ),
    )
