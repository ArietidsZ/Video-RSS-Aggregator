"""Tests for vra.service.ingester: FeedIngester + HttpFeedFetcher."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from test_service_fakes import FakeFetcher

import vra.service.ingester as ingester_module
from vra.config import Config
from vra.service.ingester import FeedIngester, HttpFeedFetcher
from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.store.repos import SqlVideoStore
from vra.types import JobKind, JobState, VideoStatus

_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Test Feed</title>
<link>https://example.com/</link>
<item>
  <title>Episode One</title>
  <link>https://example.com/ep1</link>
  <guid isPermaLink="false">urn:ep1</guid>
  <enclosure url="https://cdn.example.com/ep1.mp4" type="video/mp4"/>
  <pubDate>Tue, 10 Jun 2025 10:00:00 +0000</pubDate>
</item>
<item>
  <title>Episode Two</title>
  <link>https://example.com/ep2</link>
  <guid>https://example.com/ep2</guid>
  <pubDate>2025-06-11T08:30:00+00:00</pubDate>
</item>
</channel></rss>"""


@pytest.fixture
async def stack(
    tmp_path: Path,
) -> AsyncIterator[tuple[Database, SqlVideoStore, SqlJobQueue]]:
    db = await Database.connect(str(tmp_path / "vra.db"))
    await db.migrate()
    yield db, SqlVideoStore(db), SqlJobQueue(db)
    await db.close()


def _entries() -> list[dict[str, object]]:
    return [
        {
            "source_url": "https://youtu.be/aaa",
            "title": "A",
            "guid": "https://youtu.be/aaa",  # absolute http guid -> canonical id
            "published_at": datetime(2025, 6, 1, tzinfo=UTC),
        },
        {
            "source_url": "https://Example.com/b?utm_source=x",
            "title": "B",
            "guid": "urn:b",  # non-http guid -> url hash canonical id
            "published_at": None,
        },
    ]


async def test_ingest_new_videos_enqueued(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    fetcher = FakeFetcher(_entries())
    ingester = FeedIngester(fetcher, store, queue, Config())
    result = await ingester.ingest("https://example.com/rss")

    assert (result.new_videos, result.enqueued, result.skipped_done) == (2, 2, 0)
    assert result.feed.title == "Feed Title"
    assert result.feed.site_url == "https://example.com"
    assert result.feed.etag == '"etag-1"'
    assert result.feed.last_checked is not None
    assert result.feed.error is None

    # first entry: absolute http guid becomes the canonical id
    first = await store.get_video_by_canonical("https://youtu.be/aaa")
    assert first is not None
    assert first.status == VideoStatus.PENDING
    assert first.guid == "https://youtu.be/aaa"
    assert first.published_at == datetime(2025, 6, 1, tzinfo=UTC)

    # second entry: normalized url (host lowered, query dropped), sha1[:16]
    expected = "url:" + hashlib.sha1(b"https://example.com/b").hexdigest()[:16]
    second = await store.get_video_by_canonical(expected)
    assert second is not None
    assert second.source_url == "https://Example.com/b?utm_source=x"
    assert second.guid == "urn:b"

    jobs = await queue.list(kind=JobKind.PROCESS)
    assert len(jobs) == 2
    assert {job.video_id for job in jobs} == {first.id, second.id}
    assert all(job.state == JobState.QUEUED for job in jobs)

    # conditional headers were empty on the first poll (no stored feed)
    assert fetcher.calls[0]["etag"] is None


async def test_ingest_done_skipped_and_pending_deduped(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    ingester = FeedIngester(FakeFetcher(_entries()), store, queue, Config())
    await ingester.ingest("https://example.com/rss")

    # second poll with the same entries: everything pending -> enqueue calls
    # happen but the durable dedup keeps a single queued job each.
    result = await ingester.ingest("https://example.com/rss")
    assert (result.new_videos, result.enqueued, result.skipped_done) == (0, 2, 0)
    assert len(await queue.list(kind=JobKind.PROCESS)) == 2

    # mark both done: third poll skips entirely
    for video in [
        await store.get_video_by_canonical("https://youtu.be/aaa"),
        await store.get_video_by_canonical(
            "url:" + hashlib.sha1(b"https://example.com/b").hexdigest()[:16]
        ),
    ]:
        assert video is not None
        await store.set_video_status(video.id, VideoStatus.DONE)
    result = await ingester.ingest("https://example.com/rss")
    assert (result.new_videos, result.enqueued, result.skipped_done) == (0, 0, 2)
    assert len(await queue.list(kind=JobKind.PROCESS)) == 2


async def test_ingest_failed_video_reenqueued(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    ingester = FeedIngester(FakeFetcher(_entries()), store, queue, Config())
    await ingester.ingest("https://example.com/rss")

    first = await store.get_video_by_canonical("https://youtu.be/aaa")
    second = await store.get_video_by_canonical(
        "url:" + hashlib.sha1(b"https://example.com/b").hexdigest()[:16]
    )
    assert first is not None and second is not None

    # simulate a worker run: consume both jobs, complete one, fail the other
    # terminally, and mark the matching video states.
    job_a = await queue.claim((JobKind.PROCESS,))
    job_b = await queue.claim((JobKind.PROCESS,))
    assert job_a is not None and job_b is not None
    await queue.complete(job_a.id if job_a.video_id == first.id else job_b.id)
    failed_job = job_b if job_a.video_id == first.id else job_a
    await queue.fail(failed_job.id, "boom", retry=False)
    await store.set_video_status(first.id, VideoStatus.DONE)
    await store.set_video_status(second.id, VideoStatus.FAILED, error="x")

    result = await ingester.ingest("https://example.com/rss")
    assert (result.new_videos, result.enqueued, result.skipped_done) == (0, 1, 1)

    # the terminally-failed video got a fresh queued process job
    jobs = await queue.list(kind=JobKind.PROCESS)
    fresh = [j for j in jobs if j.id not in {job_a.id, job_b.id}]
    assert len(fresh) == 1
    assert fresh[0].video_id == second.id
    assert fresh[0].state == JobState.QUEUED


async def test_ingest_not_modified_short_circuit(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    feed = await store.upsert_feed("https://example.com/rss", "Old", 15)
    await store.save_feed_state(feed.id, etag='"keep-me"', error="old boom")
    fetcher = FakeFetcher(not_modified=True)
    ingester = FeedIngester(fetcher, store, queue, Config())

    result = await ingester.ingest("https://example.com/rss")
    assert (result.new_videos, result.enqueued, result.skipped_done) == (0, 0, 0)
    assert result.feed.id == feed.id

    # conditional headers were sent, error cleared, etag preserved
    assert fetcher.calls[0]["etag"] == '"keep-me"'
    fresh = await store.get_feed_by_url("https://example.com/rss")
    assert fresh is not None
    assert fresh.error is None
    assert fresh.etag == '"keep-me"'
    assert fresh.last_checked is not None
    assert await queue.list() == []


async def test_ingest_fetch_error_recorded_and_raised(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    feed = await store.upsert_feed("https://example.com/rss", "Old", 15)
    fetcher = FakeFetcher(error=httpx.ConnectError("network down"))
    ingester = FeedIngester(fetcher, store, queue, Config())

    with pytest.raises(httpx.ConnectError):
        await ingester.ingest("https://example.com/rss")

    fresh = await store.get_feed_by_url("https://example.com/rss")
    assert fresh is not None
    assert fresh.error is not None and "network down" in fresh.error
    assert fresh.last_checked is not None
    assert fresh.id == feed.id
    assert await queue.list() == []


async def test_ingest_skips_entries_without_source_url(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    entries = [
        {"source_url": None, "title": "x", "guid": None, "published_at": None},
        {"source_url": "   ", "title": "y", "guid": None, "published_at": None},
        {"source_url": "https://x/1", "title": "ok", "guid": None,
         "published_at": None},
    ]
    ingester = FeedIngester(FakeFetcher(entries), store, queue, Config())
    result = await ingester.ingest("https://example.com/rss")
    assert (result.new_videos, result.enqueued) == (1, 1)
    assert await store.get_video_by_canonical(
        "url:" + hashlib.sha1(b"https://x/1").hexdigest()[:16]
    ) is not None


async def test_ingest_max_items_truncates(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    entries = [
        {"source_url": f"https://x/{i}", "title": None, "guid": None,
         "published_at": None}
        for i in range(5)
    ]
    ingester = FeedIngester(FakeFetcher(entries), store, queue, Config())
    result = await ingester.ingest("https://example.com/rss", max_items=2)
    assert (result.new_videos, result.enqueued) == (2, 2)
    assert len(await queue.list(kind=JobKind.PROCESS)) == 2


# --------------------------------------------------------------------------
# HttpFeedFetcher
# --------------------------------------------------------------------------

def _mock_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin host resolution to a public IP (sandbox DNS may map domains
    into reserved ranges, which the SSRF guard rightly rejects)."""
    monkeypatch.setattr(
        ingester_module, "_resolve_addresses", lambda host: ["93.184.216.34"]
    )


async def test_fetcher_parses_entries_etag_and_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    async with _mock_client(
        lambda request: httpx.Response(200, text=_RSS, headers={"ETag": '"abc"'})
    ) as client:
        fetcher = HttpFeedFetcher(client)
        entries, title, site_url, etag, not_modified = await fetcher.fetch(
            "https://example.com/feed.xml"
        )

    assert not_modified is False
    assert title == "Test Feed"
    assert site_url == "https://example.com/"
    assert etag == '"abc"'
    assert len(entries) == 2

    # enclosure preferred over link, guid from <id>, RFC-822 pubDate parsed
    first = entries[0]
    assert first["source_url"] == "https://cdn.example.com/ep1.mp4"
    assert first["guid"] == "urn:ep1"
    assert first["title"] == "Episode One"
    assert first["published_at"] == datetime(2025, 6, 10, 10, 0, tzinfo=UTC)

    # links[0].href fallback, ISO date via fromisoformat fallback
    second = entries[1]
    assert second["source_url"] == "https://example.com/ep2"
    assert second["guid"] == "https://example.com/ep2"
    assert second["published_at"] == datetime(2025, 6, 11, 8, 30, tzinfo=UTC)


async def test_fetcher_304_not_modified_with_conditional_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["if-none-match"] = request.headers.get("if-none-match")
        seen["if-modified-since"] = request.headers.get("if-modified-since")
        if request.headers.get("if-none-match") == '"old-etag"':
            return httpx.Response(304)
        return httpx.Response(200, text=_RSS)

    async with _mock_client(handler) as client:
        fetcher = HttpFeedFetcher(client)
        result = await fetcher.fetch(
            "https://example.com/feed.xml",
            etag='"old-etag"',
            last_modified="Tue, 10 Jun 2025 09:00:00 GMT",
        )
    entries, title, site_url, etag, not_modified = result

    assert seen == {
        "if-none-match": '"old-etag"',
        "if-modified-since": "Tue, 10 Jun 2025 09:00:00 GMT",
    }
    assert entries == []
    assert not_modified is True
    assert title is None and site_url is None and etag is None


async def test_fetcher_http_errors_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    async with _mock_client(
        lambda request: httpx.Response(500, text="nope")
    ) as client:
        fetcher = HttpFeedFetcher(client)
        with pytest.raises(httpx.HTTPStatusError):
            await fetcher.fetch("https://example.com/feed.xml")


async def test_fetcher_rejects_non_http_scheme() -> None:
    async with _mock_client(
        lambda request: httpx.Response(200, text=_RSS)
    ) as client:
        fetcher = HttpFeedFetcher(client)
        with pytest.raises(ValueError, match="scheme"):
            await fetcher.fetch("ftp://example.com/feed.xml")
        with pytest.raises(ValueError, match="scheme"):
            await fetcher.fetch("file:///etc/passwd")


async def test_fetcher_rejects_private_hosts() -> None:
    async with _mock_client(
        lambda request: httpx.Response(200, text=_RSS)
    ) as client:
        fetcher = HttpFeedFetcher(client)
        for url in (
            "http://127.0.0.1/feed.xml",
            "http://[::1]/feed.xml",
            "http://10.1.2.3/feed.xml",
            "http://192.168.0.5/feed.xml",
        ):
            with pytest.raises(ValueError, match="private/reserved"):
                await fetcher.fetch(url)


async def test_fetcher_allows_private_when_enabled() -> None:
    async with _mock_client(
        lambda request: httpx.Response(200, text=_RSS)
    ) as client:
        fetcher = HttpFeedFetcher(client, allow_private=True)
        entries, *_ = await fetcher.fetch("http://127.0.0.1:8080/feed.xml")
    assert len(entries) == 2


@pytest.mark.parametrize('target', [
    'http://127.0.0.1/private', 'http://10.0.0.1/private',
    'http://169.254.169.254/latest', 'file:///etc/passwd',
])
async def test_redirect_destination_checked_before_request(monkeypatch, target):
    seen = []
    monkeypatch.setattr(ingester_module, '_resolve_addresses',
                        lambda host: ['93.184.216.34'] if host == 'public.example' else [host])
    def handle(request):
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(302, headers={'Location': target})
        return httpx.Response(200, text=_RSS)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError):
            await HttpFeedFetcher(client).fetch('https://public.example/feed')
    assert seen == ['https://public.example/feed']


async def test_relative_redirect_and_conditional_response(monkeypatch):
    seen = []
    monkeypatch.setattr(ingester_module, '_resolve_addresses', lambda host: ['93.184.216.34'])
    def handle(request):
        seen.append(str(request.url))
        assert request.headers['if-none-match'] == 'saved'
        if len(seen) == 1:
            return httpx.Response(307, headers={'Location': '/new'})
        return httpx.Response(304)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await HttpFeedFetcher(client).fetch('https://public.example/feed', etag='saved')
    assert seen == ['https://public.example/feed', 'https://public.example/new']
    assert result[-1] is True


async def test_redirect_loop_is_bounded():
    seen = []
    def handle(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={'Location': '/again'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match='limit'):
            await HttpFeedFetcher(client, allow_private=True).fetch('http://localhost/start')
    assert len(seen) == 6
