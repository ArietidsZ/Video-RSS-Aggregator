from datetime import datetime, timezone

import pytest

from video_rss_aggregator.domain.models import SummaryResult
from video_rss_aggregator.domain.models import PreparedMedia
from video_rss_aggregator.domain.publication import PublicationRecord
from video_rss_aggregator.infrastructure.sqlite_repositories import (
    SQLiteFeedRepository,
    SQLiteFeedVideoRepository,
    SQLitePublicationRepository,
    SQLiteSummaryRepository,
    SQLiteVideoRepository,
)
from video_rss_aggregator.application.ports import FetchedFeed, FetchedFeedEntry
from video_rss_aggregator.storage import Database


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def build_summary(*, error: str | None = None) -> SummaryResult:
    return SummaryResult(
        summary="A compact summary",
        key_points=("Point one", "Point two"),
        visual_highlights=("Blue slide",),
        model_used="qwen",
        vram_mb=512.0,
        transcript_chars=120,
        frame_count=3,
        error=error,
    )


@pytest.mark.anyio
async def test_sqlite_summary_repository_saves_domain_summary_result(tmp_path) -> None:
    db = await Database.connect(str(tmp_path / "rss.db"))
    await db.migrate()
    video_id = await db.upsert_video(
        feed_id=None,
        guid=None,
        title="Video title",
        source_url="https://example.com/watch?v=1",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    repository = SQLiteSummaryRepository(db)

    await repository.save(video_id, build_summary())

    publications = await SQLitePublicationRepository(db).latest_publications(limit=5)

    await db.close()

    assert publications == [
        PublicationRecord(
            title="Video title",
            source_url="https://example.com/watch?v=1",
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            summary="A compact summary",
            key_points=("Point one", "Point two"),
            visual_highlights=("Blue slide",),
            model_used="qwen",
            vram_mb=512.0,
        )
    ]


@pytest.mark.anyio
async def test_sqlite_publication_repository_returns_newest_publications_first(
    tmp_path,
) -> None:
    db = await Database.connect(str(tmp_path / "rss.db"))
    await db.migrate()

    older_video_id = await db.upsert_video(
        feed_id=None,
        guid=None,
        title="Older video",
        source_url="https://example.com/watch?v=older",
        published_at=None,
    )
    await SQLiteSummaryRepository(db).save(
        older_video_id, build_summary(error="degraded")
    )

    newer_video_id = await db.upsert_video(
        feed_id=None,
        guid=None,
        title="Newer video",
        source_url="https://example.com/watch?v=newer",
        published_at=None,
    )
    await SQLiteSummaryRepository(db).save(newer_video_id, build_summary())

    publications = await SQLitePublicationRepository(db).latest_publications(limit=1)

    await db.close()

    assert [publication.title for publication in publications] == ["Newer video"]
    assert publications[0].summary == "A compact summary"


@pytest.mark.anyio
async def test_sqlite_video_repository_saves_media_and_transcript(tmp_path) -> None:
    db = await Database.connect(str(tmp_path / "rss.db"))
    await db.migrate()

    repository = SQLiteVideoRepository(db)

    video_id = await repository.save(
        PreparedMedia(
            source_url="https://example.com/watch?v=prepared",
            title="Prepared title",
            transcript="transcript text",
            media_path="/tmp/video.mp4",
            frame_paths=("/tmp/frame-1.jpg",),
        )
    )

    publications = await db.latest_publications(limit=5)

    await db.close()

    assert isinstance(video_id, str)
    assert publications == []


@pytest.mark.anyio
async def test_sqlite_feed_adapters_persist_feed_and_video_metadata(tmp_path) -> None:
    db = await Database.connect(str(tmp_path / "rss.db"))
    await db.migrate()

    await SQLiteFeedRepository(db).save(
        "https://example.com/feed.xml",
        FetchedFeed(title="Feed title", site_url=None, entries=()),
    )
    await SQLiteFeedVideoRepository(db).save_feed_item(
        "https://example.com/feed.xml",
        FetchedFeedEntry(
            source_url="https://example.com/watch?v=from-feed",
            title="Feed item",
            guid="guid-1",
        ),
    )

    async with db._conn.execute(
        "SELECT title FROM feeds WHERE url = ?", ("https://example.com/feed.xml",)
    ) as cur:
        feed_row = await cur.fetchone()
    async with db._conn.execute(
        "SELECT title, guid FROM videos WHERE source_url = ?",
        ("https://example.com/watch?v=from-feed",),
    ) as cur:
        video_row = await cur.fetchone()

    await db.close()

    assert dict(feed_row) == {"title": "Feed title"}
    assert dict(video_row) == {"title": "Feed item", "guid": "guid-1"}
