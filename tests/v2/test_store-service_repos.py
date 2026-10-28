"""Tests for vra.store.repos: SqlVideoStore round-trips and readouts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vra.store import repos
from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.store.repos import SqlVideoStore
from vra.types import (
    Frame,
    FrameSet,
    JobKind,
    Summary,
    Transcript,
    TranscriptSeg,
    VideoMeta,
    VideoStatus,
)


@pytest.fixture
async def stack(tmp_path: Path) -> AsyncIterator[tuple[Database, SqlVideoStore, SqlJobQueue]]:
    db = await Database.connect(str(tmp_path / "vra.db"))
    await db.migrate()
    yield db, SqlVideoStore(db), SqlJobQueue(db)
    await db.close()


async def test_upsert_feed_created_then_updated(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    feed = await store.upsert_feed("https://example.com/rss", "Title", 15)
    assert feed.url == "https://example.com/rss"
    assert feed.title == "Title"
    assert feed.poll_minutes == 15
    assert feed.last_checked is None

    again = await store.upsert_feed("https://example.com/rss", None, 30)
    assert again.id == feed.id  # same row
    assert again.title == "Title"  # None keeps existing title
    assert again.poll_minutes == 30  # explicit poll interval wins


async def test_save_feed_state_semantics(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    feed = await store.upsert_feed("https://example.com/rss", "T", 15)
    await store.save_feed_state(
        feed.id,
        site_url="https://example.com",
        etag='"v1"',
        last_modified="yesterday",
        error="boom",
    )
    saved = await store.get_feed_by_url("https://example.com/rss")
    assert saved is not None
    assert saved.site_url == "https://example.com"
    assert saved.etag == '"v1"'
    assert saved.last_modified == "yesterday"
    assert saved.error == "boom"
    assert saved.last_checked is not None  # touched

    # error=None clears; other None fields keep their values.
    await store.save_feed_state(feed.id, error=None)
    cleared = await store.get_feed_by_url("https://example.com/rss")
    assert cleared is not None
    assert cleared.error is None
    assert cleared.etag == '"v1"'
    assert cleared.site_url == "https://example.com"


async def test_get_list_delete_feed(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    assert await store.get_feed_by_url("https://missing") is None
    a = await store.upsert_feed("https://a", "A", 15)
    b = await store.upsert_feed("https://b", "B", 15)
    feeds = await store.list_feeds()
    assert sorted(f.id for f in feeds) == sorted([a.id, b.id])
    await store.delete_feed(a.id)
    assert await store.get_feed_by_url("https://a") is None
    assert [f.id for f in await store.list_feeds()] == [b.id]


async def test_upsert_video_ref_created_and_updated(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    feed = await store.upsert_feed("https://a", None, 15)
    published = datetime(2025, 6, 1, tzinfo=UTC)
    rec, created = await store.upsert_video_ref(
        canonical_id="https://x/1",
        source_url="https://x/1?utm=1",
        feed_id=feed.id,
        guid="https://x/1",
        title="First",
        published_at=published,
    )
    assert created is True
    assert rec.status == VideoStatus.PENDING
    assert rec.published_at == published

    # Second insert with same canonical id updates, does not create.
    rec2, created2 = await store.upsert_video_ref(
        canonical_id="https://x/1",
        source_url="https://x/1?utm=2",
        feed_id=None,  # keeps existing feed_id
        guid=None,  # keeps existing guid
        title="Second",
        published_at=None,  # keeps existing published_at
    )
    assert created2 is False
    assert rec2.id == rec.id
    assert rec2.feed_id == feed.id
    assert rec2.guid == "https://x/1"
    assert rec2.published_at == published
    assert rec2.title == "Second"
    assert rec2.source_url == "https://x/1?utm=2"  # follows the entry
    assert await store.get_video_by_canonical("https://x/1") is not None
    assert await store.get_video_by_canonical("nope") is None


async def test_set_video_status(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    rec, _ = await store.upsert_video_ref(
        canonical_id="c1",
        source_url="https://x/1",
        feed_id=None,
        guid=None,
        title=None,
        published_at=None,
    )
    await store.set_video_status(rec.id, VideoStatus.FAILED, error="bad")
    failed = await store.get_video(rec.id)
    assert failed is not None and failed.status == VideoStatus.FAILED
    assert failed.error == "bad"

    await store.set_video_status(rec.id, VideoStatus.ACQUIRING, error=None)
    running = await store.get_video(rec.id)
    assert running is not None and running.status == VideoStatus.ACQUIRING
    assert running.error is None  # authoritative clear


async def test_save_meta_roundtrip(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    rec, _ = await store.upsert_video_ref(
        canonical_id="c1",
        source_url="https://x/1",
        feed_id=None,
        guid=None,
        title="Feed Title",
        published_at=None,
    )
    meta = VideoMeta(
        canonical_id="c1",
        title="Platform Title",
        description="Desc",
        duration_s=61.5,
        webpage_url="https://x/1",
        chapters=((0.0, "intro"), (30.0, "main")),
    )
    await store.save_meta(rec.id, meta, "/media/c1.mp4")
    detail = await store.video_detail(rec.id)
    assert detail is not None
    video = detail["video"]
    assert video["title"] == "Platform Title"
    assert video["description"] == "Desc"
    assert video["duration_s"] == 61.5
    assert video["chapters"] == [[0.0, "intro"], [30.0, "main"]]
    assert video["media_path"] == "/media/c1.mp4"


async def test_save_transcript_replaces(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    rec, _ = await store.upsert_video_ref(
        canonical_id="c1", source_url="https://x/1", feed_id=None,
        guid=None, title=None, published_at=None,
    )
    first = Transcript(
        text="one",
        source="subtitle",
        lang="en",
        segments=(TranscriptSeg(start_s=0.0, end_s=1.0, text="one"),),
    )
    second = Transcript(text="two", source="asr", lang="en")
    await store.save_transcript(rec.id, first)
    await store.save_transcript(rec.id, second)
    detail = await store.video_detail(rec.id)
    assert detail is not None
    transcript = detail["transcript"]
    assert transcript is not None
    assert transcript["text"] == "two"
    assert transcript["source"] == "asr"
    assert transcript["segments"] == []  # second has no segments
    assert detail["video"]["id"] == rec.id


async def test_save_frames_replaces(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    rec, _ = await store.upsert_video_ref(
        canonical_id="c1", source_url="https://x/1", feed_id=None,
        guid=None, title=None, published_at=None,
    )
    await store.save_frames(
        rec.id,
        FrameSet(
            (
                Frame(path="/a.jpg", ts_s=0.0, scene_score=0.1, dhash="aa"),
                Frame(path="/b.jpg", ts_s=5.0, scene_score=0.9, dhash="bb"),
            )
        ),
    )
    await store.save_frames(rec.id, FrameSet((Frame(path="/c.jpg", ts_s=9.0),)))
    detail = await store.video_detail(rec.id)
    assert detail is not None
    assert detail["frames"] == [
        {
            "idx": 0,
            "path": "/c.jpg",
            "ts_s": 9.0,
            "scene_score": 0.0,
            "dhash": "",
        }
    ]


async def test_latest_publications_newest_per_video_excludes_failed(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    feed = await store.upsert_feed("https://a", None, 15)
    done, _ = await store.upsert_video_ref(
        canonical_id="done", source_url="https://x/done", feed_id=feed.id,
        guid=None, title="Done Video", published_at=None,
    )
    failed, _ = await store.upsert_video_ref(
        canonical_id="failed", source_url="https://x/failed", feed_id=feed.id,
        guid=None, title="Failed Video", published_at=None,
    )
    await store.set_video_status(done.id, VideoStatus.DONE)
    await store.set_video_status(failed.id, VideoStatus.FAILED)

    await store.save_summary(
        done.id, Summary(summary="old summary", key_points=("a",))
    )
    await asyncio.sleep(0.002)
    await store.save_summary(
        done.id,
        Summary(
            summary="new summary",
            key_points=("k1", "k2"),
            visual_highlights=("v1",),
            model="m",
        ),
    )
    await store.save_summary(
        failed.id, Summary(summary="failed summary", key_points=())
    )

    pubs = await store.latest_publications(10)
    assert [p["summary"] for p in pubs] == ["new summary"]
    pub = pubs[0]
    assert set(pub) == {
        "video_id", "canonical_id", "title", "source_url", "published_at",
        "summary", "key_points", "visual_highlights", "model", "created_at",
    }
    assert pub["video_id"] == done.id
    assert pub["canonical_id"] == "done"
    assert pub["title"] == "Done Video"
    assert pub["source_url"] == "https://x/done"
    assert pub["key_points"] == ["k1", "k2"]
    assert pub["visual_highlights"] == ["v1"]
    assert pub["model"] == "m"
    assert isinstance(pub["created_at"], datetime)

    # limit respected, ordering newest summary first
    other, _ = await store.upsert_video_ref(
        canonical_id="other", source_url="https://x/other", feed_id=None,
        guid=None, title="Other", published_at=None,
    )
    await store.set_video_status(other.id, VideoStatus.DONE)
    await asyncio.sleep(0.002)
    await store.save_summary(other.id, Summary(summary="latest of all"))
    pubs = await store.latest_publications(1)
    assert [p["summary"] for p in pubs] == ["latest of all"]
    assert len(await store.latest_publications(10)) == 2


async def test_video_detail_shape_and_missing(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    assert await store.video_detail("missing") is None
    rec, _ = await store.upsert_video_ref(
        canonical_id="c1", source_url="https://x/1", feed_id=None,
        guid=None, title="T", published_at=None,
    )
    detail = await store.video_detail(rec.id)
    assert detail is not None
    assert set(detail) == {"video", "transcript", "frames", "summaries"}
    assert detail["transcript"] is None
    assert detail["frames"] == []
    assert detail["summaries"] == []
    await store.save_summary(rec.id, Summary(summary="s1"))
    await asyncio.sleep(0.002)
    await store.save_summary(rec.id, Summary(summary="s2"))
    detail = await store.video_detail(rec.id)
    assert detail is not None
    assert [s["summary"] for s in detail["summaries"]] == ["s2", "s1"]  # newest first
    assert detail["summaries"][0]["model"] == ""
    assert detail["video"]["status"] == "pending"


async def test_counts(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    db, store, queue = stack
    feed = await store.upsert_feed("https://a", None, 15)
    rec, _ = await store.upsert_video_ref(
        canonical_id="c1", source_url="https://x/1", feed_id=feed.id,
        guid=None, title=None, published_at=None,
    )
    other, _ = await store.upsert_video_ref(
        canonical_id="c2", source_url="https://x/2", feed_id=None,
        guid=None, title=None, published_at=None,
    )
    await store.set_video_status(other.id, VideoStatus.FAILED)
    await store.save_summary(rec.id, Summary(summary="s"))
    job = await queue.enqueue(JobKind.PROCESS, {"video_id": rec.id})
    await queue.enqueue(JobKind.INGEST, {"url": "https://a"})
    await queue.claim((JobKind.PROCESS,))
    await db.execute("UPDATE jobs SET state = 'failed' WHERE id = ?", (job.id,))

    counts = await store.counts()
    assert counts["jobs_by_state"]["failed"] == 1
    assert counts["jobs_by_state"]["queued"] == 1
    assert counts["jobs_by_state"]["done"] == 0
    assert counts["videos_by_status"] == {
        "pending": 1, "acquiring": 0, "summarizing": 0, "done": 0, "failed": 1,
    }
    assert counts["feeds"] == 1
    assert counts["summaries"] == 1


async def test_feed_order_uses_timestamp_then_id(stack, monkeypatch):
    _, store, _ = stack
    ids = iter(["z", "a", "m"])
    monkeypatch.setattr(repos.uuid, "uuid4", lambda: next(ids))
    monkeypatch.setattr(repos, "_utc_now", lambda: datetime(2026, 1, 2, tzinfo=UTC))
    await store.upsert_feed("https://z", "Z", 15)
    await store.upsert_feed("https://a", "A", 15)
    monkeypatch.setattr(repos, "_utc_now", lambda: datetime(2026, 1, 1, tzinfo=UTC))
    await store.upsert_feed("https://m", "M", 15)
    assert [feed.id for feed in await store.list_feeds()] == ["m", "a", "z"]
