"""Tests for vra.store.queue: SqlJobQueue dedup, claim, backoff, recovery."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.types import JobKind, JobState


@pytest.fixture
async def queue(tmp_path: Path) -> AsyncIterator[SqlJobQueue]:
    db = await Database.connect(str(tmp_path / "vra.db"))
    await db.migrate()
    yield SqlJobQueue(db)
    await db.close()


async def test_enqueue_returns_queued_job(queue: SqlJobQueue) -> None:
    job = await queue.enqueue(JobKind.PROCESS, {"video_id": "v1"}, video_id=None)
    assert job.kind == JobKind.PROCESS
    assert job.state == JobState.QUEUED
    assert job.payload == {"video_id": "v1"}
    assert job.video_id is None
    assert job.attempts == 0
    assert job.max_attempts == 3
    assert job.created_at is not None and job.updated_at is not None
    assert job.run_at is None


async def test_enqueue_dedup_on_same_kind_and_payload(
    queue: SqlJobQueue,
) -> None:
    first = await queue.enqueue(JobKind.PROCESS, {"video_id": "v1"}, video_id=None)
    second = await queue.enqueue(
        JobKind.PROCESS, {"video_id": "v1"}, dedup_key="process:v1"
    )
    assert second.id == first.id  # unchanged existing job
    # key order in the payload dict must not matter (canonical json)
    reordered = await queue.enqueue(JobKind.PROCESS, {"video_id": "v1"})
    assert reordered.id == first.id


async def test_enqueue_video_id_foreign_key_and_cascade(
    queue: SqlJobQueue,
) -> None:
    """jobs.video_id is FK-enforced; deleting the video cascades to jobs."""
    db = queue._db
    from vra.store.repos import SqlVideoStore

    store = SqlVideoStore(db)
    video, _ = await store.upsert_video_ref(
        canonical_id="c1", source_url="https://x/1", feed_id=None,
        guid=None, title=None, published_at=None,
    )
    job = await queue.enqueue(
        JobKind.PROCESS, {"video_id": video.id}, video_id=video.id
    )
    assert job.video_id == video.id
    with pytest.raises(Exception, match="FOREIGN KEY"):
        await queue.enqueue(JobKind.PROCESS, {"video_id": "ghost"}, video_id="ghost")

    await db.execute("DELETE FROM videos WHERE id = ?", (video.id,))
    assert await queue.get(job.id) is None  # cascade deleted


async def test_enqueue_dedup_distinct_kind_or_payload(
    queue: SqlJobQueue,
) -> None:
    base = await queue.enqueue(JobKind.PROCESS, {"video_id": "v1"})
    other_kind = await queue.enqueue(JobKind.INGEST, {"video_id": "v1"})
    other_payload = await queue.enqueue(JobKind.PROCESS, {"video_id": "v2"})
    assert other_kind.id != base.id
    assert other_payload.id != base.id


async def test_enqueue_dedup_only_covers_queued_and_running(
    queue: SqlJobQueue,
) -> None:
    first = await queue.enqueue(JobKind.PROCESS, {"video_id": "v1"})
    await queue.claim((JobKind.PROCESS,))
    await queue.complete(first.id)
    again = await queue.enqueue(JobKind.PROCESS, {"video_id": "v1"})
    assert again.id != first.id  # done jobs no longer dedup
    assert again.attempts == 0


async def test_claim_fifo_order_and_attempts(queue: SqlJobQueue) -> None:
    a = await queue.enqueue(JobKind.PROCESS, {"video_id": "a"})
    b = await queue.enqueue(JobKind.PROCESS, {"video_id": "b"})
    await queue.enqueue(JobKind.INGEST, {"url": "https://x"})  # other kind

    claimed = await queue.claim((JobKind.PROCESS,))
    assert claimed is not None
    assert claimed.id == a.id  # oldest first (created_at, rowid)
    assert claimed.state == JobState.RUNNING
    assert claimed.attempts == 1

    claimed_b = await queue.claim((JobKind.PROCESS, JobKind.INGEST))
    assert claimed_b is not None and claimed_b.id == b.id

    assert await queue.claim((JobKind.PROCESS,)) is None
    fetched = await queue.get(a.id)
    assert fetched is not None and fetched.state == JobState.RUNNING


async def test_claim_skips_future_run_at(queue: SqlJobQueue) -> None:
    future = datetime.now(UTC) + timedelta(seconds=120)
    await queue.enqueue(JobKind.PROCESS, {"v": 1}, run_at=future)
    assert await queue.claim((JobKind.PROCESS,)) is None
    await queue.enqueue(JobKind.PROCESS, {"v": 2})  # due immediately
    claimed = await queue.claim((JobKind.PROCESS,))
    assert claimed is not None and claimed.payload == {"v": 2}


async def test_claim_empty_kinds_returns_none(queue: SqlJobQueue) -> None:
    await queue.enqueue(JobKind.PROCESS, {"v": 1})
    assert await queue.claim(()) is None


async def test_fail_retry_backoff_run_at_future(queue: SqlJobQueue) -> None:
    job = await queue.enqueue(JobKind.PROCESS, {"v": 1})
    await queue.claim((JobKind.PROCESS,))

    before = datetime.now(UTC)
    await queue.fail(job.id, "transient boom", retry=True)
    failed_once = await queue.get(job.id)
    assert failed_once is not None
    assert failed_once.state == JobState.QUEUED
    assert failed_once.error == "transient boom"
    assert failed_once.attempts == 1
    assert failed_once.run_at is not None
    # attempts=1 after first claim -> backoff 30 * 2**1 = 60s
    assert failed_once.run_at > before + timedelta(seconds=55)
    assert failed_once.run_at <= before + timedelta(seconds=65)

    # while backed off, the job is not claimable
    assert await queue.claim((JobKind.PROCESS,)) is None


async def test_fail_terminal_after_max_attempts(queue: SqlJobQueue) -> None:
    job = await queue.enqueue(JobKind.PROCESS, {"v": 1})
    now_iso = datetime.now(UTC).isoformat()

    # Three claim/fail cycles; between cycles we pretend the backoff window
    # elapsed by re-arming run_at (otherwise claim skips the backed-off job).
    for _ in range(3):
        claimed = await queue.claim((JobKind.PROCESS,))
        assert claimed is not None and claimed.id == job.id
        await queue.fail(job.id, "boom", retry=True)
        await queue._db.execute(
            "UPDATE jobs SET run_at = ? WHERE id = ?", (now_iso, job.id)
        )

    terminal = await queue.get(job.id)
    assert terminal is not None
    assert terminal.state == JobState.FAILED
    assert terminal.attempts == 3
    assert terminal.error == "boom"


async def test_fail_without_retry_marks_failed(queue: SqlJobQueue) -> None:
    job = await queue.enqueue(JobKind.PROCESS, {"v": 1})
    await queue.claim((JobKind.PROCESS,))
    await queue.fail(job.id, "fatal", retry=False)
    done = await queue.get(job.id)
    assert done is not None and done.state == JobState.FAILED
    assert done.error == "fatal"


async def test_complete_marks_done(queue: SqlJobQueue) -> None:
    job = await queue.enqueue(JobKind.PROCESS, {"v": 1})
    await queue.claim((JobKind.PROCESS,))
    await queue.complete(job.id)
    done = await queue.get(job.id)
    assert done is not None and done.state == JobState.DONE


async def test_requeue_stale(queue: SqlJobQueue) -> None:
    fresh = await queue.enqueue(JobKind.PROCESS, {"v": 1})
    stale = await queue.enqueue(JobKind.PROCESS, {"v": 2})
    await queue.claim((JobKind.PROCESS,))  # fresh running
    claimed = await queue.claim((JobKind.PROCESS,))  # stale running
    assert claimed is not None and claimed.id == stale.id

    # nothing is older than an hour yet
    assert await queue.requeue_stale(older_than_s=3600.0) == 0

    # zero window: every running job counts as stale, run_at reset to now
    assert await queue.requeue_stale(older_than_s=0.0) == 2
    requeued = await queue.get(stale.id)
    assert requeued is not None
    assert requeued.state == JobState.QUEUED
    assert requeued.run_at is not None and requeued.run_at <= datetime.now(UTC)

    # requeued job is immediately claimable again; attempts keep accumulating
    again = await queue.claim((JobKind.PROCESS,))
    assert again is not None and again.id == fresh.id
    assert again.attempts == 2


async def test_list_filters_and_order(queue: SqlJobQueue) -> None:
    p1 = await queue.enqueue(JobKind.PROCESS, {"v": 1})
    i1 = await queue.enqueue(JobKind.INGEST, {"url": "https://a"})
    p2 = await queue.enqueue(JobKind.PROCESS, {"v": 2})

    process_jobs = await queue.list(kind=JobKind.PROCESS)
    assert [j.id for j in process_jobs] == [p2.id, p1.id]  # newest first

    ingest_jobs = await queue.list(kind=JobKind.INGEST)
    assert [j.id for j in ingest_jobs] == [i1.id]

    await queue.claim((JobKind.PROCESS,))  # claims oldest process job (p1)
    running = await queue.list(state=JobState.RUNNING)
    assert [j.id for j in running] == [p1.id]

    limited = await queue.list(limit=2)
    assert [j.id for j in limited] == [p2.id, i1.id]


async def test_get_missing_and_payload_roundtrip(queue: SqlJobQueue) -> None:
    payload = {"video_id": "v1", "nested": {"b": 2, "a": 1}, "n": 3}
    job = await queue.enqueue(JobKind.PROCESS, payload)
    fetched = await queue.get(job.id)
    assert fetched is not None
    assert fetched.payload == payload  # JSON round-trip preserves content
    assert await queue.get("missing-job") is None


async def test_queue_close_delegates_to_database(tmp_path: Path) -> None:
    db = await Database.connect(str(tmp_path / "vra.db"))
    await db.migrate()
    queue = SqlJobQueue(db)
    await queue.close()
    assert db.path == str(tmp_path / "vra.db")  # attributes intact after close
