"""Tests for vra.service.scheduler: due windows, jitter bounds, lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_service_fakes import FakeIngester

import vra.service.scheduler as scheduler_module
from vra.config import Config
from vra.service.scheduler import PollScheduler
from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.store.repos import SqlVideoStore
from vra.types import JobKind


@pytest.fixture
async def stack(
    tmp_path: Path,
) -> AsyncIterator[tuple[Database, SqlVideoStore, SqlJobQueue]]:
    db = await Database.connect(str(tmp_path / "vra.db"))
    await db.migrate()
    yield db, SqlVideoStore(db), SqlJobQueue(db)
    await db.close()


def _scheduler(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> PollScheduler:
    _, store, queue = stack
    return PollScheduler(FakeIngester(), store, queue, Config())


async def _set_last_checked(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
    feed_id: str,
    *,
    seconds_ago: float,
) -> None:
    """Backdate last_checked relative to the current wall clock."""
    db, _, _ = stack
    checked = datetime.now(UTC) - timedelta(seconds=seconds_ago)
    await db.execute(
        "UPDATE feeds SET last_checked = ? WHERE id = ?",
        (checked.isoformat(), feed_id),
    )


async def test_scan_enqueues_due_feed_never_checked(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    await store.upsert_feed("https://example.com/a", None, 15)
    scheduler = _scheduler(stack)

    await scheduler._scan()  # last_checked NULL -> due

    jobs = await queue.list(kind=JobKind.INGEST)
    assert len(jobs) == 1
    assert jobs[0].payload == {"url": "https://example.com/a"}
    assert jobs[0].video_id is None
    assert jobs[0].kind == JobKind.INGEST


async def test_scan_skips_recently_checked_feed(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    feed = await store.upsert_feed("https://example.com/a", None, 15)
    await _set_last_checked(stack, feed.id, seconds_ago=10.0)
    scheduler = _scheduler(stack)

    await scheduler._scan()
    assert await queue.list(kind=JobKind.INGEST) == []


async def test_due_window_jitter_bounds(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, store, _ = stack
    feed = await store.upsert_feed("https://example.com/a", None, 15)  # 900s
    scheduler = _scheduler(stack)
    now = datetime.now(UTC)
    db, _, _ = stack

    cases = [
        # (jitter factor, elapsed seconds, expected due)
        (1.0, 900.0, True),    # elapsed == interval, factor at lower bound
        (1.0, 899.9, False),   # just under interval
        (1.1, 900.0, False),   # elapsed < interval * upper-bound factor
        (1.1, 990.5, True),    # elapsed beyond interval * upper-bound factor
    ]
    for factor, elapsed, expected in cases:
        checked = now - timedelta(seconds=elapsed)
        await db.execute(
            "UPDATE feeds SET last_checked = ? WHERE id = ?",
            (checked.isoformat(), feed.id),
        )
        monkeypatch.setattr(scheduler_module, "_jitter_factor", lambda f=factor: f)
        record = (await store.list_feeds())[0]
        assert scheduler._is_due(record, now) is expected, (factor, elapsed)


async def test_scan_enqueue_deduped_across_scans(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, queue = stack
    await store.upsert_feed("https://example.com/a", None, 15)
    scheduler = _scheduler(stack)

    await scheduler._scan()
    await scheduler._scan()

    jobs = await queue.list(kind=JobKind.INGEST)
    assert len(jobs) == 1  # dedup key keeps overlapping scans idempotent


async def test_scan_multiple_feeds_only_due_enqueued(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, store, queue = stack
    await store.upsert_feed("https://example.com/due", None, 15)
    recent = await store.upsert_feed("https://example.com/recent", None, 15)
    await _set_last_checked(stack, recent.id, seconds_ago=5.0)
    monkeypatch.setattr(scheduler_module, "_jitter_factor", lambda: 1.0)
    scheduler = _scheduler(stack)

    await scheduler._scan()
    jobs = await queue.list(kind=JobKind.INGEST)
    assert [job.payload["url"] for job in jobs] == ["https://example.com/due"]


async def test_start_stop_lifecycle(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, store, queue = stack
    await store.upsert_feed("https://example.com/a", None, 15)
    monkeypatch.setattr(scheduler_module, "SCAN_INTERVAL_S", 0.01)
    scheduler = _scheduler(stack)

    scheduler.start()
    task = scheduler._task
    assert task is not None
    scheduler.start()  # idempotent: same task
    assert scheduler._task is task

    await asyncio.sleep(0.08)
    assert task.done() is False  # loop still running between scans
    assert len(await queue.list(kind=JobKind.INGEST)) >= 1

    await scheduler.stop()
    assert scheduler._task is None
    assert task.done() and task.cancelled()

    await scheduler.stop()  # idempotent, no error


async def test_stop_before_start_is_safe(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    scheduler = _scheduler(stack)
    await scheduler.stop()  # must not raise
    assert scheduler._task is None
