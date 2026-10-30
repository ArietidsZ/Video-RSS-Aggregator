"""Offline wakeup/cancellation regressions; SQLite, not the notification, owns jobs."""

import asyncio
import contextlib
import gc
import weakref
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from vra.runtime import _ingest_worker, _process_worker
from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.store.repos import SqlVideoStore
from vra.types import JobKind


@pytest.fixture
async def queue(tmp_path):
    db = await Database.connect(str(tmp_path / "queue.sqlite"))
    await db.migrate()
    try:
        yield SqlJobQueue(db)
    finally:
        await db.close()


async def cancel(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_enqueue_notifies_after_commit_and_only_its_kind(queue, monkeypatch):
    ingest = queue.work_wakeup(JobKind.INGEST)
    process = asyncio.create_task(queue.work_wakeup(JobKind.PROCESS)())
    states = []
    original = queue._notify_work

    def notify(kind):
        states.append(queue._db._conn.in_transaction)
        original(kind)

    monkeypatch.setattr(queue, "_notify_work", notify)
    try:
        job = await queue.enqueue(JobKind.INGEST, {"url": "fixture://one"})
        assert await asyncio.wait_for(ingest(), 1)
        assert states == [False]
        await asyncio.sleep(0)
        assert not process.done()
        assert (await queue.claim((JobKind.INGEST,))).id == job.id
    finally:
        await cancel(process)


async def test_enqueue_between_empty_claim_and_wait_is_not_lost(queue):
    wakeup = queue.work_wakeup(JobKind.PROCESS)
    assert await queue.claim((JobKind.PROCESS,)) is None
    job = await queue.enqueue(JobKind.PROCESS, {"video_id": "one"})
    assert await asyncio.wait_for(wakeup(), 1)
    assert (await queue.claim((JobKind.PROCESS,))).id == job.id


async def test_multiple_waiters_do_not_consume_each_others_generation(queue):
    first = queue.work_wakeup(JobKind.PROCESS)
    second = queue.work_wakeup(JobKind.PROCESS)
    job = await queue.enqueue(JobKind.PROCESS, {"video_id": "one"})
    assert await asyncio.wait_for(asyncio.gather(first(), second()), 1) == [True, True]
    claims = await asyncio.gather(
        queue.claim((JobKind.PROCESS,)), queue.claim((JobKind.PROCESS,))
    )
    assert [claim.id for claim in claims if claim] == [job.id]
    next_generation = asyncio.create_task(queue.work_wakeup(JobKind.PROCESS)())
    try:
        await asyncio.sleep(0)
        assert not next_generation.done()
        await queue.enqueue(JobKind.PROCESS, {"video_id": "two"})
        assert await asyncio.wait_for(next_generation, 1)
    finally:
        await cancel(next_generation)


async def test_queued_duplicate_wakes_but_running_duplicate_does_not(queue):
    shadow = SqlJobQueue(queue._db)
    job = await shadow.enqueue(JobKind.PROCESS, {"video_id": "one"})
    wakeup = queue.work_wakeup(JobKind.PROCESS)
    duplicate = await queue.enqueue(JobKind.PROCESS, {"video_id": "one"})
    assert duplicate.id == job.id
    assert await asyncio.wait_for(wakeup(), 1)
    await queue.claim((JobKind.PROCESS,))
    waiting = asyncio.create_task(queue.work_wakeup(JobKind.PROCESS)())
    try:
        assert (await queue.enqueue(JobKind.PROCESS, {"video_id": "one"})).id == job.id
        await asyncio.sleep(0)
        assert not waiting.done()
    finally:
        await cancel(waiting)


async def test_failed_transaction_does_not_signal_and_future_job_stays_not_due(queue):
    waiting = asyncio.create_task(queue.work_wakeup(JobKind.PROCESS)())
    try:
        with pytest.raises(Exception, match="FOREIGN KEY"):
            await queue.enqueue(JobKind.PROCESS, {"video_id": "missing"}, video_id="missing")
        await asyncio.sleep(0)
        assert not waiting.done()
        await queue.enqueue(
            JobKind.PROCESS, {"video_id": "future"},
            run_at=datetime.now(UTC) + timedelta(minutes=1),
        )
        await asyncio.sleep(0)
        assert not waiting.done()
        await queue.enqueue(JobKind.PROCESS, {"video_id": "future"})
        await asyncio.sleep(0)
        assert not waiting.done()  # dedup preserves the existing future deadline
        assert await queue.claim((JobKind.PROCESS,)) is None
    finally:
        await cancel(waiting)


async def test_stale_recovery_notifies_after_making_jobs_due(queue):
    job = await queue.enqueue(JobKind.PROCESS, {"video_id": "stale"})
    await queue.claim((JobKind.PROCESS,))
    await queue._db.execute(
        "UPDATE jobs SET updated_at = ? WHERE id = ?", ("2000-01-01T00:00:00.000000+00:00", job.id)
    )
    wakeup = queue.work_wakeup(JobKind.PROCESS)
    assert await queue.requeue_stale(60) == 1
    assert await asyncio.wait_for(wakeup(), 1)
    assert (await queue.claim((JobKind.PROCESS,))).id == job.id


async def test_rotation_does_not_retain_notification_history(queue):
    references = []
    for index in range(100):
        token = queue.work_wakeup(JobKind.PROCESS)
        references.append(weakref.ref(token.__self__))
        await queue.enqueue(JobKind.PROCESS, {"video_id": str(index)})
    del token
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("kind", [JobKind.INGEST, JobKind.PROCESS])
@pytest.mark.parametrize("other_instance", [False, True])
async def test_real_worker_observes_local_wakeup_or_external_poll_fallback(
    queue, monkeypatch, kind, other_instance
):
    empty = asyncio.Event()
    started = asyncio.Event()
    original_claim = queue.claim

    async def claim(kinds):
        job = await original_claim(kinds)
        if job is None:
            empty.set()
        return job

    async def consume(*_args, **_kwargs):
        started.set()
        return None

    monkeypatch.setattr(queue, "claim", claim)
    service = SimpleNamespace(ingest=consume, process=consume)
    stop = asyncio.Event()
    worker = _ingest_worker if kind is JobKind.INGEST else _process_worker
    task = asyncio.create_task(worker(queue, SqlVideoStore(queue._db), service, stop))
    try:
        await asyncio.wait_for(empty.wait(), 3)
        target = SqlJobQueue(queue._db) if other_instance else queue
        await target.enqueue(kind, {"url": "fixture://one", "video_id": "one"})
        await asyncio.wait_for(started.wait(), 3)
    finally:
        stop.set()
        await cancel(task)
    assert task.done()


@pytest.mark.parametrize("kind", [JobKind.INGEST, JobKind.PROCESS])
async def test_local_worker_awaits_notification_before_claiming_again(queue, monkeypatch, kind):
    """A polling-only worker never enters this gate, regardless of its speed."""
    # Keep the fallback outside the watchdog; the gate, not elapsed speed, proves wakeup.
    monkeypatch.setattr("vra.runtime._POLL_INTERVAL_S", 10.0)
    waiting = asyncio.Event()
    notified = asyncio.Event()
    release = asyncio.Event()
    started = asyncio.Event()
    original = queue.work_wakeup

    def snapshot(job_kind):
        token = original(job_kind)

        async def wait():
            waiting.set()
            result = await token()
            notified.set()
            await release.wait()
            return result

        return wait

    async def consume(*_args, **_kwargs):
        started.set()

    monkeypatch.setattr(queue, "work_wakeup", snapshot)
    service = SimpleNamespace(ingest=consume, process=consume)
    stop = asyncio.Event()
    worker = _ingest_worker if kind is JobKind.INGEST else _process_worker
    task = asyncio.create_task(worker(queue, SqlVideoStore(queue._db), service, stop))
    try:
        await asyncio.wait_for(waiting.wait(), 3)
        await queue.enqueue(kind, {"url": "fixture://one", "video_id": "one"})
        await asyncio.wait_for(notified.wait(), 3)
        assert not started.is_set()
        release.set()
        await asyncio.wait_for(started.wait(), 3)
    finally:
        stop.set()
        await cancel(task)


async def test_cancelled_waiter_does_not_consume_survivor_notification(queue):
    for index in range(20):
        token = queue.work_wakeup(JobKind.PROCESS)
        event = token.__self__
        cancelled = asyncio.create_task(token())
        survivor = asyncio.create_task(token())
        try:
            await asyncio.sleep(0)
            await cancel(cancelled)
            assert len(event._waiters) == 1
            job = await queue.enqueue(JobKind.PROCESS, {"video_id": f"cancel-{index}"})
            assert await asyncio.wait_for(survivor, 1)
            assert not event._waiters
            assert (await queue.claim((JobKind.PROCESS,))).id == job.id
            assert await queue.claim((JobKind.PROCESS,)) is None
        finally:
            await cancel(cancelled)
            await cancel(survivor)


async def test_future_local_job_runs_via_unchanged_fallback(queue, monkeypatch):
    started = asyncio.Event()
    waiting = asyncio.Event()
    original = queue.work_wakeup
    notified = []

    def snapshot(kind):
        token = original(kind)

        async def wait():
            waiting.set()
            result = await token()
            notified.append(True)
            return result

        return wait

    monkeypatch.setattr(queue, "work_wakeup", snapshot)
    deadline = None

    async def consume(*_args, **_kwargs):
        assert deadline is not None and datetime.now(UTC) >= deadline
        started.set()

    stop = asyncio.Event()
    task = asyncio.create_task(_process_worker(
        queue, SqlVideoStore(queue._db), SimpleNamespace(process=consume), stop
    ))
    try:
        await asyncio.wait_for(waiting.wait(), 3)
        deadline = datetime.now(UTC) + timedelta(milliseconds=100)
        await queue.enqueue(JobKind.PROCESS, {"video_id": "later"}, run_at=deadline)
        await asyncio.wait_for(started.wait(), 3)
        assert notified == []  # the runtime's unchanged timeout, not a local signal, drove the claim
    finally:
        stop.set()
        await cancel(task)
