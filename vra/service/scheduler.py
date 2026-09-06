"""Background per-feed poll loop. OWNER: agent-C.

``PollScheduler`` implements contracts.FeedScheduler with a single asyncio
task: every 60s it lists feeds and, for each due feed, enqueues an
``ingest`` job. A feed is due when it has never been checked or when its
last check is older than ``poll_minutes * 60`` seconds times a jitter factor
uniformly drawn from [1.0, 1.1) — this spreads polls and lets the durable
queue dedup handle overlapping scans. ``start()`` is idempotent;
``stop()`` cancels the task and is safe to call any number of times (and
before ``start()``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from datetime import UTC, datetime

from vra.config import Config
from vra.contracts import IngestService, JobQueue, VideoStore
from vra.types import FeedRecord, JobKind

log = logging.getLogger(__name__)

SCAN_INTERVAL_S = 60.0


def _jitter_factor() -> float:
    """Poll-interval multiplier in [1.0, 1.1) (module-level for tests)."""
    return 1.0 + random.random() * 0.1


class PollScheduler:
    """Single-task feed polling scheduler."""

    def __init__(
        self,
        ingester: IngestService,
        store: VideoStore,
        queue: JobQueue,
        config: Config,
    ) -> None:
        self._ingester = ingester
        self._store = store
        self._queue = queue
        self._config = config
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Spawn the scan loop once (idempotent; restarts after a crash)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="vra-poll-scheduler")

    async def stop(self) -> None:
        """Cancel the scan loop; idempotent and safe before start()."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        while True:
            try:
                await self._scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler scan failed")
            await asyncio.sleep(SCAN_INTERVAL_S)

    async def _scan(self) -> None:
        """Enqueue ingest jobs for every due feed (dedup guards overlap)."""
        feeds = await self._store.list_feeds()
        now = datetime.now(UTC)
        for feed in feeds:
            if not self._is_due(feed, now):
                continue
            try:
                await self._queue.enqueue(
                    JobKind.INGEST,
                    {"url": feed.url},
                    dedup_key=f"ingest:{feed.id}",
                )
            except Exception:
                log.warning(
                    "failed to enqueue ingest job for %s", feed.url, exc_info=True
                )

    def _is_due(self, feed: FeedRecord, now: datetime) -> bool:
        if feed.last_checked is None:
            return True
        elapsed = (now - feed.last_checked).total_seconds()
        return elapsed >= feed.poll_minutes * 60 * _jitter_factor()
