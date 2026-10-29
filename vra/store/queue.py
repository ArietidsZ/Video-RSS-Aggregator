"""Durable asyncio job queue on the jobs table. OWNER: agent-C.

Implements contracts.JobQueue on top of :class:`vra.store.db.Database`.

- Dedup identity is ``(kind, canonical payload JSON)`` — payloads are stored
  in canonical form (sorted keys, compact separators) so byte-equal canonical
  text means equal payloads. A queued/running job with the same identity is
  returned unchanged; the ``dedup_key`` parameter is advisory (callers use
  it to name identities like ``process:<video_id>``) and is not persisted
  because the frozen schema has no column for it.
- ``claim`` atomically moves the oldest due queued job to running via a
  single ``UPDATE ... RETURNING`` (state->running, attempts+1).
- ``fail`` backs off ``30 * 2**attempts`` seconds while ``attempts <
  max_attempts``, else marks the job failed. ``attempts`` is already
  incremented by claim, so the first retry waits 60s.
- ``requeue_stale`` recovers running jobs whose ``updated_at`` went stale
  (crashed workers): back to queued, ``run_at=now`` so they are immediately
  claimable.

Timestamps compare lexicographically — all writers use
UTC ISO-8601 with an explicit six-digit microsecond field. Legacy stored
non-UTC deadlines need normalization before replay; serialization only
normalizes new writes.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from vra.store.db import Database
from vra.types import Job, JobKind, JobState

_BACKOFF_BASE_S = 30.0


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _iso(value: Any) -> str | None:
    """Canonical UTC deadline; naive datetimes/ISO strings are treated as UTC."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise TypeError("run_at must be a datetime, ISO-8601 string, or None")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _job_from_row(row: aiosqlite.Row) -> Job:
    payload = _safe_payload(row["payload"])
    return Job(
        id=str(row["id"]),
        kind=JobKind(str(row["kind"])),
        state=JobState(str(row["state"])),
        payload=payload,
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        run_at=_parse_dt(row["run_at"]),
        error=row["error"],
        video_id=row["video_id"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _safe_payload(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(str(value)) if value is not None else {}
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


class SqlJobQueue:
    """SQLite-backed implementation of contracts.JobQueue."""

    def __init__(self, db: Database, poll_interval_s: float = 1.0) -> None:
        self._db = db
        #: Hint for worker loops polling claim() (runtime-owned consumers).
        self.poll_interval_s = poll_interval_s

    async def enqueue(
        self,
        kind: JobKind,
        payload: dict[str, Any],
        *,
        video_id: str | None = None,
        run_at: Any = None,
        dedup_key: str | None = None,
    ) -> Job:
        """Enqueue a job, deduplicating on (kind, canonical payload).

        Returns the existing queued/running job unchanged when one matches;
        otherwise inserts a fresh queued job. ``dedup_key`` is accepted for
        caller readability and not persisted.
        """
        normalized_run_at = _iso(run_at)
        del dedup_key  # advisory only; identity is kind + canonical payload
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        now = _now_iso()
        async with self._db.transaction():
            existing = await self._db.fetchone(
                """
                SELECT * FROM jobs
                WHERE kind = ? AND payload = ? AND state IN ('queued', 'running')
                ORDER BY created_at, rowid
                LIMIT 1
                """,
                (kind.value, payload_json),
            )
            if existing is not None:
                return _job_from_row(existing)
            job_id = str(uuid.uuid4())
            await self._db.execute(
                """
                INSERT INTO jobs (
                    id, kind, state, payload, attempts, max_attempts,
                    run_at, video_id, created_at, updated_at
                )
                VALUES (?, ?, 'queued', ?, 0, 3, ?, ?, ?, ?)
                """,
                (job_id, kind.value, payload_json, normalized_run_at, video_id, now, now),
                commit=False,
            )
            row = await self._db.fetchone(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            )
        if row is None:  # pragma: no cover - defensive
            raise RuntimeError(f"job insert failed: {job_id}")
        return _job_from_row(row)

    async def claim(self, kinds: tuple[JobKind, ...]) -> Job | None:
        """Atomically claim the oldest due queued job of any given kind."""
        if not kinds:
            return None
        now = _now_iso()
        placeholders = ", ".join("?" for _ in kinds)
        async with self._db.lock:
            rows = await self._db.execute(
                f"""
                UPDATE jobs
                SET state = 'running', attempts = attempts + 1, updated_at = ?
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE state = 'queued'
                      AND kind IN ({placeholders})
                      AND (run_at IS NULL OR run_at <= ?)
                    ORDER BY created_at, rowid
                    LIMIT 1
                )
                RETURNING *
                """,
                (now, *(kind.value for kind in kinds), now),
            )
        return _job_from_row(rows[0]) if rows else None

    async def complete(self, job_id: str) -> None:
        """Mark a job done."""
        await self._db.execute(
            "UPDATE jobs SET state = 'done', updated_at = ? WHERE id = ?",
            (_now_iso(), job_id),
        )

    async def fail(self, job_id: str, error: str, *, retry: bool) -> None:
        """Record a failure: retry with backoff, or mark terminally failed."""
        now = _now_iso()
        async with self._db.transaction():
            row = await self._db.fetchone(
                "SELECT attempts, max_attempts FROM jobs WHERE id = ?", (job_id,)
            )
            if row is None:
                return
            attempts = int(row["attempts"])
            max_attempts = int(row["max_attempts"])
            if retry and attempts < max_attempts:
                delay = _BACKOFF_BASE_S * (2**attempts)
                run_at = _iso(_utc_now() + timedelta(seconds=delay))

                await self._db.execute(
                    """
                    UPDATE jobs
                    SET state = 'queued', run_at = ?, error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (run_at, error, now, job_id),
                    commit=False,
                )
            else:
                await self._db.execute(
                    """
                    UPDATE jobs
                    SET state = 'failed', error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (error, now, job_id),
                    commit=False,
                )

    async def requeue_stale(self, older_than_s: float) -> int:
        """Requeue running jobs with stale updated_at; returns the count."""
        now_dt = _utc_now()
        cutoff = _iso(now_dt - timedelta(seconds=older_than_s))
        now = _iso(now_dt)
        async with self._db.lock:
            rows = await self._db.execute(
                """
                UPDATE jobs
                SET state = 'queued', run_at = ?, updated_at = ?
                WHERE state = 'running' AND updated_at <= ?
                RETURNING id
                """,
                (now, now, cutoff),
            )
        return len(rows)

    async def list(
        self,
        *,
        state: JobState | None = None,
        kind: JobKind | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs newest-first with optional state/kind filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = await self._db.fetchall(
            f"""
            SELECT * FROM jobs {where}
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            params,
        )
        return [_job_from_row(row) for row in rows]

    async def get(self, job_id: str) -> Job | None:
        row = await self._db.fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return _job_from_row(row) if row is not None else None

    async def close(self) -> None:
        """Close the underlying database (runtime calls this at shutdown)."""
        await self._db.close()
