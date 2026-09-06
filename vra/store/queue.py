"""Durable asyncio job queue on jobs table. OWNER: agent-C. Implements contracts.JobQueue.

SqlJobQueue(db, poll_interval_s=1.0):
- enqueue(dedup_key): if a queued/running job with same kind+payload canonical json
  exists -> return it (idempotent).
- claim(kinds): UPDATE ... SET state='running', attempts=attempts+1 WHERE id=
  (SELECT id FROM jobs WHERE state='queued' AND kind IN (...) AND (run_at IS NULL OR
  run_at<=now) ORDER BY created_at LIMIT 1) RETURNING * — single writer (aiosqlite
  connection serialized via asyncio.Lock).
- fail(retry=True): attempts<max -> state=queued, run_at=now+30*2^attempts else failed.
- requeue_stale(1800s): running jobs with updated_at older -> queued.
"""
