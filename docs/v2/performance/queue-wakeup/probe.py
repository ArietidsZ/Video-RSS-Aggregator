import asyncio
import json
import os
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from vra.runtime import _process_worker
from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.store.repos import SqlVideoStore
from vra.types import JobKind


async def run(n):
    with tempfile.TemporaryDirectory() as tmp:
        db = await Database.connect(str(Path(tmp) / "q.sqlite"))
        await db.migrate()
        q = SqlJobQueue(db)
        claims = 0
        empty = 0
        observed = asyncio.Queue()
        starts = {}
        samples = []
        completed = []
        original = q.claim

        async def claim(kinds):
            nonlocal claims, empty
            job = await original(kinds)
            claims += 1
            if job is None:
                empty += 1
            else:
                key = job.payload["video_id"]
                samples.append((time.perf_counter() - starts[key]) * 1000)
                observed.put_nowait(key)
            return job

        q.claim = claim

        async def process(key, **kw):
            completed.append(key)

        stop = asyncio.Event()
        tasks = [
            asyncio.create_task(
                _process_worker(
                    q, SqlVideoStore(db), SimpleNamespace(process=process), stop
                )
            )
            for _ in range(n)
        ]
        cpu = time.process_time()
        wall = time.perf_counter()
        try:
            await asyncio.sleep(0.1)
            for i in range(20):
                key = f"sparse-{i}"
                starts[key] = time.perf_counter()
                await q.enqueue(JobKind.PROCESS, {"video_id": key})
                assert await asyncio.wait_for(observed.get(), 3) == key
                await asyncio.sleep(0.03)
            sparse = {
                "claims": claims,
                "empty": empty,
                "p50_ms": statistics.median(samples),
                "p95_ms": sorted(samples)[18],
            }
            for i in range(32):
                key = f"burst-{i}"
                starts[key] = time.perf_counter()
                await q.enqueue(JobKind.PROCESS, {"video_id": key})
            burst = [await asyncio.wait_for(observed.get(), 3) for _ in range(32)]
            assert len(set(burst)) == 32
            await asyncio.sleep(0.1)
            before = empty
            for _ in range(10):
                await q.enqueue(
                    JobKind.PROCESS,
                    {"video_id": "future"},
                    run_at=datetime.now(UTC) + timedelta(minutes=1),
                )
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.1)
            assert len(completed) == 52 and len(set(completed)) == 52
            result = {
                "workers": n,
                "sparse": sparse,
                "total_claims": claims,
                "total_empty": empty,
                "future_duplicate_empty_delta": empty - before,
                "completed": len(completed),
                "cpu_s": time.process_time() - cpu,
                "wall_s": time.perf_counter() - wall,
            }
        finally:
            stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            result["workers_pending_after_shutdown"] = sum(not t.done() for t in tasks)
            await db.close()
        return result


async def main():
    print(
        json.dumps(
            {
                "label": os.environ["PROBE_LABEL"],
                "matrix": [await run(n) for n in [2, 8, 32]],
            }
        )
    )


asyncio.run(main())
