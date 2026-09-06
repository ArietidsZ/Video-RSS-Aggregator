"""Background per-feed poll loop. OWNER: agent-C. Implements contracts.FeedScheduler.

PollScheduler(ingester, store, queue, config): single asyncio task scanning feeds
every 60s; feed due when last_checked is NULL or older than poll_minutes (+jitter
0-10%); due -> queue.enqueue(JobKind.INGEST, {"url": feed.url},
dedup_key="ingest:"+feed.id). start() idempotent; stop() cancels task.
"""
