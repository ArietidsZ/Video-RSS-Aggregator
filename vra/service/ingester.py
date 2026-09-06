"""Feed polling + enqueue. OWNER: agent-C. Implements contracts.IngestService.

FeedIngester(fetcher, store, queue, config):
- ingest(url): get_feed_by_url -> fetch(etag,last_modified) -> if not_modified:
  touch last_checked, return zero counts. Upsert feed, for each entry:
  canonical resolution: guid if absolute http(s) guid else source_url ->
  sha1... canonical_id = sha1(normalized source_url) when platform unknown
  (cheap path); upsert_video_ref; created or status in (pending,failed) -> enqueue
  process job (dedup_key="process:"+video_id, force=False); count new/enqueued/skipped.
- Force flag comes from API /process (force=True bypasses done-skip) — ingest never forces.
"""
