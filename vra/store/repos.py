"""Repositories implementing contracts.VideoStore. OWNER: agent-C.

SqlVideoStore(db): all methods from contracts.VideoStore, JSON columns for
key_points/visual_highlights/segments/chapters/payload. timestamps ISO-8601 UTC.
latest_publications joins videos+latest summary per video ordered by
summary.created_at desc. video_detail returns video + transcript + frames +
summaries. counts(): jobs_by_state, videos_by_status, feeds, summaries.
"""
