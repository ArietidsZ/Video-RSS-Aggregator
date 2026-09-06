"""yt-dlp integration: metadata, subtitles, download. OWNER: agent-A.

Requirements:
- YtDlpAcquirer implements contracts.VideoAcquirer.
- resolve(): single `yt_dlp --dump-single-json --no-playlist` call (python -m yt_dlp),
  extracting canonical_id (extractor_key lowercased + ':' + id), title,
  description, duration, chapters. Never downloads media.
- subtitles(): --skip-download --write-subs --write-auto-subs --sub-langs
  "<preferred>.*,<preferred>,-live_chat" (preferred = VRA default 'en' +
  interface language zh), vtt/srt; parse via vra.platform.subtitles.parse.
  Cache under data_dir/subtitles/<canonical_id>.* (idempotent).
- download(): bv*+ba/b merge mp4 into data_dir/raw/<canonical_id>.mp4;
  skip if file exists (idempotency); returns path. Bounded by config.timeout_download_s.
- All subprocess via asyncio.create_subprocess_exec, no blocking on loop.
- SSRF: caller validates URLs; here treat input as already-validated.
"""
