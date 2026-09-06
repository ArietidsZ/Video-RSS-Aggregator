"""VTT/SRT parsing to Transcript with timestamps + rolling-caption dedupe.

OWNER: agent-A.
parse_file(path, max_chars) -> Transcript: strip cues, keep start/end per segment,
merge consecutive identical texts (auto-subs roll), drop markup <c>/<00:00>,
non-speech tags ([Music] kept as-is), clamp to max_chars mid-segment.
"""
