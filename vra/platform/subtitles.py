"""VTT/SRT subtitle parsing into Transcript with rolling-caption dedupe.

parse_file(path, max_chars) -> Transcript (sync; callers on the event loop
should wrap in asyncio.to_thread). Handles both WebVTT ("00:00:01.000 -->"
with "." milliseconds) and SRT ("00:00:01,000 -->" with "," milliseconds).

Parsing rules:
- Cue state machine: a timestamp line starts a cue; text lines accumulate
  until a blank line. Everything outside cues (WEBVTT header, NOTE blocks,
  Kind:/Language: headers, cue identifiers, SRT sequence numbers) is dropped.
- Markup stripped: ``<c>``/``<00:00:01.000>`` angle tags and ``{\\an8}`` brace
  tags become spaces; whitespace collapsed. Non-speech markers such as
  ``[Music]`` are kept as-is.
- Rolling auto-sub handling: a text line identical to the previously accepted
  line (across cue boundaries) is dropped, and a cue whose merged text equals
  the previous cue's text is merged away.
- Output is clamped to max_chars, cutting mid-segment at a character boundary.

OWNER: agent-A. See docs/v2/SPEC.md.
"""

from __future__ import annotations

import re
from pathlib import Path

from vra.types import Transcript, TranscriptSeg

_TIMESTAMP = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_ANGLE_TAG = re.compile(r"<[^>]*>")
_BRACE_TAG = re.compile(r"\{[^}]*\}")
_WHITESPACE = re.compile(r"\s+")


def _to_seconds(match: re.Match[str]) -> float:
    hours, minutes, seconds, millis = match.groups()
    total = int(minutes) * 60 + int(seconds)
    if hours:
        total += int(hours) * 3600
    return total + int(millis.ljust(3, "0")) / 1000.0


def _clean_line(raw: str) -> str:
    """Strip markup and collapse whitespace from one cue text line."""
    cleaned = _ANGLE_TAG.sub(" ", raw)
    cleaned = _BRACE_TAG.sub(" ", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def _parse_cues(text: str) -> list[tuple[float, float, list[str]]]:
    """Extract (start, end, raw_lines) cues with a line-oriented state machine."""
    cues: list[tuple[float, float, list[str]]] = []
    start = end = 0.0
    in_cue = False
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if "-->" in line:
            stamps = list(_TIMESTAMP.finditer(line))
            if len(stamps) >= 2:
                if in_cue and lines:
                    cues.append((start, end, lines))
                start = _to_seconds(stamps[0])
                end = _to_seconds(stamps[1])
                in_cue = True
                lines = []
            continue
        if not line:
            if in_cue and lines:
                cues.append((start, end, lines))
            in_cue = False
            lines = []
            continue
        if in_cue:
            lines.append(line)
    if in_cue and lines:
        cues.append((start, end, lines))
    return cues


def parse_file(path: Path, max_chars: int) -> Transcript:
    """Parse a .vtt/.srt file into a deduplicated, clamped Transcript."""
    if max_chars <= 0:
        return Transcript(text="", source="subtitle")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"cannot read subtitle file {path}: {exc}") from exc

    segments: list[TranscriptSeg] = []
    total = 0
    prev_line: str | None = None
    prev_text: str | None = None
    for start, end, raw_lines in _parse_cues(text):
        kept: list[str] = []
        for raw in raw_lines:
            cleaned = _clean_line(raw)
            if not cleaned or cleaned == prev_line:
                continue
            prev_line = cleaned
            kept.append(cleaned)
        merged = " ".join(kept)
        if not merged or merged == prev_text:
            continue
        prev_text = merged
        sep = 1 if segments else 0
        available = max_chars - total - sep
        if available <= 0:
            break
        take = merged[:available]
        segments.append(TranscriptSeg(start_s=start, end_s=end, text=take))
        total += sep + len(take)
        if len(take) < len(merged) or total >= max_chars:
            break
    return Transcript(
        text=" ".join(seg.text for seg in segments),
        source="subtitle",
        segments=tuple(segments),
    )
