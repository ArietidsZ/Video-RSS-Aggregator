from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedMedia:
    source_url: str
    title: str
    transcript: str
    media_path: str
    frame_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_paths", tuple(self.frame_paths))


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    key_points: tuple[str, ...]
    visual_highlights: tuple[str, ...]
    model_used: str
    vram_mb: float
    transcript_chars: int
    frame_count: int
    error: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_points", tuple(self.key_points))
        object.__setattr__(self, "visual_highlights", tuple(self.visual_highlights))
