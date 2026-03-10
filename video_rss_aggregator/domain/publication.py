from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PublicationRecord:
    title: str | None
    source_url: str
    published_at: datetime | None
    summary: str
    key_points: tuple[str, ...]
    visual_highlights: tuple[str, ...]
    model_used: str | None
    vram_mb: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_points", tuple(self.key_points))
        object.__setattr__(self, "visual_highlights", tuple(self.visual_highlights))
