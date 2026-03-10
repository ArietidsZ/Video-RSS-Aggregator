from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from video_rss_aggregator.domain.models import PreparedMedia, SummaryResult


@dataclass(frozen=True)
class Success:
    media: PreparedMedia
    summary: SummaryResult
    status: str = field(init=False, default="success")


@dataclass(frozen=True)
class PartialSuccess:
    media: PreparedMedia
    reason: str
    summary: SummaryResult
    status: str = field(init=False, default="partial_success")

    def __post_init__(self) -> None:
        if self.summary is None:
            raise ValueError("summary is required")


@dataclass(frozen=True)
class Failure:
    source_url: str
    reason: str
    status: str = field(init=False, default="failure")


ProcessOutcome: TypeAlias = Success | PartialSuccess | Failure
