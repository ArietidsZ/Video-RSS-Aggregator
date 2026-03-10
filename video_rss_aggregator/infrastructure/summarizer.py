from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from service_summarize import SummaryResult as LegacySummaryResult
from video_rss_aggregator.domain.models import PreparedMedia, SummaryResult


class _SummarizationEngine(Protocol):
    async def summarize(
        self,
        *,
        source_url: str,
        title: str | None,
        transcript: str,
        frame_paths: list[Path],
    ) -> LegacySummaryResult: ...


def _map_summary_result(result: LegacySummaryResult) -> SummaryResult:
    return SummaryResult(
        summary=result.summary,
        key_points=tuple(result.key_points),
        visual_highlights=tuple(result.visual_highlights),
        model_used=result.model_used,
        vram_mb=result.vram_mb,
        transcript_chars=result.transcript_chars,
        frame_count=result.frame_count,
        error=result.error,
    )


@dataclass(frozen=True)
class LegacySummarizer:
    engine: _SummarizationEngine

    async def summarize(self, prepared_media: PreparedMedia) -> SummaryResult:
        result = await self.engine.summarize(
            source_url=prepared_media.source_url,
            title=prepared_media.title,
            transcript=prepared_media.transcript,
            frame_paths=[Path(path) for path in prepared_media.frame_paths],
        )
        return _map_summary_result(result)
