from __future__ import annotations

from service_summarize import SummaryResult as LegacySummaryResult
from video_rss_aggregator.domain.models import SummaryResult
from video_rss_aggregator.storage import Database as PackageDatabase, SummaryRecord


class Database(PackageDatabase):
    async def insert_summary(self, video_id: str, result: SummaryResult) -> str:
        if isinstance(result, LegacySummaryResult):
            result = SummaryResult(
                summary=result.summary,
                key_points=tuple(result.key_points),
                visual_highlights=tuple(result.visual_highlights),
                model_used=result.model_used,
                vram_mb=result.vram_mb,
                transcript_chars=result.transcript_chars,
                frame_count=result.frame_count,
                error=result.error,
            )
        return await super().insert_summary(video_id, result)


__all__ = ["Database", "SummaryRecord"]
