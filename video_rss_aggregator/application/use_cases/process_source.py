from __future__ import annotations

from dataclasses import dataclass

from video_rss_aggregator.application.ports import (
    MediaPreparationService,
    SummaryRepository,
    Summarizer,
    VideoRepository,
)
from video_rss_aggregator.domain.outcomes import (
    Failure,
    PartialSuccess,
    ProcessOutcome,
    Success,
)


@dataclass(frozen=True)
class ProcessSource:
    media_service: MediaPreparationService
    summarizer: Summarizer
    videos: VideoRepository
    summaries: SummaryRepository

    async def execute(self, source_url: str, title: str | None) -> ProcessOutcome:
        try:
            media = await self.media_service.prepare(source_url, title)
            summary = await self.summarizer.summarize(media)

            video_id = await self.videos.save(media)
            await self.summaries.save(video_id, summary)
        except Exception as exc:
            return Failure(source_url=source_url, reason=str(exc))

        if summary.error is not None:
            return PartialSuccess(media=media, reason=summary.error, summary=summary)

        return Success(media=media, summary=summary)
