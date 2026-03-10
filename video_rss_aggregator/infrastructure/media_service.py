from __future__ import annotations

from dataclasses import dataclass

import httpx

import service_media
from video_rss_aggregator.domain.models import PreparedMedia


def _map_prepared_media(
    source_url: str,
    title: str | None,
    legacy_media: service_media.PreparedMedia,
) -> PreparedMedia:
    resolved_title = title or legacy_media.title or source_url
    return PreparedMedia(
        source_url=source_url,
        title=resolved_title,
        transcript=legacy_media.transcript,
        media_path=str(legacy_media.media_path),
        frame_paths=tuple(str(path) for path in legacy_media.frame_paths),
    )


@dataclass(frozen=True)
class LegacyMediaPreparationService:
    client: httpx.AsyncClient
    storage_dir: str
    max_frames: int
    scene_detection: bool
    scene_threshold: float
    scene_min_frames: int
    max_transcript_chars: int

    async def prepare(self, source_url: str, title: str | None) -> PreparedMedia:
        legacy_media = await service_media.prepare_media(
            client=self.client,
            source=source_url,
            storage_dir=self.storage_dir,
            max_frames=self.max_frames,
            scene_detection=self.scene_detection,
            scene_threshold=self.scene_threshold,
            scene_min_frames=self.scene_min_frames,
            max_transcript_chars=self.max_transcript_chars,
        )
        return _map_prepared_media(source_url, title, legacy_media)
