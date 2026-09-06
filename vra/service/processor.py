"""Per-video pipeline orchestrator. OWNER: agent-C.

``Processor`` implements contracts.SourceProcessor: acquire (metadata +
subtitles, ASR fallback when enabled, media download only when frames are
needed or ASR requires it) → extract frames → persist rows → summarize →
mark done. Any pipeline exception marks the video failed (error truncated to
800 chars) and is reported as ``"failed"`` — it never re-raises, because a
failed video is a durable state the queue/scheduler reasons about. An unknown
video_id raises ValueError (caller error, not a video state).

Concurrency note: callers (queue workers) own parallelism; GPU serialization
happens inside the summarizer. Downloads are reused — when ASR already
fetched the media, frame extraction reuses the same file, and vice versa the
media is downloaded at most once per run.
"""

from __future__ import annotations

import logging

from vra.config import AsrMode, Config
from vra.contracts import (
    FrameExtractor,
    Summarizer,
    Transcriber,
    VideoAcquirer,
    VideoStore,
)
from vra.types import FrameSet, MediaAsset, Transcript, VideoStatus

log = logging.getLogger(__name__)

_ERROR_MAX_CHARS = 800


class Processor:
    """contracts.SourceProcessor implementation wired via constructor DI."""

    def __init__(
        self,
        store: VideoStore,
        acquirer: VideoAcquirer,
        transcriber: Transcriber,
        extractor: FrameExtractor,
        summarizer: Summarizer,
        config: Config,
    ) -> None:
        self._store = store
        self._acquirer = acquirer
        self._transcriber = transcriber
        self._extractor = extractor
        self._summarizer = summarizer
        self._config = config

    async def process(self, video_id: str, *, force: bool = False) -> str:
        """Run the full pipeline for one video; returns ``done``/``failed``."""
        rec = await self._store.get_video(video_id)
        if rec is None:
            raise ValueError(f"unknown video: {video_id}")
        if rec.status == VideoStatus.DONE and not force:
            return "done"

        try:
            await self._store.set_video_status(
                video_id, VideoStatus.ACQUIRING, error=None
            )
            meta = await self._acquirer.resolve(rec.source_url)
            transcript = await self._acquirer.subtitles(
                rec.source_url, meta.canonical_id
            )
            warnings: list[str] = []
            media_path: str | None = None

            if (
                transcript is None
                and self._config.asr == AsrMode.AUTO
                and await self._transcriber.available()
            ):
                media_path = await self._acquirer.download(
                    rec.source_url, meta.canonical_id
                )
                transcript = await self._transcriber.transcribe(media_path)

            transcript = transcript or Transcript(text="", source="none")

            frames = FrameSet()
            if self._config.max_frames > 0:
                if media_path is None:
                    media_path = await self._acquirer.download(
                        rec.source_url, meta.canonical_id
                    )
                frames = await self._extractor.extract(
                    media_path,
                    canonical_id=meta.canonical_id,
                    duration_s=meta.duration_s,
                    max_frames=self._config.max_frames,
                )

            await self._store.save_meta(video_id, meta, media_path)
            await self._store.save_transcript(video_id, transcript)
            await self._store.save_frames(video_id, frames)
            await self._store.set_video_status(
                video_id, VideoStatus.SUMMARIZING, error=None
            )
            asset = MediaAsset(
                meta=meta,
                media_path=media_path,
                transcript=transcript,
                frames=frames,
                warnings=tuple(warnings),
            )
            summary = await self._summarizer.summarize(asset)
            await self._store.save_summary(video_id, summary)
            await self._store.set_video_status(video_id, VideoStatus.DONE, error=None)
            return "done"
        except Exception as exc:
            log.warning("process failed for video %s: %s", video_id, exc)
            try:
                await self._store.set_video_status(
                    video_id,
                    VideoStatus.FAILED,
                    error=str(exc)[:_ERROR_MAX_CHARS],
                )
            except Exception:
                log.exception("could not mark video %s as failed", video_id)
            return "failed"
