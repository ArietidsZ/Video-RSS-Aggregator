from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import struct_time
from typing import Any

import feedparser
import httpx

from adapter_rss import render_feed
from adapter_storage import Database
from core_config import Config
from service_media import prepare_media
from service_summarize import SummarizationEngine, SummaryResult


@dataclass(slots=True)
class IngestReport:
    feed_title: str | None = None
    item_count: int = 0
    processed_count: int = 0


@dataclass(slots=True)
class ProcessReport:
    source_url: str
    title: str | None
    transcript_chars: int
    frame_count: int
    summary: SummaryResult


class Pipeline:
    def __init__(
        self,
        config: Config,
        db: Database,
        summarizer: SummarizationEngine,
    ) -> None:
        self._config = config
        self._db = db
        self._summarizer = summarizer
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=300.0, pool=60.0),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

    @classmethod
    async def create(cls, config: Config) -> Pipeline:
        db = await Database.connect(config.database_path)
        await db.migrate()

        summarizer = SummarizationEngine(config)
        await summarizer.prepare_models()

        return cls(config, db, summarizer)

    async def close(self) -> None:
        await self._client.aclose()
        await self._summarizer.close()
        await self._db.close()

    async def runtime_status(self) -> dict[str, Any]:
        status = await self._summarizer.runtime_status()
        status["database_path"] = self._db.path
        status["storage_dir"] = self._config.storage_dir
        status["models"] = list(self._config.model_priority)
        return status

    async def bootstrap_models(self) -> dict[str, Any]:
        prepared = await self._summarizer.prepare_models()
        return {
            "models_prepared": prepared,
            "runtime": await self.runtime_status(),
        }

    async def ingest_feed(
        self,
        feed_url: str,
        process: bool = False,
        max_items: int | None = None,
    ) -> IngestReport:
        resp = await self._client.get(feed_url)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.text)

        feed_title = parsed.feed.get("title")
        feed_id = await self._db.upsert_feed(feed_url, feed_title)

        entries = parsed.entries
        if max_items is not None:
            entries = entries[:max_items]

        report = IngestReport(feed_title=feed_title)

        for entry in entries:
            source_url = _pick_source_url(entry)
            if not source_url:
                continue

            title = entry.get("title")
            guid = entry.get("id") or None
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            published_at = _struct_to_dt(published) if published else None

            video_id = await self._db.upsert_video(
                feed_id,
                guid,
                title,
                source_url,
                published_at,
            )
            report.item_count += 1

            if process:
                processed = await self._process_with_video(video_id, source_url, title)
                if processed.summary.summary:
                    report.processed_count += 1

        return report

    async def process_source(
        self,
        source_url: str,
        title: str | None = None,
    ) -> ProcessReport:
        video_id = await self._db.upsert_video(
            feed_id=None,
            guid=None,
            title=title,
            source_url=source_url,
            published_at=None,
        )
        return await self._process_with_video(video_id, source_url, title)

    async def rss_feed(self, limit: int = 20) -> str:
        records = await self._db.latest_summaries(limit)
        return render_feed(
            self._config.rss_title,
            self._config.rss_link,
            self._config.rss_description,
            records,
        )

    async def _process_with_video(
        self,
        video_id: str,
        source_url: str,
        title: str | None,
    ) -> ProcessReport:
        prepared = await prepare_media(
            client=self._client,
            source=source_url,
            storage_dir=self._config.storage_dir,
            max_frames=self._config.max_frames,
            scene_detection=self._config.frame_scene_detection,
            scene_threshold=self._config.frame_scene_threshold,
            scene_min_frames=self._config.frame_scene_min_frames,
            max_transcript_chars=self._config.max_transcript_chars,
        )

        resolved_title = title or prepared.title
        await self._db.upsert_video(
            feed_id=None,
            guid=None,
            title=resolved_title,
            source_url=source_url,
            published_at=None,
            media_path=str(prepared.media_path),
        )

        if prepared.transcript:
            await self._db.insert_transcript(video_id, prepared.transcript)

        summary = await self._summarizer.summarize(
            source_url=source_url,
            title=resolved_title,
            transcript=prepared.transcript,
            frame_paths=prepared.frame_paths,
        )
        await self._db.insert_summary(video_id, summary)

        return ProcessReport(
            source_url=source_url,
            title=resolved_title,
            transcript_chars=len(prepared.transcript),
            frame_count=len(prepared.frame_paths),
            summary=summary,
        )


def _pick_source_url(entry: Any) -> str | None:
    enclosures = entry.get("enclosures", [])
    if enclosures:
        return enclosures[0].get("href") or enclosures[0].get("url")
    links = entry.get("links", [])
    if links:
        return links[0].get("href")
    return entry.get("link")


def _struct_to_dt(value: struct_time) -> datetime | None:
    try:
        return datetime(*value[:6], tzinfo=timezone.utc)
    except Exception:
        return None
