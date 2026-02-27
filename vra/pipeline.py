from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from io import BytesIO

import feedparser
import httpx

from .config import Config
from .media import prepare_audio
from .rss import render_feed
from .storage import Database
from .summarize import SummarizationEngine, SummaryResult
from .transcribe import TranscriptionEngine, TranscriptionResult

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestReport:
    feed_title: str | None = None
    item_count: int = 0
    processed_count: int = 0


@dataclass(slots=True)
class ProcessReport:
    source_url: str = ""
    title: str | None = None
    transcription: TranscriptionResult | None = None
    summary: SummaryResult | None = None


class Pipeline:
    def __init__(
        self,
        config: Config,
        db: Database,
        transcriber: TranscriptionEngine,
        summarizer: SummarizationEngine,
    ) -> None:
        self._config = config
        self._db = db
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._client = httpx.AsyncClient(timeout=300, follow_redirects=True)

    @classmethod
    async def create(cls, config: Config) -> Pipeline:
        db = await Database.connect(config.database_url)
        await db.migrate()

        # Load models in background threads to keep event loop responsive
        transcriber = await asyncio.to_thread(TranscriptionEngine.get, config)
        summarizer = await asyncio.to_thread(SummarizationEngine.get, config)

        return cls(config, db, transcriber, summarizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_feed(
        self,
        feed_url: str,
        process: bool = False,
        max_items: int | None = None,
    ) -> IngestReport:
        resp = await self._client.get(feed_url)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)

        feed_title = feed.feed.get("title")
        feed_id = await self._db.upsert_feed(feed_url, feed_title)

        entries = feed.entries
        if max_items is not None:
            entries = entries[:max_items]

        report = IngestReport(feed_title=feed_title)

        for entry in entries:
            title = entry.get("title")
            guid = entry.get("id") or None
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            pub_dt = _struct_to_dt(published) if published else None

            source_url = _pick_source_url(entry)
            if not source_url:
                continue

            video_id = await self._db.upsert_video(feed_id, guid, title, source_url, pub_dt)
            report.item_count += 1

            if process:
                pr = await self._process_with_video(video_id, source_url, title)
                if pr.summary and pr.summary.summary:
                    report.processed_count += 1

        return report

    async def process_source(self, source_url: str, title: str | None = None) -> ProcessReport:
        video_id = await self._db.upsert_video(None, None, title, source_url, None)
        return await self._process_with_video(video_id, source_url, title)

    async def rss_feed(self, title: str, link: str, description: str, limit: int = 20) -> str:
        records = await self._db.latest_summaries(limit)
        return render_feed(title, link, description, records)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process_with_video(
        self,
        video_id,
        source_url: str,
        title: str | None,
    ) -> ProcessReport:
        audio_path = await prepare_audio(self._client, source_url, self._config.storage_dir)
        audio_str = str(audio_path)

        tr = await asyncio.to_thread(self._transcriber.transcribe, audio_str)
        sr = await asyncio.to_thread(self._summarizer.summarize, tr.text)

        await self._db.insert_transcript(video_id, tr)
        await self._db.insert_summary(video_id, sr)

        return ProcessReport(source_url=source_url, title=title, transcription=tr, summary=sr)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pick_source_url(entry) -> str | None:
    enclosures = entry.get("enclosures", [])
    if enclosures:
        return enclosures[0].get("href") or enclosures[0].get("url")
    links = entry.get("links", [])
    if links:
        return links[0].get("href")
    return entry.get("link")


def _struct_to_dt(st):
    from datetime import datetime, timezone
    try:
        return datetime(*st[:6], tzinfo=timezone.utc)
    except Exception:
        return None
