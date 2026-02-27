from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

import asyncpg

from .transcribe import TranscriptionResult
from .summarize import SummaryResult

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS feeds (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    last_checked TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY,
    feed_id UUID REFERENCES feeds(id),
    source_url TEXT NOT NULL UNIQUE,
    guid TEXT,
    title TEXT,
    published_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS transcripts (
    id UUID PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(id),
    language TEXT,
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS summaries (
    id UUID PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(id),
    summary TEXT NOT NULL,
    key_points JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);"""


@dataclass(slots=True)
class SummaryRecord:
    title: str | None
    source_url: str
    published_at: datetime | None
    summary: str
    key_points: list[str]
    created_at: datetime


class Database:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, database_url: str) -> Database:
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        return cls(pool)

    async def migrate(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)
        log.info("Database schema ready")

    async def upsert_feed(self, url: str, title: str | None) -> uuid.UUID:
        fid = uuid.uuid4()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO feeds (id, url, title, last_checked)
                   VALUES ($1, $2, $3, NOW())
                   ON CONFLICT (url)
                   DO UPDATE SET title = COALESCE(EXCLUDED.title, feeds.title),
                                 last_checked = NOW()
                   RETURNING id""",
                fid, url, title,
            )
        return row["id"]

    async def upsert_video(
        self,
        feed_id: uuid.UUID | None,
        guid: str | None,
        title: str | None,
        source_url: str,
        published_at: datetime | None,
    ) -> uuid.UUID:
        vid = uuid.uuid4()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO videos (id, feed_id, source_url, guid, title, published_at)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (source_url)
                   DO UPDATE SET
                       title = COALESCE(EXCLUDED.title, videos.title),
                       published_at = COALESCE(EXCLUDED.published_at, videos.published_at),
                       guid = COALESCE(EXCLUDED.guid, videos.guid),
                       feed_id = COALESCE(EXCLUDED.feed_id, videos.feed_id)
                   RETURNING id""",
                vid, feed_id, source_url, guid, title, published_at,
            )
        return row["id"]

    async def insert_transcript(self, video_id: uuid.UUID, tr: TranscriptionResult) -> uuid.UUID:
        tid = uuid.uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO transcripts (id, video_id, language, text)
                   VALUES ($1, $2, $3, $4)""",
                tid, video_id, tr.language, tr.text,
            )
        return tid

    async def insert_summary(self, video_id: uuid.UUID, sr: SummaryResult) -> uuid.UUID:
        sid = uuid.uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO summaries (id, video_id, summary, key_points)
                   VALUES ($1, $2, $3, $4::jsonb)""",
                sid, video_id, sr.summary, json.dumps(sr.key_points),
            )
        return sid

    async def latest_summaries(self, limit: int = 20) -> list[SummaryRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT v.title, v.source_url, v.published_at,
                          s.summary, s.key_points, s.created_at
                   FROM summaries s
                   JOIN videos v ON v.id = s.video_id
                   ORDER BY s.created_at DESC
                   LIMIT $1""",
                limit,
            )
        out: list[SummaryRecord] = []
        for r in rows:
            kp = r["key_points"]
            if isinstance(kp, str):
                kp = json.loads(kp)
            out.append(SummaryRecord(
                title=r["title"],
                source_url=r["source_url"],
                published_at=r["published_at"],
                summary=r["summary"],
                key_points=kp if isinstance(kp, list) else [],
                created_at=r["created_at"],
            ))
        return out
