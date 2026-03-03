from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from service_summarize import SummaryResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    last_checked TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    feed_id TEXT,
    source_url TEXT NOT NULL UNIQUE,
    guid TEXT,
    title TEXT,
    published_at TEXT,
    media_path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_points TEXT NOT NULL,
    visual_highlights TEXT NOT NULL,
    model_used TEXT,
    vram_mb REAL NOT NULL,
    transcript_chars INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_videos_feed_id ON videos(feed_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id);
CREATE INDEX IF NOT EXISTS idx_summaries_video_id ON summaries(video_id);
CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at DESC);
"""

_PRAGMAS = (
    "PRAGMA foreign_keys = ON",
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA busy_timeout = 5000",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(slots=True)
class SummaryRecord:
    title: str | None
    source_url: str
    published_at: datetime | None
    summary: str
    key_points: list[str]
    visual_highlights: list[str]
    model_used: str | None
    vram_mb: float
    created_at: datetime


class Database:
    def __init__(self, conn: aiosqlite.Connection, database_path: str) -> None:
        self._conn = conn
        self._database_path = database_path

    @classmethod
    async def connect(cls, database_path: str) -> Database:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(str(path))
        conn.row_factory = aiosqlite.Row
        for pragma in _PRAGMAS:
            await conn.execute(pragma)
        await conn.commit()
        return cls(conn, str(path))

    @property
    def path(self) -> str:
        return self._database_path

    async def close(self) -> None:
        await self._conn.close()

    async def migrate(self) -> None:
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def upsert_feed(self, url: str, title: str | None) -> str:
        now = _utc_now()
        fid = str(uuid.uuid4())
        await self._conn.execute(
            """
            INSERT INTO feeds (id, url, title, last_checked)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url)
            DO UPDATE SET
              title = COALESCE(excluded.title, feeds.title),
              last_checked = excluded.last_checked
            """,
            (fid, url, title, now),
        )
        await self._conn.commit()

        async with self._conn.execute(
            "SELECT id FROM feeds WHERE url = ?", (url,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to upsert feed")
        return str(row["id"])

    async def upsert_video(
        self,
        feed_id: str | None,
        guid: str | None,
        title: str | None,
        source_url: str,
        published_at: datetime | None,
        media_path: str | None = None,
    ) -> str:
        now = _utc_now()
        vid = str(uuid.uuid4())
        pub_text = published_at.isoformat() if published_at else None

        await self._conn.execute(
            """
            INSERT INTO videos (
                id, feed_id, source_url, guid, title, published_at, media_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_url)
            DO UPDATE SET
                feed_id = COALESCE(excluded.feed_id, videos.feed_id),
                guid = COALESCE(excluded.guid, videos.guid),
                title = COALESCE(excluded.title, videos.title),
                published_at = COALESCE(excluded.published_at, videos.published_at),
                media_path = COALESCE(excluded.media_path, videos.media_path)
            """,
            (vid, feed_id, source_url, guid, title, pub_text, media_path, now),
        )
        await self._conn.commit()

        async with self._conn.execute(
            "SELECT id FROM videos WHERE source_url = ?",
            (source_url,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to upsert video")
        return str(row["id"])

    async def insert_transcript(self, video_id: str, text: str) -> str:
        tid = str(uuid.uuid4())
        await self._conn.execute(
            """
            INSERT INTO transcripts (id, video_id, text, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (tid, video_id, text, _utc_now()),
        )
        await self._conn.commit()
        return tid

    async def insert_summary(self, video_id: str, result: SummaryResult) -> str:
        sid = str(uuid.uuid4())
        await self._conn.execute(
            """
            INSERT INTO summaries (
                id,
                video_id,
                summary,
                key_points,
                visual_highlights,
                model_used,
                vram_mb,
                transcript_chars,
                frame_count,
                error,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                video_id,
                result.summary,
                json.dumps(result.key_points),
                json.dumps(result.visual_highlights),
                result.model_used,
                float(result.vram_mb),
                int(result.transcript_chars),
                int(result.frame_count),
                result.error,
                _utc_now(),
            ),
        )
        await self._conn.commit()
        return sid

    async def latest_summaries(self, limit: int = 20) -> list[SummaryRecord]:
        async with self._conn.execute(
            """
            SELECT
                v.title,
                v.source_url,
                v.published_at,
                s.summary,
                s.key_points,
                s.visual_highlights,
                s.model_used,
                s.vram_mb,
                s.created_at
            FROM summaries s
            JOIN videos v ON v.id = s.video_id
            ORDER BY s.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()

        out: list[SummaryRecord] = []
        for row in rows:
            key_points_raw = row["key_points"]
            visual_raw = row["visual_highlights"]
            try:
                key_points = json.loads(key_points_raw) if key_points_raw else []
            except json.JSONDecodeError:
                key_points = []
            try:
                visual = json.loads(visual_raw) if visual_raw else []
            except json.JSONDecodeError:
                visual = []

            created = _parse_dt(str(row["created_at"]))
            out.append(
                SummaryRecord(
                    title=row["title"],
                    source_url=row["source_url"],
                    published_at=_parse_dt(row["published_at"]),
                    summary=row["summary"],
                    key_points=key_points if isinstance(key_points, list) else [],
                    visual_highlights=visual if isinstance(visual, list) else [],
                    model_used=row["model_used"],
                    vram_mb=float(row["vram_mb"] or 0.0),
                    created_at=created or datetime.now(timezone.utc),
                )
            )
        return out
