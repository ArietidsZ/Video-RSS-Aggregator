"""Repositories implementing contracts.VideoStore. OWNER: agent-C.

SqlVideoStore maps the frozen schema rows to vra.types dataclasses. List
columns (key_points, visual_highlights, segments, chapters) are JSON-encoded
text; all timestamps are ISO-8601 UTC (``datetime.now(UTC).isoformat()``).
Semantics worth noting:

- ``save_feed_state``: ``error=None`` *clears* the error column (a successful
  poll marker); every other ``None`` field is left unchanged. Each call also
  touches ``last_checked``/``updated_at``.
- ``set_video_status``: always authoritative — ``error=None`` clears any
  previous error, so success transitions wipe stale failure text.
- ``save_transcript`` / ``save_frames`` replace the previous rows for the
  video (latest wins, atomically).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from vra.store.db import Database
from vra.types import (
    FeedRecord,
    FrameSet,
    JobState,
    Summary,
    Transcript,
    VideoMeta,
    VideoRecord,
    VideoStatus,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp; naive values are assumed UTC."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _parse_published(value: Any) -> datetime | None:
    """Parse published_at accepting datetime or ISO text."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return _parse_dt(value)


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return default


def _json_list(value: Any) -> list[Any]:
    decoded = _json_loads(value, [])
    return decoded if isinstance(decoded, list) else []


def _json_str_list(value: Any) -> list[str]:
    return [str(item) for item in _json_list(value)]


def _feed_from_row(row: aiosqlite.Row) -> FeedRecord:
    return FeedRecord(
        id=str(row["id"]),
        url=str(row["url"]),
        title=row["title"],
        site_url=row["site_url"],
        etag=row["etag"],
        last_modified=row["last_modified"],
        last_checked=_parse_dt(row["last_checked"]),
        poll_minutes=int(row["poll_minutes"]),
        error=row["error"],
    )


def _video_from_row(row: aiosqlite.Row) -> VideoRecord:
    return VideoRecord(
        id=str(row["id"]),
        canonical_id=str(row["canonical_id"]),
        source_url=str(row["source_url"]),
        feed_id=row["feed_id"],
        guid=row["guid"],
        title=row["title"],
        published_at=_parse_dt(row["published_at"]),
        status=VideoStatus(str(row["status"])),
        error=row["error"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _summary_from_row(row: aiosqlite.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "summary": str(row["summary"]),
        "key_points": _json_str_list(row["key_points"]),
        "visual_highlights": _json_str_list(row["visual_highlights"]),
        "model": str(row["model"]),
        "think": str(row["think"]),
        "prompt_tokens": int(row["prompt_tokens"]),
        "completion_tokens": int(row["completion_tokens"]),
        "latency_ms": int(row["latency_ms"]),
        "vram_mb": float(row["vram_mb"]),
        "transcript_chars": int(row["transcript_chars"]),
        "frame_count": int(row["frame_count"]),
        "error": row["error"],
        "created_at": _parse_dt(row["created_at"]),
    }


def _status_text(status: Any) -> str:
    return status.value if isinstance(status, VideoStatus) else str(status)


class SqlVideoStore:
    """SQLite implementation of contracts.VideoStore."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------- feeds --

    async def upsert_feed(self, url: str, title: str | None, poll_minutes: int) -> FeedRecord:
        """Insert or update a feed keyed by URL, returning the fresh record."""
        now = _iso(_utc_now())
        async with self._db.lock:
            await self._db.execute(
                """
                INSERT INTO feeds (id, url, title, poll_minutes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = COALESCE(excluded.title, feeds.title),
                    poll_minutes = excluded.poll_minutes,
                    updated_at = excluded.updated_at
                """,
                (str(uuid.uuid4()), url, title, poll_minutes, now, now),
            )
            row = await self._db.fetchone("SELECT * FROM feeds WHERE url = ?", (url,))
        if row is None:  # pragma: no cover - defensive
            raise RuntimeError(f"feed upsert failed for {url!r}")
        return _feed_from_row(row)

    async def save_feed_state(
        self,
        feed_id: str,
        *,
        title: str | None = None,
        site_url: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        error: str | None = None,
    ) -> None:
        """Touch feed state after a poll.

        ``error=None`` clears the stored error (success marker); the other
        ``None`` fields keep their current values. ``last_checked`` is always
        refreshed.
        """
        now = _iso(_utc_now())
        await self._db.execute(
            """
            UPDATE feeds SET
                title = COALESCE(?, title),
                site_url = COALESCE(?, site_url),
                etag = COALESCE(?, etag),
                last_modified = COALESCE(?, last_modified),
                error = ?,
                last_checked = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (title, site_url, etag, last_modified, error, now, now, feed_id),
        )

    async def get_feed_by_url(self, url: str) -> FeedRecord | None:
        row = await self._db.fetchone("SELECT * FROM feeds WHERE url = ?", (url,))
        return _feed_from_row(row) if row is not None else None

    async def list_feeds(self) -> list[FeedRecord]:
        rows = await self._db.fetchall(
            "SELECT * FROM feeds ORDER BY created_at, id"
        )
        return [_feed_from_row(row) for row in rows]

    async def delete_feed(self, feed_id: str) -> None:
        await self._db.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))

    # ------------------------------------------------------------ videos --

    async def upsert_video_ref(
        self,
        *,
        canonical_id: str,
        source_url: str,
        feed_id: str | None,
        guid: str | None,
        title: str | None,
        published_at: Any,
    ) -> tuple[VideoRecord, bool]:
        """Insert a pending video ref or refresh an existing one.

        Returns ``(record, created)`` where ``created`` is True only when a
        new row was inserted. Existing rows keep their status; only ``None``
        fields are left unchanged (COALESCE), except ``source_url`` which
        always follows the feed entry.
        """
        now = _iso(_utc_now())
        pub = _iso(_parse_published(published_at))
        async with self._db.lock:
            existing = await self._db.fetchone(
                "SELECT id FROM videos WHERE canonical_id = ?", (canonical_id,)
            )
            if existing is None:
                video_id = str(uuid.uuid4())
                await self._db.execute(
                    """
                    INSERT INTO videos (
                        id, canonical_id, source_url, feed_id, guid, title,
                        published_at, status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        video_id, canonical_id, source_url, feed_id, guid, title,
                        pub, now, now,
                    ),
                )
                created = True
            else:
                video_id = str(existing["id"])
                await self._db.execute(
                    """
                    UPDATE videos SET
                        feed_id = COALESCE(?, feed_id),
                        guid = COALESCE(?, guid),
                        title = COALESCE(?, title),
                        published_at = COALESCE(?, published_at),
                        source_url = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (feed_id, guid, title, pub, source_url, now, video_id),
                )
                created = False
            row = await self._db.fetchone("SELECT * FROM videos WHERE id = ?", (video_id,))
        if row is None:  # pragma: no cover - defensive
            raise RuntimeError(f"video upsert failed for {canonical_id!r}")
        return _video_from_row(row), created

    async def set_video_status(
        self,
        video_id: str,
        status: Any,
        *,
        error: str | None = None,
    ) -> None:
        """Authoritatively set status; ``error=None`` clears prior errors."""
        await self._db.execute(
            "UPDATE videos SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (_status_text(status), error, _iso(_utc_now()), video_id),
        )

    async def save_meta(
        self, video_id: str, meta: VideoMeta, media_path: str | None
    ) -> None:
        """Persist platform metadata; None meta fields keep existing values."""
        chapters = json.dumps(
            [[float(start), str(name)] for start, name in meta.chapters]
        )
        await self._db.execute(
            """
            UPDATE videos SET
                title = COALESCE(?, title),
                description = COALESCE(?, description),
                duration_s = COALESCE(?, duration_s),
                chapters = ?,
                media_path = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                meta.title, meta.description, meta.duration_s, chapters,
                media_path, _iso(_utc_now()), video_id,
            ),
        )

    async def save_transcript(self, video_id: str, transcript: Transcript) -> None:
        """Replace the stored transcript for the video (latest wins)."""
        segments = json.dumps(
            [
                {"start_s": seg.start_s, "end_s": seg.end_s, "text": seg.text}
                for seg in transcript.segments
            ]
        )
        async with self._db.transaction():
            await self._db.execute(
                "DELETE FROM transcripts WHERE video_id = ?", (video_id,),
                commit=False,
            )
            await self._db.execute(
                """
                INSERT INTO transcripts (id, video_id, source, lang, text, segments, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), video_id, transcript.source, transcript.lang,
                    transcript.text, segments, _iso(_utc_now()),
                ),
                commit=False,
            )

    async def save_frames(self, video_id: str, frames: FrameSet) -> None:
        """Replace the stored frame rows for the video (latest wins)."""
        now = _iso(_utc_now())
        async with self._db.transaction():
            await self._db.execute(
                "DELETE FROM frames WHERE video_id = ?", (video_id,), commit=False
            )
            for idx, frame in enumerate(frames.frames):
                await self._db.execute(
                    """
                    INSERT INTO frames (
                        id, video_id, idx, path, ts_s, scene_score, dhash, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()), video_id, idx, frame.path, frame.ts_s,
                        frame.scene_score, frame.dhash, now,
                    ),
                    commit=False,
                )

    async def save_summary(self, video_id: str, summary: Summary) -> None:
        """Append a summary row (history is kept, latest used for RSS)."""
        await self._db.execute(
            """
            INSERT INTO summaries (
                id, video_id, summary, key_points, visual_highlights, model,
                think, prompt_tokens, completion_tokens, latency_ms, vram_mb,
                transcript_chars, frame_count, error, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()), video_id, summary.summary,
                json.dumps(list(summary.key_points)),
                json.dumps(list(summary.visual_highlights)),
                summary.model, summary.think, summary.prompt_tokens,
                summary.completion_tokens, summary.latency_ms, summary.vram_mb,
                summary.transcript_chars, summary.frame_count, summary.error,
                _iso(_utc_now()),
            ),
        )

    async def get_video(self, video_id: str) -> VideoRecord | None:
        row = await self._db.fetchone("SELECT * FROM videos WHERE id = ?", (video_id,))
        return _video_from_row(row) if row is not None else None

    async def get_video_by_canonical(self, canonical_id: str) -> VideoRecord | None:
        row = await self._db.fetchone(
            "SELECT * FROM videos WHERE canonical_id = ?", (canonical_id,)
        )
        return _video_from_row(row) if row is not None else None

    # --------------------------------------------------------- readouts --

    async def latest_publications(self, limit: int) -> list[dict[str, Any]]:
        """Latest summary per non-failed video, newest summary first."""
        rows = await self._db.fetchall(
            """
            SELECT
                v.id AS video_id,
                v.canonical_id AS canonical_id,
                v.title AS title,
                v.source_url AS source_url,
                v.published_at AS published_at,
                s.summary AS summary,
                s.key_points AS key_points,
                s.visual_highlights AS visual_highlights,
                s.model AS model,
                s.created_at AS created_at
            FROM videos v
            JOIN summaries s ON s.video_id = v.id
            WHERE v.status != 'failed'
              AND s.id = (
                  SELECT s2.id FROM summaries s2
                  WHERE s2.video_id = v.id
                  ORDER BY s2.created_at DESC, s2.rowid DESC
                  LIMIT 1
              )
            ORDER BY s.created_at DESC, s.rowid DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "video_id": str(row["video_id"]),
                "canonical_id": str(row["canonical_id"]),
                "title": row["title"],
                "source_url": str(row["source_url"]),
                "published_at": _parse_dt(row["published_at"]),
                "summary": str(row["summary"]),
                "key_points": _json_str_list(row["key_points"]),
                "visual_highlights": _json_str_list(row["visual_highlights"]),
                "model": str(row["model"]),
                "created_at": _parse_dt(row["created_at"]),
            }
            for row in rows
        ]

    async def video_detail(self, video_id: str) -> dict[str, Any] | None:
        """Full detail bundle: video, latest transcript, frames, summaries."""
        row = await self._db.fetchone("SELECT * FROM videos WHERE id = ?", (video_id,))
        if row is None:
            return None
        video = {
            "id": str(row["id"]),
            "canonical_id": str(row["canonical_id"]),
            "source_url": str(row["source_url"]),
            "feed_id": row["feed_id"],
            "guid": row["guid"],
            "title": row["title"],
            "description": row["description"],
            "duration_s": row["duration_s"],
            "chapters": _json_list(row["chapters"]),
            "published_at": _parse_dt(row["published_at"]),
            "status": str(row["status"]),
            "error": row["error"],
            "media_path": row["media_path"],
            "created_at": _parse_dt(row["created_at"]),
            "updated_at": _parse_dt(row["updated_at"]),
        }

        trow = await self._db.fetchone(
            """
            SELECT * FROM transcripts WHERE video_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT 1
            """,
            (video_id,),
        )
        transcript: dict[str, Any] | None = None
        if trow is not None:
            transcript = {
                "id": str(trow["id"]),
                "source": str(trow["source"]),
                "lang": trow["lang"],
                "text": str(trow["text"]),
                "segments": _json_list(trow["segments"]),
                "created_at": _parse_dt(trow["created_at"]),
            }

        frows = await self._db.fetchall(
            "SELECT * FROM frames WHERE video_id = ? ORDER BY idx", (video_id,)
        )
        frames = [
            {
                "idx": int(frow["idx"]),
                "path": str(frow["path"]),
                "ts_s": float(frow["ts_s"]),
                "scene_score": float(frow["scene_score"]),
                "dhash": str(frow["dhash"]),
            }
            for frow in frows
        ]

        srows = await self._db.fetchall(
            """
            SELECT * FROM summaries WHERE video_id = ?
            ORDER BY created_at DESC, rowid DESC
            """,
            (video_id,),
        )
        return {
            "video": video,
            "transcript": transcript,
            "frames": frames,
            "summaries": [_summary_from_row(srow) for srow in srows],
        }

    async def counts(self) -> dict[str, Any]:
        """Aggregate counters for the metrics endpoint."""
        job_rows = await self._db.fetchall(
            "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
        )
        video_rows = await self._db.fetchall(
            "SELECT status, COUNT(*) AS n FROM videos GROUP BY status"
        )
        feed_row = await self._db.fetchone("SELECT COUNT(*) AS n FROM feeds")
        summary_row = await self._db.fetchone("SELECT COUNT(*) AS n FROM summaries")

        jobs_by_state = {state.value: 0 for state in JobState}
        for row in job_rows:
            key = str(row["state"])
            jobs_by_state[key] = jobs_by_state.get(key, 0) + int(row["n"])
        videos_by_status = {status.value: 0 for status in VideoStatus}
        for row in video_rows:
            key = str(row["status"])
            videos_by_status[key] = videos_by_status.get(key, 0) + int(row["n"])
        return {
            "jobs_by_state": jobs_by_state,
            "videos_by_status": videos_by_status,
            "feeds": int(feed_row["n"]) if feed_row is not None else 0,
            "summaries": int(summary_row["n"]) if summary_row is not None else 0,
        }
