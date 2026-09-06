"""Domain types for VRA v2. FROZEN — changes go through the lead."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class VideoStatus(StrEnum):
    PENDING = "pending"
    ACQUIRING = "acquiring"
    SUMMARIZING = "summarizing"
    DONE = "done"
    FAILED = "failed"


class JobKind(StrEnum):
    INGEST = "ingest"
    PROCESS = "process"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """Platform metadata from yt-dlp (subset we persist)."""

    canonical_id: str
    title: str | None = None
    description: str | None = None
    duration_s: float | None = None
    webpage_url: str | None = None
    chapters: tuple[tuple[float, str], ...] = ()


@dataclass(frozen=True, slots=True)
class TranscriptSeg:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    source: str  # "subtitle" | "asr"
    lang: str | None = None
    segments: tuple[TranscriptSeg, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))


@dataclass(frozen=True, slots=True)
class Frame:
    path: str
    ts_s: float
    scene_score: float = 0.0
    dhash: str = ""


@dataclass(frozen=True, slots=True)
class FrameSet:
    frames: tuple[Frame, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))


@dataclass(frozen=True, slots=True)
class MediaAsset:
    meta: VideoMeta
    media_path: str | None
    transcript: Transcript
    frames: FrameSet
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class Summary:
    summary: str
    key_points: tuple[str, ...] = ()
    visual_highlights: tuple[str, ...] = ()
    model: str = ""
    think: str = "off"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    vram_mb: float = 0.0
    transcript_chars: int = 0
    frame_count: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_points", tuple(self.key_points))
        object.__setattr__(self, "visual_highlights", tuple(self.visual_highlights))


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    kind: JobKind
    state: JobState
    payload: dict[str, object]
    attempts: int = 0
    max_attempts: int = 3
    run_at: datetime | None = None
    error: str | None = None
    video_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class FeedRecord:
    id: str
    url: str
    title: str | None = None
    site_url: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    last_checked: datetime | None = None
    poll_minutes: int = 15
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VideoRecord:
    id: str
    canonical_id: str
    source_url: str
    feed_id: str | None = None
    guid: str | None = None
    title: str | None = None
    published_at: datetime | None = None
    status: VideoStatus = VideoStatus.PENDING
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class IngestResult:
    feed: FeedRecord
    new_videos: int
    enqueued: int
    skipped_done: int
