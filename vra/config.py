"""Config for VRA v2. FROZEN — env keys per docs/v2/SPEC.md.

No-fallback mandate (product directive): exactly one engine path
(OpenAI-compatible serving) and one ASR path (/v1/audio/transcriptions
hosting Qwen3-ASR). Every inference decision is server-side; the app owns
routing, budgets, and quality control only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse


class ThinkLevel(StrEnum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class FrameMode(StrEnum):
    AUTO = "auto"
    SCENE = "scene"
    UNIFORM = "uniform"


class AsrMode(StrEnum):
    AUTO = "auto"
    OFF = "off"


class Select(StrEnum):
    AUTO = "auto"
    OFF = "off"
    ON = "on"


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def _enum(value: str | None, values: set[str], default: str) -> str:
    v = (value or "").strip().lower()
    return v if v in values else default


def _parse_bind(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if not host:
        return "127.0.0.1", 8080
    try:
        return host, int(port)
    except ValueError:
        return host, 8080


@dataclass(frozen=True, slots=True)
class Config:
    bind_host: str = "127.0.0.1"
    bind_port: int = 8080
    api_key: str | None = None
    data_dir: str = ".data"
    database_path: str = ".data/vra.db"
    engine_base_url: str = "http://127.0.0.1:8000/v1"
    engine_api_key: str | None = None
    model: str = "qwen3.5-4b"
    model_ctx: int = 32768
    think: ThinkLevel = ThinkLevel.OFF
    asr_base_url: str | None = None
    asr_model: str = "Qwen3-ASR-1.7B"
    asr: AsrMode = AsrMode.AUTO
    max_frames: int = 8
    frame_mode: FrameMode = FrameMode.AUTO
    scene_threshold: float = 0.3
    frame_max_dim: int = 768
    max_transcript_chars: int = 24000
    max_output_tokens: int = 1024
    poll_minutes: int = 15
    cpu_workers: int = 2
    timeout_download_s: int = 1800
    timeout_infer_s: int = 900
    allow_local_path: bool = False
    allow_private_net: bool = False
    rss_title: str = "Video RSS Aggregator"
    rss_link: str = "http://127.0.0.1:8080/rss"
    rss_description: str = "Video summaries"
    raw: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    @property
    def bind_address(self) -> str:
        return f"{self.bind_host}:{self.bind_port}"

    @property
    def effective_asr_base_url(self) -> str:
        return self.asr_base_url or self.engine_base_url

    def storage_subdir(self, name: str) -> Path:
        return Path(self.data_dir) / name

    def allow_url(self, url: str, *, local_paths_enabled: bool) -> bool:
        """Validate a user-supplied source/feed URL against policy."""
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return True
        return local_paths_enabled and parsed.scheme in {"", "file"}

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        e = env if env is not None else os.environ
        host, port = _parse_bind(e.get("BIND_ADDRESS", "127.0.0.1:8080"))
        data_dir = e.get("VRA_DATA_DIR", ".data")
        db_path = e.get("VRA_DATABASE_PATH") or str(Path(data_dir) / "vra.db")

        think_level = ThinkLevel(
            _enum(e.get("VRA_THINK"), {t.value for t in ThinkLevel}, "off")
        )
        frame_mode_e = FrameMode(
            _enum(e.get("VRA_FRAME_MODE"), {f.value for f in FrameMode}, "auto")
        )
        asr_e = AsrMode(_enum(e.get("VRA_ASR"), {a.value for a in AsrMode}, "auto"))

        return cls(
            bind_host=host,
            bind_port=port,
            api_key=e.get("API_KEY") or None,
            data_dir=data_dir,
            database_path=db_path,
            engine_base_url=e.get("VRA_ENGINE_BASE_URL", "http://127.0.0.1:8000/v1"),
            engine_api_key=e.get("VRA_ENGINE_API_KEY") or None,
            model=e.get("VRA_MODEL", "qwen3.5-4b"),
            model_ctx=max(4096, int(e.get("VRA_MODEL_CTX", "32768"))),
            think=think_level,
            asr_base_url=e.get("VRA_ASR_BASE_URL") or None,
            asr_model=e.get("VRA_ASR_MODEL", "Qwen3-ASR-1.7B"),
            asr=asr_e,
            max_frames=max(1, int(e.get("VRA_MAX_FRAMES", "8"))),
            frame_mode=frame_mode_e,
            scene_threshold=float(e.get("VRA_SCENE_THRESHOLD", "0.3")),
            frame_max_dim=max(224, int(e.get("VRA_FRAME_MAX_DIM", "768"))),
            max_transcript_chars=max(
                1000, int(e.get("VRA_MAX_TRANSCRIPT_CHARS", "24000"))
            ),
            max_output_tokens=max(256, int(e.get("VRA_MAX_OUTPUT_TOKENS", "1024"))),
            poll_minutes=max(1, int(e.get("VRA_POLL_MINUTES", "15"))),
            cpu_workers=max(1, int(e.get("VRA_CPU_WORKERS", "2"))),
            timeout_download_s=max(60, int(e.get("VRA_TIMEOUT_DOWNLOAD", "1800"))),
            timeout_infer_s=max(60, int(e.get("VRA_TIMEOUT_INFER", "900"))),
            allow_local_path=_bool(e.get("VRA_ALLOW_LOCAL_PATH"), False),
            allow_private_net=_bool(e.get("VRA_ALLOW_PRIVATE_NET"), False),
            rss_title=e.get("VRA_RSS_TITLE", "Video RSS Aggregator"),
            rss_link=e.get("VRA_RSS_LINK", "http://127.0.0.1:8080/rss"),
            rss_description=e.get("VRA_RSS_DESCRIPTION", "Video summaries"),
            raw=dict(e),
        )
