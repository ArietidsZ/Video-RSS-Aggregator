"""Config for VRA v2. FROZEN — env keys per docs/v2/SPEC.md."""

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


class AsrBackend(StrEnum):
    AUTO = "auto"
    MLX = "mlx"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    FASTER_WHISPER = "faster_whisper"


class Engine(StrEnum):
    AUTO = "auto"
    OLLAMA = "ollama"
    OPENAI = "openai"


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


def _select(value: str | None, default: Select) -> Select:
    if value is None or not value.strip():
        return default
    try:
        return Select(value.strip().lower())
    except ValueError:
        return default


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
    ollama_base_url: str = "http://127.0.0.1:11434"
    models: tuple[str, ...] = ("qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b")
    prefer_mlx: Select = Select.AUTO
    think: ThinkLevel = ThinkLevel.OFF
    keep_alive: str = "15m"
    vram_budget_mb: int = 8192
    reserve_mb: int = 768
    max_frames: int = 8
    frame_mode: FrameMode = FrameMode.AUTO
    scene_threshold: float = 0.3
    frame_max_dim: int = 768
    max_transcript_chars: int = 24000
    ctx_tokens: str = "auto"
    max_output_tokens: int = 1024
    asr: AsrMode = AsrMode.AUTO
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    asr_backend: AsrBackend = AsrBackend.AUTO
    asr_whisper_model: str = "small"
    engine: Engine = Engine.AUTO
    engine_base_url: str | None = None
    engine_api_key: str | None = None
    asr_device: Select = Select.AUTO
    poll_minutes: int = 15
    cpu_workers: int = 2
    gpu_workers: int = 1
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
    def ctx_tokens_int(self) -> int | None:
        v = self.ctx_tokens.strip().lower()
        if v in {"", "auto"}:
            return None
        try:
            return max(1024, int(v))
        except ValueError:
            return None

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
        models = tuple(
            m.strip()
            for m in e.get("VRA_MODELS", "qwen3.5:4b,qwen3.5:2b,qwen3.5:0.8b").split(",")
            if m.strip()
        ) or ("qwen3.5:4b", "qwen3.5:2b", "qwen3.5:0.8b")

        think = e.get("VRA_THINK", "off").strip().lower()
        try:
            think_level = ThinkLevel(think)
        except ValueError:
            think_level = ThinkLevel.OFF
        frame_mode = e.get("VRA_FRAME_MODE", "auto").strip().lower()
        try:
            frame_mode_e = FrameMode(frame_mode)
        except ValueError:
            frame_mode_e = FrameMode.AUTO
        asr_mode = e.get("VRA_ASR", "auto").strip().lower()
        try:
            asr_e = AsrMode(asr_mode)
        except ValueError:
            asr_e = AsrMode.AUTO

        return cls(
            bind_host=host,
            bind_port=port,
            api_key=e.get("API_KEY") or None,
            data_dir=data_dir,
            database_path=db_path,
            ollama_base_url=e.get("VRA_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            models=models,
            prefer_mlx=_select(e.get("VRA_PREFER_MLX"), Select.AUTO),
            think=think_level,
            keep_alive=e.get("VRA_KEEP_ALIVE", "15m"),
            vram_budget_mb=int(e.get("VRA_VRAM_BUDGET_MB", "8192")),
            reserve_mb=int(e.get("VRA_RESERVE_MB", "768")),
            max_frames=max(1, int(e.get("VRA_MAX_FRAMES", "8"))),
            frame_mode=frame_mode_e,
            scene_threshold=float(e.get("VRA_SCENE_THRESHOLD", "0.3")),
            frame_max_dim=max(224, int(e.get("VRA_FRAME_MAX_DIM", "768"))),
            max_transcript_chars=max(1000, int(e.get("VRA_MAX_TRANSCRIPT_CHARS", "24000"))),
            ctx_tokens=e.get("VRA_CTX_TOKENS", "auto"),
            max_output_tokens=max(256, int(e.get("VRA_MAX_OUTPUT_TOKENS", "1024"))),
            asr=asr_e,
            asr_model=e.get("VRA_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
            asr_backend=(lambda v: AsrBackend(v) if v in {b.value for b in AsrBackend} else AsrBackend.AUTO)(
                e.get("VRA_ASR_BACKEND", "auto").strip().lower()
            ),
            asr_whisper_model=e.get("VRA_ASR_WHISPER_MODEL", "small"),
            engine=(lambda v: Engine(v) if v in {x.value for x in Engine} else Engine.AUTO)(
                e.get("VRA_ENGINE", "auto").strip().lower()
            ),
            engine_base_url=e.get("VRA_ENGINE_BASE_URL") or None,
            engine_api_key=e.get("VRA_ENGINE_API_KEY") or None,
            asr_device=_select(e.get("VRA_ASR_DEVICE"), Select.AUTO),
            poll_minutes=max(1, int(e.get("VRA_POLL_MINUTES", "15"))),
            cpu_workers=max(1, int(e.get("VRA_CPU_WORKERS", "2"))),
            gpu_workers=max(1, int(e.get("VRA_GPU_WORKERS", "1"))),
            timeout_download_s=max(60, int(e.get("VRA_TIMEOUT_DOWNLOAD", "1800"))),
            timeout_infer_s=max(60, int(e.get("VRA_TIMEOUT_INFER", "900"))),
            allow_local_path=_bool(e.get("VRA_ALLOW_LOCAL_PATH"), False),
            allow_private_net=_bool(e.get("VRA_ALLOW_PRIVATE_NET"), False),
            rss_title=e.get("VRA_RSS_TITLE", "Video RSS Aggregator"),
            rss_link=e.get("VRA_RSS_LINK", "http://127.0.0.1:8080/rss"),
            rss_description=e.get("VRA_RSS_DESCRIPTION", "Video summaries"),
            raw=dict(e),
        )
