from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
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
    storage_dir: str = ".data"
    database_path: str = ".data/vra.db"
    ollama_base_url: str = "http://127.0.0.1:11434"
    model_primary: str = "qwen3.5:4b-q4_K_M"
    model_fallback: str = "qwen3.5:2b-q4_K_M"
    model_min: str = "qwen3.5:0.8b-q8_0"
    auto_pull_models: bool = True
    vram_budget_mb: int = 8192
    model_size_budget_ratio: float = 0.75
    model_selection_reserve_mb: int = 768
    context_tokens: int = 3072
    max_output_tokens: int = 768
    max_frames: int = 5
    frame_scene_detection: bool = True
    frame_scene_threshold: float = 0.28
    frame_scene_min_frames: int = 2
    max_transcript_chars: int = 16000
    rss_title: str = "Video RSS Aggregator"
    rss_link: str = "http://127.0.0.1:8080/rss"
    rss_description: str = "Video summaries"

    @property
    def model_priority(self) -> tuple[str, ...]:
        out: list[str] = []
        for name in (self.model_primary, self.model_fallback, self.model_min):
            item = name.strip()
            if item and item not in out:
                out.append(item)
        return tuple(out)

    @classmethod
    def from_env(cls) -> Config:
        bind = os.environ.get("BIND_ADDRESS", "127.0.0.1:8080")
        host, port = _parse_bind(bind)

        storage_dir = os.environ.get("VRA_STORAGE_DIR", ".data")
        db_path = os.environ.get("VRA_DATABASE_PATH")
        if not db_path:
            db_path = str(Path(storage_dir) / "vra.db")

        return cls(
            bind_host=host,
            bind_port=port,
            api_key=os.environ.get("API_KEY"),
            storage_dir=storage_dir,
            database_path=db_path,
            ollama_base_url=os.environ.get(
                "VRA_OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
            ),
            model_primary=os.environ.get("VRA_MODEL_PRIMARY", "qwen3.5:4b-q4_K_M"),
            model_fallback=os.environ.get("VRA_MODEL_FALLBACK", "qwen3.5:2b-q4_K_M"),
            model_min=os.environ.get("VRA_MODEL_MIN", "qwen3.5:0.8b-q8_0"),
            auto_pull_models=_to_bool(
                os.environ.get("VRA_AUTO_PULL_MODELS"),
                True,
            ),
            vram_budget_mb=int(os.environ.get("VRA_VRAM_BUDGET_MB", "8192")),
            model_size_budget_ratio=float(
                os.environ.get("VRA_MODEL_SIZE_BUDGET_RATIO", "0.75")
            ),
            model_selection_reserve_mb=int(
                os.environ.get("VRA_MODEL_SELECTION_RESERVE_MB", "768")
            ),
            context_tokens=int(os.environ.get("VRA_CONTEXT_TOKENS", "3072")),
            max_output_tokens=int(os.environ.get("VRA_MAX_OUTPUT_TOKENS", "768")),
            max_frames=int(os.environ.get("VRA_MAX_FRAMES", "5")),
            frame_scene_detection=_to_bool(
                os.environ.get("VRA_FRAME_SCENE_DETECTION"),
                True,
            ),
            frame_scene_threshold=float(
                os.environ.get("VRA_FRAME_SCENE_THRESHOLD", "0.28")
            ),
            frame_scene_min_frames=max(
                1,
                int(os.environ.get("VRA_FRAME_SCENE_MIN_FRAMES", "2")),
            ),
            max_transcript_chars=int(
                os.environ.get("VRA_MAX_TRANSCRIPT_CHARS", "16000")
            ),
            rss_title=os.environ.get("VRA_RSS_TITLE", "Video RSS Aggregator"),
            rss_link=os.environ.get("VRA_RSS_LINK", "http://127.0.0.1:8080/rss"),
            rss_description=os.environ.get(
                "VRA_RSS_DESCRIPTION",
                "Video summaries",
            ),
        )
