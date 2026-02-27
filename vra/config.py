from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Config:
    database_url: str
    bind_host: str = "0.0.0.0"
    bind_port: int = 8080
    api_key: str | None = None
    storage_dir: str = ".data"
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    llm_model: str = "Qwen/Qwen3-8B-AWQ"
    gpu_memory_utilization: float = 0.8
    asr_device: str = "cuda:0"
    asr_max_tokens: int = 4096
    llm_max_tokens: int = 2048

    @classmethod
    def from_env(cls) -> Config:
        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError("DATABASE_URL must be set")

        bind = os.environ.get("BIND_ADDRESS", "0.0.0.0:8080")
        parts = bind.rsplit(":", 1)
        host = parts[0] if len(parts) == 2 else "0.0.0.0"
        port = int(parts[1]) if len(parts) == 2 else 8080

        gpu_mem = float(os.environ.get("VRA_GPU_MEMORY_UTILIZATION", "0.8"))

        return cls(
            database_url=database_url,
            bind_host=host,
            bind_port=port,
            api_key=os.environ.get("API_KEY"),
            storage_dir=os.environ.get("VRA_STORAGE_DIR", ".data"),
            asr_model=os.environ.get("VRA_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
            llm_model=os.environ.get("VRA_LLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
            gpu_memory_utilization=gpu_mem,
            asr_device=os.environ.get("VRA_ASR_DEVICE", "cuda:0"),
            asr_max_tokens=int(os.environ.get("VRA_ASR_MAX_TOKENS", "4096")),
            llm_max_tokens=int(os.environ.get("VRA_LLM_MAX_TOKENS", "2048")),
        )
