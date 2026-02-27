from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

import torch
from qwen_asr import Qwen3ASRModel

from .config import Config

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str


class TranscriptionEngine:
    """Singleton wrapper around Qwen3-ASR-1.7B (transformers backend)."""

    _instance: TranscriptionEngine | None = None
    _lock = Lock()

    def __init__(self, config: Config) -> None:
        log.info("Loading ASR model %s on %s", config.asr_model, config.asr_device)
        self._model = Qwen3ASRModel.from_pretrained(
            config.asr_model,
            dtype=torch.bfloat16,
            device_map=config.asr_device,
            max_new_tokens=config.asr_max_tokens,
        )
        log.info("ASR model loaded")

    @classmethod
    def get(cls, config: Config) -> TranscriptionEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        """Run ASR on a local audio file. Blocking — call from a worker thread."""
        results = self._model.transcribe(audio=audio_path, language=language)
        r = results[0]
        return TranscriptionResult(text=r.text, language=r.language)
