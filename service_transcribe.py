from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str = "unknown"


class TranscriptionEngine:
    """Legacy shim kept for compatibility with earlier versions."""

    @classmethod
    def get(cls, _config) -> TranscriptionEngine:
        return cls()

    def transcribe(
        self, _audio_path: str, language: str | None = None
    ) -> TranscriptionResult:
        return TranscriptionResult(text="", language=language or "unknown")
