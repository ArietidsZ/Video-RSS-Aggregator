"""ASR fallback via an OpenAI-compatible server hosting Qwen3-ASR.

OpenAiAsrTranscriber implements contracts.Transcriber against a server that
exposes ``GET /models`` and ``POST /audio/transcriptions``:

- vLLM running the official QwenLM/Qwen3-ASR toolkit (CUDA), or
- llama-server with ggml-org/Qwen3-ASR-*-GGUF (anywhere, incl. Mac Metal).

This is the single ASR path by design (lead decision, no backend ladder): if
the server is down, available() returns False and the processor proceeds
without ASR — the video then has no transcript unless subtitles existed.

Response shapes normalized (see llama.cpp issue #26749 for the asr_text
quirk):
- OpenAI verbose JSON: {"text", "language"?, "segments": [{start, end, text}]}
- plain OpenAI JSON: {"text"}
- llama.cpp quirk: {"asr_text": "..."}

The POST is one-shot (no retries beyond httpx transport defaults): ASR is
cheap to re-run at job level via the durable queue's retry policy.

OWNER: agent-A. See docs/v2/SPEC.md.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
from pathlib import Path

import httpx

from vra.types import Transcript, TranscriptSeg

log = logging.getLogger(__name__)

_MODELS_TIMEOUT_S = 2.0
_MODELS_CACHE_S = 60.0
_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_ASR_MODEL = "Qwen3-ASR-1.7B"


def _config_base_url(config: object) -> str:
    """Base URL from config, merge-safe across the pending config rewrite.

    Prefers config.effective_asr_base_url (new Config), then asr_base_url /
    engine_base_url, then the hardcoded vLLM default. The value must include
    the /v1 prefix (e.g. "http://127.0.0.1:8000/v1").
    """
    for attr in ("effective_asr_base_url", "asr_base_url", "engine_base_url"):
        value = getattr(config, attr, None)
        if isinstance(value, str) and value.strip():
            return value.rstrip("/")
    return _DEFAULT_BASE_URL


class OpenAiAsrTranscriber:
    """Transcribe media through an OpenAI-compatible Qwen3-ASR endpoint."""

    def __init__(
        self, config: object, client: httpx.AsyncClient | None = None
    ) -> None:
        self._config = config
        self._base_url = _config_base_url(config)
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(timeout=_MODELS_TIMEOUT_S)
        )
        self._owns_client = client is None
        self._available_cache: tuple[bool, float] | None = None

    def _http(self) -> httpx.AsyncClient:
        """The owned or injected client; raise if this instance was closed."""
        if self._client is None:
            raise RuntimeError("OpenAiAsrTranscriber is closed")
        return self._client

    async def close(self) -> None:
        """Close the owned httpx client (injected clients belong to callers)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def available(self) -> bool:
        """True when the endpoint lists a model whose id contains "asr"."""
        if self._available_cache is not None:
            ok, at = self._available_cache
            if time.monotonic() - at < _MODELS_CACHE_S:
                return ok
        ok = False
        try:
            resp = await self._http().get(
                f"{self._base_url}/models", timeout=_MODELS_TIMEOUT_S
            )
            resp.raise_for_status()
            payload = resp.json()
            ids = [
                str(item.get("id", ""))
                for item in (payload.get("data") or [])
                if isinstance(item, dict)
            ]
            ok = any("asr" in model_id.lower() for model_id in ids)
        except Exception as exc:
            log.warning("ASR endpoint probe failed at %s: %s", self._base_url, exc)
            ok = False
        self._available_cache = (ok, time.monotonic())
        return ok

    async def transcribe(self, media_path: str) -> Transcript:
        """POST the media file to /audio/transcriptions and normalize."""
        model = str(
            getattr(self._config, "asr_model", None) or _DEFAULT_ASR_MODEL
        )
        path = Path(media_path)
        data = await asyncio.to_thread(path.read_bytes)
        basename = path.name
        mimetype = mimetypes.guess_type(basename)[0] or "application/octet-stream"

        http = self._http()
        resp = await http.post(
            f"{self._base_url}/audio/transcriptions",
            data={"model": model},
            files={"file": (basename, data, mimetype)},
            timeout=float(
                getattr(self._config, "timeout_infer_s", None) or 900
            ),
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"ASR transcription failed (HTTP {resp.status_code}): "
                f"{resp.text.strip()[-500:]}"
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"ASR server returned non-JSON response: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("ASR server response was not a JSON object")
        return self._normalize(payload)

    def _normalize(self, payload: dict[str, object]) -> Transcript:
        """Map any supported response shape onto Transcript(source="asr")."""
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            text = payload.get("asr_text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("ASR server returned empty transcript")

        language = payload.get("language")
        lang = language if isinstance(language, str) and language else None

        raw_segments = payload.get("segments")
        segments: list[TranscriptSeg] = []
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                seg_text = seg.get("text")
                if not isinstance(seg_text, str) or not seg_text.strip():
                    continue
                segments.append(
                    TranscriptSeg(
                        start_s=_as_float(seg.get("start"), 0.0),
                        end_s=_as_float(seg.get("end"), 0.0),
                        text=seg_text.strip(),
                    )
                )
        if not segments:
            segments = [TranscriptSeg(start_s=0.0, end_s=0.0, text=text.strip())]
        return Transcript(
            text=" ".join(seg.text for seg in segments),
            source="asr",
            lang=lang,
            segments=tuple(segments),
        )


def _as_float(value: object, default: float) -> float:
    """Coerce a JSON timestamp (int/float) to float; anything else -> default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)
