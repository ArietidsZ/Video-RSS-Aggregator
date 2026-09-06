"""Summarization pipeline implementing contracts.Summarizer (owner: agent-B).

``VlmSummarizer(config, engine)`` summarizes an acquired MediaAsset
through one OpenAI-compatible engine (see vra.inference.engine). There
is exactly ONE engine path: no extractive fallback, no model ladder —
failures surface so jobs fail visibly.

Pipeline per asset:

1. ``resolve_model()`` (cached): ``config.model`` must be served by the
   engine (tolerant id matching, see engine module); otherwise a
   RuntimeError with guidance is raised — surfaced as setup/diagnostics
   info, never retried.
2. Transcript is capped to ``config.max_transcript_chars``; frames are
   sized by a Pillow-free JPEG SOF parser (``read_jpeg_size``) and
   base64-encoded (bounded by ``config.max_frames``).
3. Payload sizing (Qwen family heuristics): ``estimate_tokens`` counts
   CJK chars as ~1 token each and other chars as ~3.6 chars/token;
   ``image_tokens`` is Σ ceil(w/28)*ceil(h/28)//4 (28px vision patches).
   Single pass when prompt + image tokens fit
   ``config.model_ctx - config.max_output_tokens - 256`` (256 = response
   margin). ``config.model_ctx`` mirrors the server's ``max_model_len``.
4. Map-reduce otherwise: the transcript is chunked at sentence
   boundaries into ~3000-token chunks (shrunk to the ctx window when the
   server context is small), each condensed with a text-only map call
   (no reasoning effort, temperature 0.2, ``{"summary"}`` schema); the
   final pass runs the full prompt with the chunk summaries joined as
   the transcript and the frames attached.
5. Think levels: ``config.think`` maps to OpenAI ``reasoning_effort`` on
   the final pass (omitted when off). Engines that do not implement it
   (e.g. llama-server) ignore the field. Unlike Ollama there is no
   thinking+schema restriction, so the strict JSON schema is kept on
   every call, including the gate retry.
6. Quality gate (legacy ``_is_meaningful_summary``): summary >= 48 chars
   or >= 2 key points with one >= 24 chars. On failure: ONE retry with
   rejection feedback appended and ``reasoning_effort="low"``. If the
   retry still fails the gate, RuntimeError("summary quality gate
   failed: ...") is raised; infrastructure errors also raise.

Every call's usage tokens and wall-time latency are accumulated into the
Summary. ``vram_mb`` is always 0.0 (inference is server-side; the field
is kept for schema compatibility).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import struct
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vra.config import Config, ThinkLevel
from vra.inference.engine import ChatResult, EngineClient, encode_images
from vra.types import MediaAsset, Summary

__all__ = [
    "VlmSummarizer",
    "chunk_transcript",
    "estimate_tokens",
    "image_tokens",
    "read_jpeg_size",
]

log = logging.getLogger(__name__)

_CTX_MARGIN_TOKENS = 256
_CHUNK_TOKEN_BUDGET = 3000
_MAP_REDUCE_JOIN = "\n\n"
_NO_TRANSCRIPT_PROMPT = "(No subtitles/transcript available.)"

_SYSTEM_PROMPT = """\
You summarize web videos using the transcript and the attached sampled frames.

Rules:
1) Respond in the language of the transcript.
2) Never invent facts; if evidence is insufficient, say so explicitly.
3) Frames are supporting evidence, not the primary source.
4) Output valid JSON with keys "summary", "key_points" and \
"visual_highlights".
"""

_MAP_SYSTEM_PROMPT = (
    "You condense one chunk of a video transcript. Respond in the "
    "chunk's language; never invent facts; output JSON with key "
    '"summary".'
)

_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "visual_highlights": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_points", "visual_highlights"],
}

_MAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

_RETRY_FEEDBACK = (
    "\n\nYour previous response was rejected ({reason}). Respond with "
    "ONLY a JSON object with keys \"summary\" (a detailed paragraph), "
    "\"key_points\" (array of strings) and \"visual_highlights\" (array "
    "of strings). The summary must be at least 48 characters."
)


# --------------------------------------------------------------------------
# Token / context math (Qwen family heuristics)
# --------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count: CJK chars ~1 token each, others ~3.6 chars/token.

    CJK detection uses ``unicodedata.east_asian_width`` in {"W", "F"};
    the result is the ceil of the weighted sum so budgets never
    underestimate.
    """
    cjk = 0
    other = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ("W", "F"):
            cjk += 1
        else:
            other += 1
    return math.ceil(cjk + other / 3.6)


def image_tokens(width: int, height: int) -> int:
    """Vision tokens for one frame: ceil(w/28) * ceil(h/28) // 4 (28px patches)."""
    return math.ceil(width / 28) * math.ceil(height / 28) // 4


def fits_ctx(prompt_tokens: int, image_tokens_sum: int, *, model_ctx: int,
             max_output_tokens: int) -> bool:
    """True when the payload fits ``model_ctx - max_output_tokens - 256``."""
    return prompt_tokens + image_tokens_sum <= (
        model_ctx - max_output_tokens - _CTX_MARGIN_TOKENS
    )


def chunk_budget_tokens(model_ctx: int, max_output_tokens: int) -> int:
    """Per-chunk token budget: ~3000, shrunk to fit a small server ctx window."""
    return max(256, min(_CHUNK_TOKEN_BUDGET,
                        model_ctx - max_output_tokens - _CTX_MARGIN_TOKENS))


# --------------------------------------------------------------------------
# JPEG dimension parsing (struct-only, no Pillow)
# --------------------------------------------------------------------------

_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Parse (width, height) from JPEG bytes via a SOF marker scan."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise ValueError("not a JPEG stream (missing SOI marker)")
    offset = 2
    size = len(data)
    while offset + 4 <= size:
        if data[offset] != 0xFF:
            offset += 1  # resync inside entropy-coded data
            continue
        marker = data[offset + 1]
        if marker == 0xFF:  # fill byte
            offset += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            offset += 2  # standalone markers carry no length
            continue
        if marker == 0xD9:  # EOI before any SOF
            break
        (segment_len,) = struct.unpack_from(">H", data, offset + 2)
        if segment_len < 2:
            raise ValueError(
                f"corrupt JPEG segment length {segment_len} at offset {offset}"
            )
        if marker in _SOF_MARKERS:
            height, width = struct.unpack_from(">HH", data, offset + 5)
            return width, height
        offset += 2 + segment_len
    raise ValueError("no SOF marker found in JPEG stream")


async def read_jpeg_size(path: str | Path) -> tuple[int, int]:
    """(width, height) of a JPEG file; ValueError when it is not a JPEG."""
    data = await asyncio.to_thread(Path(path).read_bytes)
    return _jpeg_dimensions(data)


# --------------------------------------------------------------------------
# Transcript chunking (sentence boundaries)
# --------------------------------------------------------------------------


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split an over-long sentence into <= max_chars windows on whitespace."""
    parts: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", max_chars // 2, max_chars)
        if cut <= 0:
            cut = max_chars
        head = text[:cut].strip()
        if head:
            parts.append(head)
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def chunk_transcript(
    text: str,
    *,
    budget_tokens: int,
    estimate: Callable[[str], int],
) -> list[str]:
    """Split a transcript into chunks of ~``budget_tokens`` at sentence ends.

    Sentences are split after [.!?。！？…] or newlines. A single sentence
    larger than the budget is hard-split into windows of at most
    ``budget_tokens`` characters, so each window's token estimate can
    never exceed the budget (CJK is 1 token/char at worst).
    """
    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?。！？…])\s+|\n+", text) if part.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        if estimate(sentence) > budget_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.extend(part for part in _hard_split(sentence, budget_tokens) if part)
            continue
        cost = estimate(sentence)
        separator = 1 if current else 0  # the space inserted by " ".join
        if current and current_tokens + separator + cost > budget_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
            separator = 0
        current.append(sentence)
        current_tokens += separator + cost
    if current:
        chunks.append(" ".join(current))
    return chunks


# --------------------------------------------------------------------------
# Response parsing + quality gate (ported from legacy service_summarize)
# --------------------------------------------------------------------------


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction (schema output should be plain JSON)."""
    cleaned = _strip_markdown_fence(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return {"summary": cleaned, "key_points": [], "visual_highlights": []}


def _as_string_list(value: Any, limit: int) -> list[str]:
    """Coerce a JSON value into a deduplicated list of non-empty strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_payload(payload: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    summary = str(payload.get("summary", "") or "").strip()
    key_points = _as_string_list(payload.get("key_points"), 8)
    visual_highlights = _as_string_list(payload.get("visual_highlights"), 5)
    return summary, key_points, visual_highlights


def _is_meaningful_summary(summary: str, key_points: list[str]) -> bool:
    """Legacy quality gate: informative paragraph OR >= 2 substantial points."""
    return len(summary.strip()) >= 48 or (
        len(key_points) >= 2 and any(len(item) >= 24 for item in key_points)
    )


def _user_prompt_parts(
    *, source: str, title: str | None, frame_count: int
) -> tuple[str, str]:
    """Head/tail of the user prompt; the transcript body goes in between."""
    head = (
        f"Source: {source}\n"
        f"Title: {title or '(Unknown title)'}\n"
        f"Attached Frames: {frame_count}\n"
        "\n"
        "Transcript:\n"
    )
    tail = "\n\nReturn JSON only according to the schema."
    return head, tail


@dataclass(slots=True)
class _RunStats:
    """Accumulated usage across all chat calls of one summarize() run."""

    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class VlmSummarizer:
    """contracts.Summarizer backed by a single OpenAI-compatible engine.

    Requires the rewritten Config fields: ``engine_base_url``,
    ``engine_api_key``, ``model``, ``model_ctx``, ``think``,
    ``max_output_tokens``, ``timeout_infer_s``, ``max_transcript_chars``,
    ``max_frames`` and ``asr_model``.
    """

    def __init__(
        self,
        config: Config,
        engine: EngineClient,
        asr_client: EngineClient | None = None,
    ) -> None:
        self._config = config
        self._engine = engine
        self._asr_client = asr_client
        self._model: str | None = None

    async def close(self) -> None:
        """Shut down the engine and any dedicated ASR probe client."""
        await self._engine.close()
        if self._asr_client is not None and self._asr_client is not self._engine:
            await self._asr_client.close()

    async def resolve_model(self) -> str:
        """Verify config.model is served; cache the resolution.

        Raises RuntimeError with setup guidance when absent — this is
        surfaced as diagnostics info and deliberately never retried.
        """
        if self._model is None:
            if not await self._engine.has_model(self._config.model):
                raise RuntimeError(
                    f"model {self._config.model} not served by "
                    f"{self._engine.base_url}; serve it or set VRA_MODEL"
                )
            self._model = self._config.model
        return self._model

    async def runtime_status(self) -> dict[str, Any]:
        """Delegate to the engine client (setup/diagnostics shape)."""
        return await self._engine.runtime_status(
            model=self._config.model,
            asr_model=self._config.asr_model,
            asr_client=self._asr_client,
        )

    async def prepare_models(self) -> list[str]:
        """No pulling on an OpenAI-compatible server; verify and report."""
        return [await self.resolve_model()]

    async def summarize(self, asset: MediaAsset) -> Summary:
        """Summarize one acquired asset (single pass or map-reduce)."""
        model = await self.resolve_model()
        stats = _RunStats()
        transcript = asset.transcript.text.strip()
        if len(transcript) > self._config.max_transcript_chars:
            transcript = transcript[: self._config.max_transcript_chars]

        frames = asset.frames.frames[: max(self._config.max_frames, 0)]
        sized: list[tuple[str, int, int]] = []
        for frame in frames:
            try:
                width, height = await read_jpeg_size(frame.path)
            except (OSError, ValueError) as exc:
                log.warning("summarize: skipping unreadable frame %s: %s",
                            frame.path, exc)
                continue
            sized.append((frame.path, width, height))
        images = await encode_images([Path(item[0]) for item in sized], len(sized))
        image_token_total = sum(image_tokens(w, h) for _, w, h in sized)

        head, tail = _user_prompt_parts(
            source=asset.meta.webpage_url or asset.meta.canonical_id,
            title=asset.meta.title,
            frame_count=len(images),
        )
        skeleton_tokens = estimate_tokens(_SYSTEM_PROMPT + head + tail)
        prompt_tokens = skeleton_tokens + estimate_tokens(
            transcript if transcript else _NO_TRANSCRIPT_PROMPT
        )

        if fits_ctx(
            prompt_tokens,
            image_token_total,
            model_ctx=self._config.model_ctx,
            max_output_tokens=self._config.max_output_tokens,
        ):
            transcript_block = transcript
        else:
            log.info(
                "summarize: payload ~%d tokens exceeds ctx window; map-reduce",
                prompt_tokens + image_token_total,
            )
            transcript_block = await self._map_reduce(stats, model, transcript)

        body = transcript_block if transcript_block.strip() else _NO_TRANSCRIPT_PROMPT
        user = head + body + tail
        effort = (
            None
            if self._config.think is ThinkLevel.OFF
            else self._config.think.value
        )
        think_used = self._config.think.value

        result = await self._run_chat(
            stats, model=model, system_prompt=_SYSTEM_PROMPT, user_prompt=user,
            images=images, schema=_SUMMARY_SCHEMA, reasoning_effort=effort,
        )
        summary, key_points, visual_highlights = _normalize_payload(
            _extract_json_payload(result.text)
        )
        if _is_meaningful_summary(summary, key_points):
            return self._build_summary(
                model=model, think=think_used, stats=stats, transcript=transcript,
                frame_count=len(images), summary=summary, key_points=key_points,
                visual_highlights=visual_highlights,
            )

        reason = "low-information model response"
        log.warning("summarize: quality gate failed (%s); one retry", reason)
        retry = await self._run_chat(
            stats, model=model, system_prompt=_SYSTEM_PROMPT,
            user_prompt=user + _RETRY_FEEDBACK.format(reason=reason),
            images=images, schema=_SUMMARY_SCHEMA, reasoning_effort="low",
        )
        summary, key_points, visual_highlights = _normalize_payload(
            _extract_json_payload(retry.text)
        )
        if _is_meaningful_summary(summary, key_points):
            return self._build_summary(
                model=model, think=ThinkLevel.LOW.value, stats=stats,
                transcript=transcript, frame_count=len(images), summary=summary,
                key_points=key_points, visual_highlights=visual_highlights,
            )
        raise RuntimeError(f"summary quality gate failed: {reason}")

    def _build_summary(
        self,
        *,
        model: str,
        think: str,
        stats: _RunStats,
        transcript: str,
        frame_count: int,
        summary: str,
        key_points: list[str],
        visual_highlights: list[str],
    ) -> Summary:
        return Summary(
            summary=summary,
            key_points=tuple(key_points),
            visual_highlights=tuple(visual_highlights),
            model=model,
            think=think,
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            latency_ms=stats.latency_ms,
            vram_mb=0.0,  # server-side inference; field kept for schema compat
            transcript_chars=len(transcript),
            frame_count=frame_count,
            error=None,
        )

    async def _map_reduce(
        self, stats: _RunStats, model: str, transcript: str
    ) -> str:
        """Condense the transcript chunk-by-chunk, then join the summaries."""
        budget = chunk_budget_tokens(
            self._config.model_ctx, self._config.max_output_tokens
        )
        chunks = chunk_transcript(transcript, budget_tokens=budget,
                                  estimate=estimate_tokens)
        log.info("summarize: map-reduce over %d chunk(s)", len(chunks))
        summaries: list[str] = []
        for index, chunk in enumerate(chunks):
            user = (
                f"Transcript chunk {index + 1}/{len(chunks)}:\n\n{chunk}\n\n"
                "Return JSON only according to the schema."
            )
            result = await self._run_chat(
                stats, model=model, system_prompt=_MAP_SYSTEM_PROMPT,
                user_prompt=user, images=[], schema=_MAP_SCHEMA,
                reasoning_effort=None,
            )
            payload = _extract_json_payload(result.text)
            condensed = str(payload.get("summary", "") or "").strip()
            summaries.append(condensed or chunk[:4000])
        return _MAP_REDUCE_JOIN.join(summaries)

    async def _run_chat(
        self,
        stats: _RunStats,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: list[str],
        schema: dict[str, Any] | None,
        reasoning_effort: str | None,
    ) -> ChatResult:
        """One engine chat call with usage/latency accounting."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = await self._engine.chat(
            model=model,
            messages=messages,
            schema=schema,
            images=images or None,
            max_tokens=self._config.max_output_tokens,
            temperature=0.2,
            reasoning_effort=reasoning_effort,
        )
        stats.latency_ms += result.latency_ms
        stats.prompt_tokens += result.prompt_tokens
        stats.completion_tokens += result.completion_tokens
        return result
