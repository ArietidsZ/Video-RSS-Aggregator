from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core_config import Config
from service_ollama import (
    OllamaClient,
    encode_images_to_base64,
    model_size_hint_bytes,
)

log = logging.getLogger(__name__)

_MB = 1024 * 1024

_SYSTEM_PROMPT = """\
You summarize web videos using both transcript text and sampled video frames.

Output JSON with exactly these keys:
- "summary": one concise paragraph.
- "key_points": 4-8 factual bullets.
- "visual_highlights": 0-5 bullets describing key visual evidence.

Rules:
1) Do not invent facts.
2) Prefer transcript evidence when present.
3) Use frames to fill context only when transcript is sparse.
4) If uncertain, say so explicitly.
5) Keep wording neutral and information dense.
6) Use the transcript language when possible.
7) Output valid JSON only.
"""

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 40},
        "key_points": {
            "type": "array",
            "minItems": 4,
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "visual_highlights": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string"},
        },
    },
    "required": ["summary", "key_points", "visual_highlights"],
}


@dataclass(slots=True)
class SummaryResult:
    summary: str
    key_points: list[str] = field(default_factory=list)
    visual_highlights: list[str] = field(default_factory=list)
    model_used: str = ""
    vram_mb: float = 0.0
    transcript_chars: int = 0
    frame_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class ModelSelection:
    model: str
    current_vram_mb: float
    available_budget_mb: float
    estimated_overhead_mb: int
    reason: str


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
    cleaned = _strip_markdown_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {"summary": cleaned, "key_points": [], "visual_highlights": []}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"summary": cleaned, "key_points": [], "visual_highlights": []}


def _as_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _looks_like_oom(message: str) -> bool:
    lower = message.lower()
    return "out of memory" in lower or "cuda" in lower and "memory" in lower


def _is_meaningful_summary(summary: str, key_points: list[str]) -> bool:
    text = summary.strip()
    if len(text) >= 48 and text not in {"...", "…"}:
        return True
    if len(key_points) >= 2 and any(len(item) >= 24 for item in key_points):
        return True
    return False


def _fallback_key_points(transcript: str, summary: str, limit: int = 6) -> list[str]:
    source = transcript.strip() or summary.strip()
    if not source:
        return []

    chunks = re.split(r"(?<=[.!?])\s+|\n+", source)
    out: list[str] = []
    for raw in chunks:
        item = re.sub(r"\s+", " ", raw).strip(" -\t\n\r")
        if len(item) < 18:
            continue
        if len(item) > 220:
            item = item[:217].rstrip() + "..."
        if item in out:
            continue
        out.append(item)
        if len(out) >= limit:
            break

    if out:
        return out
    text = summary.strip()
    if text:
        return [text if len(text) <= 220 else text[:217].rstrip() + "..."]
    return []


def _fallback_key_points_from_summary(summary: str, limit: int = 4) -> list[str]:
    text = summary.strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|;\s+|\n+", text)
    out: list[str] = []
    for raw in chunks:
        item = re.sub(r"\s+", " ", raw).strip(" -\t\n\r")
        if len(item) < 14:
            continue
        if item in out:
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _first_nonempty_string(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _synthesize_summary(
    payload: dict[str, Any],
    transcript: str,
    key_points: list[str],
    visual_highlights: list[str],
) -> str:
    summary = _first_nonempty_string(
        payload,
        ["summary", "overview", "synopsis", "description", "abstract"],
    )
    if summary:
        return summary

    if key_points:
        head = key_points[:2]
        text = " ".join(item.rstrip(".") + "." for item in head).strip()
        if text:
            return text

    if visual_highlights:
        text = visual_highlights[0].strip()
        if text:
            return f"Video highlights: {text}"

    transcript_text = transcript.strip()
    if transcript_text:
        snippet = re.sub(r"\s+", " ", transcript_text[:400]).strip()
        if snippet:
            return snippet
    return ""


class SummarizationEngine:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._ollama = OllamaClient(config.ollama_base_url)
        self._known_local_models: set[str] = set()

    async def close(self) -> None:
        await self._ollama.close()

    async def _refresh_local_models(self) -> None:
        try:
            model_sizes = await self._ollama.list_local_models()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self._config.ollama_base_url}. "
                "Start Ollama and verify the URL with VRA_OLLAMA_BASE_URL."
            ) from exc
        self._known_local_models = {name.lower() for name in model_sizes}

    async def _ensure_model(self, model: str) -> None:
        if not self._known_local_models:
            await self._refresh_local_models()

        if model.lower() in self._known_local_models:
            return

        if not self._config.auto_pull_models:
            raise RuntimeError(
                f"Model {model} is not available locally. "
                "Enable VRA_AUTO_PULL_MODELS or run `ollama pull` manually."
            )

        log.info("Pulling missing model: %s", model)
        await self._ollama.pull_model(model)
        self._known_local_models.add(model.lower())

    def _budget_filtered_models(self) -> list[str]:
        budget_bytes = self._config.vram_budget_mb * 1024 * 1024
        model_limit = int(budget_bytes * self._config.model_size_budget_ratio)
        selected: list[str] = []

        for model in self._config.model_priority:
            hint = model_size_hint_bytes(model)
            if hint is not None and hint > model_limit:
                log.info(
                    "Skipping %s due to VRAM budget (hint=%d MB, limit=%d MB)",
                    model,
                    hint // (1024 * 1024),
                    model_limit // (1024 * 1024),
                )
                continue
            selected.append(model)

        if not selected:
            selected.append(self._config.model_min)
        return selected

    def _profile_for_model(
        self,
        model: str,
        image_payload: list[str],
        transcript_chars: int,
    ) -> tuple[list[str], int]:
        num_ctx = self._config.context_tokens
        images = image_payload

        if transcript_chars <= 1200 and len(images) > 4:
            images = images[:4]
            num_ctx = min(num_ctx, 2048)
        elif transcript_chars >= 12000 and len(images) > 5:
            images = images[:5]
            num_ctx = min(num_ctx, 4096)

        hint = model_size_hint_bytes(model)
        if hint is not None:
            hint_mb = hint // _MB
            if hint_mb >= 6000:
                num_ctx = min(num_ctx, 2048)
                images = image_payload[: min(4, len(image_payload))]
            elif hint_mb >= 3000:
                num_ctx = min(num_ctx, 3072)
                images = image_payload[: min(5, len(image_payload))]

        return images, num_ctx

    def _estimate_request_overhead_mb(
        self,
        transcript_chars: int,
        frame_count: int,
    ) -> int:
        transcript_scale = min(max(transcript_chars, 0) / 3200.0, 6.0)
        frame_scale = min(max(frame_count, 0), max(self._config.max_frames, 1))
        return int(320 + transcript_scale * 110 + frame_scale * 90)

    async def _select_model_for_job(
        self,
        candidates: list[str],
        transcript_chars: int,
        frame_count: int,
    ) -> ModelSelection:
        if not candidates:
            return ModelSelection(
                model=self._config.model_min,
                current_vram_mb=0.0,
                available_budget_mb=float(self._config.vram_budget_mb),
                estimated_overhead_mb=0,
                reason="No eligible candidates after budget filtering",
            )

        running_models = await self._ollama.list_running_models()
        running_names = {item.name.lower() for item in running_models}
        current_vram_mb = sum(item.vram_bytes for item in running_models) / _MB
        reserve_mb = max(self._config.model_selection_reserve_mb, 0)
        available_budget_mb = max(
            self._config.vram_budget_mb - current_vram_mb - reserve_mb,
            0.0,
        )
        estimated_overhead_mb = self._estimate_request_overhead_mb(
            transcript_chars,
            frame_count,
        )

        fallback_model = min(
            candidates,
            key=lambda name: model_size_hint_bytes(name) or (1 << 62),
        )

        for model in candidates:
            hint_bytes = model_size_hint_bytes(model)
            hint_mb = float(hint_bytes // _MB) if hint_bytes is not None else 2048.0
            loaded = model.lower() in running_names
            required_mb = estimated_overhead_mb + (0.0 if loaded else hint_mb)
            if required_mb <= available_budget_mb:
                return ModelSelection(
                    model=model,
                    current_vram_mb=round(current_vram_mb, 2),
                    available_budget_mb=round(available_budget_mb, 2),
                    estimated_overhead_mb=estimated_overhead_mb,
                    reason=(
                        f"selected from budget fit: required={int(required_mb)}MB, "
                        f"available={int(available_budget_mb)}MB, loaded={loaded}"
                    ),
                )

        return ModelSelection(
            model=fallback_model,
            current_vram_mb=round(current_vram_mb, 2),
            available_budget_mb=round(available_budget_mb, 2),
            estimated_overhead_mb=estimated_overhead_mb,
            reason=(
                "no model fit current available budget; "
                f"pinning smallest candidate {fallback_model}"
            ),
        )

    async def prepare_models(self) -> list[str]:
        pulled: list[str] = []
        for model in self._budget_filtered_models():
            await self._ensure_model(model)
            pulled.append(model)
        return pulled

    async def runtime_status(self) -> dict[str, Any]:
        try:
            version = await self._ollama.version()
            local_models = await self._ollama.list_local_models()
            running_models = await self._ollama.list_running_models()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to query Ollama runtime at {self._config.ollama_base_url}. "
                "Start Ollama and retry."
            ) from exc
        return {
            "ollama_version": version,
            "local_models": local_models,
            "running_models": [
                {
                    "name": item.name,
                    "size_mb": round(item.size_bytes / (1024 * 1024), 2),
                    "vram_mb": round(item.vram_bytes / (1024 * 1024), 2),
                    "expires_at": item.expires_at,
                }
                for item in running_models
            ],
            "total_vram_mb": round(
                sum(item.vram_bytes for item in running_models) / _MB,
                2,
            ),
            "vram_budget_mb": self._config.vram_budget_mb,
            "model_selection_reserve_mb": self._config.model_selection_reserve_mb,
            "model_priority": self._budget_filtered_models(),
        }

    async def summarize(
        self,
        *,
        source_url: str,
        title: str | None,
        transcript: str,
        frame_paths: list[Path],
    ) -> SummaryResult:
        transcript_text = transcript.strip()
        if len(transcript_text) > self._config.max_transcript_chars:
            transcript_text = transcript_text[: self._config.max_transcript_chars]

        image_payload = encode_images_to_base64(frame_paths, self._config.max_frames)
        candidates = self._budget_filtered_models()

        errors: list[str] = []
        selected_model = ""
        try:
            selection = await self._select_model_for_job(
                candidates,
                transcript_chars=len(transcript_text),
                frame_count=len(image_payload),
            )
            selected_model = selection.model
            log.info("Model selected: %s (%s)", selected_model, selection.reason)

            await self._ensure_model(selected_model)
            profile_images, profile_ctx = self._profile_for_model(
                selected_model,
                image_payload,
                transcript_chars=len(transcript_text),
            )
            prompt = self._build_user_prompt(
                source_url=source_url,
                title=title,
                transcript=transcript_text,
                image_count=len(profile_images),
            )
            chat = await self._ollama.chat_json(
                model=selected_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                schema=_SUMMARY_SCHEMA,
                images=profile_images,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_ctx": profile_ctx,
                    "num_predict": self._config.max_output_tokens,
                },
                keep_alive=0,
            )

            payload = _extract_json_payload(chat.text)
            summary = _synthesize_summary(
                payload,
                transcript_text,
                key_points=[],
                visual_highlights=[],
            )
            if len(summary) > 900:
                summary = summary[:897].rstrip() + "..."
            key_points = _as_string_list(payload.get("key_points"), 8)
            visual_highlights = _as_string_list(payload.get("visual_highlights"), 5)

            if not summary:
                summary = _synthesize_summary(
                    payload,
                    transcript_text,
                    key_points=key_points,
                    visual_highlights=visual_highlights,
                )

            if len(key_points) < 4:
                fallback_points = _fallback_key_points(
                    transcript_text, summary, limit=6
                )
                for item in fallback_points:
                    if item not in key_points:
                        key_points.append(item)
                    if len(key_points) >= 6:
                        break
            if len(key_points) < 3:
                summary_points = _fallback_key_points_from_summary(summary, limit=4)
                for item in summary_points:
                    if item not in key_points:
                        key_points.append(item)
                    if len(key_points) >= 4:
                        break

            if not summary:
                raise RuntimeError("Model response did not include a summary")
            if not _is_meaningful_summary(summary, key_points):
                raise RuntimeError("Model response was too low-information")

            current_vram = await self._ollama.total_vram_bytes()
            budget_bytes = self._config.vram_budget_mb * _MB
            if current_vram > budget_bytes:
                errors.append(
                    f"{selected_model}: exceeded VRAM budget "
                    f"({current_vram // _MB} MB > {self._config.vram_budget_mb} MB)"
                )

            return SummaryResult(
                summary=summary,
                key_points=key_points,
                visual_highlights=visual_highlights,
                model_used=selected_model,
                vram_mb=round(current_vram / _MB, 2),
                transcript_chars=len(transcript_text),
                frame_count=len(profile_images),
                error=" | ".join(errors) if errors else None,
            )
        except Exception as exc:
            label = selected_model or "model-selection"
            errors.append(f"{label}: {exc}")
            log.warning("Model attempt failed for %s: %s", label, exc)
            if selected_model and _looks_like_oom(str(exc)):
                await self._ollama.unload_model(selected_model)

        fallback_summary = "Unable to generate a model summary for this source."
        if transcript_text:
            fallback_summary = transcript_text[:400].strip()

        return SummaryResult(
            summary=fallback_summary,
            key_points=[],
            visual_highlights=[],
            model_used=selected_model,
            vram_mb=0.0,
            transcript_chars=len(transcript_text),
            frame_count=len(image_payload),
            error=" | ".join(errors) if errors else "No eligible models",
        )

    @staticmethod
    def _build_user_prompt(
        *,
        source_url: str,
        title: str | None,
        transcript: str,
        image_count: int,
    ) -> str:
        transcript_block = transcript or "(No subtitles/transcript available.)"
        title_line = title if title else "(Unknown title)"
        return (
            f"Source: {source_url}\n"
            f"Title: {title_line}\n"
            f"Attached Frames: {image_count}\n\n"
            "Transcript:\n"
            f"{transcript_block}\n\n"
            "Return JSON only according to the schema."
        )
