"""OpenAI-compatible inference engine client (owner: agent-B).

Single engine path for the whole product: any OpenAI-compatible server
(vLLM / SGLang / oMLX / llama-server / frontier APIs). Replaces the
former Ollama client + VRAM governor — the server owns model loading,
context windows (``max_model_len``) and scheduling, so this module is a
thin, reliable HTTP adapter:

- ``models()``: list served model ids (GET ``{base}/models``).
- ``has_model(name, ttl_s=60)``: cached presence check with tolerant
  matching (see normalization note below).
- ``chat(...)``: chat completion with optional strict JSON-schema
  output, base64 images on the last user message, and
  ``reasoning_effort`` passthrough. Retries 3 attempts with exponential
  backoff (1s/2s/4s schedule; the 1s and 2s gaps separate the three
  attempts) on connect errors, timeouts, remote-protocol errors and 5xx;
  4xx fails fast with a RuntimeError carrying the trimmed body.
- ``runtime_status(...)``: diagnostics shape for setup UI
  (engine + asr sections; the ASR section defaults to the same server,
  callers may pass a second EngineClient for a dedicated ASR server).
- ``encode_images``: base64-encode frame files off the event loop.

Model-id normalization (used by ``has_model`` and ``runtime_status``):
both the requested name and every served id are lowercased and stripped
of directory prefixes (everything before the last ``/``), so
``Qwen/Qwen3.5-4B-AWQ`` matches config ``qwen3.5-4b``. Matching is exact
first, then normalized-equal, then normalized-prefix (served id starts
with the requested name), which tolerates quantization suffixes like
``-awq``/``-gptq``/``-q4_k_m``; the prefix rule can in principle also
match unrelated longer names sharing the prefix — accepted trade-off.

``reasoning_effort`` is sent verbatim when provided; servers that do not
implement it (e.g. llama-server) ignore unknown fields per the OpenAI
wire convention, so no capability probing is done.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

__all__ = ["ChatResult", "EngineClient", "encode_images"]

log = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)

_IMAGE_DATA_URL_PREFIX = "data:image/jpeg;base64,"


def _monotonic() -> float:
    """Indirection over time.monotonic so tests can manipulate the clock."""
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    """Indirection over asyncio.sleep so tests can intercept backoff."""
    await asyncio.sleep(seconds)


def _normalize_model_id(model_id: str) -> str:
    """Lowercase ``model_id`` and keep only the segment after the last ``/``."""
    return model_id.strip().rpartition("/")[2].strip().lower()


def _model_matches(served: list[str], name: str) -> bool:
    """Tolerant presence check of ``name`` against served model ids.

    Raw exact match first, then normalized equality, then normalized
    prefix (served id starts with the requested name) — see module
    docstring for the normalization rules.
    """
    if name in served:
        return True
    target = _normalize_model_id(name)
    if not target:
        return False
    for served_id in served:
        normalized = _normalize_model_id(served_id)
        if normalized == target or normalized.startswith(target):
            return True
    return False


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Outcome of one ``/chat/completions`` call.

    ``latency_ms`` is the total wall time of the call including any
    retry backoff; tokens come from the server ``usage`` block.
    """

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


async def encode_images(paths: Sequence[Path], limit: int) -> list[str]:
    """Base64-encode up to ``limit`` image files.

    File reads happen via ``asyncio.to_thread`` so large JPEGs never
    block the event loop.
    """
    out: list[str] = []
    for path in list(paths)[: max(limit, 0)]:
        data = await asyncio.to_thread(Path(path).read_bytes)
        out.append(base64.b64encode(data).decode("ascii"))
    return out


class EngineClient:
    """Async client for one OpenAI-compatible inference server."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout_s: float = 900.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=30.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            transport=transport,
        )
        self._models_cache: list[str] | None = None
        self._models_cached_at: float = float("-inf")

    @property
    def base_url(self) -> str:
        """Base URL this client talks to (trailing slash stripped)."""
        return self._base_url

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Issue a request with retry/backoff; return a 2xx response.

        Network errors (ConnectError, TimeoutException,
        RemoteProtocolError) and 5xx responses are retried with
        exponential backoff. 4xx raises RuntimeError immediately with
        the trimmed response body.
        """
        last_error: Exception | None = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                kwargs: dict[str, Any] = {}
                if json_body is not None:
                    kwargs["json"] = json_body
                response = await self._client.request(method, path, **kwargs)
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
            ) as exc:
                last_error = exc
                log.warning(
                    "Engine %s %s network error on attempt %d/%d: %s",
                    method,
                    path,
                    attempt,
                    _RETRY_ATTEMPTS,
                    exc,
                )
            else:
                if response.status_code < 400:
                    return response
                body = response.text.strip()
                detail = f": {body}" if body else ""
                if response.status_code < 500:
                    raise RuntimeError(
                        f"Engine HTTP {response.status_code} on {path}{detail}"
                    )
                last_error = RuntimeError(
                    f"Engine HTTP {response.status_code} on {path}{detail}"
                )
                log.warning(
                    "Engine HTTP %d on %s (attempt %d/%d)",
                    response.status_code,
                    path,
                    attempt,
                    _RETRY_ATTEMPTS,
                )
            if attempt < _RETRY_ATTEMPTS:
                await _sleep(_RETRY_DELAYS[attempt - 1])
        if last_error is None:  # pragma: no cover - loop always assigns it
            raise RuntimeError(f"Engine request failed: {method} {path}")
        raise last_error

    async def models(self) -> list[str]:
        """Served model ids from GET ``{base}/models``."""
        response = await self._request("GET", "/models")
        data = response.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        out: list[str] = []
        for item in items:
            model_id = str(item.get("id", "")).strip()
            if model_id:
                out.append(model_id)
        return out

    async def _models_cached(self, ttl_s: float) -> list[str]:
        """``models()`` with a time-based cache (``ttl_s`` <= 0 = always refresh)."""
        now = _monotonic()
        if self._models_cache is None or now - self._models_cached_at >= ttl_s:
            self._models_cache = await self.models()
            self._models_cached_at = now
        return self._models_cache

    async def has_model(self, name: str, ttl_s: float = 60.0) -> bool:
        """True when ``name`` is served (exact, normalized or prefix match)."""
        served = await self._models_cached(ttl_s)
        return _model_matches(served, name)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        images: Sequence[str] | None = None,
        max_tokens: int,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        """Run one non-streaming chat completion.

        ``images`` (base64 JPEG strings) are attached to the last user
        message as ``image_url`` parts with ``data:image/jpeg;base64,``
        URLs. When ``schema`` is given the request carries a strict
        ``response_format`` json_schema block. ``reasoning_effort`` is
        sent verbatim when not None; engines without support ignore it.
        ``latency_ms`` measures wall time including retry backoff.
        """
        payload_messages = [dict(message) for message in messages]
        if images:
            for index in range(len(payload_messages) - 1, -1, -1):
                if payload_messages[index].get("role") == "user":
                    content = payload_messages[index].get("content", "")
                    parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
                    parts.extend(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"{_IMAGE_DATA_URL_PREFIX}{image}"},
                        }
                        for image in images
                    )
                    payload_messages[index]["content"] = parts
                    break
        payload: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        started = _monotonic()
        response = await self._request("POST", "/chat/completions", json_body=payload)
        latency_ms = int((_monotonic() - started) * 1000)
        data = response.json()
        if not isinstance(data, dict):
            return ChatResult(text="", model=model, prompt_tokens=0,
                              completion_tokens=0, latency_ms=latency_ms)
        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else None
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content", "") or "")
        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return ChatResult(
            text=text,
            model=str(data.get("model", model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=latency_ms,
        )

    async def runtime_status(
        self,
        *,
        model: str = "",
        asr_model: str = "",
        asr_client: EngineClient | None = None,
    ) -> dict[str, Any]:
        """Diagnostics snapshot for the setup UI.

        ``model`` / ``asr_model`` are checked against the served ids of
        the engine (and of ``asr_client`` when a dedicated ASR server is
        configured, defaulting to this same server). All probes are
        best-effort: a failing server yields ``reachable: False`` and
        empty model lists instead of raising.
        """
        try:
            engine_models = await self.models()
            reachable = True
        except Exception as exc:
            log.warning("runtime_status: engine probe failed: %s", exc)
            engine_models, reachable = [], False
        asr_models = engine_models
        if asr_client is not None and asr_client is not self:
            try:
                asr_models = await asr_client.models()
            except Exception as exc:
                log.warning("runtime_status: asr probe failed: %s", exc)
                asr_models = []
        return {
            "engine": {
                "base_url": self._base_url,
                "reachable": reachable,
                "models": engine_models,
                "model_present": _model_matches(engine_models, model) if model else False,
            },
            "asr": {
                "models": asr_models,
                "model_present": _model_matches(asr_models, asr_model) if asr_model else False,
            },
        }
