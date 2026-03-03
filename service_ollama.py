from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx


MODEL_SIZE_HINT_MB: dict[str, int] = {
    "qwen3.5:0.8b": 1024,
    "qwen3.5:0.8b-q8_0": 1024,
    "qwen3.5:2b": 2765,
    "qwen3.5:2b-q4_k_m": 1945,
    "qwen3.5:4b": 3482,
    "qwen3.5:4b-q4_k_m": 3482,
    "qwen3.5:9b": 6758,
    "qwen3.5:9b-q4_k_m": 6758,
}


@dataclass(slots=True)
class RunningModel:
    name: str
    size_bytes: int
    vram_bytes: int
    expires_at: str | None = None


@dataclass(slots=True)
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


def model_size_hint_bytes(model: str) -> int | None:
    hint_mb = MODEL_SIZE_HINT_MB.get(model.lower().strip(), None)
    if hint_mb is None:
        return None
    return hint_mb * 1024 * 1024


def encode_images_to_base64(paths: Sequence[Path], limit: int) -> list[str]:
    out: list[str] = []
    for path in paths[: max(limit, 0)]:
        out.append(base64.b64encode(path.read_bytes()).decode("ascii"))
    return out


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=60.0, pool=30.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def version(self) -> str:
        resp = await self._client.get("/api/version")
        resp.raise_for_status()
        return str(resp.json().get("version", "unknown"))

    async def list_local_models(self) -> dict[str, int]:
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        out: dict[str, int] = {}
        for item in data.get("models", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            out[name] = int(item.get("size", 0) or 0)
        return out

    async def list_running_models(self) -> list[RunningModel]:
        resp = await self._client.get("/api/ps")
        resp.raise_for_status()
        data = resp.json()
        out: list[RunningModel] = []
        for item in data.get("models", []):
            out.append(
                RunningModel(
                    name=str(item.get("name", "")),
                    size_bytes=int(item.get("size", 0) or 0),
                    vram_bytes=int(item.get("size_vram", 0) or 0),
                    expires_at=item.get("expires_at"),
                )
            )
        return out

    async def total_vram_bytes(self) -> int:
        running = await self.list_running_models()
        return sum(item.vram_bytes for item in running)

    async def pull_model(self, model: str) -> None:
        resp = await self._client.post(
            "/api/pull",
            json={"model": model, "stream": False},
            timeout=1800,
        )
        resp.raise_for_status()
        status = str(resp.json().get("status", "")).lower()
        if status and status != "success":
            raise RuntimeError(f"Failed to pull model {model}: {status}")

    async def unload_model(self, model: str) -> None:
        await self._client.post(
            "/api/chat",
            json={"model": model, "messages": [], "keep_alive": 0},
            timeout=60,
        )

    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        images: list[str] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: int | str = 0,
    ) -> ChatResult:
        payload_messages = [dict(item) for item in messages]
        if images:
            for idx in range(len(payload_messages) - 1, -1, -1):
                if payload_messages[idx].get("role") == "user":
                    payload_messages[idx]["images"] = images
                    break

        payload = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": keep_alive,
        }
        if options:
            payload["options"] = options

        resp = await self._client.post("/api/chat", json=payload, timeout=600)
        if resp.status_code >= 400:
            raise RuntimeError(resp.text.strip() or f"Ollama error {resp.status_code}")

        data = resp.json()
        msg = data.get("message", {}) if isinstance(data, dict) else {}
        content = str(msg.get("content", ""))
        return ChatResult(
            text=content,
            model=str(data.get("model", model)),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
        )
