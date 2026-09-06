"""Tests for vra.inference.engine (httpx.MockTransport, no live server)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

import vra.inference.engine as engine_module
from vra.inference.engine import ChatResult, EngineClient, encode_images

Handler = Callable[[httpx.Request], httpx.Response]


@asynccontextmanager
async def fake_engine(
    handler: Handler, api_key: str | None = None
) -> AsyncIterator[EngineClient]:
    client = EngineClient(
        "http://engine.test/v1",
        api_key,
        timeout_s=30.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        yield client
    finally:
        await client.close()


def models_response(ids: list[str]) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


def completion_body(
    text: str = "ok",
    *,
    prompt_tokens: int = 111,
    completion_tokens: int = 22,
    model: str = "qwen3.5-4b",
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


async def test_models_lists_ids_and_sends_bearer() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        return models_response(["qwen3.5-4b", "Qwen/Qwen3.5-4B-AWQ"])

    async with fake_engine(handler, api_key="sk-secret") as client:
        assert await client.models() == ["qwen3.5-4b", "Qwen/Qwen3.5-4B-AWQ"]
    assert seen["path"] == "/v1/models"
    assert seen["auth"] == "Bearer sk-secret"


async def test_models_without_key_has_no_auth_header() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return models_response(["m"])

    async with fake_engine(handler, api_key=None) as client:
        await client.models()
    assert seen["auth"] is None


async def test_has_model_exact_match() -> None:
    async with fake_engine(lambda req: models_response(["qwen3.5-4b"])) as client:
        assert await client.has_model("qwen3.5-4b") is True
        assert await client.has_model("qwen3.5-2b") is False


async def test_has_model_normalizes_org_prefix_and_case() -> None:
    async with fake_engine(
        lambda req: models_response(["Qwen/Qwen3.5-4B-AWQ"])
    ) as client:
        assert await client.has_model("qwen3.5-4b") is True
        assert await client.has_model("Qwen/Qwen3.5-4B-AWQ") is True


async def test_has_model_quant_suffix_prefix_match() -> None:
    async with fake_engine(lambda req: models_response(["m-awq"])) as client:
        assert await client.has_model("m") is True
        assert await client.has_model("other") is False


async def test_has_model_uses_ttl_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    calls = {"n": 0}
    monkeypatch.setattr(engine_module, "_monotonic", lambda: clock["now"])

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return models_response(["qwen3.5-4b"])

    async with fake_engine(handler) as client:
        assert await client.has_model("qwen3.5-4b") is True
        clock["now"] += 10.0  # within the 60s TTL
        assert await client.has_model("qwen3.5-4b") is True
        assert calls["n"] == 1
        clock["now"] += 61.0  # TTL expired -> refetch
        assert await client.has_model("qwen3.5-2b") is False
        assert calls["n"] == 2


async def test_chat_payload_schema_and_usage_mapping() -> None:
    seen: dict[str, Any] = {}
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=completion_body(prompt_tokens=500, completion_tokens=200),
        )

    async with fake_engine(handler) as client:
        result = await client.chat(
            "qwen3.5-4b",
            [{"role": "user", "content": "hi"}],
            schema=schema,
            max_tokens=1024,
            temperature=0.2,
        )
    payload = seen["payload"]
    assert seen["path"] == "/v1/chat/completions"
    assert payload["model"] == "qwen3.5-4b"
    assert payload["max_tokens"] == 1024
    assert payload["temperature"] == 0.2
    assert payload["stream"] is False
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }
    assert "reasoning_effort" not in payload
    assert result.text == "ok"
    assert result.model == "qwen3.5-4b"
    assert result.prompt_tokens == 500
    assert result.completion_tokens == 200
    assert isinstance(result.latency_ms, int)
    assert result.latency_ms >= 0


async def test_chat_sends_reasoning_effort_when_given() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body())

    async with fake_engine(handler) as client:
        await client.chat(
            "m",
            [{"role": "user", "content": "hi"}],
            max_tokens=256,
            reasoning_effort="low",
        )
    assert seen["payload"]["reasoning_effort"] == "low"


async def test_chat_without_schema_has_no_response_format() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body())

    async with fake_engine(handler) as client:
        await client.chat("m", [{"role": "user", "content": "hi"}], max_tokens=8)
    assert "response_format" not in seen["payload"]


async def test_chat_images_attach_to_last_user_message() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body())

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "mid"},
        {"role": "user", "content": "look"},
    ]
    async with fake_engine(handler) as client:
        await client.chat("m", messages, images=["QUJD", "RUZH"], max_tokens=8)

    sent = seen["payload"]["messages"]
    assert sent[1]["content"] == "first"
    parts = sent[3]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1:] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,RUZH"}},
    ]
    # caller's message list must not be mutated
    assert messages[3]["content"] == "look"


async def test_chat_retry_then_success_on_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(engine_module, "_sleep", fake_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(500, text="overloaded")
        return httpx.Response(200, json=completion_body("recovered"))

    async with fake_engine(handler) as client:
        result = await client.chat("m", [{"role": "user", "content": "x"}],
                                   max_tokens=8)
    assert calls["n"] == 3
    assert delays == [1.0, 2.0]
    assert result.text == "recovered"


async def test_chat_4xx_fails_fast_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_sleep(seconds: float) -> None:
        raise AssertionError("no backoff expected on 4xx")

    monkeypatch.setattr(engine_module, "_sleep", fail_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text='{"error": {"message": "bad model"}}')

    async with fake_engine(handler) as client:
        with pytest.raises(RuntimeError, match="bad model"):
            await client.chat("missing", [{"role": "user", "content": "x"}],
                              max_tokens=8)
    assert calls["n"] == 1


async def test_chat_retries_exhausted_on_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(engine_module, "_sleep", fake_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="down")

    async with fake_engine(handler) as client:
        with pytest.raises(RuntimeError, match="down"):
            await client.chat("m", [{"role": "user", "content": "x"}],
                              max_tokens=8)
    assert calls["n"] == 3
    assert delays == [1.0, 2.0]


async def test_runtime_status_shape_and_reachability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(engine_module, "_sleep", fake_sleep)

    def ok_handler(request: httpx.Request) -> httpx.Response:
        return models_response(["qwen3.5-4b", "Qwen/Qwen3-ASR-1.7B"])

    async with fake_engine(ok_handler) as client:
        status = await client.runtime_status(
            model="qwen3.5-4b", asr_model="Qwen3-ASR-1.7B"
        )
    assert status["engine"]["reachable"] is True
    assert status["engine"]["base_url"] == "http://engine.test/v1"
    assert status["engine"]["models"] == ["qwen3.5-4b", "Qwen/Qwen3-ASR-1.7B"]
    assert status["engine"]["model_present"] is True
    assert status["asr"]["models"] == ["qwen3.5-4b", "Qwen/Qwen3-ASR-1.7B"]
    assert status["asr"]["model_present"] is True

    def down_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="nope")

    async with fake_engine(down_handler) as client:
        status = await client.runtime_status(model="qwen3.5-4b")
    assert status["engine"]["reachable"] is False
    assert status["engine"]["models"] == []
    assert status["engine"]["model_present"] is False
    assert status["asr"]["models"] == []


async def test_runtime_status_with_dedicated_asr_client() -> None:
    async with (
        fake_engine(lambda req: models_response(["qwen3.5-4b"])) as engine,
        fake_engine(lambda req: models_response(["whisper-large"])) as asr,
    ):
        status = await engine.runtime_status(
            model="qwen3.5-4b", asr_model="whisper-large", asr_client=asr
        )
    assert status["engine"]["models"] == ["qwen3.5-4b"]
    assert status["asr"]["models"] == ["whisper-large"]
    assert status["asr"]["model_present"] is True


async def test_encode_images_limit_and_content(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(3):
        path = tmp_path / f"{index}.bin"
        await asyncio.to_thread(path.write_bytes, bytes([index]) * 5)
        paths.append(path)
    encoded = await encode_images(paths, 2)
    assert len(encoded) == 2
    assert base64.b64decode(encoded[0]) == bytes([0]) * 5
    assert await encode_images(paths, 0) == []


async def test_chat_result_dataclass_shape() -> None:
    result = ChatResult(text="t", model="m", prompt_tokens=1,
                        completion_tokens=2, latency_ms=3)
    assert (result.text, result.model, result.prompt_tokens,
            result.completion_tokens, result.latency_ms) == ("t", "m", 1, 2, 3)
