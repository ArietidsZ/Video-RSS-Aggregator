"""Tests for vra.inference.summarize (httpx.MockTransport, no live server).

The lead's rewritten vra/config.py (engine_base_url / engine_api_key /
model / model_ctx) was not present in any git ref visible from this
worktree at implementation time, so these tests construct a local frozen
stand-in mirroring the addendum's field names verbatim. When the lead's
config lands, the stand-in can be swapped for vra.config.Config.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import struct
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

import vra.inference.engine as engine_module
from vra.config import ThinkLevel
from vra.inference.engine import EngineClient
from vra.inference.summarize import (
    _SUMMARY_SCHEMA,
    VlmSummarizer,
    _extract_json_payload,
    _is_meaningful_summary,
    chunk_transcript,
    estimate_tokens,
    image_tokens,
    read_jpeg_size,
)
from vra.types import Frame, FrameSet, MediaAsset, Transcript, VideoMeta


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Stand-in for the lead's rewritten Config (see module docstring)."""

    engine_base_url: str = "http://engine.test/v1"
    engine_api_key: str | None = None
    model: str = "qwen3.5-4b"
    model_ctx: int = 32768
    think: ThinkLevel = ThinkLevel.OFF
    max_output_tokens: int = 1024
    timeout_infer_s: int = 900
    max_transcript_chars: int = 24000
    max_frames: int = 8
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"


Handler = Callable[[httpx.Request], httpx.Response]

GOOD_JSON = json.dumps(
    {
        "summary": "A detailed and factual summary of the video content.",
        "key_points": [
            "First key point that is long enough for the gate here",
            "Second key point also long enough for the gate",
        ],
        "visual_highlights": ["A red title card at the opening"],
    }
)


def minimal_jpeg(width: int = 320, height: int = 200, sof: int = 0xC0) -> bytes:
    """Handcrafted JPEG: SOI + APP0 + SOF(+sof) + EOI with given dimensions."""
    sof_payload = struct.pack(">BHH", 8, height, width)
    segment = struct.pack(">H", len(sof_payload) + 2) + sof_payload
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + bytes(9)
    return b"\xff\xd8" + app0 + bytes([0xFF, sof]) + segment + b"\xff\xd9"


def make_asset(transcript: str, frame_paths: list[Path]) -> MediaAsset:
    return MediaAsset(
        meta=VideoMeta(
            canonical_id="yt:test123",
            title="Test video",
            webpage_url="https://example.com/watch?v=test",
        ),
        media_path=None,
        transcript=Transcript(text=transcript, source="subtitle"),
        frames=FrameSet(
            frames=tuple(
                Frame(path=str(path), ts_s=float(i))
                for i, path in enumerate(frame_paths)
            )
        ),
    )


def make_handler(
    served: list[str], chat_fn: Callable[[dict[str, Any]], httpx.Response]
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": i} for i in served]}
            )
        if path == "/v1/chat/completions":
            return chat_fn(json.loads(request.content))
        raise AssertionError(f"unexpected path {path}")

    return handler


@asynccontextmanager
async def make_summarizer(
    config: EngineConfig, served: list[str], chat_fn: Callable[[dict], httpx.Response]
) -> AsyncIterator[tuple[VlmSummarizer, EngineClient, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    def chat_fn_recording(payload: dict[str, Any]) -> httpx.Response:
        requests.append(payload)
        return chat_fn(payload)

    engine = EngineClient(
        config.engine_base_url,
        config.engine_api_key,
        timeout_s=30.0,
        transport=httpx.MockTransport(make_handler(served, chat_fn_recording)),
    )
    try:
        yield VlmSummarizer(config, engine), engine, requests
    finally:
        await engine.close()


def completion(
    text: str, *, prompt_tokens: int = 111, completion_tokens: int = 22
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "qwen3.5-4b",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    )


# --------------------------------------------------------------------------
# JPEG size parsing
# --------------------------------------------------------------------------


async def test_read_jpeg_size_handcrafted(tmp_path: Path) -> None:
    path = tmp_path / "f.jpg"
    await asyncio.to_thread(path.write_bytes, minimal_jpeg(320, 200))
    assert await read_jpeg_size(path) == (320, 200)


async def test_read_jpeg_size_progressive_sof2(tmp_path: Path) -> None:
    path = tmp_path / "p.jpg"
    await asyncio.to_thread(path.write_bytes, minimal_jpeg(64, 48, sof=0xC2))
    assert await read_jpeg_size(path) == (64, 48)


async def test_read_jpeg_size_rejects_non_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    await asyncio.to_thread(path.write_bytes, b"\x89PNG\r\n\x1a\n" + bytes(32))
    with pytest.raises(ValueError, match="not a JPEG"):
        await read_jpeg_size(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_read_jpeg_size_ffmpeg_generated(tmp_path: Path) -> None:
    out = tmp_path / "gen.jpg"
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=red:s=64x32", "-frames:v", "1", str(out),
    )
    await asyncio.wait_for(proc.wait(), 60)
    assert await read_jpeg_size(out) == (64, 32)


# --------------------------------------------------------------------------
# Token / chunk math
# --------------------------------------------------------------------------


def test_estimate_tokens_cjk_vs_latin() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("abcdefgh") == math.ceil(8 / 3.6)
    assert estimate_tokens("hi你好") == math.ceil(2 + 2 / 3.6)
    assert estimate_tokens("你好") > estimate_tokens("hi")


def test_image_tokens_patch_math() -> None:
    assert image_tokens(224, 224) == 16  # 8*8//4
    assert image_tokens(56, 56) == 1  # 2*2//4
    assert image_tokens(28, 28) == 0  # 1*1//4
    assert image_tokens(224, 56) == 4  # 8*2//4


def test_chunk_transcript_respects_sentence_boundaries() -> None:
    text = " ".join(
        f"The quick brown fox jumps over lazy dog {i}." for i in range(60)
    )
    chunks = chunk_transcript(text, budget_tokens=120, estimate=estimate_tokens)
    assert len(chunks) >= 2
    assert " ".join(chunks) == text
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 120


def test_chunk_transcript_hard_splits_oversized_sentence() -> None:
    text = " ".join(["word"] * 1000)
    chunks = chunk_transcript(text, budget_tokens=100, estimate=estimate_tokens)
    assert len(chunks) > 1
    assert " ".join(chunks) == text
    for chunk in chunks:
        assert estimate_tokens(chunk) <= 100


# --------------------------------------------------------------------------
# Parsing + gate helpers
# --------------------------------------------------------------------------


def test_extract_json_payload_variants() -> None:
    assert _extract_json_payload('{"summary": "x"}')["summary"] == "x"
    fenced = "```json\n{\"summary\": \"x\"}\n```"
    assert _extract_json_payload(fenced)["summary"] == "x"
    embedded = 'Sure! {"summary": "x"} hope that helps'
    assert _extract_json_payload(embedded)["summary"] == "x"
    assert _extract_json_payload("no json at all")["summary"] == "no json at all"


def test_is_meaningful_summary_gate() -> None:
    assert _is_meaningful_summary("x" * 48, [])
    assert not _is_meaningful_summary("too short", [])
    assert _is_meaningful_summary("short", ["ok", "y" * 24])
    assert not _is_meaningful_summary("short", ["ok", "y"])
    assert _is_meaningful_summary("", ["a" * 24, "b" * 24])


# --------------------------------------------------------------------------
# Summarizer behaviour
# --------------------------------------------------------------------------


async def test_summarize_single_pass(tmp_path: Path) -> None:
    frame = tmp_path / "f0.jpg"
    await asyncio.to_thread(frame.write_bytes, minimal_jpeg(64, 32))
    config = EngineConfig()
    transcript = "Short transcript about cats. Second sentence here."

    async with make_summarizer(
        config, ["qwen3.5-4b"], lambda p: completion(GOOD_JSON)
    ) as (summarizer, _engine, requests):
        summary = await summarizer.summarize(make_asset(transcript, [frame]))

    assert summary.error is None
    assert summary.model == "qwen3.5-4b"
    assert summary.think == "off"
    assert summary.frame_count == 1
    assert summary.transcript_chars == len(transcript)
    assert summary.prompt_tokens == 111
    assert summary.completion_tokens == 22
    assert summary.latency_ms >= 0
    assert summary.vram_mb == 0.0
    assert len(summary.summary) >= 48
    assert len(summary.key_points) == 2

    payload = requests[0]
    assert payload["max_tokens"] == config.max_output_tokens
    assert payload["temperature"] == 0.2
    assert payload["stream"] is False
    assert payload["response_format"]["json_schema"]["schema"] == _SUMMARY_SCHEMA
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "reasoning_effort" not in payload
    parts = payload["messages"][1]["content"]
    assert parts[0]["type"] == "text" and "cats" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_summarize_sends_reasoning_effort_when_think_on(
    tmp_path: Path,
) -> None:
    config = EngineConfig(think=ThinkLevel.LOW)

    async with make_summarizer(
        config, ["qwen3.5-4b"], lambda p: completion(GOOD_JSON)
    ) as (summarizer, _engine, requests):
        summary = await summarizer.summarize(make_asset("Brief text.", []))

    assert requests[0]["reasoning_effort"] == "low"
    assert summary.think == "low"
    assert summary.frame_count == 0


async def test_summarize_map_reduce_decision(tmp_path: Path) -> None:
    config = EngineConfig(model_ctx=2048, max_output_tokens=256)
    transcript = " ".join(
        f"Sentence {i} talks about topic {i} in considerable detail today." for i in range(250)
    )
    assert len(transcript) > 12000  # ~3.3k tokens > the 1536-token window

    map_count = {"n": 0}

    def chat_fn(payload: dict[str, Any]) -> httpx.Response:
        schema = payload["response_format"]["json_schema"]["schema"]
        if schema["properties"].keys() == {"summary"}:
            map_count["n"] += 1
            return completion(
                json.dumps({"summary": f"Chunk summary {map_count['n']} with details."})
            )
        return completion(GOOD_JSON)

    async with make_summarizer(config, ["qwen3.5-4b"], chat_fn) as (
        summarizer,
        _engine,
        requests,
    ):
        summary = await summarizer.summarize(make_asset(transcript, []))

    map_requests = [
        p
        for p in requests
        if p["response_format"]["json_schema"]["schema"]["properties"].keys()
        == {"summary"}
    ]
    final_request = requests[-1]
    assert len(map_requests) >= 2
    assert final_request["response_format"]["json_schema"]["schema"] == _SUMMARY_SCHEMA
    final_user = final_request["messages"][1]["content"]
    assert isinstance(final_user, str)
    assert "Chunk summary" in final_user
    assert "image_url" not in json.dumps(final_request["messages"])
    assert summary.error is None
    assert summary.transcript_chars == len(transcript)
    assert summary.frame_count == 0


async def test_summarize_quality_gate_retry_then_success(tmp_path: Path) -> None:
    config = EngineConfig()
    responses = iter(["too short", GOOD_JSON])

    def chat_fn(_payload: dict[str, Any]) -> httpx.Response:
        return completion(next(responses), prompt_tokens=111, completion_tokens=22)

    async with make_summarizer(
        config, ["qwen3.5-4b"], chat_fn
    ) as (summarizer, _engine, requests):
        summary = await summarizer.summarize(make_asset("Some transcript.", []))

    assert summary.error is None
    assert summary.think == "low"  # retry used reasoning_effort=low
    assert summary.prompt_tokens == 222
    assert summary.completion_tokens == 44
    assert len(requests) == 2
    retry = requests[1]
    retry_user = retry["messages"][1]["content"]
    assert "rejected" in retry_user
    assert retry["reasoning_effort"] == "low"
    # schema is KEPT on retry (no Ollama-style think+schema restriction)
    assert retry["response_format"]["json_schema"]["schema"] == _SUMMARY_SCHEMA


async def test_summarize_gate_failure_after_retry_raises(tmp_path: Path) -> None:
    config = EngineConfig()

    async with make_summarizer(
        config, ["qwen3.5-4b"], lambda p: completion("nope")
    ) as (summarizer, _engine, requests):
        with pytest.raises(RuntimeError, match="summary quality gate failed"):
            await summarizer.summarize(make_asset("Some transcript.", []))
    assert len(requests) == 2  # initial pass + exactly ONE retry


async def test_resolve_model_not_served_raises_without_chat(
    tmp_path: Path,
) -> None:
    config = EngineConfig(model="absent-model")

    async with make_summarizer(
        config, [], lambda p: completion(GOOD_JSON)
    ) as (summarizer, _engine, requests):
        with pytest.raises(RuntimeError, match="not served by.*serve it or set VRA_MODEL"):
            await summarizer.summarize(make_asset("Some transcript.", []))
        with pytest.raises(RuntimeError, match="not served by"):
            await summarizer.prepare_models()
    assert requests == []  # never reached inference


async def test_prepare_models_reports_resolved_model(tmp_path: Path) -> None:
    config = EngineConfig(model="qwen3.5-4b")

    async with make_summarizer(
        config, ["Qwen/Qwen3.5-4B-AWQ"], lambda p: completion(GOOD_JSON)
    ) as (summarizer, _engine, _requests):
        assert await summarizer.prepare_models() == ["qwen3.5-4b"]


async def test_runtime_status_delegates_to_engine(tmp_path: Path) -> None:
    config = EngineConfig(model="qwen3.5-4b")

    async with make_summarizer(
        config, ["qwen3.5-4b", "Qwen/Qwen3-ASR-1.7B"], lambda p: completion(GOOD_JSON)
    ) as (summarizer, _engine, _requests):
        status = await summarizer.runtime_status()
    assert status["engine"]["reachable"] is True
    assert status["engine"]["model_present"] is True
    assert status["asr"]["model_present"] is True


async def test_summarize_close_shuts_engine_down(tmp_path: Path) -> None:
    async with make_summarizer(
        EngineConfig(), ["qwen3.5-4b"], lambda p: completion(GOOD_JSON)
    ) as (summarizer, engine, _requests):
        await summarizer.close()
        assert engine._client.is_closed


async def test_summarize_infra_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(engine_module, "_sleep", fake_sleep)
    config = EngineConfig()

    def always_503(payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(503, text="down")

    async with make_summarizer(config, ["qwen3.5-4b"], always_503) as (
        summarizer,
        _engine,
        _requests,
    ):
        with pytest.raises(RuntimeError, match="down"):
            await summarizer.summarize(make_asset("Some transcript.", []))


async def test_summarize_skips_unreadable_frames(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    await asyncio.to_thread(good.write_bytes, minimal_jpeg(64, 32))
    bad = tmp_path / "bad.jpg"
    await asyncio.to_thread(bad.write_bytes, b"\x89PNG not a jpeg")
    config = EngineConfig()

    async with make_summarizer(
        config, ["qwen3.5-4b"], lambda p: completion(GOOD_JSON)
    ) as (summarizer, _engine, requests):
        summary = await summarizer.summarize(
            make_asset("Some transcript.", [bad, good])
        )
    assert summary.frame_count == 1
    payload = requests[0]
    parts = payload["messages"][1]["content"]
    assert len(parts) == 2  # text + exactly one image part
