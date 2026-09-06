"""Unit tests for OpenAiAsrTranscriber — httpx MockTransport, no network."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from vra.config import AsrMode, Config
from vra.platform.asr import OpenAiAsrTranscriber

BASE = "http://asr.test/v1"


def make_config(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "effective_asr_base_url": BASE,
        "asr_model": "Qwen3-ASR-1.7B",
        "timeout_infer_s": 5,
        "asr": AsrMode.AUTO,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def transport_with(
    models_payload: dict[str, object] | None,
    *,
    transcribe_payload: dict[str, object] | None = None,
    transcribe_status: int = 200,
    calls: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if request.url.path.endswith("/models"):
            if models_payload is None:
                raise httpx.ConnectError("server down")
            return httpx.Response(200, json=models_payload)
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(
                transcribe_status,
                json=transcribe_payload if transcribe_status < 400 else {"error": "x"},
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


async def test_available_true_when_asr_model_listed() -> None:
    calls: list[httpx.Request] = []
    client = httpx.AsyncClient(
        transport=transport_with(
            {"data": [{"id": "other"}, {"id": "Qwen3-ASR-1.7B"}]}, calls=calls
        )
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    assert await transcriber.available() is True
    assert await transcriber.available() is True  # served from the 60s cache
    assert len(calls) == 1
    await client.aclose()


async def test_available_case_insensitive_match() -> None:
    client = httpx.AsyncClient(
        transport=transport_with({"data": [{"id": "qwen3-asr-1.7b-gguf"}]})
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    assert await transcriber.available() is True
    await client.aclose()


async def test_available_false_without_asr_model() -> None:
    client = httpx.AsyncClient(
        transport=transport_with({"data": [{"id": "gpt-4o-audio"}]})
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    assert await transcriber.available() is False
    await client.aclose()


async def test_available_false_on_connection_error() -> None:
    client = httpx.AsyncClient(transport=transport_with(None))
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    assert await transcriber.available() is False
    await client.aclose()


async def test_available_false_on_bad_payload() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="garbage"))
    client = httpx.AsyncClient(transport=transport)
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    assert await transcriber.available() is False
    await client.aclose()


async def test_transcribe_openai_verbose_shape(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake-audio-bytes")
    payload = {
        "text": " hi there ",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": " hi "},
            {"start": 1.5, "end": 2.5, "text": "there "},
        ],
    }
    calls: list[httpx.Request] = []
    client = httpx.AsyncClient(
        transport=transport_with(
            {"data": [{"id": "Qwen3-ASR-1.7B"}]},
            transcribe_payload=payload,
            calls=calls,
        )
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    transcript = await transcriber.transcribe(str(media))

    assert transcript.source == "asr"
    assert transcript.lang == "en"
    assert [seg.text for seg in transcript.segments] == ["hi", "there"]
    assert transcript.segments[0].start_s == 0.0
    assert transcript.segments[1].end_s == 2.5
    assert transcript.text == "hi there"

    request = calls[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/audio/transcriptions")
    body = (await request.aread()).decode("latin-1")
    assert 'name="model"' in body
    assert "Qwen3-ASR-1.7B" in body
    assert 'name="file"' in body
    assert 'filename="clip.mp4"' in body
    assert "video/mp4" in body
    await client.aclose()


async def test_transcribe_plain_text_shape(tmp_path: Path) -> None:
    media = tmp_path / "clip.mkv"
    media.write_bytes(b"data")
    client = httpx.AsyncClient(
        transport=transport_with(
            {"data": []}, transcribe_payload={"text": "plain words"}
        )
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    transcript = await transcriber.transcribe(str(media))
    assert transcript.lang is None
    (seg,) = transcript.segments
    assert (seg.start_s, seg.end_s, seg.text) == (0.0, 0.0, "plain words")
    assert transcript.text == "plain words"
    await client.aclose()


async def test_transcribe_asr_text_quirk(tmp_path: Path) -> None:
    """llama.cpp #26749: {"asr_text": ...} instead of {"text": ...}."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"data")
    client = httpx.AsyncClient(
        transport=transport_with(
            {"data": []}, transcribe_payload={"asr_text": "bonjour monde"}
        )
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    transcript = await transcriber.transcribe(str(media))
    assert transcript.text == "bonjour monde"
    assert transcript.source == "asr"
    await client.aclose()


async def test_transcribe_segments_without_timestamps(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"data")
    client = httpx.AsyncClient(
        transport=transport_with(
            {"data": []},
            transcribe_payload={
                "text": "full text",
                "segments": [{"text": "full"}, {"text": ""}, "junk"],
            },
        )
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    transcript = await transcriber.transcribe(str(media))
    assert [seg.text for seg in transcript.segments] == ["full"]
    await client.aclose()


async def test_transcribe_empty_text_raises(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"data")
    client = httpx.AsyncClient(
        transport=transport_with({"data": []}, transcribe_payload={"text": "   "})
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    with pytest.raises(RuntimeError, match="empty transcript"):
        await transcriber.transcribe(str(media))
    await client.aclose()


async def test_transcribe_http_error_raises(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"data")
    client = httpx.AsyncClient(
        transport=transport_with(
            {"data": []}, transcribe_payload={}, transcribe_status=500
        )
    )
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    with pytest.raises(RuntimeError, match="500"):
        await transcriber.transcribe(str(media))
    await client.aclose()


async def test_transcribe_non_json_raises(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"data")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<html>not json</html>")
    )
    client = httpx.AsyncClient(transport=transport)
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    with pytest.raises(RuntimeError, match="non-JSON"):
        await transcriber.transcribe(str(media))
    await client.aclose()


async def test_config_fallback_base_url(tmp_path: Path) -> None:
    """Without effective_asr_base_url (pre-merge Config), the vLLM default is used."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"data")
    calls: list[httpx.Request] = []
    transport = httpx.MockTransport(
        lambda request: (
            calls.append(request)
            or httpx.Response(200, json={"text": "ok", "language": "de"})
        )
    )
    client = httpx.AsyncClient(transport=transport)
    transcriber = OpenAiAsrTranscriber(Config(), client=client)
    transcript = await transcriber.transcribe(str(media))
    assert str(calls[0].url).startswith("http://127.0.0.1:8000/v1/")
    assert transcript.lang == "de"
    await client.aclose()


async def test_owned_client_closed_via_close(tmp_path: Path) -> None:
    transcriber = OpenAiAsrTranscriber(make_config())
    client = transcriber._http()
    assert not client.is_closed
    await transcriber.close()
    assert client.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        transcriber._http()


async def test_injected_client_not_closed() -> None:
    client = httpx.AsyncClient(transport=transport_with({"data": []}))
    transcriber = OpenAiAsrTranscriber(make_config(), client=client)
    await transcriber.close()
    assert not client.is_closed
    await client.aclose()


def test_base_url_strips_trailing_slash() -> None:
    from vra.platform.asr import _config_base_url

    assert _config_base_url(make_config()) == BASE
    assert _config_base_url(
        make_config(effective_asr_base_url="http://x/v1/")
    ) == "http://x/v1"
    assert _config_base_url(object()) == "http://127.0.0.1:8000/v1"
