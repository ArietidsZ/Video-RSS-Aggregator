from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from fastapi.testclient import TestClient

from adapter_api import create_app
from core_config import Config
from service_summarize import SummaryResult


@dataclass
class _ProcessReport:
    source_url: str
    title: str | None
    transcript_chars: int
    frame_count: int
    summary: SummaryResult


class _DummyPipeline:
    async def ingest_feed(self, *_args, **_kwargs):
        class _Report:
            feed_title = "dummy"
            item_count = 1
            processed_count = 0

        return _Report()

    async def process_source(self, source_url: str, title: str | None = None):
        return _ProcessReport(
            source_url=source_url,
            title=title,
            transcript_chars=42,
            frame_count=3,
            summary=SummaryResult(
                summary="ok",
                key_points=["a", "b", "c", "d"],
                visual_highlights=[],
                model_used="qwen3.5:2b-q4_K_M",
                vram_mb=2048.0,
                transcript_chars=42,
                frame_count=3,
            ),
        )

    async def rss_feed(self, _limit: int = 20) -> str:
        return "<rss></rss>"

    async def runtime_status(self):
        return {"ok": True}

    async def bootstrap_models(self):
        return {"models_prepared": ["qwen3.5:2b-q4_K_M"]}


def test_gui_and_setup_routes() -> None:
    app = create_app(cast(Any, _DummyPipeline()), Config())
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert "Video RSS Aggregator Installation" in home.text

    setup = client.get("/setup/config")
    assert setup.status_code == 200
    payload = setup.json()
    assert payload["ollama_base_url"] == "http://127.0.0.1:11434"

    diagnostics = client.get("/setup/diagnostics")
    assert diagnostics.status_code == 200
    diag_payload = diagnostics.json()
    assert "platform" in diag_payload
    assert "dependencies" in diag_payload
    assert "ready" in diag_payload
    assert "ffmpeg" in diag_payload["dependencies"]
    assert "yt_dlp" in diag_payload["dependencies"]

    bootstrap = client.post("/setup/bootstrap")
    assert bootstrap.status_code == 200
    assert bootstrap.json()["models_prepared"] == ["qwen3.5:2b-q4_K_M"]


def test_runtime_requires_api_key_when_enabled() -> None:
    app = create_app(cast(Any, _DummyPipeline()), Config(api_key="secret"))
    client = TestClient(app)

    unauthorized = client.get("/runtime")
    assert unauthorized.status_code == 401

    authorized = client.get("/runtime", headers={"X-API-Key": "secret"})
    assert authorized.status_code == 200
    assert authorized.json() == {"ok": True}
