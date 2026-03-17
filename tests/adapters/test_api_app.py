from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from core_config import Config
from video_rss_aggregator.api import create_app
from video_rss_aggregator.application.use_cases.ingest_feed import IngestReport
from video_rss_aggregator.bootstrap import AppRuntime, AppUseCases
from video_rss_aggregator.domain.models import PreparedMedia, SummaryResult
from video_rss_aggregator.domain.outcomes import PartialSuccess


@dataclass
class RecordingUseCase:
    result: Any
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    async def execute(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.result


def _build_runtime() -> tuple[AppRuntime, dict[str, RecordingUseCase]]:
    runtime_status = RecordingUseCase(
        {
            "ollama_version": "0.6.0",
            "local_models": {"qwen": {"size": 1}},
            "reachable": True,
            "database_path": ".data/runtime.db",
            "storage_dir": ".data",
            "models": ["qwen", "qwen:min"],
        }
    )
    bootstrap_runtime = RecordingUseCase({"models": ["qwen"]})
    ingest_feed = RecordingUseCase(
        IngestReport(feed_title="Example Feed", item_count=2, processed_count=1)
    )
    process_source = RecordingUseCase(
        PartialSuccess(
            media=PreparedMedia(
                source_url="https://example.com/watch?v=1",
                title="Example Title",
                transcript="transcript text",
                media_path="/tmp/video.mp4",
                frame_paths=("/tmp/frame-1.jpg", "/tmp/frame-2.jpg"),
            ),
            reason="model degraded",
            summary=SummaryResult(
                summary="Compact summary",
                key_points=("one", "two"),
                visual_highlights=("frame",),
                model_used="qwen",
                vram_mb=512.0,
                transcript_chars=15,
                frame_count=2,
                error="model degraded",
            ),
        )
    )
    render_rss_feed = RecordingUseCase("<rss>feed</rss>")

    use_cases = {
        "get_runtime_status": runtime_status,
        "bootstrap_runtime": bootstrap_runtime,
        "ingest_feed": ingest_feed,
        "process_source": process_source,
        "render_rss_feed": render_rss_feed,
    }

    return (
        AppRuntime(
            config=Config(),
            use_cases=AppUseCases(**use_cases),
            close=lambda: None,
        ),
        use_cases,
    )


def test_routes_delegate_to_runtime_use_cases_and_keep_http_shapes() -> None:
    runtime, use_cases = _build_runtime()
    client = TestClient(create_app(runtime))

    ingest = client.post(
        "/ingest",
        json={
            "feed_url": "https://example.com/feed.xml",
            "process": True,
            "max_items": 3,
        },
    )
    process = client.post(
        "/process",
        json={
            "source_url": "https://example.com/watch?v=1",
            "title": "Example Title",
        },
    )
    rss = client.get("/rss?limit=5")
    runtime_response = client.get("/runtime")
    bootstrap = client.post("/setup/bootstrap")
    runtime_payload = runtime_response.json()
    bootstrap_payload = bootstrap.json()

    assert ingest.status_code == 200
    assert ingest.json() == {
        "feed_title": "Example Feed",
        "item_count": 2,
        "processed_count": 1,
    }
    assert process.status_code == 200
    assert process.json() == {
        "source_url": "https://example.com/watch?v=1",
        "title": "Example Title",
        "transcript_chars": 15,
        "frame_count": 2,
        "summary": {
            "summary": "Compact summary",
            "key_points": ["one", "two"],
            "visual_highlights": ["frame"],
            "model_used": "qwen",
            "vram_mb": 512.0,
            "error": "model degraded",
        },
    }
    assert rss.status_code == 200
    assert rss.text == "<rss>feed</rss>"
    assert runtime_response.status_code == 200
    assert runtime_payload == {
        "ollama_version": "0.6.0",
        "local_models": {"qwen": {"size": 1}},
        "reachable": True,
        "database_path": ".data/runtime.db",
        "storage_dir": ".data",
        "models": ["qwen", "qwen:min"],
        "setup_view": {
            "state": "blocked",
            "missing_models": ["qwen:min"],
            "next_action": "Bootstrap required models",
        },
    }
    assert runtime_payload["setup_view"] == {
        "state": "blocked",
        "missing_models": ["qwen:min"],
        "next_action": "Bootstrap required models",
    }
    assert bootstrap.status_code == 200
    assert bootstrap_payload == {
        "models_prepared": ["qwen"],
        "runtime": {
            "ollama_version": "0.6.0",
            "local_models": {"qwen": {"size": 1}},
            "reachable": True,
            "database_path": ".data/runtime.db",
            "storage_dir": ".data",
            "models": ["qwen", "qwen:min"],
        },
    }

    assert use_cases["ingest_feed"].calls == [("https://example.com/feed.xml", True, 3)]
    assert use_cases["process_source"].calls == [
        ("https://example.com/watch?v=1", "Example Title")
    ]
    assert use_cases["render_rss_feed"].calls == [(5,)]
    assert use_cases["get_runtime_status"].calls == [(), ()]
    assert use_cases["bootstrap_runtime"].calls == [()]
