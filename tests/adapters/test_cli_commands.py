from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from click.testing import CliRunner

import cli as cli_module
from core_config import Config
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
    passthrough = RecordingUseCase(None)

    use_cases = {
        "get_runtime_status": runtime_status,
        "bootstrap_runtime": bootstrap_runtime,
        "ingest_feed": passthrough,
        "process_source": process_source,
        "render_rss_feed": passthrough,
    }

    closed = {"value": False}

    async def _close() -> None:
        closed["value"] = True

    return (
        AppRuntime(
            config=Config(),
            use_cases=AppUseCases(**use_cases),
            close=_close,
        ),
        {**use_cases, "closed": closed},
    )


def test_bootstrap_uses_runtime_use_cases_and_keeps_json_shape(monkeypatch) -> None:
    runtime, use_cases = _build_runtime()

    async def fake_build_runtime(config: Config | None = None) -> AppRuntime:
        return runtime

    class LegacySummarizationEngine:
        def __init__(self, config: Config) -> None:
            raise AssertionError("legacy summarizer wiring used")

    monkeypatch.setattr(cli_module, "build_runtime", fake_build_runtime, raising=False)
    monkeypatch.setattr(cli_module, "SummarizationEngine", LegacySummarizationEngine)

    result = CliRunner().invoke(cli_module.cli, ["bootstrap"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
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
    assert use_cases["bootstrap_runtime"].calls == [()]
    assert use_cases["get_runtime_status"].calls == [()]
    assert use_cases["closed"]["value"] is True


def test_status_uses_runtime_use_case_and_keeps_json_shape(monkeypatch) -> None:
    runtime, use_cases = _build_runtime()

    async def fake_build_runtime(config: Config | None = None) -> AppRuntime:
        return runtime

    class LegacySummarizationEngine:
        def __init__(self, config: Config) -> None:
            raise AssertionError("legacy summarizer wiring used")

    monkeypatch.setattr(cli_module, "build_runtime", fake_build_runtime, raising=False)
    monkeypatch.setattr(cli_module, "SummarizationEngine", LegacySummarizationEngine)

    result = CliRunner().invoke(cli_module.cli, ["status"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "ollama_version": "0.6.0",
        "local_models": {"qwen": {"size": 1}},
        "reachable": True,
        "database_path": ".data/runtime.db",
        "storage_dir": ".data",
        "models": ["qwen", "qwen:min"],
    }
    assert use_cases["get_runtime_status"].calls == [()]
    assert use_cases["closed"]["value"] is True


def test_verify_uses_runtime_process_source_and_keeps_metrics_shape(
    monkeypatch,
) -> None:
    runtime, use_cases = _build_runtime()
    monotonic_values = iter([100.0, 100.25])

    async def fake_build_runtime(config: Config | None = None) -> AppRuntime:
        return runtime

    monkeypatch.setattr(cli_module, "build_runtime", fake_build_runtime, raising=False)
    monkeypatch.setattr(cli_module, "monotonic", lambda: next(monotonic_values))

    result = CliRunner().invoke(
        cli_module.cli,
        [
            "verify",
            "--source",
            "https://example.com/watch?v=1",
            "--title",
            "Example Title",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "source_url": "https://example.com/watch?v=1",
        "title": "Example Title",
        "transcript_chars": 15,
        "frame_count": 2,
        "model_used": "qwen",
        "vram_mb": 512.0,
        "error": "model degraded",
        "summary_chars": 15,
        "key_points": 2,
        "total_ms": 250,
    }
    assert use_cases["process_source"].calls == [
        ("https://example.com/watch?v=1", "Example Title")
    ]
    assert use_cases["closed"]["value"] is True


def test_serve_builds_runtime_then_passes_it_to_app_factory(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_create_app(
        runtime: AppRuntime | None = None,
        config: Config | None = None,
    ) -> object:
        calls["create_app_runtime"] = runtime
        calls["create_app_config"] = config
        return object()

    class FakeUvicornConfig:
        def __init__(self, app: object, host: str, port: int, log_level: str) -> None:
            calls["uvicorn_app"] = app
            calls["uvicorn_host"] = host
            calls["uvicorn_port"] = port
            calls["uvicorn_log_level"] = log_level

    class FakeUvicornServer:
        def __init__(self, config: FakeUvicornConfig) -> None:
            calls["uvicorn_config"] = config

        async def serve(self) -> None:
            calls["served"] = True

    monkeypatch.setattr("adapter_api.create_app", fake_create_app)
    monkeypatch.setattr(cli_module.uvicorn, "Config", FakeUvicornConfig)
    monkeypatch.setattr(cli_module.uvicorn, "Server", FakeUvicornServer)

    result = CliRunner().invoke(cli_module.cli, ["serve", "--bind", "127.0.0.1:8080"])

    assert result.exit_code == 0, result.output
    assert calls["create_app_runtime"] is None
    assert calls["create_app_config"] == Config(
        database_path=str(Path(".data") / "vra.db")
    )
    assert calls["uvicorn_host"] == "127.0.0.1"
    assert calls["uvicorn_port"] == 8080
    assert calls["uvicorn_log_level"] == "info"
    assert calls["served"] is True
