from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from adapter_api import create_app
from core_config import Config
from video_rss_aggregator.bootstrap import AppRuntime, AppUseCases


@dataclass
class _AsyncValue:
    value: Any

    async def execute(self, *_args, **_kwargs) -> Any:
        return self.value


def _build_runtime(config: Config) -> AppRuntime:
    runtime_status = {
        "ollama_version": "0.6.0",
        "local_models": {"qwen3.5:2b-q4_K_M": {}},
        "reachable": True,
        "database_path": config.database_path,
        "storage_dir": config.storage_dir,
        "models": list(config.model_priority),
    }
    return AppRuntime(
        config=config,
        use_cases=AppUseCases(
            get_runtime_status=_AsyncValue(runtime_status),
            bootstrap_runtime=_AsyncValue({"models": ["qwen3.5:2b-q4_K_M"]}),
            ingest_feed=cast(Any, _AsyncValue(None)),
            process_source=cast(Any, _AsyncValue(None)),
            render_rss_feed=_AsyncValue("<rss></rss>"),
        ),
        close=lambda: None,
    )


def _is_javascript_content_type(content_type: str) -> bool:
    return content_type.startswith(("text/javascript", "application/javascript"))


def test_gui_and_setup_routes() -> None:
    config = Config()
    app = create_app(_build_runtime(config))
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert "Video RSS Aggregator Installation" in home.text
    assert '<base href="/" />' in home.text
    assert 'href="static/setup.css"' in home.text
    assert 'src="static/setup.js"' in home.text
    assert "fonts.googleapis.com" not in home.text
    assert "fonts.gstatic.com" not in home.text

    css = client.get("/static/setup.css")
    assert css.status_code == 200
    assert ".shell" in css.text

    js = client.get("/static/setup.js")
    assert js.status_code == 200
    assert _is_javascript_content_type(js.headers["content-type"])

    setup_state = client.get("/static/setup_state.js")
    assert setup_state.status_code == 200
    assert _is_javascript_content_type(setup_state.headers["content-type"])

    setup = client.get("/setup/config")
    assert setup.status_code == 200
    payload = setup.json()
    assert payload["ollama_base_url"] == "http://127.0.0.1:11434"
    assert payload["model_primary"] == "qwen3.5:4b-q4_K_M"
    assert payload["model_fallback"] == "qwen3.5:2b-q4_K_M"
    assert payload["model_min"] == "qwen3.5:0.8b-q8_0"
    assert payload["context_tokens"] == 3072
    assert payload["max_output_tokens"] == 768
    assert payload["rss_title"] == "Video RSS Aggregator"
    assert payload["rss_description"] == "Video summaries"
    assert payload["auto_pull_models"] is True
    assert "api_key" not in payload

    diagnostics = client.get("/setup/diagnostics")
    assert diagnostics.status_code == 200
    diag_payload = diagnostics.json()
    setup_view = diag_payload["setup_view"]
    assert set(setup_view) >= {"state", "checks", "blockers", "next_action"}
    assert setup_view["state"] in {"ready", "blocked"}
    assert isinstance(setup_view["checks"], list)
    assert {check["id"] for check in setup_view["checks"]} >= {
        "python",
        "ffmpeg",
        "ffprobe",
        "yt_dlp",
        "ollama",
    }
    assert "platform" in diag_payload
    assert "dependencies" in diag_payload
    assert "ready" in diag_payload
    assert "ffmpeg" in diag_payload["dependencies"]
    assert "yt_dlp" in diag_payload["dependencies"]

    bootstrap = client.post("/setup/bootstrap")
    assert bootstrap.status_code == 200
    assert bootstrap.json()["models_prepared"] == ["qwen3.5:2b-q4_K_M"]


def test_runtime_requires_api_key_when_enabled() -> None:
    app = create_app(_build_runtime(Config(api_key="secret")))
    client = TestClient(app)

    unauthorized = client.get("/runtime")
    assert unauthorized.status_code == 401

    authorized = client.get("/runtime", headers={"X-API-Key": "secret"})
    assert authorized.status_code == 200
    assert authorized.json()["setup_view"]["state"] == "blocked"


def test_setup_config_omits_api_key_and_uses_current_bind_in_commands() -> None:
    config = Config(api_key="secret", bind_host="0.0.0.0", bind_port=9090)
    app = create_app(_build_runtime(config))
    client = TestClient(app)

    payload = client.get("/setup/config").json()

    assert "api_key" not in payload
    assert payload["quick_commands"] == {
        "bootstrap": "python -m vra bootstrap",
        "status": "python -m vra status",
        "serve": "python -m vra serve --bind 0.0.0.0:9090",
    }


def test_setup_home_uses_root_path_aware_base_href() -> None:
    app = create_app(_build_runtime(Config()))
    client = TestClient(app, root_path="/vra")

    home = client.get("/")

    assert home.status_code == 200
    assert '<base href="/vra/" />' in home.text


def test_create_app_rejects_non_runtime_objects() -> None:
    with pytest.raises(TypeError, match="AppRuntime"):
        create_app(cast(Any, object()), Config())
