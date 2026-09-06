"""GUI module tests: template rendering, setup payload, step-view builders."""

from __future__ import annotations

import json

from vra.api.gui import (
    build_diagnostics_view,
    build_runtime_view,
    render_setup_page,
    setup_payload,
)
from vra.config import Config


def _config(**overrides: object) -> Config:
    return Config(rss_link="http://127.0.0.1:8080/rss", **overrides)  # type: ignore[arg-type]


def test_setup_payload_contains_config_surface_and_flags() -> None:
    payload = setup_payload(_config(api_key="hush", engine_api_key="eh"))
    assert payload["model"] == "qwen3.5-9b"
    assert payload["bind_address"] == "127.0.0.1:8080"
    assert payload["api_key_required"] is True
    assert payload["engine_api_key_required"] is True
    assert "api_key" not in payload and "engine_api_key" not in payload
    assert set(payload["quick_commands"]) == {"doctor", "list", "poll_now", "serve"}


def test_render_setup_page_replaces_placeholders() -> None:
    html = render_setup_page(_config(), root_path="")
    assert "__BASE_HREF__" not in html
    assert "__DEFAULTS_JSON__" not in html
    marker = '<script id="setup-defaults" type="application/json">'
    idx = html.find(marker)
    assert idx != -1
    start = idx + len(marker)
    end = html.find("</script>", start)
    decoded = json.loads(html[start:end].replace("<\\/", "</"))
    assert decoded["model"] == "qwen3.5-9b"


def test_render_setup_page_uses_root_path_for_base_href() -> None:
    html = render_setup_page(_config(), root_path="/vra")
    assert '<base href="/vra/" />' in html


def test_build_diagnostics_view_ready_and_blocked() -> None:
    good = {
        "platform": {"python_version": "3.13.0", "python_executable": "/py"},
        "dependencies": {
            "ffmpeg": {"available": True, "resolved": "/usr/bin/ffmpeg"},
            "ffprobe": {"available": True, "resolved": "/usr/bin/ffprobe"},
            "yt_dlp": {"available": True, "command": "yt-dlp"},
            "engine": {
                "base_url": "http://e/v1",
                "reachable": True,
                "models": ["qwen3.5-9b"],
                "model_present": True,
            },
            "asr": {"base_url": "http://e/v1", "model_present": True},
        },
        "ready": True,
    }
    view = build_diagnostics_view(good, asr_enabled=True)
    assert view["state"] == "ready"
    assert view["blockers"] == []

    blocked = {
        "platform": {},
        "dependencies": {
            "ffmpeg": {"available": False},
            "engine": {"base_url": "http://e/v1", "model_present": False},
        },
        "ready": False,
    }
    view = build_diagnostics_view(blocked, asr_enabled=True)
    assert view["state"] == "blocked"
    ids = [check["id"] for check in view["checks"]]
    assert {"python", "ffmpeg", "engine", "asr"} <= set(ids)


def test_build_runtime_view_states() -> None:
    ready = build_runtime_view(
        {"reachable": True, "model_present": True, "model": "qwen3.5-9b"},
        {"enabled": True, "reachable": True, "model_present": True, "model": "Qwen3-ASR-1.7B"},
    )
    assert ready["state"] == "ready"
    assert ready["missing_models"] == []

    blocked = build_runtime_view(
        {"reachable": True, "model_present": False, "model": "qwen3.5-9b"},
        {"enabled": True, "reachable": True, "model_present": False, "model": "Qwen3-ASR-1.7B"},
    )
    assert blocked["state"] == "blocked"
    assert blocked["missing_models"] == ["qwen3.5-9b", "Qwen3-ASR-1.7B"]
