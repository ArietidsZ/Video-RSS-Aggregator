"""Setup studio GUI. OWNER: agent-D.

Port of video_rss_aggregator/gui.py + setup_view_models.py adapted to the v2
engine posture (single OpenAI-compatible engine + Qwen3-ASR; no Ollama).

- render_setup_page(config, root_path): reads templates/setup.html, injects
  setup_payload(config) as JSON plus the base href, and returns the HTML.
  Assets are served from /static (vra/static).
- build_diagnostics_view / build_runtime_view: the setup-studio "step view"
  models the JS consumes (state: ready|blocked, per-check fix hints).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from functools import lru_cache
from importlib.resources import files
from typing import Any

from vra.config import Config

_OK = "complete"
_BLOCKED = "blocked"


def setup_payload(config: Config) -> dict[str, Any]:
    """Setup-studio payload: full config surface minus secret values."""
    payload: dict[str, Any] = {}
    for key, value in asdict(config).items():
        if key in {"api_key", "engine_api_key", "raw"}:
            continue
        payload[key] = value.value if hasattr(value, "value") else value
    payload["bind_address"] = config.bind_address
    payload["api_key_required"] = config.api_key is not None
    payload["engine_api_key_required"] = config.engine_api_key is not None
    payload["quick_commands"] = {
        "doctor": "vra doctor",
        "list": "vra list",
        "poll_now": "vra poll-now",
        "serve": f"vra serve --bind {config.bind_address}",
    }
    return payload


@lru_cache(maxsize=1)
def _template_text() -> str:
    """Template text (cached; called via asyncio.to_thread from endpoints)."""
    return (
        files("vra")
        .joinpath("templates", "setup.html")
        .read_text(encoding="utf-8")
    )


def render_setup_page(config: Config, root_path: str = "") -> str:
    """Render the setup studio HTML with the config payload injected."""
    defaults_json = json.dumps(setup_payload(config)).replace("</", "<\\/")
    base_href = f"{root_path.rstrip('/')}/" if root_path else "/"
    return (
        _template_text()
        .replace("__BASE_HREF__", base_href)
        .replace("__DEFAULTS_JSON__", defaults_json)
    )


def _dependency_check(
    check_id: str,
    label: str,
    dependency: object,
    *,
    fix: str,
) -> dict[str, Any]:
    payload = dependency if isinstance(dependency, dict) else {}
    detail = payload.get("resolved") or payload.get("command")
    available = bool(payload.get("available"))
    return {
        "id": check_id,
        "label": label,
        "state": _OK if available else _BLOCKED,
        "detail": str(detail) if detail else "Not detected",
        "fix": None if available else fix,
    }


def build_diagnostics_view(report: dict[str, Any], *, asr_enabled: bool) -> dict[str, Any]:
    """Setup-studio view over the /setup/diagnostics report (v2 checks)."""
    platform = report.get("platform") or {}
    dependencies = report.get("dependencies") or {}

    python_version = str(platform.get("python_version") or "unknown")
    major_ok = sys.version_info >= (3, 11)

    checks: list[dict[str, Any]] = [
        {
            "id": "python",
            "label": "Python 3.11+",
            "state": _OK if major_ok else _BLOCKED,
            "detail": f"{python_version} via {platform.get('python_executable', 'unknown')}",
            "fix": None if major_ok else "Install Python 3.11+",
        },
        _dependency_check(
            "ffmpeg", "FFmpeg", dependencies.get("ffmpeg"), fix="Install FFmpeg and add it to PATH"
        ),
        _dependency_check(
            "ffprobe", "FFprobe", dependencies.get("ffprobe"), fix="Install FFprobe and add it to PATH"
        ),
        _dependency_check(
            "yt_dlp", "yt-dlp", dependencies.get("yt_dlp"), fix="Install yt-dlp (pip install yt-dlp)"
        ),
    ]

    engine = dependencies.get("engine") or {}
    engine_ready = bool(engine.get("model_present"))
    engine_detail = (
        f"{engine.get('model')} served at {engine.get('base_url')}"
        if engine_ready
        else str(engine.get("error") or f"model {engine.get('model')} not served at {engine.get('base_url')}")
    )
    checks.append(
        {
            "id": "engine",
            "label": "Engine model",
            "state": _OK if engine_ready else _BLOCKED,
            "detail": engine_detail,
            "fix": None
            if engine_ready
            else "Serve the configured VRA_MODEL at VRA_ENGINE_BASE_URL",
        }
    )

    asr = dependencies.get("asr") or {}
    asr_ready = bool(asr.get("model_present")) or not asr_enabled
    asr_detail = (
        "ASR disabled (VRA_ASR=off)"
        if not asr_enabled
        else str(
            asr.get("model_present") and f"{asr.get('model')} served at {asr.get('base_url')}"
            or asr.get("error")
            or f"model {asr.get('model')} not served at {asr.get('base_url')}"
        )
    )
    checks.append(
        {
            "id": "asr",
            "label": "ASR model",
            "state": _OK if asr_ready else _BLOCKED,
            "detail": asr_detail,
            "fix": None
            if asr_ready
            else "Serve the configured VRA_ASR_MODEL or set VRA_ASR=off",
        }
    )

    blockers = [check["fix"] for check in checks if check["state"] == _BLOCKED and check["fix"]]
    state = "ready" if not blockers and bool(report.get("ready")) else _BLOCKED
    return {
        "state": state,
        "blockers": blockers,
        "checks": checks,
        "next_action": "Continue to configuration"
        if state == "ready"
        else "Resolve the blocked checks and run diagnostics again",
    }


def build_runtime_view(engine: dict[str, Any], asr: dict[str, Any]) -> dict[str, Any]:
    """Runtime-step view: engine reachable + configured model served."""
    reachable = bool(engine.get("reachable"))
    model_present = bool(engine.get("model_present"))
    asr_enabled = bool(asr.get("enabled", True))
    asr_present = bool(asr.get("model_present")) or not asr_enabled

    missing: list[str] = []
    if reachable and not model_present:
        missing.append(str(engine.get("model") or "engine model"))
    if asr_enabled and reachable and not asr_present:
        missing.append(str(asr.get("model") or "asr model"))

    state = "ready" if reachable and model_present and asr_present else _BLOCKED
    if not reachable:
        next_action = "Start the server at VRA_ENGINE_BASE_URL"
    elif missing:
        next_action = f"Serve the configured model(s): {', '.join(missing)}"
    else:
        next_action = "Run the first processing test"
    return {
        "state": state,
        "missing_models": missing,
        "next_action": next_action,
    }
