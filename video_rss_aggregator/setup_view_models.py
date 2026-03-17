from __future__ import annotations


def build_diagnostics_view(report: dict[str, object]) -> dict[str, object]:
    platform = _as_dict(report.get("platform"))
    dependencies = _as_dict(report.get("dependencies"))

    python_version = str(platform.get("python_version") or "unknown")
    python_executable = str(platform.get("python_executable") or "unknown")
    python_ready = _is_supported_python(python_version)

    checks = [
        {
            "id": "python",
            "label": "Python 3.11+",
            "state": "complete" if python_ready else "blocked",
            "detail": f"{python_version} via {python_executable}",
            "fix": None if python_ready else "Install Python 3.11+",
        },
        _dependency_check(
            "ffmpeg",
            "FFmpeg",
            dependencies.get("ffmpeg"),
            fix="Install FFmpeg and add it to PATH",
        ),
        _dependency_check(
            "ffprobe",
            "FFprobe",
            dependencies.get("ffprobe"),
            fix="Install FFprobe and add it to PATH",
        ),
        _dependency_check(
            "yt_dlp",
            "yt-dlp",
            dependencies.get("yt_dlp"),
            fix="Install yt-dlp and add it to PATH",
        ),
        _ollama_check(dependencies.get("ollama")),
    ]

    blockers = [
        check["fix"] for check in checks if check["state"] == "blocked" and check["fix"]
    ]
    state = "ready" if not blockers and bool(report.get("ready")) else "blocked"

    return {
        "state": state,
        "blockers": blockers,
        "checks": checks,
        "next_action": "Continue to configuration"
        if state == "ready"
        else "Resolve the blocked checks and run diagnostics again",
    }


def build_runtime_view(runtime: dict[str, object]) -> dict[str, object]:
    local_models = _local_model_names(runtime.get("local_models"))
    models = [str(model) for model in _as_list(runtime.get("models"))]
    missing_models = [model for model in models if model not in local_models]

    reachable = bool(runtime["reachable"]) if "reachable" in runtime else True
    state = "ready" if reachable and not missing_models else "blocked"
    if not reachable:
        next_action = "Check runtime"
    elif missing_models:
        next_action = "Bootstrap required models"
    else:
        next_action = "Run the first processing test"

    return {
        "state": state,
        "missing_models": missing_models,
        "next_action": next_action,
    }


def _dependency_check(
    check_id: str,
    label: str,
    dependency: object,
    *,
    fix: str,
) -> dict[str, object]:
    payload = _as_dict(dependency)
    command = payload.get("command")
    available = bool(payload.get("available"))

    return {
        "id": check_id,
        "label": label,
        "state": "complete" if available else "blocked",
        "detail": str(command) if command else "Not detected",
        "fix": None if available else fix,
    }


def _ollama_check(dependency: object) -> dict[str, object]:
    payload = _as_dict(dependency)
    reachable = bool(payload.get("reachable"))
    error = payload.get("error")
    version = payload.get("version")

    return {
        "id": "ollama",
        "label": "Ollama API",
        "state": "complete" if reachable else "blocked",
        "detail": str(version)
        if reachable and version
        else ("Reachable" if reachable else str(error or "Unavailable")),
        "fix": None if reachable else "Start Ollama so the local API is reachable",
    }


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _is_supported_python(version: str) -> bool:
    parts = version.split(".")
    if len(parts) < 2:
        return False
    if not parts[0].isdigit() or not parts[1].isdigit():
        return False
    major = int(parts[0])
    minor = int(parts[1])
    return major > 3 or (major == 3 and minor >= 11)


def _local_model_names(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(model) for model in value}
    if isinstance(value, list):
        return {str(model) for model in value}
    return set()
