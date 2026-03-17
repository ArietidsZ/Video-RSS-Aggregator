from video_rss_aggregator.setup_view_models import (
    build_diagnostics_view,
    build_runtime_view,
)


def test_build_diagnostics_view_marks_missing_dependencies_blocked() -> None:
    report = {
        "ready": False,
        "platform": {
            "python_version": "3.11.9",
            "python_executable": "/usr/bin/python3",
        },
        "dependencies": {
            "ffmpeg": {"available": False, "command": None},
            "ffprobe": {"available": True, "command": "/usr/bin/ffprobe"},
            "yt_dlp": {"available": True, "command": "/usr/bin/yt-dlp"},
            "ollama": {"reachable": False, "error": "connection refused"},
        },
    }

    view = build_diagnostics_view(report)

    assert view["state"] == "blocked"
    assert view["blockers"] == [
        "Install FFmpeg and add it to PATH",
        "Start Ollama so the local API is reachable",
    ]
    assert view["checks"][0] == {
        "id": "python",
        "label": "Python 3.11+",
        "state": "complete",
        "detail": "3.11.9 via /usr/bin/python3",
        "fix": None,
    }
    assert view["checks"][1]["id"] == "ffmpeg"
    assert view["checks"][1]["state"] == "blocked"
    assert view["next_action"] == "Resolve the blocked checks and run diagnostics again"


def test_build_diagnostics_view_marks_ready_path_complete() -> None:
    report = {
        "ready": True,
        "platform": {
            "python_version": "3.11.9",
            "python_executable": "/usr/bin/python3",
        },
        "dependencies": {
            "ffmpeg": {"available": True, "command": "/usr/bin/ffmpeg"},
            "ffprobe": {"available": True, "command": "/usr/bin/ffprobe"},
            "yt_dlp": {"available": True, "command": "/usr/bin/yt-dlp"},
            "ollama": {"reachable": True, "error": None},
        },
    }

    view = build_diagnostics_view(report)

    assert view["state"] == "ready"
    assert view["blockers"] == []
    assert view["next_action"] == "Continue to configuration"


def test_build_diagnostics_view_uses_ollama_version_for_reachable_detail() -> None:
    report = {
        "ready": True,
        "platform": {
            "python_version": "3.11.9",
            "python_executable": "/usr/bin/python3",
        },
        "dependencies": {
            "ffmpeg": {"available": True, "command": "/usr/bin/ffmpeg"},
            "ffprobe": {"available": True, "command": "/usr/bin/ffprobe"},
            "yt_dlp": {"available": True, "command": "/usr/bin/yt-dlp"},
            "ollama": {"reachable": True, "error": None, "version": "0.6.0"},
        },
    }

    view = build_diagnostics_view(report)

    assert view["checks"][4]["id"] == "ollama"
    assert view["checks"][4]["detail"] == "0.6.0"


def test_build_diagnostics_view_blocks_python_below_311() -> None:
    report = {
        "ready": False,
        "platform": {
            "python_version": "3.10.14",
            "python_executable": "/usr/bin/python3",
        },
        "dependencies": {
            "ffmpeg": {"available": True, "command": "/usr/bin/ffmpeg"},
            "ffprobe": {"available": True, "command": "/usr/bin/ffprobe"},
            "yt_dlp": {"available": True, "command": "/usr/bin/yt-dlp"},
            "ollama": {"reachable": True, "error": None, "version": "0.6.0"},
        },
    }

    view = build_diagnostics_view(report)

    assert view["state"] == "blocked"
    assert view["checks"][0]["state"] == "blocked"
    assert view["checks"][0]["fix"] == "Install Python 3.11+"
    assert "Install Python 3.11+" in view["blockers"]


def test_build_runtime_view_marks_missing_models_blocked() -> None:
    runtime = {
        "reachable": True,
        "local_models": {"qwen3.5:2b-q4_K_M": {}},
        "models": ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"],
    }

    view = build_runtime_view(runtime)

    assert view["state"] == "blocked"
    assert view["missing_models"] == ["qwen3.5:4b-q4_K_M"]
    assert view["next_action"] == "Bootstrap required models"


def test_build_runtime_view_marks_unreachable_runtime_for_check() -> None:
    runtime = {
        "reachable": False,
        "local_models": {
            "qwen3.5:4b-q4_K_M": {},
            "qwen3.5:2b-q4_K_M": {},
        },
        "models": ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"],
    }

    view = build_runtime_view(runtime)

    assert view["state"] == "blocked"
    assert view["missing_models"] == []
    assert view["next_action"] == "Check runtime"


def test_build_runtime_view_accepts_list_shaped_local_models() -> None:
    runtime = {
        "reachable": True,
        "local_models": ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"],
        "models": ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"],
    }

    view = build_runtime_view(runtime)

    assert view["state"] == "ready"
    assert view["missing_models"] == []
    assert view["next_action"] == "Run the first processing test"


def test_build_runtime_view_marks_ready_path_complete() -> None:
    runtime = {
        "reachable": True,
        "local_models": {
            "qwen3.5:4b-q4_K_M": {},
            "qwen3.5:2b-q4_K_M": {},
        },
        "models": ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"],
    }

    view = build_runtime_view(runtime)

    assert view["state"] == "ready"
    assert view["missing_models"] == []
    assert view["next_action"] == "Run the first processing test"
