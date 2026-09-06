"""FFmpeg/ffprobe binary resolution (Windows winget-aware) + dependency report.

Ported from legacy service_media._resolve_binary/_binary_path_state/
runtime_dependency_report. Resolution is lazy: the module imports cheap and
stays side-effect free; results are memoized per binary name.

Search order for a binary name:
1. ``shutil.which(name)`` (PATH).
2. Windows only: newest ``Gyan.FFmpeg_*`` winget package under
   ``%LOCALAPPDATA%/Microsoft/WinGet/Packages``.
3. Fall back to the bare name (ffmpeg resolves it via its own PATH search).

OWNER: agent-A. See docs/v2/SPEC.md.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE: dict[str, str] = {}


def _safe_mtime(path: Path) -> float:
    """st_mtime that survives files vanishing mid-glob (race-safe sort key)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _winget_matches(name: str) -> list[Path]:
    """Newest-first Gyan.FFmpeg winget package binaries for *name* (Windows)."""
    local = os.environ.get("LOCALAPPDATA", "")
    if not local:
        return []
    winget_root = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if not winget_root.is_dir():
        return []
    return sorted(
        winget_root.glob(f"Gyan.FFmpeg_*/*/bin/{name}.exe"),
        key=_safe_mtime,
        reverse=True,
    )


def resolve_binary(name: str) -> str:
    """Resolve *name* ("ffmpeg"/"ffprobe") to a usable path; memoized."""
    cached = _CACHE.get(name)
    if cached is not None:
        return cached
    direct = shutil.which(name)
    resolved = direct or ""
    if not resolved and os.name == "nt":
        matches = _winget_matches(name)
        if matches:
            resolved = str(matches[0])
    result = resolved or name
    log.debug("resolved binary %s -> %s", name, result)
    _CACHE[name] = result
    return result


def resolve_ffmpeg() -> str:
    """Return the resolved ffmpeg path (or bare "ffmpeg" when not found)."""
    return resolve_binary("ffmpeg")


def resolve_ffprobe() -> str:
    """Return the resolved ffprobe path (or bare "ffprobe" when not found)."""
    return resolve_binary("ffprobe")


def _binary_path_state(binary: str) -> dict[str, str | bool]:
    """Describe one resolved binary for diagnostics (legacy-compatible)."""
    path = Path(binary)
    if path.is_absolute():
        resolved = str(path)
        available = path.exists()
    else:
        found = shutil.which(binary)
        resolved = found or binary
        available = found is not None
    return {
        "configured": binary,
        "resolved": resolved,
        "available": available,
    }


def dependency_report() -> dict[str, dict[str, str | bool]]:
    """Report resolved ffmpeg/ffprobe state for /setup/diagnostics."""
    return {
        "ffmpeg": _binary_path_state(resolve_ffmpeg()),
        "ffprobe": _binary_path_state(resolve_ffprobe()),
    }
