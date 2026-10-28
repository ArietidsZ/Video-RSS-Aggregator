"""Tests for vra.platform.binaries — resolution, memoization, report shape."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vra.platform import binaries


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binaries, "_CACHE", {})


def test_resolve_returns_nonempty_string() -> None:
    ffmpeg = binaries.resolve_ffmpeg()
    ffprobe = binaries.resolve_ffprobe()
    assert isinstance(ffmpeg, str) and ffmpeg
    assert isinstance(ffprobe, str) and ffprobe


def test_results_memoized() -> None:
    assert binaries.resolve_ffmpeg() is binaries.resolve_ffmpeg()
    assert binaries.resolve_ffprobe() is binaries.resolve_ffprobe()


def test_dependency_report_shape() -> None:
    report = binaries.dependency_report()
    assert set(report) == {"ffmpeg", "ffprobe"}
    for entry in report.values():
        assert set(entry) == {"configured", "resolved", "available"}
        assert isinstance(entry["configured"], str)
        assert isinstance(entry["resolved"], str)
        assert isinstance(entry["available"], bool)


def test_winget_resolution_newest_package_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On Windows with no PATH hit, the Gyan.FFmpeg glob resolves the binary."""
    winget_root = tmp_path / "Microsoft" / "WinGet" / "Packages"
    old_bin = winget_root / "Gyan.FFmpeg_8.0" / "x" / "bin" / "ffmpeg.exe"
    new_bin = winget_root / "Gyan.FFmpeg_8.1" / "y" / "bin" / "ffmpeg.exe"
    old_bin.parent.mkdir(parents=True)
    new_bin.parent.mkdir(parents=True)
    old_bin.write_bytes(b"mz")
    new_bin.write_bytes(b"mz")
    past, now = time.time() - 100, time.time()
    os.utime(old_bin, (past, past))
    os.utime(new_bin, (now, now))

    # Emulate only the module platform check; keep native pathlib and global os.
    monkeypatch.setattr(binaries, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)

    resolved = binaries.resolve_binary("ffmpeg")
    assert resolved == str(new_bin)


def test_fallback_to_bare_name_when_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    monkeypatch.setattr(binaries, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "missing"))
    assert binaries.resolve_binary("ffmpeg") == "ffmpeg"
