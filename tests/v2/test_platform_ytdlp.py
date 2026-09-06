"""Unit tests for YtDlpAcquirer — command construction + JSON parsing, no network.

asyncio.create_subprocess_exec is monkeypatched with a fake process factory;
every scenario runs offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from vra.config import Config
from vra.platform.ytdlp import YtDlpAcquirer, sanitize_canonical_id


@dataclass
class FakeProc:
    """Minimal stand-in for asyncio.subprocess.Process."""

    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int = 0
    delay: float = 0.0
    on_start: object = None
    pid: int = 1234

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.delay:
            await asyncio.sleep(self.delay)
        if callable(self.on_start):
            self.on_start()
        return self.stdout, self.stderr

    def kill(self) -> None:  # pragma: no cover - mirrors Process.kill
        self.returncode = -9


@dataclass
class Harness:
    """Captures every spawned command and dispatches scripted responses."""

    responses: list[FakeProc] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    counter: int = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = self

        async def fake_exec(*cmd: str, **kwargs: object) -> FakeProc:
            harness.commands.append(list(cmd))
            idx = min(harness.counter, len(harness.responses) - 1)
            harness.counter += 1
            scripted = harness.responses[idx]
            if callable(scripted.on_start):
                scripted.on_start()
            return FakeProc(
                stdout=scripted.stdout,
                stderr=scripted.stderr,
                returncode=scripted.returncode,
                delay=scripted.delay,
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


def sha1_prefix(source: str) -> str:
    return "url:" + hashlib.sha1(source.encode()).hexdigest()[:16]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(data_dir=str(tmp_path), timeout_download_s=30)


def test_sanitize_canonical_id() -> None:
    assert sanitize_canonical_id("youtube:dQ w/x") == "youtube_dQ_w_x"
    long = "a" * 300
    assert sanitize_canonical_id(long) == "a" * 120
    assert sanitize_canonical_id("///") == "___"


async def test_resolve_parses_metadata(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    payload = (
        b'{"extractor_key": "Youtube", "id": "abc123", "title": "T", '
        b'"description": "D", "duration": 61, '
        b'"webpage_url": "https://x/y", '
        b'"chapters": [{"start_time": 0.0, "end_time": 10, "title": "Intro"}, '
        b'{"start_time": "bad"}, {"start_time": 15}]}'
    )
    harness = Harness(responses=[FakeProc(stdout=payload)])
    harness.install(monkeypatch)
    acquirer = YtDlpAcquirer(config)

    source = "https://example.com/watch?v=abc123"
    meta = await acquirer.resolve(source)

    assert meta.canonical_id == "youtube:abc123"
    assert meta.title == "T"
    assert meta.description == "D"
    assert meta.duration_s == 61.0
    assert meta.webpage_url == "https://x/y"
    assert meta.chapters == ((0.0, "Intro"), (15.0, ""))

    cmd = harness.commands[0]
    assert cmd[:3] == [sys.executable, "-m", "yt_dlp"]
    assert "--dump-single-json" in cmd
    assert "--no-playlist" in cmd
    assert cmd[-1] == source


async def test_resolve_fallback_canonical_id(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    harness = Harness(responses=[FakeProc(stdout=b'{"title": "only title"}')])
    harness.install(monkeypatch)
    source = "https://example.com/v/1"
    meta = await YtDlpAcquirer(config).resolve(source)
    assert meta.canonical_id == sha1_prefix(source)
    assert meta.title == "only title"
    assert meta.duration_s is None


async def test_resolve_local_path(config: Config) -> None:
    meta = await YtDlpAcquirer(config).resolve("/data/clip.mp4")
    assert meta.canonical_id == sha1_prefix("/data/clip.mp4")
    assert meta.title == "clip"


async def test_resolve_failure_raises(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    harness = Harness(responses=[FakeProc(stderr=b"boom happened", returncode=1)])
    harness.install(monkeypatch)
    with pytest.raises(RuntimeError, match="boom happened"):
        await YtDlpAcquirer(config).resolve("https://example.com/x")


async def test_resolve_timeout_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = Config(data_dir=str(tmp_path), timeout_download_s=0)
    harness = Harness(responses=[FakeProc(delay=0.05)])
    harness.install(monkeypatch)
    with pytest.raises(RuntimeError, match="timed out"):
        await YtDlpAcquirer(config).resolve("https://example.com/x")


async def test_subtitles_uses_cache(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    cache_dir = config.storage_subdir("subtitles") / "cid1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "cid1.en.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8"
    )
    harness = Harness()  # no responses: any spawn would fail the test
    harness.install(monkeypatch)

    transcript = await YtDlpAcquirer(config).subtitles("https://x/y", "cid1")
    assert transcript is not None
    assert transcript.source == "subtitle"
    assert transcript.lang == "en"
    assert transcript.text == "hello"
    assert harness.commands == []


async def test_subtitles_downloads_preferred_langs(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    cache_dir = config.storage_subdir("subtitles") / "cid2"

    def write_subtitle() -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "cid2.en.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhi there\n",
            encoding="utf-8",
        )

    harness = Harness(responses=[FakeProc(on_start=write_subtitle)])
    harness.install(monkeypatch)

    transcript = await YtDlpAcquirer(config).subtitles("https://x/y", "cid2")
    assert transcript is not None
    assert transcript.lang == "en"
    assert transcript.text == "hi there"

    cmd = harness.commands[0]
    langs_idx = cmd.index("--sub-langs")
    assert cmd[langs_idx + 1] == "en.*,en,zh.*,zh,-live_chat"
    assert "--skip-download" in cmd
    assert "--write-subs" in cmd
    assert "--write-auto-subs" in cmd
    assert cmd[cmd.index("--sub-format") + 1] == "vtt/srt/best"
    template = cmd[cmd.index("--output") + 1]
    assert template.endswith("cid2.%(ext)s")


async def test_subtitles_retries_with_all(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    cache_dir = config.storage_subdir("subtitles") / "cid3"

    def write_second_pass() -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "cid3.fr.vtt").write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nsalut\n", encoding="utf-8"
        )

    # First attempt yields nothing; second (all) writes the file.
    harness = Harness(responses=[FakeProc(), FakeProc(on_start=write_second_pass)])
    harness.install(monkeypatch)

    transcript = await YtDlpAcquirer(config).subtitles("https://x/y", "cid3")
    assert transcript is not None
    assert transcript.lang == "fr"
    assert harness.commands[1][harness.commands[1].index("--sub-langs") + 1] == (
        "all,-live_chat"
    )


async def test_subtitles_none_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    harness = Harness(
        responses=[
            FakeProc(stderr=b"no subs", returncode=1),
            FakeProc(stderr=b"no subs", returncode=1),
        ]
    )
    harness.install(monkeypatch)
    transcript = await YtDlpAcquirer(config).subtitles("https://x/y", "cid4")
    assert transcript is None


async def test_subtitles_local_source_is_none(config: Config) -> None:
    assert await YtDlpAcquirer(config).subtitles("/data/clip.mp4", "cid") is None


async def test_download_idempotent_skip(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    raw_dir = config.storage_subdir("raw")
    raw_dir.mkdir(parents=True)
    existing = raw_dir / "cid5.mp4"
    with existing.open("wb") as fh:
        fh.truncate(17 * 1024 * 1024)
    harness = Harness()
    harness.install(monkeypatch)

    path = await YtDlpAcquirer(config).download("https://x/y", "cid5")
    assert path == str(existing)
    assert harness.commands == []  # no network activity


async def test_download_small_existing_file_redownloads(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    raw_dir = config.storage_subdir("raw")
    raw_dir.mkdir(parents=True)
    (raw_dir / "cid5b.mp4").write_bytes(b"tiny")
    harness = Harness(responses=[FakeProc(stderr=b"fail", returncode=1)])
    harness.install(monkeypatch)
    # <16MiB artifact does not short-circuit: a download is attempted, and its
    # failure falls back to the largest existing artifact for the canonical id.
    path = await YtDlpAcquirer(config).download("https://x/y", "cid5b")
    assert len(harness.commands) == 1
    assert path == str(raw_dir / "cid5b.mp4")


async def test_download_command_and_printed_path(
    monkeypatch: pytest.MonkeyPatch, config: Config, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").write_text("#!/bin/sh\n")
    monkeypatch.setattr("vra.platform.binaries.resolve_ffmpeg", lambda: str(bin_dir / "ffmpeg"))

    raw_dir = config.storage_subdir("raw")
    final = raw_dir / "cid6.mp4"
    harness = Harness(
        responses=[
            FakeProc(
                stdout=f"filepath: {final}\n".encode(),
                on_start=lambda: _touch(raw_dir, final),
            )
        ]
    )
    harness.install(monkeypatch)

    path = await YtDlpAcquirer(config).download("https://x/y", "cid6")
    assert path == str(final)

    cmd = harness.commands[0]
    assert cmd[:3] == [sys.executable, "-m", "yt_dlp"]
    assert cmd[cmd.index("-f") + 1] == "bv*+ba/b"
    assert cmd[cmd.index("--merge-output-format") + 1] == "mp4"
    assert "--no-playlist" in cmd
    assert cmd[cmd.index("--ffmpeg-location") + 1] == str(bin_dir)
    assert cmd[cmd.index("--output") + 1].endswith("cid6.%(ext)s")
    assert cmd[cmd.index("--print") + 1] == "after_move:filepath:%(filepath)s"


def _touch(raw_dir: Path, final: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"x" * 1024)


async def test_download_falls_back_to_largest_file(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    raw_dir = config.storage_subdir("raw")

    def write_mkv() -> None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "cid7.mkv").write_bytes(b"m" * 4096)

    harness = Harness(responses=[FakeProc(stdout=b"nothing printed\n")])
    harness.responses[0].on_start = write_mkv
    harness.install(monkeypatch)

    path = await YtDlpAcquirer(config).download("https://x/y", "cid7")
    assert path == str(raw_dir / "cid7.mkv")


async def test_download_failure_raises(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    harness = Harness(responses=[FakeProc(stderr=b"unsupported url", returncode=1)])
    harness.install(monkeypatch)
    with pytest.raises(RuntimeError, match="unsupported url"):
        await YtDlpAcquirer(config).download("https://x/y", "cid8")


async def test_download_local_source(
    config: Config, tmp_path: Path
) -> None:
    local = tmp_path / "movie.mp4"
    local.write_bytes(b"contents")
    assert await YtDlpAcquirer(config).download(str(local), "cid9") == str(local)
    with pytest.raises(FileNotFoundError):
        await YtDlpAcquirer(config).download(str(tmp_path / "missing.mp4"), "cid9")
