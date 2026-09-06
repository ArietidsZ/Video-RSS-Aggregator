"""Integration tests for FfmpegFrameExtractor (real ffmpeg, no network).

All tests skip when ffmpeg/ffprobe are unavailable. Tiny clips are generated
once per session with lavfi sources.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from vra.config import Config, FrameMode
from vra.platform.media import FfmpegFrameExtractor, _parse_scene_log
from vra.types import FrameSet

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _gen(args: list[str], target: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", *args, str(target)],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture(scope="session")
def tiny_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4s moving pattern (testsrc2), 128x64 @ 10fps."""
    out = tmp_path_factory.mktemp("media") / "tiny.mp4"
    return _gen(
        ["-f", "lavfi", "-i", "testsrc2=duration=4:size=128x64:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p"],
        out,
    )


@pytest.fixture(scope="session")
def cuts_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """15s: red 0-5s, blue 5-10s, white 10-15s -> hard cuts at 5s and 10s."""
    out = tmp_path_factory.mktemp("media") / "cuts.mp4"
    return _gen(
        [
            "-f", "lavfi", "-i", "color=red:size=128x64:rate=10:duration=5",
            "-f", "lavfi", "-i", "color=blue:size=128x64:rate=10:duration=5",
            "-f", "lavfi", "-i", "color=white:size=128x64:rate=10:duration=5",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        ],
        out,
    )


@pytest.fixture(scope="session")
def solid_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """30s solid red: every sampled frame is visually identical."""
    out = tmp_path_factory.mktemp("media") / "solid.mp4"
    return _gen(
        ["-f", "lavfi", "-i", "color=red:size=128x64:rate=5:duration=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p"],
        out,
    )


@pytest.fixture(scope="session")
def audio_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("media") / "audio.wav"
    return _gen(
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "pcm_s16le"],
        out,
    )


def _config(tmp_path: Path, **overrides: object) -> Config:
    return Config(data_dir=str(tmp_path), **overrides)  # type: ignore[arg-type]


async def test_extract_caps_frames_and_timestamps(
    tmp_path: Path, tiny_clip: Path
) -> None:
    extractor = FfmpegFrameExtractor(_config(tmp_path))
    frames = await extractor.extract(
        str(tiny_clip), canonical_id="tiny", duration_s=20.0, max_frames=8
    )
    assert isinstance(frames, FrameSet)
    assert 1 <= len(frames.frames) <= 8
    for frame in frames.frames:
        assert 0.0 <= frame.ts_s <= 20.0 + 1e-6
        assert Path(frame.path).is_file()
        assert len(frame.dhash) == 16
        int(frame.dhash, 16)  # valid hex
    ts = [f.ts_s for f in frames.frames]
    assert ts == sorted(ts)


async def test_duration_none_probes_with_ffprobe(
    tmp_path: Path, tiny_clip: Path
) -> None:
    extractor = FfmpegFrameExtractor(_config(tmp_path))
    frames = await extractor.extract(
        str(tiny_clip), canonical_id="probe", duration_s=None, max_frames=4
    )
    assert 1 <= len(frames.frames) <= 4
    assert all(0.0 <= f.ts_s <= 4.0 + 1e-6 for f in frames.frames)


async def test_uniform_mode_produces_no_scene_outputs(
    tmp_path: Path, tiny_clip: Path
) -> None:
    extractor = FfmpegFrameExtractor(_config(tmp_path, frame_mode=FrameMode.UNIFORM))
    frames = await extractor.extract(
        str(tiny_clip), canonical_id="unif", duration_s=4.0, max_frames=4
    )
    out_dir = tmp_path / "frames" / "unif"
    assert not list(out_dir.glob("scene_*.jpg"))
    assert list(out_dir.glob("unif_*.jpg"))
    assert 1 <= len(frames.frames) <= 4


async def test_scene_mode_produces_no_uniform_outputs(
    tmp_path: Path, tiny_clip: Path
) -> None:
    extractor = FfmpegFrameExtractor(_config(tmp_path, frame_mode=FrameMode.SCENE))
    await extractor.extract(
        str(tiny_clip), canonical_id="scene", duration_s=4.0, max_frames=4
    )
    out_dir = tmp_path / "frames" / "scene"
    assert list(out_dir.glob("scene_*.jpg"))
    assert not list(out_dir.glob("unif_*.jpg"))


async def test_solid_color_input_deduped_to_few_frames(
    tmp_path: Path, solid_clip: Path
) -> None:
    """10+ raw candidates of identical content must collapse via dHash dedup."""
    extractor = FfmpegFrameExtractor(_config(tmp_path, max_frames=8))
    frames = await extractor.extract(
        str(solid_clip), canonical_id="solid", duration_s=30.0, max_frames=8
    )
    out_dir = tmp_path / "frames" / "solid"
    raw_candidates = len(list(out_dir.glob("scene_*.jpg"))) + len(
        list(out_dir.glob("unif_*.jpg"))
    )
    assert raw_candidates > 4
    assert len(frames.frames) <= 2


async def test_regression_cuts_scene_timestamps_hit_boundaries(
    tmp_path: Path, cuts_clip: Path
) -> None:
    """Lead regression: scene ts hits the 5s/10s cuts within +/- 1 fps
    interval; uniform emits exactly one file per UNIF_INT slot (4 for 15s/8)
    — i.e. -fps_mode passthrough prevents CFR frame duplication."""
    extractor = FfmpegFrameExtractor(_config(tmp_path, max_frames=8))
    frames = await extractor.extract(
        str(cuts_clip), canonical_id="cuts", duration_s=15.0, max_frames=8
    )
    out_dir = tmp_path / "frames" / "cuts"

    scene_files = sorted(out_dir.glob("scene_*.jpg"))
    entries = _parse_scene_log(out_dir / "scene.log")
    assert len(entries) == len(scene_files) >= 3  # anchor + 2 detected cuts
    scene_interval = 3.0  # max(15 / (8*4), 3)
    for boundary in (5.0, 10.0):
        assert any(abs(ts - boundary) <= scene_interval + 1e-6 for ts, _ in entries)
    assert entries[0][0] == 0.0  # eq(n,0) anchor

    # UNIF_INT = max(15/8, 4) = 4 -> slots 0,4,8,12, all present, none duped.
    assert len(list(out_dir.glob("unif_*.jpg"))) == 4

    assert len(frames.frames) <= 8
    assert all(0.0 <= f.ts_s <= 15.0 + 1e-6 for f in frames.frames)


async def test_uniform_files_match_expected_slots(
    tmp_path: Path, cuts_clip: Path
) -> None:
    extractor = FfmpegFrameExtractor(
        _config(tmp_path, frame_mode=FrameMode.UNIFORM, max_frames=8)
    )
    await extractor.extract(
        str(cuts_clip), canonical_id="cutsu", duration_s=15.0, max_frames=8
    )
    out_dir = tmp_path / "frames" / "cutsu"
    uniform_files = list(out_dir.glob("unif_*.jpg"))
    assert len(uniform_files) == 4  # exactly the 0,4,8,12 slots


async def test_no_video_stream_returns_empty(tmp_path: Path, audio_only: Path) -> None:
    extractor = FfmpegFrameExtractor(_config(tmp_path))
    frames = await extractor.extract(
        str(audio_only), canonical_id="audio", duration_s=1.0, max_frames=4
    )
    assert frames.frames == ()


async def test_zero_max_frames_short_circuits(tmp_path: Path, tiny_clip: Path) -> None:
    extractor = FfmpegFrameExtractor(_config(tmp_path))
    frames = await extractor.extract(
        str(tiny_clip), canonical_id="noframes", duration_s=4.0, max_frames=0
    )
    assert frames.frames == ()


async def test_corrupt_input_raises_runtime_error(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"this is not a video file" * 100)
    extractor = FfmpegFrameExtractor(_config(tmp_path))
    with pytest.raises(RuntimeError):
        await extractor.extract(
            str(garbage), canonical_id="bad", duration_s=None, max_frames=4
        )


async def test_stale_outputs_replaced_between_runs(
    tmp_path: Path, tiny_clip: Path
) -> None:
    out_dir = tmp_path / "frames" / "stale"
    out_dir.mkdir(parents=True)
    stale = out_dir / "scene_001.jpg"
    stale.write_bytes(b"old bytes")
    extractor = FfmpegFrameExtractor(_config(tmp_path))
    await extractor.extract(
        str(tiny_clip), canonical_id="stale", duration_s=4.0, max_frames=4
    )
    # The stale artifact was removed and (scene branch active) rewritten fresh.
    assert stale.read_bytes() != b"old bytes"
    assert not (out_dir / "unif_999.jpg").exists()  # never-touched slot stays empty
