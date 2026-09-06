"""ffmpeg frame extraction: ONE decode pass, scene+uniform blend, dHash dedup.

FfmpegFrameExtractor implements contracts.FrameExtractor. Performance
contract: both candidate sets come from a single ffmpeg decode invocation —
the input stream is split, the scene branch samples at fps=1/SCENE_INT and
keeps frame 0 plus select=gt(scene,T) hits, the uniform branch samples at
fps=1/UNIF_INT; both scale to frame_max_dim and encode mjpeg q=3.

Timestamp mechanism (verified on ffmpeg 8.1.2; -frame_pts does NOT expand pts
into image2 filenames on current ffmpeg):
- Scene branch: ``metadata=print:key=lavfi.scene_score:file=scene.log`` records
  one ``frame:N pts:P pts_time:S`` / ``lavfi.scene_score=X`` pair per kept
  frame. The k-th scene_%03d.jpg (image2 numbers from 1) maps to the k-th log
  entry: ts_s = pts_time (seconds), scene_score = real logged score.
- Uniform branch: analytic ts_s = (k-1) * UNIF_INT for the k-th unif_%03d.jpg
  (fps emits slots contiguously from 0).
- ``-fps_mode passthrough`` on both image2 outputs is REQUIRED: without it the
  muxer duplicates frames to fill CFR gaps (4 selected frames produced 6
  files). Log-entry count is asserted against file count; on divergence we log
  a warning and fall back to analytic scene timestamps.
- ffmpeg's select scene score is a *rise* detector (empirically
  clip(4*(mafd - prev_mafd), 0, 1)); two adjacent equal-magnitude cuts can
  absorb each other. The uniform branch guarantees coverage regardless.

dHash is a pure-python 64-bit difference hash: each candidate JPEG is piped
through ``ffmpeg -i frame.jpg -vf scale=9:8,format=gray -f rawvideo pipe:1``
(72 bytes) in a worker thread; bit = px[x] > px[x+1] per row, hex string.
Candidates with hamming distance <= 6 are near-duplicates; the copy with the
higher scene_score survives (tie: earlier ts).

Selection: scene candidates win when there are at least min(3, max_frames) of
them (spread evenly across the timeline); otherwise scene frames are kept and
the remaining slots are filled from uniform candidates by spread. Output is
ordered by timestamp.

OWNER: agent-A. See docs/v2/SPEC.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from vra.config import Config, FrameMode
from vra.platform import binaries
from vra.platform.ytdlp import sanitize_canonical_id
from vra.types import Frame, FrameSet

log = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT_S = 600.0
_DHASH_BYTES = 72
_DHASH_MAX_DISTANCE = 6
_LOG_FRAME = re.compile(r"frame:\s*\d+\s+pts:\s*\d+\s+pts_time:([0-9.eE+-]+)")
_LOG_SCORE = re.compile(r"lavfi\.scene_score=([0-9.eE+-]+)")
_FILE_NUMBER = re.compile(r"(\d+)$")


@dataclass(slots=True)
class _Candidate:
    """An extracted frame before selection."""

    path: Path
    ts_s: float
    scene_score: float
    is_scene: bool
    dhash: str = ""


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _parse_scene_log(path: Path) -> list[tuple[float, float]]:
    """(pts_time, scene_score) pairs from a metadata=print log file."""
    entries: list[tuple[float, float]] = []
    pending_ts: float | None = None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("cannot read scene log %s: %s", path, exc)
        return []
    for line in text.splitlines():
        ts_match = _LOG_FRAME.search(line)
        if ts_match:
            try:
                pending_ts = float(ts_match.group(1))
            except ValueError:
                pending_ts = None
            continue
        score_match = _LOG_SCORE.search(line)
        if score_match and pending_ts is not None:
            try:
                entries.append((pending_ts, _clip(float(score_match.group(1)), 0.0, 1.0)))
            except ValueError:
                entries.append((pending_ts, 0.0))
            pending_ts = None
    return entries


def _numeric_key(path: Path) -> int:
    match = _FILE_NUMBER.search(path.stem)
    return int(match.group(1)) if match else 0


def _hamming_distance(a: str, b: str) -> int:
    xor = int(a, 16) ^ int(b, 16)
    return xor.bit_count()


def _pick_evenly(candidates: list[_Candidate], limit: int) -> list[_Candidate]:
    """Evenly spaced picks across a ts-ordered candidate list (legacy port)."""
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        return list(candidates)
    if limit == 1:
        return [candidates[len(candidates) // 2]]
    step = (len(candidates) - 1) / (limit - 1)
    picked: list[_Candidate] = []
    used: set[int] = set()
    for i in range(limit):
        idx = min(max(int(round(i * step)), 0), len(candidates) - 1)
        if idx in used:
            continue
        used.add(idx)
        picked.append(candidates[idx])
    for idx, candidate in enumerate(candidates):
        if len(picked) >= limit:
            break
        if idx not in used:
            used.add(idx)
            picked.append(candidate)
    return picked[:limit]


class FfmpegFrameExtractor:
    """Single-decode-pass frame extraction with scene/uniform blending."""

    def __init__(self, config: Config) -> None:
        self._config = config

    async def extract(
        self,
        media_path: str,
        *,
        canonical_id: str,
        duration_s: float | None,
        max_frames: int,
    ) -> FrameSet:
        """Extract a deduplicated, spread-ranked FrameSet for the media."""
        if max_frames <= 0:
            return FrameSet()
        absolute = os.path.abspath(media_path)

        has_video, probed_duration = await self._probe(absolute)
        if has_video is False:
            log.info("no video stream in %s; skipping frame extraction", media_path)
            return FrameSet()
        duration = (
            duration_s
            if duration_s is not None and duration_s > 0
            else probed_duration
        )
        if duration is not None and duration <= 0:
            return FrameSet()

        scene_interval = (
            max(duration / (max_frames * 4), 3.0) if duration else 12.0
        )
        uniform_interval = max(duration / max_frames, 4.0) if duration else 8.0

        out_dir = self._config.storage_subdir("frames") / sanitize_canonical_id(
            canonical_id
        )
        await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(self._clean_output_dir, out_dir)

        ffmpeg = await asyncio.to_thread(binaries.resolve_ffmpeg)
        await self._extract_pass(
            ffmpeg,
            absolute,
            out_dir,
            scene_interval,
            uniform_interval,
        )

        scene_files, uniform_files = await asyncio.to_thread(
            self._collect_files, out_dir
        )
        if not scene_files and not uniform_files:
            log.warning("frame extraction produced no frames for %s", media_path)
            return FrameSet()

        scene_candidates = self._scene_candidates(out_dir, scene_files, scene_interval)
        uniform_candidates = [
            _Candidate(
                path=path,
                ts_s=idx * uniform_interval,
                scene_score=0.0,
                is_scene=False,
            )
            for idx, path in enumerate(uniform_files)
        ]

        pool = scene_candidates + uniform_candidates
        for candidate in pool:
            candidate.dhash = await self._dhash(ffmpeg, candidate.path)
        pool = await asyncio.to_thread(self._dedup, pool)

        chosen = self._select(
            pool,
            mode=self._config.frame_mode,
            max_frames=max_frames,
        )
        chosen.sort(key=lambda c: c.ts_s)
        return FrameSet(
            frames=tuple(
                Frame(
                    path=str(c.path),
                    ts_s=c.ts_s,
                    scene_score=c.scene_score,
                    dhash=c.dhash,
                )
                for c in chosen
            )
        )

    # ------------------------------------------------------------------ probe

    @staticmethod
    async def _probe(absolute: str) -> tuple[bool | None, float | None]:
        """One ffprobe call: (has_video_stream, duration). None = unknown."""
        ffprobe = await asyncio.to_thread(binaries.resolve_ffprobe)
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
            absolute,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), _SUBPROCESS_TIMEOUT_S
            )
        except TimeoutError:
            proc.kill()
            return None, None
        if proc.returncode != 0:
            return None, None
        try:
            payload = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError:
            return None, None
        if not isinstance(payload, dict):
            return None, None
        streams = payload.get("streams")
        has_video = (
            any(
                isinstance(s, dict) and s.get("codec_type") == "video"
                for s in streams
            )
            if isinstance(streams, list)
            else None
        )
        fmt = payload.get("format")
        duration: float | None = None
        if isinstance(fmt, dict):
            raw = fmt.get("duration")
            if isinstance(raw, str):
                try:
                    duration = float(raw)
                except ValueError:
                    duration = None
        return has_video, duration

    # -------------------------------------------------------------- invocation

    def _scene_chain(self, interval: float, threshold: float) -> str:
        scale = (
            f"scale={self._config.frame_max_dim}:-2:"
            "force_original_aspect_ratio=decrease"
        )
        threshold = _clip(threshold, 0.05, 0.95)
        return (
            f"fps=1/{interval:.3f},"
            f"select='eq(n\\,0)+gt(scene\\,{threshold:.3f})',"
            f"metadata=print:key=lavfi.scene_score:file=scene.log,{scale}"
        )

    def _uniform_chain(self, interval: float) -> str:
        scale = (
            f"scale={self._config.frame_max_dim}:-2:"
            "force_original_aspect_ratio=decrease"
        )
        return f"fps=1/{interval:.3f},{scale}"

    async def _extract_pass(
        self,
        ffmpeg: str,
        absolute: str,
        out_dir: Path,
        scene_interval: float,
        uniform_interval: float,
    ) -> None:
        """The single decode pass covering every enabled branch."""
        mode = self._config.frame_mode
        want_scene = mode in (FrameMode.AUTO, FrameMode.SCENE)
        want_uniform = mode in (FrameMode.AUTO, FrameMode.UNIFORM)
        if not want_scene and not want_uniform:
            return

        chains: list[str] = []
        if want_scene and want_uniform:
            graph = (
                "[0:v]split=2[a][b];"
                f"[a]{self._scene_chain(scene_interval, self._config.scene_threshold)}[s];"
                f"[b]{self._uniform_chain(uniform_interval)}[u]"
            )
            chains.append(graph)
        elif want_scene:
            chains.append(
                f"[0:v]{self._scene_chain(scene_interval, self._config.scene_threshold)}[s]"
            )
        else:
            chains.append(f"[0:v]{self._uniform_chain(uniform_interval)}[u]")

        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-hwaccel", "auto", "-i", absolute]
        cmd += ["-filter_complex", chains[0]]
        if want_scene:
            cmd += [
                "-map",
                "[s]",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "mjpeg",
                "-q:v",
                "3",
                "scene_%03d.jpg",
            ]
        if want_uniform:
            cmd += [
                "-map",
                "[u]",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "mjpeg",
                "-q:v",
                "3",
                "unif_%03d.jpg",
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(out_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), _SUBPROCESS_TIMEOUT_S
            )
        except TimeoutError:
            proc.kill()
            raise RuntimeError(
                f"ffmpeg frame extraction timed out after {_SUBPROCESS_TIMEOUT_S}s"
            ) from None
        if proc.returncode != 0:
            tail = stderr.decode(errors="replace").strip()[-2000:]
            raise RuntimeError(
                f"ffmpeg frame extraction failed (rc={proc.returncode}): {tail}"
            )

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _clean_output_dir(out_dir: Path) -> None:
        """Remove pre-existing extraction artifacts (idempotent re-runs)."""
        for pattern in (
            "scene_*.jpg",
            "unif_*.jpg",
            "out_scene_*.jpg",
            "out_unif_*.jpg",
            "scene.log",
        ):
            for path in out_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError as exc:
                    log.warning("could not remove stale artifact %s: %s", path, exc)

    @staticmethod
    def _collect_files(out_dir: Path) -> tuple[list[Path], list[Path]]:
        scene = sorted(out_dir.glob("scene_*.jpg"), key=_numeric_key)
        uniform = sorted(out_dir.glob("unif_*.jpg"), key=_numeric_key)
        return scene, uniform

    @staticmethod
    def _scene_candidates(
        out_dir: Path, files: list[Path], scene_interval: float
    ) -> list[_Candidate]:
        """Pair scene files with logged pts_time/scene_score entries."""
        entries = _parse_scene_log(out_dir / "scene.log")
        if len(entries) == len(files):
            return [
                _Candidate(path=path, ts_s=ts, scene_score=score, is_scene=True)
                for path, (ts, score) in zip(files, entries, strict=True)
            ]
        log.warning(
            "scene log entries (%d) != scene files (%d); "
            "falling back to analytic timestamps",
            len(entries),
            len(files),
        )
        return [
            _Candidate(
                path=path,
                ts_s=idx * scene_interval,
                scene_score=1.0,
                is_scene=True,
            )
            for idx, path in enumerate(files)
        ]

    async def _dhash(self, ffmpeg: str, path: Path) -> str:
        """64-bit dHash of one JPEG via a 9x8 grayscale rawvideo pipe."""
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "scale=9:8,format=gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), _SUBPROCESS_TIMEOUT_S
            )
        except TimeoutError:
            proc.kill()
            log.warning("dHash timed out for %s", path)
            return ""
        if proc.returncode != 0 or len(stdout) != _DHASH_BYTES:
            log.warning(
                "dHash failed for %s (rc=%s, %d bytes): %s",
                path,
                proc.returncode,
                len(stdout),
                stderr.decode(errors="replace").strip()[-500:],
            )
            return ""
        digest = 0
        for row in range(8):
            offset = row * 9
            for col in range(8):
                bit = 1 if stdout[offset + col] > stdout[offset + col + 1] else 0
                digest = (digest << 1) | bit
        return f"{digest:016x}"

    @staticmethod
    def _dedup(pool: list[_Candidate]) -> list[_Candidate]:
        """Drop near-duplicate dHashes (hamming <= 6); prefer score then ts."""
        kept: list[_Candidate] = []
        for candidate in pool:
            if not candidate.dhash:
                kept.append(candidate)
                continue
            match: _Candidate | None = None
            for existing in kept:
                is_match = (
                    existing.dhash
                    and _hamming_distance(candidate.dhash, existing.dhash)
                    <= _DHASH_MAX_DISTANCE
                )
                if is_match and (
                    match is None
                    or (existing.scene_score, -existing.ts_s)
                    > (match.scene_score, -match.ts_s)
                ):
                    match = existing
            if match is None:
                kept.append(candidate)
                continue
            if candidate.scene_score > match.scene_score or (
                candidate.scene_score == match.scene_score
                and candidate.ts_s < match.ts_s
            ):
                kept[kept.index(match)] = candidate
        return kept

    @staticmethod
    def _select(
        pool: list[_Candidate], *, mode: FrameMode, max_frames: int
    ) -> list[_Candidate]:
        """Blend scene/uniform candidates per frame_mode, capped to max_frames."""
        scene = sorted((c for c in pool if c.is_scene), key=lambda c: c.ts_s)
        uniform = sorted((c for c in pool if not c.is_scene), key=lambda c: c.ts_s)
        if mode == FrameMode.UNIFORM:
            return _pick_evenly(uniform, max_frames)
        if mode == FrameMode.SCENE:
            return _pick_evenly(scene, max_frames)
        if len(scene) >= min(3, max_frames):
            return _pick_evenly(scene, max_frames)
        fill = max_frames - len(scene)
        return (scene + _pick_evenly(uniform, fill))[:max_frames]
