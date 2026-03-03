from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
import logging
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx


log = logging.getLogger(__name__)


_DIRECT_MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4v",
    ".m4a",
    ".mka",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".opus",
    ".ts",
    ".wmv",
    ".wav",
    ".webm",
}


def _resolve_binary(name: str) -> str:
    direct = shutil.which(name)
    if direct:
        return direct

    if os.name != "nt":
        return name

    local = Path(os.environ.get("LOCALAPPDATA", ""))
    if local:
        winget_root = local / "Microsoft" / "WinGet" / "Packages"
        if winget_root.exists():
            matches = sorted(
                winget_root.glob(f"Gyan.FFmpeg_*/*/bin/{name}.exe"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return str(matches[0])

    return name


_FFMPEG_BIN = _resolve_binary("ffmpeg")
_FFPROBE_BIN = _resolve_binary("ffprobe")


def _binary_path_state(binary: str) -> dict[str, str | bool]:
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


def runtime_dependency_report() -> dict[str, dict[str, str | bool]]:
    return {
        "ffmpeg": _binary_path_state(_FFMPEG_BIN),
        "ffprobe": _binary_path_state(_FFPROBE_BIN),
    }


@dataclass(slots=True)
class PreparedMedia:
    media_path: Path
    title: str | None = None
    transcript: str = ""
    frame_paths: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class PreparedSource:
    media_path: Path
    title: str | None = None
    transcript: str = ""


async def ensure_storage_dirs(storage_dir: str) -> None:
    base = Path(storage_dir)
    (base / "raw").mkdir(parents=True, exist_ok=True)
    (base / "frames").mkdir(parents=True, exist_ok=True)
    (base / "subtitles").mkdir(parents=True, exist_ok=True)


async def download_to_file(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=1 << 16):
                f.write(chunk)


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _url_suffix(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def _looks_like_direct_media_url(url: str) -> bool:
    return _url_suffix(url) in _DIRECT_MEDIA_EXTENSIONS


def _title_from_url(url: str) -> str | None:
    stem = Path(unquote(urlparse(url).path)).stem.strip()
    return stem or None


def _title_from_path(path: Path) -> str | None:
    stem = path.stem.strip()
    return stem or None


async def download_with_ytdlp(
    url: str,
    raw_dir: Path,
    file_id: uuid.UUID,
) -> tuple[Path, str | None]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(raw_dir / f"{file_id}.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
    ]

    ffmpeg_path = Path(_FFMPEG_BIN)
    if ffmpeg_path.exists():
        cmd.extend(["--ffmpeg-location", str(ffmpeg_path.parent)])

    cmd.extend(
        [
            "--output",
            output_template,
            "--print",
            "title:%(title)s",
            "--print",
            "after_move:filepath:%(filepath)s",
            url,
        ]
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    out_text = stdout.decode(errors="replace")
    downloaded_paths: list[Path] = []
    title: str | None = None
    for raw_line in out_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("title:"):
            value = line.removeprefix("title:").strip()
            if value:
                title = value
            continue
        if line.startswith("filepath:"):
            value = line.removeprefix("filepath:").strip()
            p = Path(value)
            if p.exists():
                downloaded_paths.append(p)
            continue

        p = Path(line)
        if p.exists():
            downloaded_paths.append(p)

    selected = await _select_best_media_path(downloaded_paths)
    if selected:
        return selected, title

    matches = sorted(
        raw_dir.glob(f"{file_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    selected = await _select_best_media_path(matches)
    if selected:
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            if err:
                log.warning("yt-dlp returned non-zero for %s: %s", url, err[:500])
        return selected, title

    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"yt-dlp failed: {err}")

    raise RuntimeError("yt-dlp completed but no media file was found")


async def _select_best_media_path(candidates: list[Path]) -> Path | None:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            unique.append(path)

    if not unique:
        return None

    video_paths: list[Path] = []
    for path in unique:
        try:
            if await _has_video_stream(path):
                video_paths.append(path)
        except Exception:
            continue

    if video_paths:
        return max(video_paths, key=lambda p: p.stat().st_size)
    return max(unique, key=lambda p: p.stat().st_size)


async def download_subtitles_with_ytdlp(
    url: str,
    subtitle_dir: Path,
    file_id: uuid.UUID,
) -> Path | None:
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(subtitle_dir / f"{file_id}.%(ext)s")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "--skip-download",
        "--write-auto-subs",
        "--write-subs",
        "--sub-langs",
        "en.*,en,-live_chat",
        "--sub-format",
        "vtt",
        "--output",
        output_template,
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    candidates = sorted(
        subtitle_dir.glob(f"{file_id}*.vtt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            subtitle_dir.glob(f"{file_id}*.srt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if candidates:

        def _rank(path: Path) -> tuple[int, float]:
            name = path.name.lower()
            if name.endswith(".en.vtt"):
                score = 0
            elif name.endswith(".en-en.vtt"):
                score = 1
            elif name.endswith(".en-orig.vtt"):
                score = 2
            else:
                score = 3
            return (score, -path.stat().st_mtime)

        return sorted(candidates, key=_rank)[0]

    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        out = stdout.decode(errors="replace").strip()
        short = err or out
        if short:
            log.warning("Subtitle download failed for %s: %s", url, short[:500])
    return None


def subtitle_to_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    seen = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "WEBVTT":
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        if line.isdigit():
            continue
        if line.startswith("NOTE"):
            continue

        cleaned = re.sub(r"<[^>]+>", " ", line)
        cleaned = re.sub(r"\{[^}]+\}", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue

        if cleaned == seen:
            continue
        seen = cleaned
        out.append(cleaned)
        if sum(len(item) + 1 for item in out) >= max_chars:
            break
    merged = " ".join(out)
    return merged[:max_chars].strip()


async def _probe_duration_seconds(path: Path) -> float | None:
    proc = await asyncio.create_subprocess_exec(
        _FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    raw = stdout.decode(errors="replace").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


async def _has_video_stream(path: Path) -> bool:
    proc = await asyncio.create_subprocess_exec(
        _FFPROBE_BIN,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return False
    return "video" in stdout.decode(errors="replace").lower()


def _bound_scene_threshold(value: float) -> float:
    return min(max(value, 0.05), 0.95)


def _pick_evenly(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0:
        return []
    if len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[len(paths) // 2]]

    step = (len(paths) - 1) / (limit - 1)
    picked: list[Path] = []
    used: set[int] = set()
    for i in range(limit):
        idx = int(round(i * step))
        idx = min(max(idx, 0), len(paths) - 1)
        if idx in used:
            continue
        used.add(idx)
        picked.append(paths[idx])

    if len(picked) < limit:
        for idx, path in enumerate(paths):
            if idx in used:
                continue
            picked.append(path)
            if len(picked) >= limit:
                break
    return picked[:limit]


def _interval_for_uniform_frames(duration: float | None, max_frames: int) -> float:
    if duration and duration > 0:
        return max(duration / max(max_frames, 1), 4.0)
    return 8.0


def _interval_for_scene_candidates(duration: float | None, max_frames: int) -> float:
    if duration and duration > 0:
        target_samples = max(max_frames * 4, 12)
        return max(duration / target_samples, 3.0)
    return 12.0


def _digest_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 16)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _merge_unique_frames(
    primary: list[Path], secondary: list[Path], limit: int
) -> list[Path]:
    out: list[Path] = []
    seen_digest: set[str] = set()

    for path in [*primary, *secondary]:
        if len(out) >= limit:
            break
        if not path.exists():
            continue
        try:
            file_digest = _digest_file(path)
        except OSError:
            continue
        if file_digest in seen_digest:
            continue
        seen_digest.add(file_digest)
        out.append(path)
    return out


async def _extract_scene_candidates_ffmpeg(
    input_path: Path,
    output_dir: Path,
    file_id: uuid.UUID,
    max_frames: int,
    duration: float | None,
    scene_threshold: float,
) -> list[Path]:
    if max_frames <= 0:
        return []

    interval = _interval_for_scene_candidates(duration, max_frames)
    threshold = _bound_scene_threshold(scene_threshold)
    pattern = output_dir / f"{file_id}_scene_%03d.jpg"
    vf = (
        f"fps=1/{interval:.3f},"
        f"select=eq(n\\,0)+gt(scene\\,{threshold:.3f}),"
        "scale=960:-2:force_original_aspect_ratio=decrease"
    )

    proc = await asyncio.create_subprocess_exec(
        _FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        str(pattern),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg scene detection frame extraction failed: {stderr.decode(errors='replace')}"
        )
    return sorted(output_dir.glob(f"{file_id}_scene_*.jpg"))


async def _extract_uniform_frames_ffmpeg(
    input_path: Path,
    output_dir: Path,
    file_id: uuid.UUID,
    max_frames: int,
    duration: float | None,
) -> list[Path]:
    if max_frames <= 0:
        return []

    interval = _interval_for_uniform_frames(duration, max_frames)
    pattern = output_dir / f"{file_id}_uniform_%03d.jpg"

    proc = await asyncio.create_subprocess_exec(
        _FFMPEG_BIN,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"fps=1/{interval:.3f},scale=960:-2:force_original_aspect_ratio=decrease",
        "-frames:v",
        str(max_frames),
        str(pattern),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg uniform frame extraction failed: {stderr.decode(errors='replace')}"
        )

    return sorted(output_dir.glob(f"{file_id}_uniform_*.jpg"))


async def extract_frames_ffmpeg(
    input_path: Path,
    output_dir: Path,
    file_id: uuid.UUID,
    max_frames: int,
    scene_detection: bool,
    scene_threshold: float,
    min_scene_frames: int,
) -> list[Path]:
    if max_frames <= 0:
        return []
    if not await _has_video_stream(input_path):
        return []

    duration = await _probe_duration_seconds(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_candidates: list[Path] = []
    if scene_detection:
        try:
            scene_candidates = await _extract_scene_candidates_ffmpeg(
                input_path=input_path,
                output_dir=output_dir,
                file_id=file_id,
                max_frames=max_frames,
                duration=duration,
                scene_threshold=scene_threshold,
            )
        except Exception as exc:
            log.warning("Scene detection failed for %s: %s", input_path, exc)

    use_scene = len(scene_candidates) >= max(min_scene_frames, 1)
    if use_scene and len(scene_candidates) >= max_frames:
        return _pick_evenly(scene_candidates, max_frames)

    uniform_frames = await _extract_uniform_frames_ffmpeg(
        input_path=input_path,
        output_dir=output_dir,
        file_id=file_id,
        max_frames=max_frames,
        duration=duration,
    )
    if not use_scene:
        return _pick_evenly(uniform_frames, max_frames)

    scene_focus = _pick_evenly(scene_candidates, max_frames)
    uniform_focus = _pick_evenly(uniform_frames, max_frames)
    merged = _merge_unique_frames(scene_focus, uniform_focus, max_frames)
    if len(merged) < max_frames:
        merged = _merge_unique_frames(merged, scene_candidates, max_frames)
    return merged[:max_frames]


async def prepare_media(
    client: httpx.AsyncClient,
    source: str,
    storage_dir: str,
    max_frames: int,
    scene_detection: bool,
    scene_threshold: float,
    scene_min_frames: int,
    max_transcript_chars: int,
) -> PreparedMedia:
    prepared = await prepare_source(
        client=client,
        source=source,
        storage_dir=storage_dir,
        max_transcript_chars=max_transcript_chars,
    )

    frame_paths = await extract_frames_ffmpeg(
        prepared.media_path,
        Path(storage_dir) / "frames",
        uuid.uuid4(),
        max_frames,
        scene_detection=scene_detection,
        scene_threshold=scene_threshold,
        min_scene_frames=scene_min_frames,
    )

    return PreparedMedia(
        media_path=prepared.media_path,
        title=prepared.title,
        transcript=prepared.transcript,
        frame_paths=frame_paths,
    )


async def prepare_source(
    client: httpx.AsyncClient,
    source: str,
    storage_dir: str,
    max_transcript_chars: int,
) -> PreparedSource:
    await ensure_storage_dirs(storage_dir)
    base = Path(storage_dir)
    file_id = uuid.uuid4()
    inferred_title: str | None = None
    subtitle_path: Path | None = None

    if _is_url(source):
        media_path: Path | None = None

        try:
            subtitle_path = await download_subtitles_with_ytdlp(
                source,
                base / "subtitles",
                file_id,
            )
        except Exception:
            subtitle_path = None

        if _looks_like_direct_media_url(source):
            raw_path = base / "raw" / f"{file_id}{_url_suffix(source)}"
            try:
                await download_to_file(client, source, raw_path)
                media_path = raw_path
                inferred_title = _title_from_url(source)
            except Exception:
                media_path = None

        if media_path is None:
            media_path, inferred_title = await download_with_ytdlp(
                source,
                base / "raw",
                file_id,
            )

        if not inferred_title:
            inferred_title = _title_from_url(source)
    else:
        media_path = Path(source)
        if not media_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        inferred_title = _title_from_path(media_path)

    transcript = ""
    if subtitle_path and subtitle_path.exists():
        transcript = subtitle_to_text(subtitle_path, max_transcript_chars)

    return PreparedSource(
        media_path=media_path,
        title=inferred_title,
        transcript=transcript,
    )
