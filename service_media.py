from __future__ import annotations

import asyncio
from dataclasses import dataclass
import sys
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx


_DIRECT_MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mka",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


@dataclass(slots=True)
class PreparedAudio:
    audio_path: Path
    title: str | None = None


async def ensure_storage_dirs(storage_dir: str) -> None:
    base = Path(storage_dir)
    (base / "raw").mkdir(parents=True, exist_ok=True)
    (base / "audio").mkdir(parents=True, exist_ok=True)


async def download_to_file(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=1 << 16):
                f.write(chunk)


async def extract_audio_ffmpeg(input_path: Path, output_path: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")


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

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-warnings",
        "-f",
        "bestaudio/best",
        "--output",
        output_template,
        "--print",
        "title:%(title)s",
        "--print",
        "after_move:filepath:%(filepath)s",
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"yt-dlp failed: {err}")

    out_text = stdout.decode(errors="replace")
    downloaded_path: Path | None = None
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
                downloaded_path = p
            continue

        p = Path(line)
        if p.exists():
            downloaded_path = p

    if downloaded_path:
        return downloaded_path, title

    matches = sorted(
        raw_dir.glob(f"{file_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if matches:
        return matches[0], title

    raise RuntimeError("yt-dlp completed but no media file was found")


async def prepare_audio(
    client: httpx.AsyncClient,
    source: str,
    storage_dir: str,
) -> PreparedAudio:
    await ensure_storage_dirs(storage_dir)
    base = Path(storage_dir)
    file_id = uuid.uuid4()
    inferred_title: str | None = None

    if _is_url(source):
        input_path: Path | None = None

        if _looks_like_direct_media_url(source):
            raw_path = base / "raw" / f"{file_id}{_url_suffix(source)}"
            try:
                await download_to_file(client, source, raw_path)
                input_path = raw_path
                inferred_title = _title_from_url(source)
            except Exception:
                input_path = None

        if input_path is None:
            input_path, inferred_title = await download_with_ytdlp(
                source, base / "raw", file_id
            )

        if not inferred_title:
            inferred_title = _title_from_url(source)
    else:
        input_path = Path(source)
        if not input_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        inferred_title = _title_from_path(input_path)

    output_path = base / "audio" / f"{file_id}.wav"
    await extract_audio_ffmpeg(input_path, output_path)
    return PreparedAudio(audio_path=output_path, title=inferred_title)
