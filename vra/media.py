from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx


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
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


async def prepare_audio(
    client: httpx.AsyncClient,
    source: str,
    storage_dir: str,
) -> Path:
    await ensure_storage_dirs(storage_dir)
    base = Path(storage_dir)
    file_id = uuid.uuid4()

    if _is_url(source):
        raw_path = base / "raw" / str(file_id)
        await download_to_file(client, source, raw_path)
        input_path = raw_path
    else:
        input_path = Path(source)
        if not input_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

    output_path = base / "audio" / f"{file_id}.wav"
    await extract_audio_ffmpeg(input_path, output_path)
    return output_path
