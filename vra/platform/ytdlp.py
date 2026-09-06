"""yt-dlp integration: metadata resolution, subtitles, media download.

YtDlpAcquirer implements contracts.VideoAcquirer. Every yt-dlp invocation is
``sys.executable -m yt_dlp ...`` via asyncio.create_subprocess_exec (no shell),
bounded by config.timeout_download_s, with stdout/stderr captured.

- resolve(): one ``--dump-single-json --no-playlist`` call; never downloads.
  canonical_id = f"{extractor_key.lower()}:{id}" with fallback
  "url:" + sha1(source)[:16] when extractor keys are missing. Local paths are
  resolved natively (no extractor) using the same sha1 fallback.
- subtitles(): ``--skip-download`` with preferred languages first, one retry
  with "all" (live_chat always excluded); parses via
  vra.platform.subtitles.parse_file. Results are cached under
  data_dir/subtitles/<sanitized canonical_id>/.
- download(): ``-f bv*+ba/b --merge-output-format mp4`` into
  data_dir/raw/<sanitized canonical_id>.mp4; idempotent (existing >16 MiB
  artifact short-circuits before any network activity). The printed
  ``after_move:filepath`` is preferred because the merged extension may differ.

SSRF: source URLs are validated upstream (config.allow_url); this module never
fetches anything except the given source.

OWNER: agent-A. See docs/v2/SPEC.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from vra.config import Config
from vra.platform import binaries
from vra.platform.subtitles import parse_file
from vra.types import Transcript, VideoMeta

log = logging.getLogger(__name__)

_PREFERRED_SUB_LANGS = "en.*,en,zh.*,zh,-live_chat"
_ALL_SUB_LANGS = "all,-live_chat"
_SUB_FORMAT = "vtt/srt/best"
_MIN_REUSE_BYTES = 16 * 1024 * 1024
_STDERR_TAIL = 2000
_SANITIZE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(slots=True)
class _Result:
    """Completed subprocess result (byte payloads)."""

    returncode: int
    stdout: bytes
    stderr: bytes


def sanitize_canonical_id(canonical_id: str) -> str:
    """Filesystem-safe form of a canonical_id (Windows-safe, capped at 120)."""
    cleaned = _SANITIZE.sub("_", canonical_id)[:120]
    return cleaned or "_"


def _tail(stderr: bytes) -> str:
    """Decoded, whitespace-trimmed tail of a stderr stream for error messages."""
    return stderr.decode(errors="replace").strip()[-_STDERR_TAIL:]


def _is_http_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def _fallback_canonical_id(source: str) -> str:
    return "url:" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def _local_path(source: str) -> Path:
    return Path(source.removeprefix("file://"))


async def _run(cmd: list[str], timeout_s: float, what: str) -> _Result:
    """Run a subprocess to completion with a hard timeout; kill on overrun."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):  # best-effort reap after kill
            await proc.communicate()
        raise RuntimeError(f"{what} timed out after {timeout_s}s") from None
    return _Result(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout,
        stderr=stderr,
    )


def _ytdlp_cmd() -> list[str]:
    return [sys.executable, "-m", "yt_dlp"]


async def _ensure_dir(path: Path) -> None:
    await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)


class YtDlpAcquirer:
    """Acquire platform metadata, subtitles, and media via yt-dlp."""

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------ meta

    async def resolve(self, source: str) -> VideoMeta:
        """Resolve a source into VideoMeta without downloading media."""
        if not _is_http_url(source):
            return self._resolve_local(source)
        cmd = _ytdlp_cmd() + [
            "--dump-single-json",
            "--no-playlist",
            "--no-warnings",
            source,
        ]
        result = await _run(
            cmd, float(self._config.timeout_download_s), "yt-dlp resolve"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp resolve failed (rc={result.returncode}): "
                f"{_tail(result.stderr)}"
            )
        try:
            info = json.loads(result.stdout.decode(errors="replace"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"yt-dlp returned invalid JSON: {exc}") from exc
        if not isinstance(info, dict):
            raise RuntimeError("yt-dlp JSON payload was not an object")
        return self._meta_from_info(info, source)

    @staticmethod
    def _resolve_local(source: str) -> VideoMeta:
        """Metadata for a local file path (no extractor, no probing here)."""
        path = _local_path(source)
        return VideoMeta(
            canonical_id=_fallback_canonical_id(source),
            title=path.stem or None,
            webpage_url=source,
        )

    @staticmethod
    def _meta_from_info(info: dict[str, object], source: str) -> VideoMeta:
        extractor_key = info.get("extractor_key")
        video_id = info.get("id")
        if (
            isinstance(extractor_key, str)
            and extractor_key
            and isinstance(video_id, str)
            and video_id
        ):
            canonical_id = f"{extractor_key.lower()}:{video_id}"
        else:
            canonical_id = _fallback_canonical_id(source)

        raw_duration = info.get("duration")
        duration = (
            float(raw_duration)
            if isinstance(raw_duration, (int, float))
            and not isinstance(raw_duration, bool)
            else None
        )

        chapters: list[tuple[float, str]] = []
        raw_chapters = info.get("chapters")
        if isinstance(raw_chapters, list):
            for chapter in raw_chapters:
                if not isinstance(chapter, dict):
                    continue
                start = chapter.get("start_time")
                if isinstance(start, bool) or not isinstance(start, (int, float)):
                    continue
                title = chapter.get("title")
                chapters.append((float(start), str(title) if title else ""))

        def _str_field(key: str) -> str | None:
            value = info.get(key)
            return str(value) if isinstance(value, str) and value else None

        return VideoMeta(
            canonical_id=canonical_id,
            title=_str_field("title"),
            description=_str_field("description"),
            duration_s=duration,
            webpage_url=_str_field("webpage_url") or source,
            chapters=tuple(chapters),
        )

    # -------------------------------------------------------------- subtitles

    async def subtitles(self, source: str, canonical_id: str) -> Transcript | None:
        """Fetch + parse subtitles; None when none are available."""
        if not _is_http_url(source):
            log.debug("no subtitle fetch for local source %s", source)
            return None
        sanitized = sanitize_canonical_id(canonical_id)
        cache_dir = self._config.storage_subdir("subtitles") / sanitized
        await _ensure_dir(cache_dir)

        cached = await asyncio.to_thread(self._find_subtitle, cache_dir, sanitized)
        if cached is not None:
            log.debug("subtitle cache hit for %s: %s", canonical_id, cached.name)
            return await asyncio.to_thread(
                self._parse, cached, sanitized, self._config.max_transcript_chars
            )

        template = str(cache_dir / f"{sanitized}.%(ext)s")
        result = await self._subtitle_attempt(source, template, _PREFERRED_SUB_LANGS)
        if await asyncio.to_thread(self._find_subtitle, cache_dir, sanitized) is None:
            log.info(
                "no preferred-language subtitles for %s; retrying with all langs",
                canonical_id,
            )
            result = await self._subtitle_attempt(source, template, _ALL_SUB_LANGS)

        found = await asyncio.to_thread(self._find_subtitle, cache_dir, sanitized)
        if found is None:
            if result is not None and result.returncode != 0:
                log.warning(
                    "subtitle download failed for %s: %s",
                    canonical_id,
                    _tail(result.stderr),
                )
            else:
                log.info("no subtitles available for %s", canonical_id)
            return None
        return await asyncio.to_thread(
            self._parse, found, sanitized, self._config.max_transcript_chars
        )

    async def _subtitle_attempt(
        self, source: str, template: str, langs: str
    ) -> _Result | None:
        """One --skip-download subtitle fetch; None on hard failure."""
        cmd = _ytdlp_cmd() + [
            "--no-playlist",
            "--no-warnings",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            langs,
            "--sub-format",
            _SUB_FORMAT,
            "--output",
            template,
            source,
        ]
        try:
            return await _run(
                cmd, float(self._config.timeout_download_s), "yt-dlp subtitles"
            )
        except RuntimeError as exc:
            log.warning("subtitle attempt failed: %s", exc)
            return None

    @staticmethod
    def _find_subtitle(cache_dir: Path, sanitized: str) -> Path | None:
        """Best cached/downloaded subtitle file: vtt first, en before others."""
        lowered = sanitized.lower()
        candidates: list[Path] = []
        for pattern in (f"{sanitized}*.vtt", f"{sanitized}*.srt"):
            candidates.extend(cache_dir.glob(pattern))

        def rank(path: Path) -> tuple[int, int, float]:
            name = path.name.lower()
            suffix = name[len(lowered) :] if name.startswith(lowered) else name
            lang_rank = 0 if suffix.startswith(".en") else 1
            fmt_rank = 0 if name.endswith(".vtt") else 1
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (lang_rank, fmt_rank, -mtime)

        return min(candidates, key=rank) if candidates else None

    @staticmethod
    def _parse(path: Path, sanitized: str, max_chars: int) -> Transcript:
        transcript = parse_file(path, max_chars)
        name = path.stem
        lang: str | None = None
        if name.startswith(sanitized + "."):
            lang = name[len(sanitized) + 1 :] or None
        return replace(transcript, lang=lang)

    # --------------------------------------------------------------- download

    async def download(self, source: str, canonical_id: str) -> str:
        """Download (or reuse) the merged mp4 for canonical_id; return path."""
        sanitized = sanitize_canonical_id(canonical_id)
        raw_dir = self._config.storage_subdir("raw")
        expected = raw_dir / f"{sanitized}.mp4"
        if await asyncio.to_thread(self._reusable, expected):
            log.debug("raw media already present for %s", canonical_id)
            return str(expected)
        if not _is_http_url(source):
            path = _local_path(source)
            if not await asyncio.to_thread(path.is_file):
                raise FileNotFoundError(f"source file not found: {source}")
            return str(path)

        await _ensure_dir(raw_dir)
        cmd = _ytdlp_cmd() + [
            "--no-playlist",
            "--no-warnings",
            "-f",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
        ]
        ffmpeg_path = Path(await asyncio.to_thread(binaries.resolve_ffmpeg))
        if await asyncio.to_thread(ffmpeg_path.is_file):
            cmd += ["--ffmpeg-location", str(ffmpeg_path.parent)]
        cmd += [
            "--output",
            str(raw_dir / f"{sanitized}.%(ext)s"),
            "--print",
            "after_move:filepath:%(filepath)s",
            source,
        ]
        result = await _run(
            cmd, float(self._config.timeout_download_s), "yt-dlp download"
        )

        printed = await asyncio.to_thread(self._printed_paths, result.stdout, raw_dir)
        if printed is not None:
            return str(printed)
        fallback = await asyncio.to_thread(self._glob_fallback, raw_dir, sanitized)
        if fallback is not None:
            if result.returncode != 0:
                log.warning(
                    "yt-dlp returned rc=%s but media exists: %s",
                    result.returncode,
                    _tail(result.stderr)[:500],
                )
            return str(fallback)
        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp download failed (rc={result.returncode}): "
                f"{_tail(result.stderr)}"
            )
        raise RuntimeError("yt-dlp completed but no media file was found")

    @staticmethod
    def _reusable(expected: Path) -> bool:
        try:
            return expected.is_file() and expected.stat().st_size > _MIN_REUSE_BYTES
        except OSError:
            return False

    @staticmethod
    def _printed_paths(stdout: bytes, raw_dir: Path) -> Path | None:
        """First existing file yt-dlp printed via after_move:filepath."""
        for raw_line in stdout.decode(errors="replace").splitlines():
            line = raw_line.strip()
            if not line.startswith("filepath:"):
                continue
            candidate = Path(line.removeprefix("filepath:").strip())
            if not candidate.is_absolute():
                candidate = raw_dir / candidate
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _glob_fallback(raw_dir: Path, sanitized: str) -> Path | None:
        """Largest raw/<sanitized>.* artifact; exact .mp4 preferred."""
        try:
            matches = [p for p in raw_dir.glob(f"{sanitized}.*") if p.is_file()]
        except OSError:
            return None
        if not matches:
            return None
        exact = raw_dir / f"{sanitized}.mp4"
        if exact in matches:
            return exact
        return max(matches, key=lambda p: p.stat().st_size)
