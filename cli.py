from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path
from time import monotonic

import click
import httpx
import uvicorn

from core_config import Config
from service_media import extract_frames_ffmpeg, prepare_source
from service_summarize import SummarizationEngine
from video_rss_aggregator.bootstrap import AppRuntime, build_runtime
from video_rss_aggregator.domain.outcomes import Failure

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _digest_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 16)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _frame_metrics(paths: list[Path]) -> dict[str, float | int]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            seen.add(_digest_file(path))
        except OSError:
            continue

    frame_count = len(paths)
    unique_frames = len(seen)
    duplicates = max(frame_count - unique_frames, 0)
    unique_ratio = round(unique_frames / frame_count, 3) if frame_count else 0.0
    return {
        "frame_count": frame_count,
        "unique_frames": unique_frames,
        "duplicate_frames": duplicates,
        "unique_ratio": unique_ratio,
    }


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


async def _close_runtime(runtime: AppRuntime) -> None:
    close_result = runtime.close()
    if close_result is not None:
        await close_result


@click.group()
def cli() -> None:
    """Video RSS Aggregator powered by Qwen3.5 vision models."""
    _setup_logging()


@cli.command()
def bootstrap() -> None:
    """Validate Ollama connectivity and pull configured models."""
    config = Config.from_env()

    async def _run() -> None:
        runtime = await build_runtime(config)
        try:
            report = await runtime.use_cases.bootstrap_runtime.execute()
            models_prepared = report.get("models_prepared")
            if models_prepared is None:
                models_prepared = report["models"]

            runtime_payload = report.get("runtime")
            if isinstance(runtime_payload, dict):
                runtime_response = runtime_payload
            else:
                runtime_response = await runtime.use_cases.get_runtime_status.execute()

            print(
                json.dumps(
                    {
                        "models_prepared": models_prepared,
                        "runtime": runtime_response,
                    },
                    indent=2,
                )
            )
        finally:
            await _close_runtime(runtime)

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--bind", default=None, help="host:port override")
def serve(bind: str | None) -> None:
    """Start the FastAPI server."""
    config = Config.from_env()
    host, port = config.bind_host, config.bind_port

    if bind:
        host_part, sep, port_part = bind.rpartition(":")
        if sep and host_part:
            host = host_part
            try:
                port = int(port_part)
            except ValueError:
                pass

    async def _run() -> None:
        from adapter_api import create_app

        app = create_app(config=config)
        uv_cfg = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(uv_cfg)
        await server.serve()

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
def status() -> None:
    """Print runtime status including VRAM usage."""
    config = Config.from_env()

    async def _run() -> None:
        runtime = await build_runtime(config)
        try:
            status_payload = await runtime.use_cases.get_runtime_status.execute()
            print(json.dumps(status_payload, indent=2))
        finally:
            await _close_runtime(runtime)

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--source", envvar="VRA_VERIFY_SOURCE", required=True, help="URL or file")
@click.option("--title", envvar="VRA_VERIFY_TITLE", default=None)
def verify(source: str, title: str | None) -> None:
    """Run one end-to-end processing job and print metrics."""
    config = Config.from_env()

    async def _run() -> None:
        runtime = await build_runtime(config)
        try:
            t0 = monotonic()
            outcome = await runtime.use_cases.process_source.execute(source, title)
            if isinstance(outcome, Failure):
                raise click.ClickException(outcome.reason)

            total_ms = int((monotonic() - t0) * 1000)

            print(
                json.dumps(
                    {
                        "source_url": outcome.media.source_url,
                        "title": outcome.media.title,
                        "transcript_chars": outcome.summary.transcript_chars,
                        "frame_count": outcome.summary.frame_count,
                        "model_used": outcome.summary.model_used,
                        "vram_mb": outcome.summary.vram_mb,
                        "error": outcome.summary.error,
                        "summary_chars": len(outcome.summary.summary),
                        "key_points": len(outcome.summary.key_points),
                        "total_ms": total_ms,
                    },
                    indent=2,
                )
            )
        finally:
            await _close_runtime(runtime)

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--source", envvar="VRA_BENCH_SOURCE", required=True, help="URL or file")
@click.option("--title", envvar="VRA_BENCH_TITLE", default=None)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Optional override for frame count during benchmark",
)
@click.option(
    "--with-summary/--no-summary",
    default=True,
    show_default=True,
    help="Run model summarization for both extraction strategies",
)
def benchmark(
    source: str,
    title: str | None,
    max_frames: int | None,
    with_summary: bool,
) -> None:
    """Compare scene-aware and uniform frame extraction on one source."""
    config = Config.from_env()
    if max_frames is not None:
        config = replace(config, max_frames=max(max_frames, 1))

    async def _run_mode(
        *,
        mode: str,
        source_url: str,
        media_path: Path,
        resolved_title: str | None,
        transcript: str,
        scene_detection: bool,
        summarizer: SummarizationEngine | None,
        frame_output_dir: Path,
        cfg: Config,
    ) -> dict[str, object]:
        started = monotonic()
        try:
            frames = await extract_frames_ffmpeg(
                input_path=media_path,
                output_dir=frame_output_dir,
                file_id=uuid.uuid4(),
                max_frames=cfg.max_frames,
                scene_detection=scene_detection,
                scene_threshold=cfg.frame_scene_threshold,
                min_scene_frames=cfg.frame_scene_min_frames,
            )
        except Exception as exc:
            return {
                "mode": mode,
                "error": str(exc),
                "extraction_ms": int((monotonic() - started) * 1000),
            }

        result: dict[str, object] = {
            "mode": mode,
            "extraction_ms": int((monotonic() - started) * 1000),
            "frames": [str(path) for path in frames],
            "scene_frame_count": sum(1 for path in frames if "_scene_" in path.name),
            "uniform_frame_count": sum(
                1 for path in frames if "_uniform_" in path.name
            ),
            **_frame_metrics(frames),
        }

        if summarizer is None:
            return result

        summary_started = monotonic()
        summary = await summarizer.summarize(
            source_url=source_url,
            title=resolved_title,
            transcript=transcript,
            frame_paths=frames,
        )
        result["summary"] = {
            "latency_ms": int((monotonic() - summary_started) * 1000),
            "summary_chars": len(summary.summary),
            "key_points": len(summary.key_points),
            "visual_highlights": len(summary.visual_highlights),
            "model_used": summary.model_used,
            "vram_mb": summary.vram_mb,
            "error": summary.error,
        }
        return result

    async def _run() -> None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=300.0, pool=60.0),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
        summarizer: SummarizationEngine | None = None

        try:
            prepared = await prepare_source(
                client=client,
                source=source,
                storage_dir=config.storage_dir,
                max_transcript_chars=config.max_transcript_chars,
            )
            resolved_title = title or prepared.title

            if with_summary:
                summarizer = SummarizationEngine(config)
                await summarizer.prepare_models()

            frame_dir = Path(config.storage_dir) / "frames"

            scene_aware = await _run_mode(
                mode="scene_aware",
                source_url=source,
                media_path=prepared.media_path,
                resolved_title=resolved_title,
                transcript=prepared.transcript,
                scene_detection=True,
                summarizer=summarizer,
                frame_output_dir=frame_dir,
                cfg=config,
            )
            uniform_only = await _run_mode(
                mode="uniform_only",
                source_url=source,
                media_path=prepared.media_path,
                resolved_title=resolved_title,
                transcript=prepared.transcript,
                scene_detection=False,
                summarizer=summarizer,
                frame_output_dir=frame_dir,
                cfg=config,
            )

            comparison: dict[str, object] = {
                "extraction_delta_ms": int(
                    _as_int(scene_aware.get("extraction_ms", 0))
                    - _as_int(uniform_only.get("extraction_ms", 0))
                ),
                "unique_ratio_delta": round(
                    _as_float(scene_aware.get("unique_ratio", 0.0))
                    - _as_float(uniform_only.get("unique_ratio", 0.0)),
                    3,
                ),
            }

            scene_summary = scene_aware.get("summary")
            uniform_summary = uniform_only.get("summary")
            if isinstance(scene_summary, dict) and isinstance(uniform_summary, dict):
                comparison.update(
                    {
                        "summary_latency_delta_ms": int(
                            scene_summary.get("latency_ms", 0)
                            - uniform_summary.get("latency_ms", 0)
                        ),
                        "summary_chars_delta": int(
                            scene_summary.get("summary_chars", 0)
                            - uniform_summary.get("summary_chars", 0)
                        ),
                        "key_points_delta": int(
                            scene_summary.get("key_points", 0)
                            - uniform_summary.get("key_points", 0)
                        ),
                        "visual_highlights_delta": int(
                            scene_summary.get("visual_highlights", 0)
                            - uniform_summary.get("visual_highlights", 0)
                        ),
                    }
                )

            print(
                json.dumps(
                    {
                        "source_url": source,
                        "title": resolved_title,
                        "transcript_chars": len(prepared.transcript),
                        "max_frames": config.max_frames,
                        "scene_threshold": config.frame_scene_threshold,
                        "scene_min_frames": config.frame_scene_min_frames,
                        "scene_aware": scene_aware,
                        "uniform_only": uniform_only,
                        "comparison": comparison,
                    },
                    indent=2,
                )
            )
        finally:
            await client.aclose()
            if summarizer is not None:
                await summarizer.close()

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
