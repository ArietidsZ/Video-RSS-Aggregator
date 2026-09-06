"""Click CLI for VRA v2. OWNER: agent-D.

Commands (docs/v2/SPEC.md §CLI): serve | poll-now | add-feed URL | list |
doctor | engine (alias of doctor).

- serve builds the runtime via vra.runtime.build_runtime (composition root,
  lead-owned) and hands it to vra.api.app.create_app; uvicorn runs the app's
  lifespan, which owns runtime.start()/close(). Every other command builds a
  runtime the same way and closes it before exiting (no bare network use).
- doctor / engine is a config-driven doctor, NOT a launcher: it reads Config
  (engine_base_url, effective_asr_base_url, model, asr_model), probes the
  OpenAI-compatible /models endpoints and prints a posture report. It exits
  non-zero when the engine is unreachable. Zero runtime-specific flags, zero
  model download logic, zero launch recipes — the launch recipe lands in the
  model phase (docs/v2/RESEARCH.md S7.7) as docs + script, not Python.
- All command output is JSON; failures raise ClickException (exit != 0).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import click
import httpx

from vra.config import Config
from vra.types import JobKind

log = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 2.0
_T = TypeVar("_T")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _print(payload: object) -> None:
    click.echo(json.dumps(payload, indent=2, default=str))


def _parse_bind(bind: str) -> tuple[str, int]:
    """host:port -> (host, port); missing/invalid port keeps the default."""
    host, sep, port = bind.rpartition(":")
    if not sep or not host:
        return bind, 8080
    with contextlib.suppress(ValueError):
        return host, int(port)
    return host, 8080


async def _with_runtime(config: Config, run: Callable[[Any], Awaitable[_T]]) -> _T:
    """build_runtime -> await run(runtime) -> always close (lead-owned root)."""
    from vra.runtime import build_runtime  # lazy: store/service slices may lag

    runtime = await build_runtime(config)
    try:
        return await run(runtime)
    finally:
        await runtime.close()


@click.group()
def cli() -> None:
    """Video RSS Aggregator — summarize video feeds with a local vision model."""
    _setup_logging()


@cli.command()
@click.option("--bind", default=None, help="host:port override (e.g. 0.0.0.0:8080).")
def serve(bind: str | None) -> None:
    """Start the API server + queue workers + poll scheduler."""
    import uvicorn

    from vra.api.app import create_app

    config = Config.from_env()
    host, port = config.bind_host, config.bind_port
    if bind:
        host, port = _parse_bind(bind)

    async def _run() -> None:
        from vra.runtime import build_runtime

        runtime = await build_runtime(config)
        app = create_app(runtime)
        server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="info")
        )
        await server.serve()

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
def poll_now() -> None:
    """Enqueue an INGEST job for every known feed (poll all feeds now)."""
    config = Config.from_env()

    async def _run(runtime: Any) -> dict[str, object]:
        feeds = await runtime.store.list_feeds()
        job_ids: list[str] = []
        for feed in feeds:
            job = await runtime.queue.enqueue(JobKind.INGEST, {"url": feed.url})
            job_ids.append(job.id)
        return {"feeds": len(feeds), "enqueued": len(job_ids), "job_ids": job_ids}

    try:
        _print(asyncio.run(_with_runtime(config, _run)))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.argument("url")
@click.option("--poll-minutes", type=int, default=None, help="Per-feed poll interval.")
def add_feed(url: str, poll_minutes: int | None) -> None:
    """Register a feed and enqueue its first ingest."""
    from vra.api.app import _record_dict

    config = Config.from_env()

    async def _run(runtime: Any) -> dict[str, object]:
        record = await runtime.store.upsert_feed(
            url, None, poll_minutes or config.poll_minutes
        )
        job = await runtime.queue.enqueue(JobKind.INGEST, {"url": url})
        return {**_record_dict(record), "job_id": job.id}

    try:
        _print(asyncio.run(_with_runtime(config, _run)))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--limit", type=int, default=20, show_default=True, help="Videos listed.")
def list_cmd(limit: int) -> None:
    """Print known feeds and recent videos as JSON."""
    from vra.api.app import _record_dict

    config = Config.from_env()

    async def _run(runtime: Any) -> dict[str, object]:
        feeds = [_record_dict(record) for record in await runtime.store.list_feeds()]
        videos = await runtime.store.latest_publications(max(limit, 1))
        return {"feeds": feeds, "videos": videos}

    try:
        _print(asyncio.run(_with_runtime(config, _run)))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


# register the documented name `list` (shadowing the builtin only inside click)
cli.add_command(list_cmd, name="list")


async def _probe_models(base_url: str, api_key: str | None) -> dict[str, Any]:
    """GET {base_url}/models with a 2s budget; best-effort, never raises."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_PROBE_TIMEOUT_S)) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
        models = [
            str(item.get("id"))
            for item in (data.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        return {"base_url": base_url, "reachable": True, "models": models, "error": None}
    except Exception as exc:
        return {"base_url": base_url, "reachable": False, "models": [], "error": str(exc)}


async def _doctor_report(config: Config) -> dict[str, Any]:
    engine = await _probe_models(config.engine_base_url, config.engine_api_key)
    asr = await _probe_models(config.effective_asr_base_url, config.engine_api_key)
    engine["model"] = config.model
    engine["model_present"] = config.model.lower() in [
        str(m).lower() for m in engine["models"]
    ]
    asr["model"] = config.asr_model
    asr["model_present"] = config.asr_model.lower() in [
        str(m).lower() for m in asr["models"]
    ]
    return {
        "engine": engine,
        "asr": {
            "base_url": asr["base_url"],
            "reachable": asr["reachable"],
            "models": asr["models"],
            "model": asr["model"],
            "model_present": asr["model_present"],
            "error": asr["error"],
        },
        "ok": bool(engine["reachable"]),
    }


def _doctor() -> None:
    """Shared body of `vra doctor` and `vra engine`."""
    config = Config.from_env()
    try:
        report = asyncio.run(_doctor_report(config))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    _print(report)
    if not report["ok"]:
        ctx = click.get_current_context()
        ctx.exit(1)


@cli.command()
def doctor() -> None:
    """Probe the configured engine/ASR servers and print a posture report."""
    _doctor()


@cli.command()
def engine() -> None:
    """Alias of `doctor`: verify the configured engine (never launches one)."""
    _doctor()


if __name__ == "__main__":
    cli()
