from __future__ import annotations

import asyncio
import logging
import sys

import click
import uvicorn

from .config import Config

log = logging.getLogger(__name__)


@click.group()
def cli():
    """Video RSS Aggregator — Qwen3 intelligent video summarization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@cli.command()
@click.option("--bind", default=None, help="host:port override")
def serve(bind: str | None):
    """Start the HTTP API server."""
    config = Config.from_env()
    if bind:
        parts = bind.rsplit(":", 1)
        host = parts[0] if len(parts) == 2 else config.bind_host
        port = int(parts[1]) if len(parts) == 2 else config.bind_port
    else:
        host, port = config.bind_host, config.bind_port

    async def _run():
        from .api import create_app
        from .pipeline import Pipeline

        pipeline = await Pipeline.create(config)
        app = create_app(pipeline, config.api_key)
        server_config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(server_config)
        await server.serve()

    asyncio.run(_run())


@cli.command()
@click.option("--feed-url", envvar="VRA_VERIFY_FEED_URL", required=True)
@click.option("--source", envvar="VRA_VERIFY_SOURCE", required=True, help="Audio path or URL")
def verify(feed_url: str, source: str):
    """Run end-to-end verification with real data."""
    import json
    import time

    config = Config.from_env()

    async def _run():
        from .pipeline import Pipeline

        pipeline = await Pipeline.create(config)

        t0 = time.monotonic()

        t1 = time.monotonic()
        feed_report = await pipeline.ingest_feed(feed_url, process=False, max_items=5)
        feed_ms = int((time.monotonic() - t1) * 1000)

        if feed_report.item_count == 0:
            log.error("Feed returned no entries")
            sys.exit(1)

        t2 = time.monotonic()
        proc_report = await pipeline.process_source(source)
        process_ms = int((time.monotonic() - t2) * 1000)

        t3 = time.monotonic()
        rss = await pipeline.rss_feed("Verification", feed_url, "test", 10)
        rss_ms = int((time.monotonic() - t3) * 1000)

        total_ms = int((time.monotonic() - t0) * 1000)

        report = {
            "feed_url": feed_url,
            "feed_items": feed_report.item_count,
            "processed_source": source,
            "transcription_chars": len(proc_report.transcription.text) if proc_report.transcription else 0,
            "summary_chars": len(proc_report.summary.summary) if proc_report.summary else 0,
            "total_ms": total_ms,
            "feed_ms": feed_ms,
            "process_ms": process_ms,
            "rss_ms": rss_ms,
        }
        print(json.dumps(report, indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    cli()
