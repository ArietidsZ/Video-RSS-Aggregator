from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from .pipeline import Pipeline


class IngestRequest(BaseModel):
    feed_url: str
    process: bool = False
    max_items: int | None = None


class ProcessRequest(BaseModel):
    source_url: str
    title: str | None = None


def create_app(pipeline: Pipeline, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="Video RSS Aggregator", version="0.1.0")

    rss_title = os.environ.get("VRA_RSS_TITLE", "Video RSS Aggregator")
    rss_link = os.environ.get("VRA_RSS_LINK", "http://localhost:8080/rss")
    rss_desc = os.environ.get("VRA_RSS_DESCRIPTION", "Video summaries")

    def _check_auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
        if api_key is None:
            return
        token = None
        if authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        if token is None:
            token = x_api_key
        if token != api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.post("/ingest")
    async def ingest(req: IngestRequest, _=Depends(_check_auth)):
        report = await pipeline.ingest_feed(req.feed_url, req.process, req.max_items)
        return asdict(report)

    @app.post("/process")
    async def process(req: ProcessRequest, _=Depends(_check_auth)):
        report = await pipeline.process_source(req.source_url, req.title)
        return {
            "source_url": report.source_url,
            "title": report.title,
            "transcription": asdict(report.transcription) if report.transcription else None,
            "summary": asdict(report.summary) if report.summary else None,
        }

    @app.get("/rss")
    async def rss_feed(limit: int = Query(20, ge=1, le=200)):
        xml = await pipeline.rss_feed(rss_title, rss_link, rss_desc, limit)
        return Response(content=xml, media_type="application/rss+xml")

    return app
