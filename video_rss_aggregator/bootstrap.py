from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

from core_config import Config
from service_summarize import SummarizationEngine
from video_rss_aggregator.application.use_cases.ingest_feed import IngestFeed
from video_rss_aggregator.application.use_cases.process_source import ProcessSource
from video_rss_aggregator.application.use_cases.render_rss_feed import RenderRssFeed
from video_rss_aggregator.application.use_cases.runtime import (
    BootstrapRuntime,
    GetRuntimeStatus,
)
from video_rss_aggregator.infrastructure.feed_source import HttpFeedSource
from video_rss_aggregator.infrastructure.media_service import (
    LegacyMediaPreparationService,
)
from video_rss_aggregator.infrastructure.publication_renderer import (
    RssPublicationRenderer,
)
from video_rss_aggregator.infrastructure.runtime_adapters import LegacyRuntimeInspector
from video_rss_aggregator.infrastructure.sqlite_repositories import (
    SQLiteFeedRepository,
    SQLiteFeedVideoRepository,
    SQLitePublicationRepository,
    SQLiteSummaryRepository,
    SQLiteVideoRepository,
)
from video_rss_aggregator.infrastructure.summarizer import LegacySummarizer
from video_rss_aggregator.storage import Database


@dataclass(frozen=True)
class AppUseCases:
    get_runtime_status: GetRuntimeStatus
    bootstrap_runtime: BootstrapRuntime
    ingest_feed: IngestFeed
    process_source: ProcessSource
    render_rss_feed: RenderRssFeed


@dataclass(frozen=True)
class AppRuntime:
    config: Config
    use_cases: AppUseCases
    close: Callable[[], Awaitable[None] | None]


async def build_runtime(config: Config | None = None) -> AppRuntime:
    resolved_config = config or Config.from_env()
    database = await Database.connect(resolved_config.database_path)
    await database.migrate()
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=300.0, write=300.0, pool=60.0),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )
    summarization_engine = SummarizationEngine(resolved_config)

    runtime_inspector = LegacyRuntimeInspector(summarization_engine)
    process_source = ProcessSource(
        media_service=LegacyMediaPreparationService(
            client=client,
            storage_dir=resolved_config.storage_dir,
            max_frames=resolved_config.max_frames,
            scene_detection=resolved_config.frame_scene_detection,
            scene_threshold=resolved_config.frame_scene_threshold,
            scene_min_frames=resolved_config.frame_scene_min_frames,
            max_transcript_chars=resolved_config.max_transcript_chars,
        ),
        summarizer=LegacySummarizer(summarization_engine),
        videos=SQLiteVideoRepository(database),
        summaries=SQLiteSummaryRepository(database),
    )

    use_cases = AppUseCases(
        get_runtime_status=GetRuntimeStatus(
            runtime=runtime_inspector,
            storage_path=database.path,
            storage_dir=resolved_config.storage_dir,
            models=resolved_config.model_priority,
        ),
        bootstrap_runtime=BootstrapRuntime(runtime=runtime_inspector),
        ingest_feed=IngestFeed(
            feed_source=HttpFeedSource(client),
            feeds=SQLiteFeedRepository(database),
            videos=SQLiteFeedVideoRepository(database),
            process_source=process_source,
        ),
        process_source=process_source,
        render_rss_feed=RenderRssFeed(
            publications=SQLitePublicationRepository(database),
            renderer=RssPublicationRenderer(
                title=resolved_config.rss_title,
                link=resolved_config.rss_link,
                description=resolved_config.rss_description,
            ),
        ),
    )

    async def _close() -> None:
        await client.aclose()
        await summarization_engine.close()
        await database.close()

    return AppRuntime(config=resolved_config, use_cases=use_cases, close=_close)
