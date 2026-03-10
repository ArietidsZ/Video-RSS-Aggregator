from __future__ import annotations

from core_config import Config
from video_rss_aggregator.api import IngestRequest, ProcessRequest
from video_rss_aggregator.api import create_app as create_runtime_app
from video_rss_aggregator.bootstrap import AppRuntime, AppUseCases, build_runtime


def create_app(runtime: AppRuntime | None = None, config: Config | None = None):
    if runtime is not None and not isinstance(runtime, AppRuntime):
        raise TypeError("create_app expects an AppRuntime or None")

    return create_runtime_app(runtime=runtime, config=config)


__all__ = [
    "AppRuntime",
    "AppUseCases",
    "IngestRequest",
    "ProcessRequest",
    "build_runtime",
    "create_app",
]
