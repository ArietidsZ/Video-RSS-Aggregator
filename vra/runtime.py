"""Composition root (LEAD-owned). Agents code against the AppRuntime facade.

AppRuntime attributes every consumer may rely on:
    config:     vra.config.Config
    store:      contracts.VideoStore
    queue:      contracts.JobQueue
    ingester:   contracts.IngestService
    processor:  contracts.SourceProcessor
    scheduler:  contracts.FeedScheduler
    summarizer: contracts.Summarizer (runtime_status/prepare_models used by setup)
    start() -> coroutine: starts scheduler + queue workers (cpu/gpu pools)
    close() -> coroutine: stops everything, closes db/http
    worker stats: runtime.stats() -> {"queue_depth": int, ...}

Wiring (build_runtime(config)):
    binaries -> YtDlpAcquirer -> Processor; OllamaClient -> Governor ->
    VlmSummarizer; Database -> SqlVideoStore/SqlJobQueue; HttpFeedFetcher;
    PollScheduler; workers: cpu_workers x process-consumer, gpu bound inside
    summarizer semaphore. kind=ingest jobs dispatch to ingester.ingest(url).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vra.config import Config
from vra.contracts import (
    FeedScheduler,
    IngestService,
    JobQueue,
    SourceProcessor,
    Summarizer,
    VideoStore,
)


@dataclass(frozen=True)
class AppRuntime:
    config: Config
    store: VideoStore
    queue: JobQueue
    ingester: IngestService
    processor: SourceProcessor
    scheduler: FeedScheduler
    summarizer: Summarizer

    async def start(self) -> None:
        raise NotImplementedError("lead wires at integration")

    async def close(self) -> None:
        raise NotImplementedError("lead wires at integration")

    def stats(self) -> dict[str, Any]:
        return {}


async def build_runtime(config: Config | None = None) -> AppRuntime:
    raise NotImplementedError("lead wires at integration")
