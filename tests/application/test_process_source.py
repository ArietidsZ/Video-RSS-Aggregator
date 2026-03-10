import pytest

from video_rss_aggregator.application.use_cases.process_source import ProcessSource
from video_rss_aggregator.domain.models import PreparedMedia, SummaryResult
from video_rss_aggregator.domain.outcomes import Failure, PartialSuccess, Success


class FakeMediaPreparationService:
    async def prepare(self, source_url: str, title: str | None) -> PreparedMedia:
        return PreparedMedia(
            source_url=source_url,
            title=title or "Prepared title",
            transcript="transcript",
            media_path="/tmp/video.mp4",
            frame_paths=("/tmp/frame-1.jpg",),
        )


class FakeSummarizer:
    def __init__(self, result: SummaryResult) -> None:
        self.result = result

    async def summarize(self, prepared_media: PreparedMedia) -> SummaryResult:
        return self.result


class RaisingMediaPreparationService:
    async def prepare(self, source_url: str, title: str | None) -> PreparedMedia:
        raise RuntimeError("download failed")


class FakeVideoRepository:
    def __init__(self) -> None:
        self.saved: list[PreparedMedia] = []

    async def save(self, media: PreparedMedia) -> str:
        self.saved.append(media)
        return "video-123"


class FakeSummaryRepository:
    def __init__(self) -> None:
        self.saved: list[tuple[str, SummaryResult]] = []

    async def save(self, video_id: str, summary: SummaryResult) -> None:
        self.saved.append((video_id, summary))


class RaisingVideoRepository:
    async def save(self, media: PreparedMedia) -> str:
        raise RuntimeError("database write failed")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def build_summary(*, error: str | None = None) -> SummaryResult:
    return SummaryResult(
        summary="Summarized output",
        key_points=("point",),
        visual_highlights=("highlight",),
        model_used="qwen",
        vram_mb=1024.0,
        transcript_chars=10,
        frame_count=1,
        error=error,
    )


@pytest.mark.anyio
async def test_process_source_returns_success_for_valid_media() -> None:
    videos = FakeVideoRepository()
    summaries = FakeSummaryRepository()
    use_case = ProcessSource(
        media_service=FakeMediaPreparationService(),
        summarizer=FakeSummarizer(build_summary()),
        videos=videos,
        summaries=summaries,
    )

    outcome = await use_case.execute("https://example.com/video", None)

    assert isinstance(outcome, Success)
    assert outcome.summary.summary == "Summarized output"
    assert len(videos.saved) == 1
    assert len(summaries.saved) == 1
    assert summaries.saved[0][0] == "video-123"


@pytest.mark.anyio
async def test_process_source_returns_partial_success_when_summary_has_error() -> None:
    use_case = ProcessSource(
        media_service=FakeMediaPreparationService(),
        summarizer=FakeSummarizer(build_summary(error="model degraded")),
        videos=FakeVideoRepository(),
        summaries=FakeSummaryRepository(),
    )

    outcome = await use_case.execute("https://example.com/video", "Example")

    assert isinstance(outcome, PartialSuccess)
    assert outcome.reason == "model degraded"
    assert outcome.summary.error == "model degraded"


@pytest.mark.anyio
async def test_process_source_returns_failure_when_media_preparation_raises() -> None:
    use_case = ProcessSource(
        media_service=RaisingMediaPreparationService(),
        summarizer=FakeSummarizer(build_summary()),
        videos=FakeVideoRepository(),
        summaries=FakeSummaryRepository(),
    )

    outcome = await use_case.execute("https://example.com/video", None)

    assert outcome == Failure(
        source_url="https://example.com/video",
        reason="download failed",
    )


@pytest.mark.anyio
async def test_process_source_returns_failure_when_video_persistence_raises() -> None:
    use_case = ProcessSource(
        media_service=FakeMediaPreparationService(),
        summarizer=FakeSummarizer(build_summary()),
        videos=RaisingVideoRepository(),
        summaries=FakeSummaryRepository(),
    )

    outcome = await use_case.execute("https://example.com/video", "Example")

    assert outcome == Failure(
        source_url="https://example.com/video",
        reason="database write failed",
    )
