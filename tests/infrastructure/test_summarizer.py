from pathlib import Path

import pytest

from service_summarize import SummaryResult as LegacySummaryResult
from video_rss_aggregator.domain.models import PreparedMedia
from video_rss_aggregator.infrastructure.summarizer import LegacySummarizer


class FakeSummarizationEngine:
    def __init__(self, result: LegacySummaryResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def summarize(self, **kwargs) -> LegacySummaryResult:
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_legacy_summarizer_maps_domain_media_to_legacy_engine() -> None:
    engine = FakeSummarizationEngine(
        LegacySummaryResult(
            summary="Concise summary",
            key_points=["One", "Two"],
            visual_highlights=["Frame one"],
            model_used="qwen",
            vram_mb=512.0,
            transcript_chars=123,
            frame_count=2,
            error=None,
        )
    )
    adapter = LegacySummarizer(engine)
    prepared = PreparedMedia(
        source_url="https://example.com/watch?v=1",
        title="Example title",
        transcript="Transcript body",
        media_path="/tmp/video.mp4",
        frame_paths=("/tmp/frame-1.jpg", "/tmp/frame-2.jpg"),
    )

    result = await adapter.summarize(prepared)

    assert engine.calls == [
        {
            "source_url": "https://example.com/watch?v=1",
            "title": "Example title",
            "transcript": "Transcript body",
            "frame_paths": [Path("/tmp/frame-1.jpg"), Path("/tmp/frame-2.jpg")],
        }
    ]
    assert result.summary == "Concise summary"
    assert result.key_points == ("One", "Two")
    assert result.visual_highlights == ("Frame one",)
    assert result.model_used == "qwen"
    assert result.vram_mb == 512.0
    assert result.transcript_chars == 123
    assert result.frame_count == 2
    assert result.error is None
