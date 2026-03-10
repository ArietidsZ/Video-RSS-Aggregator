from datetime import datetime, timezone

import pytest

from adapter_rss import render_feed
from adapter_storage import Database, SummaryRecord
from service_summarize import SummaryResult as LegacySummaryResult
from video_rss_aggregator.domain.models import SummaryResult as DomainSummaryResult
from video_rss_aggregator.storage import Database as PackageDatabase


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def build_legacy_summary(*, error: str | None = None) -> LegacySummaryResult:
    return LegacySummaryResult(
        summary="Legacy summary",
        key_points=["Point one", "Point two"],
        visual_highlights=["Blue slide"],
        model_used="qwen",
        vram_mb=256.0,
        transcript_chars=42,
        frame_count=2,
        error=error,
    )


@pytest.mark.anyio
async def test_legacy_database_insert_summary_adapts_legacy_summary_result(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_insert_summary(
        self, video_id: str, result: DomainSummaryResult
    ) -> str:
        captured["video_id"] = video_id
        captured["result"] = result
        return "summary-id"

    monkeypatch.setattr(PackageDatabase, "insert_summary", fake_insert_summary)
    db = await Database.connect(":memory:")

    summary_id = await db.insert_summary("video-123", build_legacy_summary())

    await db.close()

    assert summary_id == "summary-id"
    assert captured["video_id"] == "video-123"
    assert captured["result"] == DomainSummaryResult(
        summary="Legacy summary",
        key_points=("Point one", "Point two"),
        visual_highlights=("Blue slide",),
        model_used="qwen",
        vram_mb=256.0,
        transcript_chars=42,
        frame_count=2,
        error=None,
    )


@pytest.mark.anyio
async def test_legacy_database_persists_legacy_summary_through_shim(tmp_path) -> None:
    db = await Database.connect(str(tmp_path / "rss.db"))
    await db.migrate()
    video_id = await db.upsert_video(
        feed_id=None,
        guid=None,
        title="Legacy title",
        source_url="https://example.com/watch?v=legacy",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    await db.insert_summary(video_id, build_legacy_summary())
    publications = await db.latest_summaries(limit=1)

    await db.close()

    assert publications == [
        SummaryRecord(
            title="Legacy title",
            source_url="https://example.com/watch?v=legacy",
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            summary="Legacy summary",
            key_points=("Point one", "Point two"),
            visual_highlights=("Blue slide",),
            model_used="qwen",
            vram_mb=256.0,
        )
    ]


def test_legacy_rss_render_feed_accepts_summary_record_shim() -> None:
    xml = render_feed(
        title="Feed title",
        link="https://example.com",
        description="Feed description",
        publications=[
            SummaryRecord(
                title="Legacy title",
                source_url="https://example.com/watch?v=legacy",
                published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                summary="Legacy summary",
                key_points=("Point one",),
                visual_highlights=("Blue slide",),
                model_used="qwen",
                vram_mb=256.0,
            )
        ],
    )

    assert "Legacy title" in xml
    assert "Legacy summary" in xml
