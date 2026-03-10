from datetime import datetime, timezone
from typing import get_type_hints

import pytest

from video_rss_aggregator.application.ports import PublicationRepository
from video_rss_aggregator.application.use_cases.render_rss_feed import RenderRssFeed
from video_rss_aggregator.domain.publication import PublicationRecord


class FakePublicationRepository:
    async def latest_publications(self, limit: int) -> list[PublicationRecord]:
        assert limit == 20
        return [
            PublicationRecord(
                title="Video",
                source_url="https://example.com/watch?v=1",
                published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                summary="A compact summary",
                key_points=("Point one", "Point two"),
                visual_highlights=("Blue slide",),
                model_used="qwen",
                vram_mb=1234.5,
            )
        ]


class FakePublicationRenderer:
    def __init__(self) -> None:
        self.received: tuple[PublicationRecord, ...] = ()

    async def render(self, publications: tuple[PublicationRecord, ...]) -> str:
        self.received = publications
        return f"<rss>{publications[0].summary}</rss>"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_render_rss_feed_uses_publication_records() -> None:
    renderer = FakePublicationRenderer()

    xml = await RenderRssFeed(
        publications=FakePublicationRepository(),
        renderer=renderer,
    ).execute(limit=20)

    assert "A compact summary" in xml
    assert renderer.received[0].title == "Video"


def test_publication_record_allows_nullable_source_fields_in_annotations() -> None:
    hints = get_type_hints(PublicationRecord)

    assert hints["title"] == str | None
    assert hints["published_at"] == datetime | None
    assert hints["model_used"] == str | None


def test_publication_repository_exposes_latest_publications_read_model() -> None:
    assert hasattr(PublicationRepository, "latest_publications")
