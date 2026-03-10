from datetime import datetime, timezone

import pytest

from video_rss_aggregator.domain.publication import PublicationRecord
from video_rss_aggregator.infrastructure.publication_renderer import (
    RssPublicationRenderer,
)
from video_rss_aggregator.rss import render_feed


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def build_publication() -> PublicationRecord:
    return PublicationRecord(
        title="Video title",
        source_url="https://example.com/watch?v=1",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        summary="A compact summary",
        key_points=("Point one", "Point two"),
        visual_highlights=("Blue slide",),
        model_used="qwen",
        vram_mb=512.0,
    )


@pytest.mark.anyio
async def test_rss_publication_renderer_delegates_to_package_render_feed() -> None:
    publications = (build_publication(),)
    renderer = RssPublicationRenderer(
        title="Feed title",
        link="https://example.com",
        description="Feed description",
    )

    xml = await renderer.render(publications)

    assert xml == render_feed(
        title="Feed title",
        link="https://example.com",
        description="Feed description",
        publications=publications,
    )


@pytest.mark.anyio
async def test_rss_publication_renderer_renders_publication_record_fields() -> None:
    renderer = RssPublicationRenderer(
        title="Feed title",
        link="https://example.com",
        description="Feed description",
    )

    xml = await renderer.render((build_publication(),))

    assert "Video title" in xml
    assert "https://example.com/watch?v=1" in xml
    assert "A compact summary" in xml
    assert "Point one" in xml
    assert "Blue slide" in xml
    assert "Model: qwen (VRAM 512.00 MB)" in xml
