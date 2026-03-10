from pathlib import Path

import pytest

import service_media
from service_media import PreparedMedia as LegacyPreparedMedia
from video_rss_aggregator.infrastructure.media_service import (
    LegacyMediaPreparationService,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_media_preparation_service_maps_legacy_prepared_media(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_prepare_media(**kwargs) -> LegacyPreparedMedia:
        captured.update(kwargs)
        return LegacyPreparedMedia(
            media_path=Path("/tmp/downloaded.mp4"),
            title="Legacy title",
            transcript="captured transcript",
            frame_paths=[Path("/tmp/frame-1.jpg"), Path("/tmp/frame-2.jpg")],
        )

    monkeypatch.setattr(service_media, "prepare_media", fake_prepare_media)

    adapter = LegacyMediaPreparationService(
        client=object(),
        storage_dir=".data",
        max_frames=4,
        scene_detection=True,
        scene_threshold=0.42,
        scene_min_frames=3,
        max_transcript_chars=5000,
    )

    prepared = await adapter.prepare("https://example.com/watch?v=1", "Feed title")

    assert captured == {
        "client": adapter.client,
        "source": "https://example.com/watch?v=1",
        "storage_dir": ".data",
        "max_frames": 4,
        "scene_detection": True,
        "scene_threshold": 0.42,
        "scene_min_frames": 3,
        "max_transcript_chars": 5000,
    }
    assert prepared.source_url == "https://example.com/watch?v=1"
    assert prepared.title == "Feed title"
    assert prepared.transcript == "captured transcript"
    assert prepared.media_path == "/tmp/downloaded.mp4"
    assert prepared.frame_paths == ("/tmp/frame-1.jpg", "/tmp/frame-2.jpg")


@pytest.mark.anyio
async def test_media_preparation_service_falls_back_to_legacy_title(
    monkeypatch,
) -> None:
    async def fake_prepare_media(**kwargs) -> LegacyPreparedMedia:
        return LegacyPreparedMedia(
            media_path=Path("/tmp/downloaded.mp4"),
            title="Legacy title",
            transcript="",
            frame_paths=[],
        )

    monkeypatch.setattr(service_media, "prepare_media", fake_prepare_media)

    adapter = LegacyMediaPreparationService(
        client=object(),
        storage_dir=".data",
        max_frames=1,
        scene_detection=False,
        scene_threshold=0.28,
        scene_min_frames=1,
        max_transcript_chars=100,
    )

    prepared = await adapter.prepare("https://example.com/watch?v=2", None)

    assert prepared.title == "Legacy title"
