"""Tests for vra.service.processor: pipeline states, persistence, failure."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from test_service_fakes import (
    FakeAcquirer,
    FakeExtractor,
    FakeSummarizer,
    FakeTranscriber,
)

from vra.config import AsrMode, Config
from vra.service.processor import Processor
from vra.store.db import Database
from vra.store.queue import SqlJobQueue
from vra.store.repos import SqlVideoStore
from vra.types import Transcript, TranscriptSeg, VideoStatus


@pytest.fixture
async def stack(
    tmp_path: Path,
) -> AsyncIterator[tuple[Database, SqlVideoStore, SqlJobQueue]]:
    db = await Database.connect(str(tmp_path / "vra.db"))
    await db.migrate()
    yield db, SqlVideoStore(db), SqlJobQueue(db)
    await db.close()


_SUBTITLED = Transcript(
    text="hello world",
    source="subtitle",
    lang="en",
    segments=(TranscriptSeg(start_s=0.0, end_s=2.0, text="hello"),),
)


def _make_processor(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
    acquirer: FakeAcquirer,
    transcriber: FakeTranscriber,
    extractor: FakeExtractor,
    summarizer: FakeSummarizer,
    config: Config | None = None,
) -> Processor:
    _, store, _ = stack
    return Processor(
        store, acquirer, transcriber, extractor, summarizer,
        config or Config(),
    )


async def _new_video(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
    canonical_id: str = "c1",
    source_url: str = "https://x/1",
) -> str:
    _, store, _ = stack
    rec, _ = await store.upsert_video_ref(
        canonical_id=canonical_id,
        source_url=source_url,
        feed_id=None,
        guid=None,
        title="Feed Title",
        published_at=None,
    )
    return rec.id


async def test_process_happy_path_persists_all_rows(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    video_id = await _new_video(stack)
    acquirer = FakeAcquirer(transcript=_SUBTITLED)
    transcriber = FakeTranscriber()
    extractor = FakeExtractor(count=2)
    summarizer = FakeSummarizer()
    processor = _make_processor(stack, acquirer, transcriber, extractor, summarizer)

    result = await processor.process(video_id)
    assert result == "done"

    detail = await store.video_detail(video_id)
    assert detail is not None
    video = detail["video"]
    assert video["status"] == "done"
    assert video["error"] is None
    assert video["title"] == "Meta Title"  # platform meta overwrites feed title
    assert video["duration_s"] == 123.5
    assert video["chapters"] == [[0.0, "intro"], [12.0, "body"]]
    assert video["media_path"] == "/media/cid:https://x/1.mp4"

    transcript = detail["transcript"]
    assert transcript is not None
    assert transcript["source"] == "subtitle"
    assert transcript["text"] == "hello world"
    assert transcript["segments"] == [{"start_s": 0.0, "end_s": 2.0, "text": "hello"}]

    assert len(detail["frames"]) == 2
    assert detail["frames"][0]["path"].endswith("-0.jpg")
    assert detail["frames"][0]["dhash"] == "0000000000000000"

    assert len(detail["summaries"]) == 1
    summary = detail["summaries"][0]
    assert summary["model"] == "fake-vlm"
    assert summary["key_points"] == ["point one", "point two"]
    assert summary["transcript_chars"] == 11
    assert summary["frame_count"] == 2

    # subtitles existed, max_frames default (8) -> exactly one download for frames
    assert len(acquirer.downloads) == 1
    assert transcriber.calls == []
    assert extractor.calls[0]["max_frames"] == 8
    assert extractor.calls[0]["duration_s"] == 123.5


async def test_process_unknown_video_raises(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    processor = _make_processor(
        stack, FakeAcquirer(), FakeTranscriber(), FakeExtractor(), FakeSummarizer()
    )
    with pytest.raises(ValueError, match="unknown video"):
        await processor.process("missing-id")


async def test_process_exception_marks_failed_without_raising(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    video_id = await _new_video(stack)
    acquirer = FakeAcquirer(transcript=_SUBTITLED)
    extractor = FakeExtractor(error=RuntimeError("ffmpeg exploded"))
    processor = _make_processor(
        stack, acquirer, FakeTranscriber(), extractor, FakeSummarizer()
    )

    result = await processor.process(video_id)
    assert result == "failed"  # never re-raises

    detail = await store.video_detail(video_id)
    assert detail is not None
    assert detail["video"]["status"] == "failed"
    assert "ffmpeg exploded" in detail["video"]["error"]
    # extraction failed before any persistence: no transcript/frames/summaries
    assert detail["transcript"] is None
    assert detail["frames"] == []
    assert detail["summaries"] == []


async def test_process_done_short_circuit(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    video_id = await _new_video(stack)
    await store.set_video_status(video_id, VideoStatus.DONE)
    acquirer = FakeAcquirer(transcript=_SUBTITLED)
    processor = _make_processor(
        stack, acquirer, FakeTranscriber(), FakeExtractor(), FakeSummarizer()
    )

    assert await processor.process(video_id) == "done"
    assert acquirer.resolves == []  # no work performed


async def test_process_force_bypasses_done_short_circuit(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    video_id = await _new_video(stack)
    acquirer = FakeAcquirer(transcript=_SUBTITLED)
    summarizer = FakeSummarizer()
    processor = _make_processor(
        stack, acquirer, FakeTranscriber(), FakeExtractor(), summarizer
    )

    assert await processor.process(video_id) == "done"
    assert await processor.process(video_id, force=True) == "done"
    assert len(acquirer.resolves) == 2  # reprocessed end to end
    detail = await store.video_detail(video_id)
    assert detail is not None
    summaries = detail["summaries"]
    assert len(summaries) == 2  # history kept, newest first
    assert summaries[0]["created_at"] >= summaries[1]["created_at"]


async def test_process_asr_fallback_downloads_once(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    video_id = await _new_video(stack)
    acquirer = FakeAcquirer(transcript=None)  # no subtitles available
    transcriber = FakeTranscriber()
    extractor = FakeExtractor(count=1)
    summarizer = FakeSummarizer()
    processor = _make_processor(stack, acquirer, transcriber, extractor, summarizer)

    assert await processor.process(video_id) == "done"

    # one download shared between ASR and frame extraction
    assert len(acquirer.downloads) == 1
    media_path = acquirer.media_paths[0]
    assert transcriber.calls == [media_path]
    assert extractor.calls[0]["media_path"] == media_path

    detail = await store.video_detail(video_id)
    assert detail is not None
    assert detail["transcript"] is not None
    assert detail["transcript"]["source"] == "asr"
    assert detail["transcript"]["text"] == "asr words"
    assert detail["summaries"][0]["transcript_chars"] == len("asr words")


async def test_process_max_frames_zero_skips_download(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    video_id = await _new_video(stack)
    acquirer = FakeAcquirer(transcript=_SUBTITLED)
    transcriber = FakeTranscriber()
    extractor = FakeExtractor(count=3)
    summarizer = FakeSummarizer()
    processor = _make_processor(
        stack, acquirer, transcriber, extractor, summarizer, Config(max_frames=0)
    )

    assert await processor.process(video_id) == "done"
    assert acquirer.downloads == []  # no media needed at all
    assert extractor.calls == []
    detail = await store.video_detail(video_id)
    assert detail is not None
    assert detail["frames"] == []
    assert detail["video"]["media_path"] is None
    assert detail["summaries"][0]["frame_count"] == 0


async def test_process_asr_off_and_no_subs_yields_empty_transcript(
    stack: tuple[Database, SqlVideoStore, SqlJobQueue],
) -> None:
    _, store, _ = stack
    video_id = await _new_video(stack)
    acquirer = FakeAcquirer(transcript=None)
    transcriber = FakeTranscriber()
    summarizer = FakeSummarizer()
    processor = _make_processor(
        stack,
        acquirer,
        transcriber,
        FakeExtractor(count=0),
        summarizer,
        Config(asr=AsrMode.OFF, max_frames=0),
    )

    assert await processor.process(video_id) == "done"
    assert transcriber.calls == []  # ASR disabled
    assert acquirer.downloads == []
    asset = summarizer.assets[0]
    assert asset.transcript.text == ""
    assert asset.transcript.source == "none"
    assert asset.frames.frames == ()
