from dataclasses import FrozenInstanceError
from typing import Any, get_args

import pytest

from video_rss_aggregator.domain import ProcessOutcome
from video_rss_aggregator.domain.models import PreparedMedia, SummaryResult
from video_rss_aggregator.domain.outcomes import Failure, PartialSuccess, Success


def test_success_outcome_exposes_status_and_payload() -> None:
    summary = SummaryResult(
        summary="ok",
        key_points=cast_list(["a", "b", "c", "d"]),
        visual_highlights=cast_list(["frame 1"]),
        model_used="fake-model",
        vram_mb=512.0,
        transcript_chars=12,
        frame_count=1,
        error=None,
    )
    media = PreparedMedia(
        source_url="https://example.com/video",
        title="Example",
        transcript="hello",
        media_path="/tmp/video.mp4",
        frame_paths=cast_list(["/tmp/frame-1.jpg"]),
    )

    outcome = Success(media=media, summary=summary)

    assert outcome.status == "success"
    assert outcome.summary.model_used == "fake-model"


def test_failure_outcome_preserves_reason() -> None:
    outcome = Failure(source_url="https://example.com/video", reason="ollama offline")

    assert outcome.status == "failure"
    assert outcome.reason == "ollama offline"


def test_domain_models_normalize_collections_to_tuples() -> None:
    summary = SummaryResult(
        summary="ok",
        key_points=cast_list(["a", "b"]),
        visual_highlights=cast_list(["frame 1"]),
        model_used="fake-model",
        vram_mb=512.0,
        transcript_chars=12,
        frame_count=1,
        error=None,
    )
    media = PreparedMedia(
        source_url="https://example.com/video",
        title="Example",
        transcript="hello",
        media_path="/tmp/video.mp4",
        frame_paths=cast_list(["/tmp/frame-1.jpg", "/tmp/frame-2.jpg"]),
    )

    assert summary.key_points == ("a", "b")
    assert summary.visual_highlights == ("frame 1",)
    assert media.frame_paths == ("/tmp/frame-1.jpg", "/tmp/frame-2.jpg")


def test_domain_models_are_frozen() -> None:
    summary = SummaryResult(
        summary="ok",
        key_points=cast_list(["a"]),
        visual_highlights=cast_list([]),
        model_used="fake-model",
        vram_mb=512.0,
        transcript_chars=12,
        frame_count=1,
        error=None,
    )

    with pytest.raises(FrozenInstanceError):
        cast_any(summary).summary = "changed"


def test_partial_success_requires_summary_and_preserves_reason() -> None:
    media = PreparedMedia(
        source_url="https://example.com/video",
        title="Example",
        transcript="hello",
        media_path="/tmp/video.mp4",
        frame_paths=cast_list([]),
    )
    summary = SummaryResult(
        summary="degraded",
        key_points=cast_list(["a"]),
        visual_highlights=cast_list([]),
        model_used="fake-model",
        vram_mb=512.0,
        transcript_chars=12,
        frame_count=0,
        error="missing frames",
    )

    outcome = PartialSuccess(media=media, summary=summary, reason="missing frames")

    assert outcome.status == "partial_success"
    assert outcome.summary is summary
    assert outcome.reason == "missing frames"


def test_partial_success_rejects_missing_summary() -> None:
    media = PreparedMedia(
        source_url="https://example.com/video",
        title="Example",
        transcript="hello",
        media_path="/tmp/video.mp4",
        frame_paths=cast_list([]),
    )

    with pytest.raises(ValueError, match="summary is required"):
        PartialSuccess(media=media, summary=cast_any(None), reason="missing frames")


def test_process_outcome_alias_is_exported_for_all_outcome_types() -> None:
    assert get_args(ProcessOutcome) == (Success, PartialSuccess, Failure)


def cast_any(value: object) -> Any:
    return value


def cast_list(values: list[str]) -> Any:
    return values
