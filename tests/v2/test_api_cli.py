"""CLI tests via CliRunner: doctor/engine posture report + runtime-backed commands.

`vra doctor` / `vra engine` are config-driven (no runtime), so their probes are
monkeypatched at vra.cli._probe_models. The runtime-backed commands (add-feed,
poll-now, list) get a fake runtime injected as a fake `vra.runtime` module,
because the real one imports the still-stubbed store/service slice.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field

import pytest
from click.testing import CliRunner

from vra.cli import _parse_bind, cli
from vra.config import Config
from vra.types import FeedRecord, Job, JobKind, JobState


def _config(**overrides: object) -> Config:
    return Config(**overrides)  # type: ignore[arg-type]


def _doctor_result(
    monkeypatch: pytest.MonkeyPatch,
    engine: dict[str, object],
    asr: dict[str, object],
    args: list[str],
) -> object:
    async def fake_probe(base_url: str, api_key: str | None) -> dict[str, object]:
        # fresh dict per call: _doctor_report mutates the returned dicts
        source = asr if "8081" in base_url else engine
        return dict(source)

    monkeypatch.setattr("vra.cli._probe_models", fake_probe)
    return CliRunner().invoke(cli, args)


def test_doctor_reports_present_model_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = {"base_url": "http://e/v1", "reachable": True, "models": ["qwen3.5-9b"], "error": None}
    asr = {"base_url": "http://e:8081/v1", "reachable": True, "models": ["Qwen3-ASR-1.7B"], "error": None}
    result = _doctor_result(monkeypatch, engine, asr, ["doctor"])
    assert result.exit_code == 0  # type: ignore[attr-defined]
    report = json.loads(result.output)  # type: ignore[attr-defined]
    assert report["engine"]["reachable"] is True
    assert report["engine"]["model_present"] is True
    assert report["asr"]["model"] == "Qwen3-ASR-1.7B"
    assert report["ok"] is True


def test_doctor_exits_nonzero_when_engine_unreachable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    engine = {"base_url": "http://e/v1", "reachable": False, "models": [], "error": "conn refused"}
    asr = {"base_url": "http://e:8081/v1", "reachable": False, "models": [], "error": None}
    result = _doctor_result(monkeypatch, engine, asr, ["doctor"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["ok"] is False
    assert report["engine"]["model_present"] is False


def test_doctor_flags_missing_configured_model(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    engine = {"base_url": "http://e/v1", "reachable": True, "models": ["other-model"], "error": None}
    asr = {"base_url": "http://e:8081/v1", "reachable": True, "models": [], "error": None}
    result = _doctor_result(monkeypatch, engine, asr, ["doctor"])
    assert result.exit_code == 0  # reachable is the only failure signal
    report = json.loads(result.output)
    assert report["engine"]["model_present"] is False
    assert report["asr"]["model_present"] is False


def test_engine_command_is_a_doctor_alias(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    engine = {"base_url": "http://e/v1", "reachable": False, "models": [], "error": "down"}
    asr = {"base_url": "http://e:8081/v1", "reachable": False, "models": [], "error": None}
    result = _doctor_result(monkeypatch, engine, asr, ["engine"])
    assert result.exit_code == 1
    assert json.loads(result.output)["ok"] is False


@dataclass
class FakeRuntime:
    config: Config
    feeds: list[FeedRecord] = field(default_factory=list)
    publications: list[dict[str, object]] = field(default_factory=list)
    enqueued: list[dict[str, object]] = field(default_factory=list)
    closed: bool = False

    async def close(self) -> None:
        self.closed = True

    @property
    def store(self) -> FakeStore:
        return FakeStore(self)

    @property
    def queue(self) -> FakeQueue:
        return FakeQueue(self)


@dataclass
class FakeStore:
    runtime: FakeRuntime

    async def list_feeds(self) -> list[FeedRecord]:
        return self.runtime.feeds

    async def upsert_feed(self, url: str, title: str | None, poll_minutes: int) -> FeedRecord:
        record = FeedRecord(id="feed-1", url=url, title=title, poll_minutes=poll_minutes)
        self.runtime.feeds.append(record)
        return record

    async def latest_publications(self, limit: int) -> list[dict[str, object]]:
        return self.runtime.publications[:limit]


@dataclass
class FakeQueue:
    runtime: FakeRuntime

    async def enqueue(
        self,
        kind: JobKind,
        payload: dict[str, object],
        *,
        video_id: str | None = None,
        run_at: object = None,
        dedup_key: str | None = None,
    ) -> Job:
        self.runtime.enqueued.append({"kind": kind, "payload": payload})
        return Job(
            id=f"job-{len(self.runtime.enqueued)}",
            kind=kind,
            state=JobState.QUEUED,
            payload=payload,
            video_id=video_id,
        )


def _install_fake_runtime(monkeypatch, runtime: FakeRuntime) -> None:  # type: ignore[no-untyped-def]
    module = types.ModuleType("vra.runtime")

    async def build_runtime(config: Config | None = None) -> FakeRuntime:
        runtime.config = config or Config.from_env()
        return runtime

    module.build_runtime = build_runtime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vra.runtime", module)


def test_add_feed_registers_and_enqueues(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runtime = FakeRuntime(config=_config())
    _install_fake_runtime(monkeypatch, runtime)
    result = CliRunner().invoke(
        cli, ["add-feed", "https://example.com/feed.xml", "--poll-minutes", "30"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["url"] == "https://example.com/feed.xml"
    assert payload["poll_minutes"] == 30
    assert payload["job_id"] == "job-1"
    assert runtime.enqueued[0]["kind"] is JobKind.INGEST
    assert runtime.closed is True


def test_poll_now_enqueues_ingest_for_every_feed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runtime = FakeRuntime(
        config=_config(),
        feeds=[
            FeedRecord(id="f1", url="https://a.xml"),
            FeedRecord(id="f2", url="https://b.xml"),
        ],
    )
    _install_fake_runtime(monkeypatch, runtime)
    result = CliRunner().invoke(cli, ["poll-now"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"feeds": 2, "enqueued": 2, "job_ids": ["job-1", "job-2"]}


def test_list_prints_feeds_and_videos(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runtime = FakeRuntime(
        config=_config(),
        feeds=[FeedRecord(id="f1", url="https://a.xml")],
        publications=[{"video_id": "v1", "title": "One"}],
    )
    _install_fake_runtime(monkeypatch, runtime)
    result = CliRunner().invoke(cli, ["list"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["feeds"][0]["url"] == "https://a.xml"
    assert payload["videos"][0]["video_id"] == "v1"


def test_poll_now_reports_empty_store(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runtime = FakeRuntime(config=_config())
    _install_fake_runtime(monkeypatch, runtime)
    result = CliRunner().invoke(cli, ["poll-now"])
    assert result.exit_code == 0
    assert json.loads(result.output)["enqueued"] == 0


def test_parse_bind_variants() -> None:
    assert _parse_bind("0.0.0.0:9000") == ("0.0.0.0", 9000)
    assert _parse_bind("localhost") == ("localhost", 8080)
    assert _parse_bind("localhost:oops") == ("localhost", 8080)
