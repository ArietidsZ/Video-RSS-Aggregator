"""API tests: create_app against a fake runtime facade (hermetic, no network).

The runtime is duck-typed (see create_app docstring), so a small in-memory
fake exercises every route: auth, feeds CRUD, ingest/process, jobs, videos,
rss, media containment, metrics, setup studio, and the 1 MB body cap.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from vra.api.app import canonical_for_source, create_app
from vra.config import AsrMode, Config
from vra.types import FeedRecord, Job, JobKind, JobState, VideoRecord


def _config(**overrides: object) -> Config:
    base: dict[str, object] = {"data_dir": "", "database_path": ""}
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


PUBLIC_IP = ipaddress.ip_address("93.184.216.34")
LOOPBACK_IP = ipaddress.ip_address("127.0.0.1")


async def _fake_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    return [LOOPBACK_IP if host.endswith(".local") else PUBLIC_IP]


@pytest.fixture(autouse=True)
def _hermetic_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vra.api.app._resolved_addresses", _fake_addresses)


@dataclass
class FakeStore:
    feeds: dict[str, FeedRecord] = field(default_factory=dict)
    videos: dict[str, VideoRecord] = field(default_factory=dict)
    publications: list[dict[str, object]] = field(default_factory=list)
    details: dict[str, dict[str, object]] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)

    async def upsert_feed(self, url: str, title: str | None, poll_minutes: int) -> FeedRecord:
        existing = next((f for f in self.feeds.values() if f.url == url), None)
        if existing:
            return existing
        record = FeedRecord(
            id=f"feed-{len(self.feeds) + 1}", url=url, title=title, poll_minutes=poll_minutes
        )
        self.feeds[record.id] = record
        return record

    async def get_feed_by_url(self, url: str) -> FeedRecord | None:
        return next((f for f in self.feeds.values() if f.url == url), None)

    async def list_feeds(self) -> list[FeedRecord]:
        return list(self.feeds.values())

    async def delete_feed(self, feed_id: str) -> None:
        self.feeds.pop(feed_id, None)
        self.deleted.append(feed_id)

    async def upsert_video_ref(self, **kwargs: object) -> tuple[VideoRecord, bool]:
        canonical = str(kwargs["canonical_id"])
        found = next(
            (v for v in self.videos.values() if v.canonical_id == canonical), None
        )
        if found:
            return found, False
        record = VideoRecord(
            id=f"video-{len(self.videos) + 1}",
            canonical_id=canonical,
            source_url=str(kwargs["source_url"]),
            feed_id=kwargs.get("feed_id"),  # type: ignore[arg-type]
            guid=kwargs.get("guid"),  # type: ignore[arg-type]
            title=kwargs.get("title"),  # type: ignore[arg-type]
            published_at=kwargs.get("published_at"),  # type: ignore[arg-type]
        )
        self.videos[record.id] = record
        return record, True

    async def get_video_by_canonical(self, canonical_id: str) -> VideoRecord | None:
        return next(
            (v for v in self.videos.values() if v.canonical_id == canonical_id), None
        )

    async def latest_publications(self, limit: int) -> list[dict[str, object]]:
        return self.publications[:limit]

    async def video_detail(self, video_id: str) -> dict[str, object] | None:
        return self.details.get(video_id)

    async def counts(self) -> dict[str, object]:
        return {
            "jobs_by_state": {"queued": 1},
            "videos_by_status": {"done": 1},
            "feeds": len(self.feeds),
            "summaries": len(self.publications),
        }


@dataclass
class FakeQueue:
    jobs: dict[str, Job] = field(default_factory=dict)

    async def enqueue(
        self,
        kind: JobKind,
        payload: dict[str, object],
        *,
        video_id: str | None = None,
        run_at: object = None,
        dedup_key: str | None = None,
    ) -> Job:
        job = Job(
            id=f"job-{len(self.jobs) + 1}",
            kind=kind,
            state=JobState.QUEUED,
            payload=payload,
            video_id=video_id,
            created_at=datetime.now(UTC),
        )
        self.jobs[job.id] = job
        return job

    async def list(
        self,
        *,
        state: JobState | None = None,
        kind: JobKind | None = None,
        limit: int = 50,
    ) -> list[Job]:
        out = [
            job
            for job in self.jobs.values()
            if (state is None or job.state == state) and (kind is None or job.kind == kind)
        ]
        return out[:limit]

    async def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


@dataclass
class FakeSummarizer:
    status: dict[str, object] = field(default_factory=dict)
    prepared: list[str] = field(default_factory=list)
    raise_on_status: bool = False
    raise_on_prepare: bool = False

    async def runtime_status(self) -> dict[str, object]:
        if self.raise_on_status:
            raise RuntimeError("engine probe exploded")
        return self.status

    async def prepare_models(self) -> list[str]:
        if self.raise_on_prepare:
            raise RuntimeError("engine not reachable")
        return self.prepared


@dataclass
class FakeRuntime:
    config: Config
    store: FakeStore = field(default_factory=FakeStore)
    queue: FakeQueue = field(default_factory=FakeQueue)
    summarizer: FakeSummarizer = field(default_factory=FakeSummarizer)
    started: bool = False
    closed: bool = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    def stats(self) -> dict[str, object]:
        return {"workers": self.config.cpu_workers, "started": self.started}


_GOOD_STATUS: dict[str, object] = {
    "engine": {
        "base_url": "http://127.0.0.1:8000/v1",
        "reachable": True,
        "models": ["qwen3.5-9b"],
        "model_present": True,
    },
    "asr": {"models": ["Qwen3-ASR-1.7B"], "model_present": True},
}


def _client(runtime: FakeRuntime) -> TestClient:
    return TestClient(create_app(runtime))


def test_lifespan_starts_and_closes_runtime() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime):
        assert runtime.started
    assert runtime.closed


def test_health_reports_version() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["version"]


def test_setup_home_renders_html_without_placeholders() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "__DEFAULTS_JSON__" not in resp.text
    assert "__BASE_HREF__" not in resp.text


def test_feeds_crud_flow() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        created = client.post("/feeds", json={"url": "https://example.com/feed.xml"})
        assert created.status_code == 201
        body = created.json()
        assert body["url"] == "https://example.com/feed.xml"
        assert body["job_id"].startswith("job-")

        listed = client.get("/feeds").json()
        assert len(listed) == 1

        deleted = client.delete(f"/feeds/{body['id']}")
        assert deleted.status_code == 204
        assert runtime.store.deleted == [body["id"]]
        assert client.get("/feeds").json() == []


def test_auth_guard_allows_gets_and_blocks_posts() -> None:
    runtime = FakeRuntime(_config(api_key="sekrit"))
    with _client(runtime) as client:
        assert client.get("/feeds").status_code == 200
        denied = client.post("/feeds", json={"url": "https://example.com/f.xml"})
        assert denied.status_code == 401
        assert denied.json() == {"detail": "unauthorized"}
        by_bearer = client.post(
            "/feeds",
            json={"url": "https://example.com/f.xml"},
            headers={"Authorization": "Bearer sekrit"},
        )
        assert by_bearer.status_code == 201
        by_header = client.post(
            "/feeds",
            json={"url": "https://example.com/g.xml"},
            headers={"X-API-Key": "sekrit"},
        )
        assert by_header.status_code == 201


def test_feed_url_resolving_to_private_address_rejected() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        resp = client.post("/feeds", json={"url": "http://box.local/feed.xml"})
    assert resp.status_code == 400
    assert "non-public address" in resp.json()["detail"]


def test_ingest_enqueues_job_and_rejects_private_urls() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        ok = client.post("/ingest", json={"url": "https://example.com/feed.xml"})
        assert ok.status_code == 200
        assert ok.json()["job_id"]
        bad = client.post("/ingest", json={"url": "http://box.local/feed.xml"})
        assert bad.status_code == 400


def test_process_by_source_url_creates_ref_and_job() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        resp = client.post(
            "/process",
            json={"source_url": "https://example.com/watch/1", "force": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["video_id"].startswith("video-")
        job = runtime.queue.jobs[body["job_id"]]
        assert job.kind is JobKind.PROCESS
        assert job.payload == {"video_id": body["video_id"], "force": True}


def test_process_requires_source_or_video_id() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        resp = client.post("/process", json={})
    assert resp.status_code == 400


def test_canonical_for_source_is_stable_versioned_source() -> None:
    assert canonical_for_source("https://x/y") == canonical_for_source("https://x/y")
    assert canonical_for_source("https://x/y") != canonical_for_source("https://x/z")
    assert canonical_for_source("https://x/y").startswith("source:v2:")


def test_jobs_list_and_filters() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        client.post("/ingest", json={"url": "https://example.com/feed.xml"})
        listed = client.get("/jobs", params={"kind": "ingest"}).json()
        assert len(listed) == 1
        assert listed[0]["kind"] == "ingest"
        assert listed[0]["state"] == "queued"
        unknown = client.get("/jobs", params={"state": "bogus"})
        assert unknown.status_code == 400


def test_get_job_detail_and_404() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        job_id = client.post("/ingest", json={"url": "https://example.com/f.xml"}).json()[
            "job_id"
        ]
        detail = client.get(f"/jobs/{job_id}").json()
        assert detail["id"] == job_id
        assert detail["payload"] == {"url": "https://example.com/f.xml"}
        assert client.get("/jobs/missing").status_code == 404


def test_videos_list_with_feed_filter_and_detail_404() -> None:
    runtime = FakeRuntime(_config())
    runtime.store.publications = [
        {"video_id": "v1", "feed_id": "feed-1", "title": "One"},
        {"video_id": "v2", "feed_id": "feed-2", "title": "Two"},
    ]
    with _client(runtime) as client:
        assert len(client.get("/videos").json()) == 2
        filtered = client.get("/videos", params={"feed_id": "feed-2"}).json()
        assert [v["video_id"] for v in filtered] == ["v2"]
        assert client.get("/videos/missing-id").status_code == 404


def test_video_detail_returned() -> None:
    runtime = FakeRuntime(_config())
    runtime.store.details["v1"] = {"video": {"id": "v1"}, "summaries": []}
    with _client(runtime) as client:
        resp = client.get("/videos/v1")
    assert resp.status_code == 200
    assert resp.json()["video"]["id"] == "v1"


def test_rss_endpoint_content_type_and_body() -> None:
    runtime = FakeRuntime(_config())
    runtime.store.publications = [
        {
            "video_id": "v1",
            "canonical_id": "url:abc",
            "source_url": "https://x/1",
            "title": "One",
            "summary": "s",
            "key_points": [],
            "frame_count": 0,
        }
    ]
    with _client(runtime) as client:
        resp = client.get("/rss")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/rss+xml")
    assert b"<rss" in resp.content and b"One" in resp.content


def test_media_serves_frames_inside_data_dir_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    frame = data_dir / "raw" / "f0.jpg"
    frame.write_bytes(b"\xff\xd8fakejpeg")
    vid = "11111111-2222-3333-4444-555555555555"
    runtime = FakeRuntime(_config(data_dir=str(data_dir)))
    runtime.store.details[vid] = {
        "frames": [
            {"path": str(frame)},
            {"path": str(tmp_path / "escape.jpg")},
        ]
    }
    with _client(runtime) as client:
        ok = client.get(f"/media/{vid}/0.jpg")
        assert ok.status_code == 200
        assert ok.content == b"\xff\xd8fakejpeg"
        traversal = client.get(f"/media/{vid}/1.jpg")
        assert traversal.status_code == 404
        bad_id = client.get("/media/not-a-uuid/0.jpg")
        assert bad_id.status_code == 404


def test_metrics_shape() -> None:
    runtime = FakeRuntime(_config(model="test-model"))
    with _client(runtime) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "test-model"
    assert body["feeds"] == 0
    assert body["runtime"] == {"workers": runtime.config.cpu_workers, "started": True}


def test_setup_config_payload_hides_secrets() -> None:
    runtime = FakeRuntime(_config(api_key="hush", engine_api_key="eh"))
    with _client(runtime) as client:
        payload = client.get("/setup/config").json()
    assert "api_key" not in payload
    assert "engine_api_key" not in payload
    assert payload["api_key_required"] is True
    assert payload["engine_api_key_required"] is True
    assert payload["model"] == runtime.config.model


def test_setup_diagnostics_consumes_runtime_status() -> None:
    runtime = FakeRuntime(_config(), summarizer=FakeSummarizer(status=_GOOD_STATUS))
    with _client(runtime) as client:
        report = client.get("/setup/diagnostics").json()
    engine = report["dependencies"]["engine"]
    assert engine["reachable"] is True
    assert engine["model_present"] is True
    assert engine["models"] == ["qwen3.5-9b"]
    assert report["dependencies"]["asr"]["model_present"] is True
    assert report["runtime_view"]["state"] in {"ready", "blocked"}
    assert report["setup_view"]["state"] in {"ready", "blocked"}


def test_setup_diagnostics_tolerates_summarizer_failure() -> None:
    runtime = FakeRuntime(
        _config(), summarizer=FakeSummarizer(raise_on_status=True)
    )
    with _client(runtime) as client:
        report = client.get("/setup/diagnostics").json()
    engine = report["dependencies"]["engine"]
    assert engine["reachable"] is False
    assert "exploded" in str(engine["error"])


def test_bootstrap_success_and_failure_shapes() -> None:
    ok_runtime = FakeRuntime(_config(), summarizer=FakeSummarizer(prepared=["qwen3.5-9b"]))
    with _client(ok_runtime) as client:
        resp = client.post("/setup/bootstrap")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "models_prepared": ["qwen3.5-9b"]}

    bad_runtime = FakeRuntime(_config(), summarizer=FakeSummarizer(raise_on_prepare=True))
    with _client(bad_runtime) as client:
        resp = client.post("/setup/bootstrap")
    assert resp.status_code == 502
    assert resp.json()["ok"] is False
    assert "not reachable" in resp.json()["error"]


def test_request_body_cap_returns_413() -> None:
    runtime = FakeRuntime(_config())
    with _client(runtime) as client:
        resp = client.post(
            "/ingest",
            content=b" " * 1_000_001,
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 413


def test_asr_off_satisfies_ready_logic() -> None:
    status: dict[str, object] = {
        "engine": {
            "base_url": "http://127.0.0.1:8000/v1",
            "reachable": True,
            "models": ["qwen3.5-9b"],
            "model_present": True,
        },
        "asr": {"models": [], "model_present": False},
    }
    runtime = FakeRuntime(
        _config(asr=AsrMode.OFF), summarizer=FakeSummarizer(status=status)
    )
    with _client(runtime) as client:
        report = client.get("/setup/diagnostics").json()
    asr = report["dependencies"]["asr"]
    assert asr["enabled"] is False
    assert report["runtime_view"]["state"] == "ready"


def test_blank_source_is_bad_request_even_when_local_paths_allowed():
    runtime = FakeRuntime(_config(allow_local_path=True))
    with _client(runtime) as client:
        response = client.post('/process', json={'source_url': '   '})
    assert response.status_code == 400
    assert response.json()['detail'] == 'source_url must not be blank'
