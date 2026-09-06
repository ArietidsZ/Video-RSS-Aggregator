"""Skeleton smoke tests — prove contracts/types/config import and validate."""

from vra import __version__
from vra.config import AsrMode, Config, FrameMode, ThinkLevel
from vra.store.schema import SCHEMA, SCHEMA_VERSION


def test_version() -> None:
    assert __version__ == "2.0.0"


def test_config_defaults() -> None:
    c = Config.from_env(env={})
    assert c.model == "qwen3.5-9b"
    assert c.model_ctx == 32768
    assert c.think == ThinkLevel.OFF
    assert c.frame_mode == FrameMode.AUTO
    assert c.asr == AsrMode.AUTO
    assert c.engine_base_url == "http://127.0.0.1:8000/v1"
    assert c.effective_asr_base_url == "http://127.0.0.1:8000/v1"
    assert c.bind_address == "127.0.0.1:8080"


def test_config_overrides() -> None:
    c = Config.from_env(
        env={
            "VRA_MODEL": "Qwen3.5-9B-AWQ",
            "VRA_THINK": "low",
            "VRA_MAX_FRAMES": "12",
            "VRA_MODEL_CTX": "8192",
            "VRA_ASR_BASE_URL": "http://127.0.0.1:8081/v1",
            "BIND_ADDRESS": "0.0.0.0:9000",
        }
    )
    assert c.model == "Qwen3.5-9B-AWQ"
    assert c.think == ThinkLevel.LOW
    assert c.max_frames == 12
    assert c.model_ctx == 8192
    assert c.effective_asr_base_url == "http://127.0.0.1:8081/v1"
    assert c.bind_address == "0.0.0.0:9000"


def test_url_policy() -> None:
    c = Config()
    assert c.allow_url("https://example.com/feed.xml", local_paths_enabled=False)
    assert not c.allow_url("/etc/passwd", local_paths_enabled=False)
    assert c.allow_url("/data/x.mp4", local_paths_enabled=True)
    assert not c.allow_url("ftp://x", local_paths_enabled=True)


def test_schema_tables() -> None:
    for table in ("feeds", "videos", "transcripts", "frames", "summaries", "jobs"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in SCHEMA
    assert SCHEMA_VERSION == 2
