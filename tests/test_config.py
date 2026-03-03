from __future__ import annotations

from core_config import Config


def test_model_priority_deduplicates_preserving_order() -> None:
    cfg = Config(model_primary="m4", model_fallback="m4", model_min="m2")
    assert cfg.model_priority == ("m4", "m2")


def test_from_env_parses_scene_detection_and_reserve(monkeypatch) -> None:
    monkeypatch.setenv("BIND_ADDRESS", "0.0.0.0:9090")
    monkeypatch.setenv("VRA_MODEL_SELECTION_RESERVE_MB", "512")
    monkeypatch.setenv("VRA_FRAME_SCENE_DETECTION", "false")
    monkeypatch.setenv("VRA_FRAME_SCENE_THRESHOLD", "0.31")
    monkeypatch.setenv("VRA_FRAME_SCENE_MIN_FRAMES", "3")

    cfg = Config.from_env()

    assert cfg.bind_host == "0.0.0.0"
    assert cfg.bind_port == 9090
    assert cfg.model_selection_reserve_mb == 512
    assert cfg.frame_scene_detection is False
    assert cfg.frame_scene_threshold == 0.31
    assert cfg.frame_scene_min_frames == 3


def test_from_env_invalid_port_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("BIND_ADDRESS", "127.0.0.1:not-a-port")
    cfg = Config.from_env()
    assert cfg.bind_host == "127.0.0.1"
    assert cfg.bind_port == 8080
