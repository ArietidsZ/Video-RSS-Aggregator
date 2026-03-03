from __future__ import annotations

from service_ollama import model_size_hint_bytes


def test_model_size_hint_returns_bytes_for_known_models() -> None:
    assert model_size_hint_bytes("qwen3.5:2b-q4_K_M") == 1945 * 1024 * 1024
    assert model_size_hint_bytes("qwen3.5:4b-q4_k_m") == 3482 * 1024 * 1024


def test_model_size_hint_unknown_returns_none() -> None:
    assert model_size_hint_bytes("qwen3.5:totally-unknown") is None
