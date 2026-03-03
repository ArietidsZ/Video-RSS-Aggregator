from __future__ import annotations

from service_summarize import _extract_json_payload, _is_meaningful_summary


def test_extract_json_payload_handles_markdown_fence() -> None:
    payload = _extract_json_payload(
        """```json
{"summary":"Short summary","key_points":["A","B"],"visual_highlights":[]}
```"""
    )
    assert payload["summary"] == "Short summary"
    assert payload["key_points"] == ["A", "B"]


def test_extract_json_payload_falls_back_when_not_json() -> None:
    payload = _extract_json_payload("No strict JSON output")
    assert payload["summary"] == "No strict JSON output"
    assert payload["key_points"] == []
    assert payload["visual_highlights"] == []


def test_is_meaningful_summary_uses_summary_or_key_points() -> None:
    assert _is_meaningful_summary(
        "This summary has enough words to clear the threshold safely.",
        [],
    )
    assert _is_meaningful_summary(
        "too short",
        [
            "This key point carries enough detail to be considered meaningful.",
            "Second key point",
        ],
    )
