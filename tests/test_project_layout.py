from pathlib import Path


def test_legacy_service_pipeline_module_has_been_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "service_pipeline.py").exists()
