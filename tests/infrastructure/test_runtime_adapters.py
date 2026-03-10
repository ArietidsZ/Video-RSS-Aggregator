import pytest

from video_rss_aggregator.infrastructure.runtime_adapters import LegacyRuntimeInspector


class FakeSummarizationEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def runtime_status(self) -> dict[str, object]:
        self.calls.append("runtime_status")
        return {"reachable": True, "local_models": ["qwen"]}

    async def prepare_models(self) -> list[str]:
        self.calls.append("prepare_models")
        return ["qwen", "min"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_runtime_inspector_delegates_to_legacy_engine() -> None:
    engine = FakeSummarizationEngine()
    adapter = LegacyRuntimeInspector(engine)

    status = await adapter.status()
    prepared = await adapter.bootstrap()

    assert status == {"reachable": True, "local_models": ["qwen"]}
    assert prepared == ["qwen", "min"]
    assert engine.calls == ["runtime_status", "prepare_models"]
