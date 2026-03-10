import pytest

from video_rss_aggregator.application.use_cases.runtime import (
    BootstrapRuntime,
    GetRuntimeStatus,
)


class FakeRuntimeInspector:
    async def status(self) -> dict[str, object]:
        return {"reachable": True, "local_models": {"qwen": {}}, "models": ["qwen"]}

    async def bootstrap(self) -> list[str]:
        return ["qwen"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_get_runtime_status_returns_app_view_model() -> None:
    use_case = GetRuntimeStatus(
        runtime=FakeRuntimeInspector(),
        storage_path=".data/vra.db",
        storage_dir=".data",
        models=("qwen", "qwen:min"),
    )

    payload = await use_case.execute()

    assert payload["reachable"] is True
    assert payload["local_models"] == {"qwen": {}}
    assert payload["database_path"] == ".data/vra.db"
    assert payload["storage_dir"] == ".data"
    assert payload["models"] == ["qwen", "qwen:min"]


@pytest.mark.anyio
async def test_bootstrap_runtime_returns_loaded_models() -> None:
    use_case = BootstrapRuntime(runtime=FakeRuntimeInspector())

    payload = await use_case.execute()

    assert payload == {"models": ["qwen"]}
