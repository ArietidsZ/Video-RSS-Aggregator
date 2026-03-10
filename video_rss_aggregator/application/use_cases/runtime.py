from __future__ import annotations

from dataclasses import dataclass

from video_rss_aggregator.application.ports import RuntimeInspector


@dataclass(frozen=True)
class GetRuntimeStatus:
    runtime: RuntimeInspector
    storage_path: str
    storage_dir: str
    models: tuple[str, ...]

    async def execute(self) -> dict[str, object]:
        return {
            **await self.runtime.status(),
            "database_path": self.storage_path,
            "storage_dir": self.storage_dir,
            "models": list(self.models),
        }


@dataclass(frozen=True)
class BootstrapRuntime:
    runtime: RuntimeInspector

    async def execute(self) -> dict[str, list[str]]:
        return {"models": await self.runtime.bootstrap()}
