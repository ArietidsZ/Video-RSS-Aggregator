from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _RuntimeEngine(Protocol):
    async def runtime_status(self) -> dict[str, object]: ...

    async def prepare_models(self) -> list[str]: ...


@dataclass(frozen=True)
class LegacyRuntimeInspector:
    engine: _RuntimeEngine

    async def status(self) -> dict[str, object]:
        return await self.engine.runtime_status()

    async def bootstrap(self) -> list[str]:
        return await self.engine.prepare_models()
