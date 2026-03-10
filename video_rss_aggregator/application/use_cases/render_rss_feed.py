from __future__ import annotations

from dataclasses import dataclass

from video_rss_aggregator.application.ports import (
    PublicationRenderer,
    PublicationRepository,
)


@dataclass(frozen=True)
class RenderRssFeed:
    publications: PublicationRepository
    renderer: PublicationRenderer

    async def execute(self, limit: int) -> str:
        publications = await self.publications.latest_publications(limit)
        return await self.renderer.render(tuple(publications))
