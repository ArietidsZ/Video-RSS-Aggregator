from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from video_rss_aggregator.domain.publication import PublicationRecord
from video_rss_aggregator.rss import render_feed


@dataclass(frozen=True)
class RssPublicationRenderer:
    title: str
    link: str
    description: str

    async def render(self, publications: Sequence[PublicationRecord]) -> str:
        return render_feed(self.title, self.link, self.description, publications)
