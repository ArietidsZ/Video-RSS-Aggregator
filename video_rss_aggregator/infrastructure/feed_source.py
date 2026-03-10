from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import feedparser
import httpx

from video_rss_aggregator.application.ports import FetchedFeed, FetchedFeedEntry


def _pick_source_url(entry: Any) -> str | None:
    enclosures = entry.get("enclosures", [])
    if enclosures:
        return enclosures[0].get("href") or enclosures[0].get("url")
    links = entry.get("links", [])
    if links:
        return links[0].get("href")
    return entry.get("link")


def _map_entry(entry: Any) -> FetchedFeedEntry:
    return FetchedFeedEntry(
        source_url=_pick_source_url(entry),
        title=entry.get("title") or None,
        guid=entry.get("id") or None,
    )


@dataclass(frozen=True)
class HttpFeedSource:
    client: httpx.AsyncClient

    async def fetch(self, feed_url: str, max_items: int | None = None) -> FetchedFeed:
        response = await self.client.get(feed_url)
        response.raise_for_status()
        parsed = feedparser.parse(response.text)
        entries = (
            parsed.entries[:max_items] if max_items is not None else parsed.entries
        )
        return FetchedFeed(
            title=parsed.feed.get("title") or None,
            site_url=parsed.feed.get("link") or None,
            entries=tuple(_map_entry(entry) for entry in entries),
        )
