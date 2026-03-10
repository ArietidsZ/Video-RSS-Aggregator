from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
        published_at=_pick_published_at(entry),
    )


def _pick_published_at(entry: Any) -> datetime | None:
    for field in ("published", "updated"):
        value = entry.get(field)
        if value:
            parsed = _parse_datetime_value(value)
            if parsed is not None:
                return parsed

    for field in ("published_parsed", "updated_parsed"):
        value = entry.get(field)
        if value is not None:
            parsed = _parse_datetime_tuple(value)
            if parsed is not None:
                return parsed

    return None


def _parse_datetime_value(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None

    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        parsed = None

    if parsed is None:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_datetime_tuple(value: Any) -> datetime | None:
    try:
        return datetime(*value[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


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
