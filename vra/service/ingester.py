"""Feed polling + enqueue, and the HTTP feed fetcher. OWNER: agent-C.

``HttpFeedFetcher`` implements contracts.FeedFetcher: conditional GET
(If-None-Match / If-Modified-Since), RSS/Atom parsing via feedparser, and an
SSRF guard (http/https only; host resolved and private/loopback/link-local/
reserved addresses rejected unless ``allow_private``). Entry mapping and the
datetime normalization logic are ported from the legacy
``video_rss_aggregator/infrastructure/feed_source.py``.

``FeedIngester`` implements contracts.IngestService: fetch (with the feed's
stored conditional headers), short-circuit on 304, upsert feed + video refs,
enqueue a process job per new/pending/failed video, and persist the feed's
post-poll state. Fetch errors are recorded on the feed row and re-raised so
the queue worker can retry with backoff.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import socket
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import feedparser
import httpx

from vra.config import Config
from vra.contracts import FeedFetcher, JobQueue, VideoStore
from vra.types import IngestResult, JobKind, VideoStatus

log = logging.getLogger(__name__)

_FETCH_TIMEOUT_S = 30.0
_ERROR_MAX_CHARS = 800
_MAX_FEED_REDIRECTS = 5


# --------------------------------------------------------------------------
# feedparser entry mapping (ported from legacy feed_source.py)
# --------------------------------------------------------------------------

def _pick_source_url(entry: Any) -> str | None:
    """Prefer enclosures, then links, then the plain link (legacy order)."""
    enclosures = entry.get("enclosures", [])
    if enclosures:
        return enclosures[0].get("href") or enclosures[0].get("url")
    links = entry.get("links", [])
    if links:
        return links[0].get("href")
    return entry.get("link")


def _parse_datetime_value(value: Any) -> datetime | None:
    """Parse RFC-822 or ISO-8601 date strings into tz-aware UTC datetimes."""
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
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_datetime_tuple(value: Any) -> datetime | None:
    """Parse feedparser's *_parsed 9-tuples into tz-aware UTC datetimes."""
    try:
        return datetime(*value[:6], tzinfo=UTC)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _pick_published_at(entry: Any) -> datetime | None:
    """String published/updated first, then the pre-parsed tuples (legacy)."""
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


def _map_entry(entry: Any) -> dict[str, Any]:
    """Normalize a feedparser entry into the frozen fetcher entry dict."""
    return {
        "source_url": _pick_source_url(entry),
        "title": entry.get("title") or None,
        "guid": entry.get("id") or None,
        "published_at": _pick_published_at(entry),
    }


# --------------------------------------------------------------------------
# HttpFeedFetcher
# --------------------------------------------------------------------------

def _resolve_addresses(host: str) -> list[str]:
    """Resolve ``host`` with getaddrinfo (blocking; run via to_thread)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve feed host {host!r}: {exc}") from exc
    return [str(info[4][0]) for info in infos]


def _as_ip(addr: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse an address string, unwrapping IPv6 zone ids and v4-mapped v6."""
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


class HttpFeedFetcher:
    """contracts.FeedFetcher over httpx + feedparser with an SSRF guard."""

    def __init__(self, client: httpx.AsyncClient, allow_private: bool = False) -> None:
        self._client = client
        self._allow_private = allow_private

    async def fetch(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, str | None, str | None, bool]:
        """Conditional GET + parse. Returns (entries, title, site_url, etag,
        not_modified); on HTTP 304 entries is empty and not_modified True."""
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        current_url = url
        for hop in range(_MAX_FEED_REDIRECTS + 1):
            await self._validate_url(current_url)
            response = await self._client.get(
                current_url, headers=headers, timeout=_FETCH_TIMEOUT_S,
                follow_redirects=False,
            )
            if response.status_code not in {301, 302, 303, 307, 308}:
                break
            location = response.headers.get("location")
            if not location:
                raise ValueError("feed redirect has no Location header")
            if hop == _MAX_FEED_REDIRECTS:
                raise ValueError("feed redirect limit exceeded")
            current_url = urljoin(str(response.url), location)
        if response.status_code == 304:
            return [], None, None, None, True
        response.raise_for_status()

        parsed = await asyncio.to_thread(feedparser.parse, response.text)
        entries = [_map_entry(entry) for entry in parsed.entries]
        feed_title = parsed.feed.get("title") or None
        site_url = parsed.feed.get("link") or None
        new_etag = response.headers.get("ETag")
        return entries, feed_title, site_url, new_etag, False

    async def _validate_url(self, url: str) -> None:
        parts = urlparse(url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"unsupported feed url scheme: {parts.scheme!r}")
        host = parts.hostname
        if not host:
            raise ValueError(f"feed url has no host: {url!r}")
        if not self._allow_private:
            try:
                addresses = await asyncio.to_thread(_resolve_addresses, host)
            except ValueError as exc:
                log.warning("ssrf guard rejected feed %s: %s", url, exc)
                raise
            if not addresses:
                raise ValueError(f"feed host {host!r} has no addresses")
            for addr in addresses:
                ip = _as_ip(addr)
                if ip is None or _is_forbidden(ip):
                    raise ValueError(
                        f"refusing feed host {host!r}: "
                        f"resolves to private/reserved {addr!r}"
                    )



# --------------------------------------------------------------------------
# canonical identity + FeedIngester
# --------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """scheme://host/path form with query/fragment dropped (host lowered)."""
    parts = urlparse(url)
    if parts.scheme and parts.netloc:
        return f"{parts.scheme}://{parts.netloc.lower()}{parts.path}"
    return parts.path or url


def _canonical_id(guid: str | None, source_url: str) -> str:
    """Absolute-http guid wins; otherwise sha1 of the normalized URL."""
    if guid and guid.startswith("http"):
        return guid
    digest = hashlib.sha1(_normalize_url(source_url).encode("utf-8")).hexdigest()
    return f"url:{digest[:16]}"


class FeedIngester:
    """contracts.IngestService: poll a feed, upsert refs, enqueue work."""

    def __init__(
        self,
        fetcher: FeedFetcher,
        store: VideoStore,
        queue: JobQueue,
        config: Config,
    ) -> None:
        self._fetcher = fetcher
        self._store = store
        self._queue = queue
        self._config = config

    async def ingest(self, url: str, *, max_items: int | None = None) -> IngestResult:
        """Fetch ``url``, sync video refs, enqueue process jobs for new work."""
        feed = await self._store.get_feed_by_url(url)
        try:
            entries, title, site_url, etag, not_modified = await self._fetcher.fetch(
                url,
                etag=feed.etag if feed is not None else None,
                last_modified=feed.last_modified if feed is not None else None,
            )
        except Exception as exc:
            log.warning("feed fetch failed for %s: %s", url, exc)
            if feed is not None:
                await self._store.save_feed_state(feed.id, error=str(exc)[:_ERROR_MAX_CHARS])
            raise

        if feed is None:
            feed = await self._store.upsert_feed(url, title, self._config.poll_minutes)
        if not_modified:
            await self._store.save_feed_state(feed.id, error=None)
            return IngestResult(feed=feed, new_videos=0, enqueued=0, skipped_done=0)

        if max_items is not None:
            entries = entries[: max(0, max_items)]

        new_videos = enqueued = skipped_done = 0
        for entry in entries:
            raw_url = entry.get("source_url")
            source_url = str(raw_url).strip() if raw_url else ""
            if not source_url:
                continue
            guid = entry.get("guid") or None
            rec, created = await self._store.upsert_video_ref(
                canonical_id=_canonical_id(guid, source_url),
                source_url=source_url,
                feed_id=feed.id,
                guid=guid,
                title=entry.get("title") or None,
                published_at=entry.get("published_at"),
            )
            if created or rec.status in (VideoStatus.PENDING, VideoStatus.FAILED):
                await self._queue.enqueue(
                    JobKind.PROCESS,
                    {"video_id": rec.id},
                    video_id=rec.id,
                    dedup_key=f"process:{rec.id}",
                )
                enqueued += 1
                if created:
                    new_videos += 1
            else:
                skipped_done += 1

        await self._store.save_feed_state(
            feed.id, title=title, site_url=site_url, etag=etag, error=None
        )
        fresh = await self._store.get_feed_by_url(url)
        return IngestResult(
            feed=fresh or feed,
            new_videos=new_videos,
            enqueued=enqueued,
            skipped_done=skipped_done,
        )
