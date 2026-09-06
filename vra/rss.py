"""RSS 2.0 writer. OWNER: agent-D.

RssWriter(config).render(items) -> XML string built with xml.etree.ElementTree
using the `media` (http://search.yahoo.com/mrss/) and `atom`
(http://www.w3.org/2005/Atom) namespaces.

Items come from VideoStore.latest_publications(); recognized keys:
    video_id: str                              - internal video id (media URL)
    canonical_id: str                          - guid content (isPermaLink=false)
    source_url: str                            - item <link>
    title: str | None                          - item <title> (fallback "Untitled video")
    published_at: datetime | str | None        - item <pubDate> (fallback: build time)
    summary: str                               - description paragraph (HTML-escaped)
    key_points: list[str] | tuple[str, ...]    - description bullet list (HTML-escaped)
    frame_count: int                           - >0 adds media:thumbnail of frame 0

Channel: title/link/description from config, atom:link rel=self pointing at
config.rss_link, lastBuildDate RFC-822 now. Output is UTF-8 with an XML
declaration. No network or filesystem access: render is pure and synchronous.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.sax.saxutils import escape

from vra.config import Config

_MEDIA_NS = "http://search.yahoo.com/mrss/"
_ATOM_NS = "http://www.w3.org/2005/Atom"


def _rfc822(value: datetime | str | None, fallback: datetime) -> str:
    """Format a datetime (or ISO-8601 / RFC-822 string) as an RFC-822 date."""
    parsed: datetime
    if isinstance(value, str):
        text = value.strip()
        if not text:
            parsed = fallback
        else:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(text)
                except (TypeError, ValueError):
                    return text  # already a date-shaped string; pass through
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return format_datetime(parsed)


def _media_base(rss_link: str) -> str:
    """Strip the trailing /rss path from the configured self link."""
    base = rss_link.rstrip("/")
    if base.endswith("/rss"):
        base = base[: -len("/rss")].rstrip("/")
    return base


def _item_description(summary: str, key_points: list[str] | tuple[str, ...]) -> str:
    """HTML body for <description>: escaped paragraph + escaped bullet list.

    The value is stored as element text, so ElementTree applies the single
    XML-escaping pass on serialization and the HTML tags survive as markup.
    """
    parts = [escape(summary or "")]
    parts.append("<br/><br/><ul>")
    for point in key_points:
        parts.append(f"<li>{escape(str(point))}</li>")
    parts.append("</ul>")
    return "".join(parts)


class RssWriter:
    """Render published videos as an RSS 2.0 XML feed."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def render(self, items: list[dict[str, Any]]) -> str:
        """Render `items` (VideoStore.latest_publications shape) to XML text."""
        now = datetime.now(UTC)
        rss = Element(
            "rss",
            {
                "version": "2.0",
                "xmlns:media": _MEDIA_NS,
                "xmlns:atom": _ATOM_NS,
            },
        )
        channel = SubElement(rss, "channel")
        SubElement(channel, "title").text = self._config.rss_title
        SubElement(channel, "link").text = self._config.rss_link
        SubElement(channel, "description").text = self._config.rss_description
        SubElement(channel, "lastBuildDate").text = _rfc822(None, now)
        SubElement(
            channel,
            f"{{{_ATOM_NS}}}link",
            {"rel": "self", "href": self._config.rss_link},
        )

        thumbnail_base = _media_base(self._config.rss_link)
        for item in items:
            entry = SubElement(channel, "item")
            SubElement(entry, "title").text = str(item.get("title") or "Untitled video")
            SubElement(entry, "link").text = str(item.get("source_url", ""))
            SubElement(entry, "guid", {"isPermaLink": "false"}).text = str(
                item.get("canonical_id", "")
            )
            SubElement(entry, "pubDate").text = _rfc822(
                item.get("published_at"), now
            )
            SubElement(entry, "description").text = _item_description(
                str(item.get("summary") or ""),
                item.get("key_points") or [],
            )
            frame_count = item.get("frame_count") or 0
            if frame_count > 0:
                SubElement(
                    entry,
                    f"{{{_MEDIA_NS}}}thumbnail",
                    {
                        "url": f"{thumbnail_base}/media/{item.get('video_id', '')}/0.jpg"
                    },
                )

        xml_bytes = tostring(rss, encoding="utf-8", xml_declaration=True)
        return xml_bytes.decode("utf-8")
