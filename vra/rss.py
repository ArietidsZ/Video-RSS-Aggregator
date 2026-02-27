from __future__ import annotations

from datetime import datetime, timezone
from xml.etree.ElementTree import Element, SubElement, tostring

from .storage import SummaryRecord


def render_feed(
    title: str,
    link: str,
    description: str,
    records: list[SummaryRecord],
) -> str:
    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    SubElement(channel, "title").text = title
    SubElement(channel, "link").text = link
    SubElement(channel, "description").text = description

    for rec in records:
        item = SubElement(channel, "item")
        SubElement(item, "title").text = rec.title or "Untitled video"
        SubElement(item, "link").text = rec.source_url

        desc = rec.summary
        if rec.key_points:
            bullets = "\n".join(f"- {p}" for p in rec.key_points)
            desc = f"{desc}\n\n{bullets}"
        SubElement(item, "description").text = desc

        if rec.published_at:
            SubElement(item, "pubDate").text = _rfc2822(rec.published_at)

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(rss, encoding="unicode")


def _rfc2822(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S %z")
