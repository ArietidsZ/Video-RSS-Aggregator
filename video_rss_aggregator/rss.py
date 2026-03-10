from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from xml.etree.ElementTree import Element, SubElement, tostring

from video_rss_aggregator.domain.publication import PublicationRecord


def render_feed(
    title: str,
    link: str,
    description: str,
    publications: Sequence[PublicationRecord],
) -> str:
    rss = Element("rss", version="2.0")
    channel = SubElement(rss, "channel")
    SubElement(channel, "title").text = title
    SubElement(channel, "link").text = link
    SubElement(channel, "description").text = description

    for publication in publications:
        item = SubElement(channel, "item")
        SubElement(item, "title").text = publication.title or "Untitled video"
        SubElement(item, "link").text = publication.source_url

        description_parts = [publication.summary]
        if publication.key_points:
            bullets = "\n".join(f"- {point}" for point in publication.key_points)
            description_parts.append(bullets)
        if publication.visual_highlights:
            visuals = "\n".join(
                f"- {highlight}" for highlight in publication.visual_highlights
            )
            description_parts.append(f"Visual Highlights:\n{visuals}")
        if publication.model_used:
            description_parts.append(
                f"Model: {publication.model_used} (VRAM {publication.vram_mb:.2f} MB)"
            )

        SubElement(item, "description").text = "\n\n".join(description_parts)

        if publication.published_at:
            SubElement(item, "pubDate").text = _rfc2822(publication.published_at)

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(
        rss, encoding="unicode"
    )


def _rfc2822(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%a, %d %b %Y %H:%M:%S %z")
