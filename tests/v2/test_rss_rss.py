"""RSS writer tests: structure, namespaces, escaping, feedparser validation."""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree.ElementTree import fromstring

import feedparser

from vra.config import Config
from vra.rss import RssWriter

_MEDIA_NS = "http://search.yahoo.com/mrss/"
_ATOM_NS = "http://www.w3.org/2005/Atom"


def _config(**overrides: object) -> Config:
    return Config(
        rss_title="Test Feed",
        rss_link="http://example.com:8080/rss",
        rss_description="Test description",
        **overrides,  # type: ignore[arg-type]
    )


def _item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "video_id": "11111111-2222-3333-4444-555555555555",
        "canonical_id": "url:abcdef0123456789",
        "source_url": "https://example.com/watch/1",
        "title": "First video",
        "published_at": datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC),
        "summary": "A summary",
        "key_points": ["point one", "point two"],
        "frame_count": 2,
    }
    item.update(overrides)
    return item


def _render(items: list[dict[str, object]], config: Config | None = None) -> str:
    return RssWriter(config or _config()).render(items)


def test_channel_metadata_and_atom_self_link() -> None:
    xml = _render([])
    root = fromstring(xml)
    assert root.tag == "rss" and root.attrib["version"] == "2.0"
    channel = root.find("channel")
    assert channel is not None
    assert channel.findtext("title") == "Test Feed"
    assert channel.findtext("link") == "http://example.com:8080/rss"
    assert channel.findtext("description") == "Test description"
    atom_self = channel.find(f"{{{_ATOM_NS}}}link")
    assert atom_self is not None
    assert atom_self.attrib["rel"] == "self"
    assert atom_self.attrib["href"] == "http://example.com:8080/rss"
    assert channel.findtext("lastBuildDate")


def test_item_core_fields_and_guid_not_permalink() -> None:
    xml = _render([_item()])
    item = fromstring(xml).find("channel/item")
    assert item is not None
    assert item.findtext("title") == "First video"
    assert item.findtext("link") == "https://example.com/watch/1"
    guid = item.find("guid")
    assert guid is not None
    assert guid.attrib["isPermaLink"] == "false"
    assert guid.text == "url:abcdef0123456789"
    pub = parsedate_to_datetime(item.findtext("pubDate") or "")
    assert pub.date() == datetime(2026, 2, 1, tzinfo=UTC).date()


def test_item_title_fallback_when_missing() -> None:
    xml = _render([_item(title=None)])
    assert "<title>Untitled video</title>" in xml


def test_description_embeds_html_bullets_with_single_escaping() -> None:
    xml = _render([_item(summary="Fish & chips <tag>", key_points=["a & b", "<x>"])])
    item = fromstring(xml).find("channel/item")
    assert item is not None
    desc = item.findtext("description") or ""
    assert "Fish &amp; chips &lt;tag&gt;" in desc
    assert "<ul><li>a &amp; b</li><li>&lt;x&gt;</li></ul>" in desc


def test_media_thumbnail_only_when_frames_exist() -> None:
    xml = _render([_item(), _item(video_id="99999999-8888-7777-6666-555555555555", frame_count=0)])
    root = fromstring(xml)
    items = root.findall("channel/item")
    assert len(items) == 2
    thumb = items[0].find(f"{{{_MEDIA_NS}}}thumbnail")
    assert thumb is not None
    assert thumb.attrib["url"] == "http://example.com:8080/media/11111111-2222-3333-4444-555555555555/0.jpg"
    assert items[1].find(f"{{{_MEDIA_NS}}}thumbnail") is None


def test_published_at_variants() -> None:
    iso = _render([_item(published_at="2026-03-05T04:05:06+00:00")])
    assert "2026" in iso and "Mar" in iso
    passthrough = _render([_item(published_at="not a date")])
    assert "not a date" in passthrough
    naive = _render([_item(published_at=datetime(2026, 1, 2, 3, 4, 5))])
    pub = parsedate_to_datetime(
        fromstring(naive).find("channel/item/pubDate").text or ""  # type: ignore[union-attr]
    )
    assert pub.tzinfo is not None
    now = _render([_item(published_at=None)])
    assert parsedate_to_datetime(fromstring(now).find("channel/item/pubDate").text or "")  # type: ignore[union-attr]


def test_feedparser_validates_feed() -> None:
    parsed = feedparser.parse(_render([_item(), _item(title="Second")]))
    assert parsed.bozo == 0
    assert parsed.feed.title == "Test Feed"
    assert parsed.feed.get("link") == "http://example.com:8080/rss"
    assert len(parsed.entries) == 2
    entry = parsed.entries[0]
    assert entry.title == "First video"
    assert entry.link == "https://example.com/watch/1"
    assert entry.get("id") == "url:abcdef0123456789"


def test_output_is_utf8_with_xml_declaration() -> None:
    xml = _render([_item(title="视频 Ünïcode")])
    assert xml.startswith("<?xml version='1.0' encoding='utf-8'?>")
    assert "视频" in xml
