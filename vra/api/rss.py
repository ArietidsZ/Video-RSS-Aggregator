"""RSS 2.0 writer. OWNER: agent-D. RssWriter(config) -> render(items) -> str xml.

- Channel from config rss_*; atom:link rel=self (xml.etree only, no new deps);
  per item: title, guid isPermaLink=false (canonical_id), link=source_url,
  pubDate RFC822, description=summary paragraph + key_points <ul>,
  media:thumbnail url=/media/{video_id}/0.jpg when frame exists.
- Escape via xml.sax.saxutils.escape; validate with feedparser.parse in test.
"""
