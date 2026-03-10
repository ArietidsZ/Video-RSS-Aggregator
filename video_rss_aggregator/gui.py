from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from video_rss_aggregator.config import Config


@lru_cache(maxsize=1)
def _template_text() -> str:
    return (
        files("video_rss_aggregator")
        .joinpath("templates", "setup.html")
        .read_text(encoding="utf-8")
    )


def render_setup_page(config: Config, root_path: str = "") -> str:
    defaults_json = json.dumps(config.as_setup_payload()).replace("</", "<\\/")
    base_href = f"{root_path.rstrip('/')}/" if root_path else "/"
    return (
        _template_text()
        .replace("__BASE_HREF__", base_href)
        .replace("__DEFAULTS_JSON__", defaults_json)
    )
