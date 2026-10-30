"""Conservative source identity shared by feed ingestion and manual requests.

Legacy lossy identities are never adopted automatically: their saved artifacts
may belong to an earlier colliding URL even when source_url now looks correct.
"""
from __future__ import annotations

import hashlib


def source_identity(source: str) -> str:
    value = source.strip()
    if not value:
        raise ValueError('source must not be empty')
    return 'source:v2:' + hashlib.sha256(value.encode('utf-8')).hexdigest()
