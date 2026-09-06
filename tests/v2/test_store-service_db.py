"""Tests for vra.store.db: connect/PRAGMAs, migrate, v1 backup handling."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from vra.store.db import Database
from vra.store.schema import SCHEMA_VERSION

_V1_DDL = """
CREATE TABLE feeds (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    last_checked TEXT NOT NULL
);
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    feed_id TEXT,
    source_url TEXT NOT NULL UNIQUE,
    guid TEXT,
    title TEXT,
    published_at TEXT,
    media_path TEXT,
    created_at TEXT NOT NULL
);
"""


def _make_v1_file(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_V1_DDL)
        conn.execute(
            "INSERT INTO feeds (id, url, last_checked) "
            "VALUES ('f1', 'https://old.example.com', '2020-01-01T00:00:00+00:00')"
        )
        conn.commit()
    finally:
        conn.close()


async def test_connect_creates_parent_dirs_and_pragmas(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "vra.db"
    db = await Database.connect(str(db_path))
    try:
        assert db_path.exists()
        assert db.path == str(db_path)
        assert (await db.fetchone("PRAGMA journal_mode"))[0] == "wal"
        assert (await db.fetchone("PRAGMA foreign_keys"))[0] == 1
        assert (await db.fetchone("PRAGMA busy_timeout"))[0] == 5000
        assert (await db.fetchone("PRAGMA synchronous"))[0] == 1  # NORMAL
        assert (await db.fetchone("PRAGMA temp_store"))[0] == 2  # MEMORY
    finally:
        await db.close()


async def test_migrate_fresh_inserts_schema_version(tmp_path: Path) -> None:
    db = await Database.connect(str(tmp_path / "vra.db"))
    try:
        await db.migrate()
        rows = await db.fetchall("SELECT version FROM schema_version")
        assert [row["version"] for row in rows] == [SCHEMA_VERSION]
    finally:
        await db.close()


async def test_migrate_idempotent(tmp_path: Path) -> None:
    db = await Database.connect(str(tmp_path / "vra.db"))
    try:
        await db.migrate()
        await db.migrate()
        rows = await db.fetchall("SELECT version FROM schema_version")
        assert [row["version"] for row in rows] == [SCHEMA_VERSION]
        assert not list(tmp_path.glob("*.v1.bak"))
    finally:
        await db.close()


async def test_migrate_renames_legacy_v1_file(tmp_path: Path) -> None:
    db_path = tmp_path / "vra.db"
    _make_v1_file(str(db_path))
    db = await Database.connect(str(db_path))
    try:
        await db.migrate()
        backup = tmp_path / "vra.db.v1.bak"
        assert backup.exists()
        # Reopened fresh: v2 schema present, legacy rows gone.
        columns = await db.fetchall("PRAGMA table_info(feeds)")
        assert "poll_minutes" in {row["name"] for row in columns}
        assert (await db.fetchone("SELECT COUNT(*) AS n FROM feeds"))["n"] == 0
        assert (await db.fetchone("SELECT version FROM schema_version"))[
            "version"
        ] == SCHEMA_VERSION
    finally:
        await db.close()
    # Original path is a live v2 database again; data preserved only in backup.
    conn = sqlite3.connect(str(backup))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "feeds" in tables
    finally:
        conn.close()


async def test_migrate_renames_version_row_below_2(tmp_path: Path) -> None:
    db_path = tmp_path / "vra.db"
    db = await Database.connect(str(db_path))
    await db.migrate()
    await db.close()

    stale = sqlite3.connect(str(db_path))
    try:
        stale.execute("UPDATE schema_version SET version = 1")
        stale.commit()
    finally:
        stale.close()

    db = await Database.connect(str(db_path))
    try:
        await db.migrate()
        assert (tmp_path / "vra.db.v1.bak").exists()
        assert (await db.fetchone("SELECT version FROM schema_version"))[
            "version"
        ] == SCHEMA_VERSION
    finally:
        await db.close()


async def test_execute_fetchone_fetchall_helpers(tmp_path: Path) -> None:
    db = await Database.connect(str(tmp_path / "vra.db"))
    try:
        await db.migrate()
        rows = await db.execute(
            "INSERT INTO feeds (id, url, title, created_at, updated_at) "
            "VALUES ('f1', 'https://a', 'A', '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00') RETURNING id"
        )
        assert [row["id"] for row in rows] == ["f1"]
        row = await db.fetchone("SELECT url FROM feeds WHERE id = ?", ("f1",))
        assert row is not None and row["url"] == "https://a"
        assert await db.fetchall("SELECT * FROM feeds") != []
        assert await db.fetchone("SELECT * FROM feeds WHERE id = ?", ("nope",)) is None
    finally:
        await db.close()


async def test_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    db = await Database.connect(str(tmp_path / "vra.db"))
    try:
        await db.migrate()
        with pytest.raises(RuntimeError, match="boom"):
            async with db.transaction():
                await db.execute(
                    "INSERT INTO feeds (id, url, created_at, updated_at) "
                    "VALUES ('f1', 'https://a', 'x', 'x')",
                    commit=False,
                )
                raise RuntimeError("boom")
        assert (await db.fetchone("SELECT COUNT(*) AS n FROM feeds"))["n"] == 0
    finally:
        await db.close()


async def test_close_is_idempotent(tmp_path: Path) -> None:
    db = await Database.connect(str(tmp_path / "vra.db"))
    await db.migrate()
    await db.close()
    await db.close()  # second close must not raise
