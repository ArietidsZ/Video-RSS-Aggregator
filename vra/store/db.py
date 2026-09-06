"""SQLite (aiosqlite) connection, PRAGMAs, and migrations for VRA v2.

One serialized connection is shared by all repositories. Writes go through
:meth:`Database.execute` (autocommit per statement) or :meth:`Database.transaction`
(atomic multi-statement groups); both run under a process-local
:class:`asyncio.Lock` — aiosqlite already serializes statements on a single
connection, the lock removes interleaving *between* logical operations.

``migrate()`` applies ``vra.store.schema.SCHEMA`` and manages the
``schema_version`` row (v2 baseline; no in-place v1 upgrade — a legacy v1 file
is detected by a ``feeds`` table without a ``poll_minutes`` column, or a
version row < 2, and is renamed to ``<name>.v1.bak`` once before a fresh v2
database is created).

Local filesystem metadata ops here (mkdir/exists/replace) are single fast
syscalls on the DB path; all actual IO (sqlite handles, WAL files) runs on the
aiosqlite worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from vra.store.schema import SCHEMA, SCHEMA_VERSION

log = logging.getLogger(__name__)

# Applied in order on every open connection (see docs/v2/SPEC.md).
_PRAGMAS: tuple[str, ...] = (
    "journal_mode=WAL",
    "synchronous=NORMAL",
    "foreign_keys=ON",
    "busy_timeout=5000",
    "temp_store=MEMORY",
)


class Database:
    """Async wrapper around one aiosqlite connection + migration logic."""

    def __init__(self, conn: aiosqlite.Connection, path: str) -> None:
        self._conn = conn
        self._path = path
        self._closed = False
        #: Serializes logical write transactions across repositories.
        self.lock = asyncio.Lock()

    @classmethod
    async def connect(cls, path: str) -> Database:
        """Open (creating parent dirs) a SQLite database with v2 PRAGMAs."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = await cls._open(str(target))
        return cls(conn, str(target))

    @staticmethod
    async def _open(path: str) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(path)
        conn.row_factory = aiosqlite.Row
        for pragma in _PRAGMAS:
            await conn.execute(f"PRAGMA {pragma}")
        await conn.commit()
        return conn

    @property
    def path(self) -> str:
        """Filesystem path of the database file."""
        return self._path

    async def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        if self._closed:
            return
        self._closed = True
        await self._conn.close()

    async def migrate(self) -> None:
        """Create the v2 schema and manage the schema_version row.

        A legacy v1 file (version row < 2, or a feeds table lacking the
        poll_minutes column) is closed, renamed to ``<path>.v1.bak`` once, and
        replaced by a freshly migrated v2 database.
        """
        version = await self._schema_version()
        if version is not None and version < SCHEMA_VERSION:
            await self._replace_legacy()
            return
        columns = await self._table_columns("feeds")
        if columns and "poll_minutes" not in columns:
            await self._replace_legacy()
            return

        await self._conn.executescript(SCHEMA)
        if version is None:
            await self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
            await self._conn.commit()

    async def _schema_version(self) -> int | None:
        """Current schema_version value, or None when absent/empty."""
        exists = await self.fetchone(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        )
        if exists is None:
            return None
        row = await self.fetchone("SELECT version FROM schema_version")
        if row is None:
            return None
        return int(row["version"])

    async def _table_columns(self, table: str) -> list[str]:
        """Column names of ``table``; empty list when the table is absent."""
        rows = await self.fetchall(f"PRAGMA table_info({table})")
        return [str(row["name"]) for row in rows]

    async def _replace_legacy(self) -> None:
        """Rename the legacy v1 file to .v1.bak and build a fresh v2 database."""
        backup = f"{self._path}.v1.bak"
        await self._conn.close()
        self._closed = True
        overwrote = os.path.exists(backup)
        os.replace(self._path, backup)
        log.warning(
            "legacy v1 database moved to %s%s", backup,
            " (previous backup overwritten)" if overwrote else "",
        )
        conn = await self._open(self._path)
        self._conn = conn
        self._closed = False
        await conn.executescript(SCHEMA)
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
        )
        await conn.commit()

    async def execute(
        self,
        sql: str,
        params: Sequence[Any] = (),
        *,
        commit: bool = True,
    ) -> list[aiosqlite.Row]:
        """Run one statement and return any rows (e.g. ``RETURNING`` output)."""
        cur = await self._conn.execute(sql, tuple(params))
        try:
            rows = await cur.fetchall()
        finally:
            await cur.close()
        if commit:
            await self._conn.commit()
        return rows

    async def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> aiosqlite.Row | None:
        """Run one read statement, returning the first row or None."""
        rows = await self.execute(sql, params, commit=False)
        return rows[0] if rows else None

    async def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[aiosqlite.Row]:
        """Run one read statement and return all rows."""
        return await self.execute(sql, params, commit=False)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Run a group of writes atomically, serialized on :attr:`lock`.

        Inside the block pass ``commit=False`` to :meth:`execute`; the commit
        (or rollback on error) happens once when the block exits. Not
        reentrant — never nest.
        """
        async with self.lock:
            try:
                yield
            except BaseException:
                await self._conn.rollback()
                raise
            await self._conn.commit()
