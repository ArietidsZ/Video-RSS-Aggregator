"""SQLite (aiosqlite) connection + migrations. OWNER: agent-C.

- Database.connect(path): WAL, synchronous=NORMAL, foreign_keys=ON, busy_timeout
  5000, migrate() applies schema.py SCHEMA + schema_version row (v2 baseline;
  no v1 migration — fresh DB; if legacy tables detected, rename file .v1.bak once).
- commit-per-write batched per logical op is fine (low volume).
"""
