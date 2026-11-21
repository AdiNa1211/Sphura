from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path


class SqliteDatabase:
    """One connection per thread (sqlite3 objects are not thread-safe); WAL for concurrent reads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if c is None:
            c = self._connect()
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        # BEGIN IMMEDIATE takes the write lock up front, avoiding upgrade deadlocks
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def _migrate(self) -> None:
        pkg = resources.files("sphura.infrastructure.persistence.sqlite").joinpath("migrations")
        files = sorted((f for f in pkg.iterdir() if f.name.endswith(".sql")), key=lambda f: f.name)
        c = self.conn
        current = c.execute("PRAGMA user_version").fetchone()[0]
        for f in files:
            version = int(f.name.split("_", 1)[0])
            if version <= current:
                continue
            c.executescript(f"BEGIN;\n{f.read_text()}\nPRAGMA user_version={version};\nCOMMIT;")
