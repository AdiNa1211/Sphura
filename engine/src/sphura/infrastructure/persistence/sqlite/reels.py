from __future__ import annotations

import json
from typing import Any

from sphura.domain.models import utcnow
from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase


class SqliteReelRepository:
    """Reels as JSON documents: the shape changes often while the feature grows."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def save(self, reel: dict[str, Any]) -> None:
        now = utcnow().isoformat()
        reel["updated_at"] = now
        reel.setdefault("created_at", now)
        with self._db.tx() as c:
            c.execute(
                """INSERT INTO reels (id, name, status, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, status=excluded.status,
                   data=excluded.data, updated_at=excluded.updated_at""",
                (reel["id"], reel["name"], reel["status"], json.dumps(reel),
                 reel["created_at"], now),
            )  # fmt: skip

    def get(self, reel_id: str) -> dict[str, Any] | None:
        row = self._db.conn.execute("SELECT data FROM reels WHERE id=?", (reel_id,)).fetchone()
        return json.loads(row["data"]) if row else None

    def all(self) -> list[dict[str, Any]]:
        rows = self._db.conn.execute("SELECT data FROM reels ORDER BY created_at DESC").fetchall()
        return [json.loads(r["data"]) for r in rows]

    def delete(self, reel_id: str) -> None:
        with self._db.tx() as c:
            c.execute("DELETE FROM reels WHERE id=?", (reel_id,))
