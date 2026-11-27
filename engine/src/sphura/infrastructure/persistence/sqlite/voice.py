from __future__ import annotations

import json
from typing import Any

from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase
from sphura.pipeline.voice_intel import Term


class SqliteVoiceRepository:
    """Pronunciation dictionaries and voice profiles, one set per game."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def terms(self, game: str) -> list[Term]:
        rows = self._db.conn.execute(
            "SELECT data FROM voice_terms WHERE game=? ORDER BY id", (game,)
        ).fetchall()
        return [Term.from_dict(json.loads(r["data"])) for r in rows]

    def term(self, game: str, term_id: str) -> Term | None:
        r = self._db.conn.execute(
            "SELECT data FROM voice_terms WHERE game=? AND id=?", (game, term_id)
        ).fetchone()
        return Term.from_dict(json.loads(r["data"])) if r else None

    def save_terms(self, game: str, terms: list[Term], now: str) -> None:
        with self._db.tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO voice_terms (game, id, data, source, verified, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(game, t.id, json.dumps(t.to_dict(), ensure_ascii=False), t.source,
                  int(t.verified), now) for t in terms],
            )  # fmt: skip

    def delete_term(self, game: str, term_id: str) -> bool:
        with self._db.tx() as c:
            cur = c.execute("DELETE FROM voice_terms WHERE game=? AND id=?", (game, term_id))
            return cur.rowcount > 0

    def profiles(self) -> dict[str, dict[str, Any]]:
        rows = self._db.conn.execute("SELECT game, data FROM voice_profiles").fetchall()
        return {r["game"]: json.loads(r["data"]) for r in rows}

    def profile(self, game: str) -> dict[str, Any] | None:
        r = self._db.conn.execute(
            "SELECT data FROM voice_profiles WHERE game=?", (game,)
        ).fetchone()
        return json.loads(r["data"]) if r else None

    def save_profile(self, game: str, data: dict[str, Any], now: str) -> None:
        with self._db.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO voice_profiles (game, data, updated_at) VALUES (?, ?, ?)",
                (game, json.dumps(data, ensure_ascii=False), now),
            )

    def settings(self, scope: str = "global") -> dict[str, Any]:
        r = self._db.conn.execute(
            "SELECT data FROM voice_settings WHERE scope=?", (scope,)
        ).fetchone()
        return json.loads(r["data"]) if r else {}

    def save_settings(self, data: dict[str, Any], now: str, scope: str = "global") -> None:
        with self._db.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO voice_settings (scope, data, updated_at) VALUES (?, ?, ?)",
                (scope, json.dumps(data), now),
            )
