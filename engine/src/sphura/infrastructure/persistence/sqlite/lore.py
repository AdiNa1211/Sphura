from __future__ import annotations

import json
from typing import Any

from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase


class SqliteLoreRepository:
    """Lore Studio storage: per-video index state, frame tags, scenes, lore passages, ideas."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    # ---------- index state ----------

    def index_state(self, asset_id: str) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM lore_index WHERE asset_id=?", (asset_id,)
        ).fetchone()
        return _row(row) if row else None

    def index_states(self) -> dict[str, dict[str, Any]]:
        rows = self._db.conn.execute("SELECT * FROM lore_index").fetchall()
        return {r["asset_id"]: _row(r) for r in rows}

    def save_index(
        self,
        asset_id: str,
        project_id: str,
        game: str | None,
        fingerprint: str,
        version: int,
        status: str,
        data: dict[str, Any],
        now: str,
    ) -> None:
        with self._db.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO lore_index"
                " (asset_id, project_id, game, fingerprint, version, status, data, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (asset_id, project_id, game, fingerprint, version, status, json.dumps(data), now),
            )

    # ---------- frames (the resumable checkpoint) ----------

    def frames(self, asset_id: str, fingerprint: str, version: int) -> dict[float, dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT t_s, data FROM lore_frames WHERE asset_id=? AND fingerprint=? AND version=?",
            (asset_id, fingerprint, version),
        ).fetchall()
        return {round(r["t_s"], 2): json.loads(r["data"]) for r in rows}

    def add_frames(
        self, asset_id: str, fingerprint: str, version: int, frames: dict[float, dict[str, Any]]
    ) -> None:
        with self._db.tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO lore_frames (asset_id, fingerprint, version, t_s, data)"
                " VALUES (?, ?, ?, ?, ?)",
                [(asset_id, fingerprint, version, t, json.dumps(d)) for t, d in frames.items()],
            )

    def drop_old_frames(self, asset_id: str, fingerprint: str, version: int) -> None:
        with self._db.tx() as c:
            c.execute(
                "DELETE FROM lore_frames WHERE asset_id=? AND (fingerprint<>? OR version<>?)",
                (asset_id, fingerprint, version),
            )

    # ---------- scenes ----------

    def replace_scenes(self, asset_id: str, scenes: list[dict[str, Any]]) -> None:
        with self._db.tx() as c:
            c.execute("DELETE FROM lore_scenes WHERE asset_id=?", (asset_id,))
            c.executemany(
                "INSERT INTO lore_scenes (id, asset_id, project_id, game, start_s, end_s, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s["id"],
                        asset_id,
                        s["project_id"],
                        s.get("game"),
                        s["start_s"],
                        s["end_s"],
                        json.dumps({k: v for k, v in s.items() if k not in _SCENE_COLS}),
                    )
                    for s in scenes
                ],
            )

    def scenes(self, game: str | None = None, asset_id: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM lore_scenes WHERE 1=1", []
        if game:
            sql += " AND game=?"
            args.append(game)
        if asset_id:
            sql += " AND asset_id=?"
            args.append(asset_id)
        rows = self._db.conn.execute(sql + " ORDER BY asset_id, start_s", args).fetchall()
        return [_scene(r) for r in rows]

    def scenes_by_id(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self._db.conn.execute(f"SELECT * FROM lore_scenes WHERE id IN ({q})", ids)
        return {r["id"]: _scene(r) for r in rows.fetchall()}

    # ---------- lore sources ----------

    def page(self, game: str, entity: str, source: str) -> dict[str, Any] | None:
        row = self._db.conn.execute(
            "SELECT * FROM lore_pages WHERE game=? AND entity=? AND source=?",
            (game, entity, source),
        ).fetchone()
        return _row(row) if row else None

    def pages(self, game: str) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT * FROM lore_pages WHERE game=? ORDER BY entity, source", (game,)
        ).fetchall()
        return [_row(r) for r in rows]

    def save_page(
        self,
        game: str,
        entity: str,
        source: str,
        status: str,
        data: dict[str, Any],
        passages: list[dict[str, Any]],
        now: str,
    ) -> None:
        """A fetched page replaces its earlier passages; a failed fetch keeps them."""
        with self._db.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO lore_pages (game, entity, source, status, data, fetched_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (game, entity, source, status, json.dumps(data), now),
            )
            if status == "failed":
                return
            c.execute(
                "DELETE FROM lore_passages WHERE game=? AND entity=? AND source=?",
                (game, entity, source),
            )
            c.executemany(
                "INSERT OR REPLACE INTO lore_passages"
                " (id, game, entity, source, kind, data, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        p["id"],
                        game,
                        entity,
                        source,
                        p["kind"],
                        json.dumps({k: v for k, v in p.items() if k not in ("id", "kind")}),
                        now,
                    )
                    for p in passages
                ],
            )

    def passages(self, game: str, entity: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM lore_passages WHERE game=?", [game]
        if entity:
            sql += " AND entity=?"
            args.append(entity)
        return [_passage(r) for r in self._db.conn.execute(sql, args).fetchall()]

    def passages_by_id(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self._db.conn.execute(f"SELECT * FROM lore_passages WHERE id IN ({q})", ids)
        return {r["id"]: _passage(r) for r in rows.fetchall()}

    # ---------- ideas ----------

    def replace_ideas(self, game: str, ideas: list[dict[str, Any]], now: str) -> None:
        with self._db.tx() as c:
            c.execute("DELETE FROM lore_ideas WHERE game=?", (game,))
            c.executemany(
                "INSERT INTO lore_ideas (id, game, category, score, data, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(i["id"], game, i["category"], i["score"], json.dumps(i), now) for i in ideas],
            )

    def ideas(self, game: str) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT data, created_at FROM lore_ideas WHERE game=? ORDER BY score DESC", (game,)
        ).fetchall()
        return [{**json.loads(r["data"]), "created_at": r["created_at"]} for r in rows]


_SCENE_COLS = ("id", "asset_id", "project_id", "game", "start_s", "end_s")


def _row(r: Any) -> dict[str, Any]:
    d = dict(r)
    d["data"] = json.loads(d["data"])
    return d


def _scene(r: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(r["data"])
    return {k: r[k] for k in _SCENE_COLS} | data


def _passage(r: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(r["data"])
    return {
        "id": r["id"],
        "entity": r["entity"],
        "source": r["source"],
        "kind": r["kind"],
        "fetched_at": r["fetched_at"],
    } | data
