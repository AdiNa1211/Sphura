from __future__ import annotations

import json
from typing import Any

from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase


class SqliteResearchRepository:
    """Researched Shorts (one row per video, never duplicated) plus metric history."""

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def upsert(self, video_id: str, game: str, data: dict[str, Any], now: str) -> bool:
        """True when the video is new to the research set."""
        with self._db.tx() as c:
            row = c.execute(
                "SELECT data FROM research_videos WHERE video_id=?", (video_id,)
            ).fetchone()
            if row:
                merged = {**json.loads(row["data"]), **data}
                queries = set(json.loads(row["data"]).get("queries", [])) | set(
                    data.get("queries", [])
                )
                merged["queries"] = sorted(queries)
                c.execute(
                    "UPDATE research_videos SET data=?, last_seen=? WHERE video_id=?",
                    (json.dumps(merged), now, video_id),
                )
                return False
            c.execute(
                "INSERT INTO research_videos (video_id, game, data, first_seen, last_seen)"
                " VALUES (?, ?, ?, ?, ?)",
                (video_id, game, json.dumps(data), now, now),
            )
            return True

    def snapshot(self, video_id: str, now: str, stats: dict[str, int | None]) -> None:
        with self._db.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO research_snapshots"
                " (video_id, collected_at, views, likes, comments, channel_subs)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (video_id, now, stats.get("views"), stats.get("likes"),
                 stats.get("comments"), stats.get("channel_subs")),
            )  # fmt: skip

    def videos(self, game: str | None = None) -> list[dict[str, Any]]:
        """Every video with its latest and previous snapshot."""
        sql = "SELECT * FROM research_videos" + (" WHERE game=?" if game else "")
        rows = self._db.conn.execute(sql, (game,) if game else ()).fetchall()
        snaps: dict[str, list[dict[str, Any]]] = {}
        for s in self._db.conn.execute(
            "SELECT * FROM research_snapshots ORDER BY collected_at DESC"
        ).fetchall():
            lst = snaps.setdefault(s["video_id"], [])
            if len(lst) < 2:
                lst.append(dict(s))
        out = []
        for r in rows:
            v = {**json.loads(r["data"]), "video_id": r["video_id"], "game": r["game"],
                 "first_seen": r["first_seen"], "last_seen": r["last_seen"]}  # fmt: skip
            hist = snaps.get(r["video_id"], [])
            v["latest"] = hist[0] if hist else None
            v["previous"] = hist[1] if len(hist) > 1 else None
            out.append(v)
        return out

    def ids(self) -> list[str]:
        return [r[0] for r in self._db.conn.execute("SELECT video_id FROM research_videos")]

    def history(self, video_id: str) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT * FROM research_snapshots WHERE video_id=? ORDER BY collected_at",
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_run(self, run: dict[str, Any]) -> None:
        with self._db.tx() as c:
            c.execute(
                """INSERT INTO research_runs (id, started_at, finished_at, status, data)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET finished_at=excluded.finished_at,
                   status=excluded.status, data=excluded.data""",
                (run["id"], run["started_at"], run.get("finished_at"), run["status"],
                 json.dumps(run)),
            )  # fmt: skip

    def runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.conn.execute(
            "SELECT data FROM research_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(r["data"]) for r in rows]
