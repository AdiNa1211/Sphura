from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from sphura.domain.models import new_id, utcnow
from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase

ACTIVE = ("queued", "downloading", "paused")


@dataclass(slots=True)
class Download:
    video_id: str
    url: str
    title: str
    quality: str
    channel: str = ""
    duration_s: float = 0.0
    thumbnail_url: str = ""
    thumbnail_path: str | None = None
    playlist_title: str | None = None
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    file_path: str | None = None
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    project_id: str | None = None
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: utcnow().isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COLS = [f.name for f in fields(Download)]


class SqliteDownloadRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def add(self, d: Download) -> None:
        marks = ", ".join("?" * len(_COLS))
        with self._db.tx() as c:
            c.execute(
                f"INSERT INTO downloads ({', '.join(_COLS)}) VALUES ({marks})",
                [getattr(d, k) for k in _COLS],
            )

    def update(self, d: Download, **changes: Any) -> None:
        changes["updated_at"] = utcnow().isoformat()
        for k, v in changes.items():
            setattr(d, k, v)
        with self._db.tx() as c:
            c.execute(
                f"UPDATE downloads SET {', '.join(f'{k}=?' for k in changes)} WHERE id=?",
                [*changes.values(), d.id],
            )

    def get(self, download_id: str) -> Download | None:
        row = self._db.conn.execute("SELECT * FROM downloads WHERE id=?", (download_id,)).fetchone()
        return Download(**dict(row)) if row else None

    def all(self) -> list[Download]:
        rows = self._db.conn.execute("SELECT * FROM downloads ORDER BY created_at").fetchall()
        return [Download(**dict(r)) for r in rows]

    def by_video(self, video_id: str) -> list[Download]:
        rows = self._db.conn.execute("SELECT * FROM downloads WHERE video_id=?", (video_id,))
        return [Download(**dict(r)) for r in rows.fetchall()]

    def delete(self, download_id: str) -> None:
        with self._db.tx() as c:
            c.execute("DELETE FROM downloads WHERE id=?", (download_id,))

    def interrupted_to_queued(self) -> int:
        """After a crash or restart, downloads that were running resume from where they were."""
        with self._db.tx() as c:
            return c.execute(
                "UPDATE downloads SET status='queued' WHERE status='downloading'"
            ).rowcount
