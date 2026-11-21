from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from sphura.domain.clip_codec import clip_from_dict, clip_to_dict
from sphura.domain.models import (
    AudioStream,
    Clip,
    Container,
    EditingMode,
    HighlightCandidate,
    Job,
    JobStatus,
    JobType,
    MediaAsset,
    MediaInfo,
    Project,
    ProjectStatus,
    StorageKind,
    TimeRange,
    VideoStream,
    utcnow,
)
from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase
from sphura.pipeline.judge import judgement_from_dict, judgement_to_dict
from sphura.pipeline.preferences import Rating


def _ts(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _media_info_to_json(info: MediaInfo | None) -> str | None:
    return json.dumps(asdict(info)) if info else None


def _media_info_from_json(s: str | None) -> MediaInfo | None:
    if not s:
        return None
    d: dict[str, Any] = json.loads(s)
    return MediaInfo(
        duration_s=d["duration_s"],
        video=tuple(VideoStream(**v) for v in d["video"]),
        audio=tuple(AudioStream(**a) for a in d["audio"]),
        format_name=d["format_name"],
        bit_rate=d.get("bit_rate"),
    )


class SqliteProjectRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> Project:
        return Project(
            id=r["id"],
            name=r["name"],
            mode=EditingMode(r["mode"]),
            status=ProjectStatus(r["status"]),
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    def add(self, project: Project) -> None:
        self._db.conn.execute(
            "INSERT INTO projects (id, name, mode, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (project.id, project.name, project.mode, project.status, _ts(project.created_at)),
        )

    def get(self, project_id: str) -> Project | None:
        r = self._db.conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, limit: int = 100) -> list[Project]:
        rows = self._db.conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def update(self, project: Project) -> None:
        self._db.conn.execute(
            "UPDATE projects SET name = ?, mode = ?, status = ? WHERE id = ?",
            (project.name, project.mode, project.status, project.id),
        )


class SqliteMediaAssetRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> MediaAsset:
        return MediaAsset(
            id=r["id"],
            project_id=r["project_id"],
            source_uri=r["source_uri"],
            storage_kind=StorageKind(r["storage_kind"]),
            container=Container(r["container"]),
            size_bytes=r["size_bytes"],
            fingerprint=r["fingerprint"],
            original_name=r["original_name"],
            sha256=r["sha256"],
            media_info=_media_info_from_json(r["media_info"]),
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    def add(self, asset: MediaAsset) -> None:
        self._db.conn.execute(
            """INSERT INTO media_assets (id, project_id, source_uri, storage_kind, container,
                 size_bytes, fingerprint, original_name, sha256, media_info, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asset.id,
                asset.project_id,
                asset.source_uri,
                asset.storage_kind,
                asset.container,
                asset.size_bytes,
                asset.fingerprint,
                asset.original_name,
                asset.sha256,
                _media_info_to_json(asset.media_info),
                _ts(asset.created_at),
            ),
        )

    def get(self, asset_id: str) -> MediaAsset | None:
        r = self._db.conn.execute("SELECT * FROM media_assets WHERE id = ?", (asset_id,)).fetchone()
        return self._row(r) if r else None

    def list_for_project(self, project_id: str) -> list[MediaAsset]:
        rows = self._db.conn.execute(
            "SELECT * FROM media_assets WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def find_by_fingerprint(self, fingerprint: str) -> list[MediaAsset]:
        rows = self._db.conn.execute(
            "SELECT * FROM media_assets WHERE fingerprint = ?", (fingerprint,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def update(self, asset: MediaAsset) -> None:
        self._db.conn.execute(
            "UPDATE media_assets SET source_uri = ?, sha256 = ?, media_info = ? WHERE id = ?",
            (asset.source_uri, asset.sha256, _media_info_to_json(asset.media_info), asset.id),
        )


class SqliteJobRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> Job:
        return Job(
            id=r["id"],
            project_id=r["project_id"],
            type=JobType(r["type"]),
            status=JobStatus(r["status"]),
            stage=r["stage"],
            progress=r["progress"],
            payload=json.loads(r["payload"]),
            result=json.loads(r["result"]) if r["result"] else None,
            error=r["error"],
            attempts=r["attempts"],
            created_at=datetime.fromisoformat(r["created_at"]),
            started_at=_dt(r["started_at"]),
            finished_at=_dt(r["finished_at"]),
        )

    def add(self, job: Job) -> None:
        self._db.conn.execute(
            """INSERT INTO jobs (id, project_id, type, status, stage, progress, payload, result,
                 error, attempts, created_at, started_at, finished_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.project_id,
                job.type,
                job.status,
                job.stage,
                job.progress,
                json.dumps(job.payload),
                json.dumps(job.result) if job.result is not None else None,
                job.error,
                job.attempts,
                _ts(job.created_at),
                _ts(job.started_at),
                _ts(job.finished_at),
            ),
        )

    def get(self, job_id: str) -> Job | None:
        r = self._db.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row(r) if r else None

    def status_of(self, job_id: str) -> JobStatus | None:
        r = self._db.conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return JobStatus(r["status"]) if r else None

    def list_for_project(self, project_id: str) -> list[Job]:
        rows = self._db.conn.execute(
            "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def update(self, job: Job) -> None:
        self._db.conn.execute(
            """UPDATE jobs SET status = ?, stage = ?, progress = ?, result = ?, error = ?,
                 attempts = ?, started_at = ?, finished_at = ?, heartbeat_at = ?
               WHERE id = ?""",
            (
                job.status,
                job.stage,
                job.progress,
                json.dumps(job.result) if job.result is not None else None,
                job.error,
                job.attempts,
                _ts(job.started_at),
                _ts(job.finished_at),
                _ts(utcnow()),
                job.id,
            ),
        )

    def claim_next(self) -> Job | None:
        now = _ts(utcnow())
        with self._db.tx() as c:
            r = c.execute(
                """UPDATE jobs SET status = 'running', started_at = ?, heartbeat_at = ?,
                     attempts = attempts + 1
                   WHERE id = (SELECT id FROM jobs WHERE status = 'queued'
                               ORDER BY created_at LIMIT 1)
                   RETURNING *""",
                (now, now),
            ).fetchone()
        return self._row(r) if r else None

    def heartbeat(self, job_id: str) -> None:
        self._db.conn.execute(
            "UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (_ts(utcnow()), job_id)
        )

    def update_progress(self, job_id: str, stage: str | None, progress: float) -> None:
        self._db.conn.execute(
            "UPDATE jobs SET stage = ?, progress = ?, heartbeat_at = ? WHERE id = ?",
            (stage, progress, _ts(utcnow()), job_id),
        )

    def try_cancel(self, job_id: str) -> bool:
        cur = self._db.conn.execute(
            """UPDATE jobs SET status = 'cancelled', finished_at = ?
               WHERE id = ? AND status IN ('queued', 'running')""",
            (_ts(utcnow()), job_id),
        )
        return cur.rowcount == 1

    def requeue_stale(self, older_than_s: float) -> int:
        cutoff = _ts(utcnow() - timedelta(seconds=older_than_s))
        cur = self._db.conn.execute(
            """UPDATE jobs SET status = 'queued', stage = NULL
               WHERE status = 'running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)""",
            (cutoff,),
        )
        return cur.rowcount


class SqliteCandidateRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def replace_for_asset(self, asset_id: str, candidates: list[HighlightCandidate]) -> None:
        now = _ts(utcnow())
        with self._db.tx() as c:
            c.execute("DELETE FROM candidates WHERE asset_id = ?", (asset_id,))
            c.executemany(
                """INSERT INTO candidates (id, project_id, asset_id, start_s, end_s, peak_s,
                     signal_score, peak_score, drivers, features, judgement, compare_score,
                     created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        x.id,
                        x.project_id,
                        x.asset_id,
                        x.range.start_s,
                        x.range.end_s,
                        x.peak_s,
                        x.signal_score,
                        x.peak_score,
                        json.dumps(list(x.drivers)),
                        json.dumps(x.features),
                        json.dumps(judgement_to_dict(x.judgement)) if x.judgement else None,
                        x.compare_score,
                        now,
                    )
                    for x in candidates
                ],
            )

    def list_for_project(self, project_id: str) -> list[HighlightCandidate]:
        rows = self._db.conn.execute(
            "SELECT * FROM candidates WHERE project_id = ? ORDER BY start_s", (project_id,)
        ).fetchall()
        return [
            HighlightCandidate(
                id=r["id"],
                project_id=r["project_id"],
                asset_id=r["asset_id"],
                range=TimeRange(r["start_s"], r["end_s"]),
                peak_s=r["peak_s"],
                signal_score=r["signal_score"],
                peak_score=r["peak_score"],
                drivers=tuple(json.loads(r["drivers"])),
                features=json.loads(r["features"]),
                judgement=judgement_from_dict(json.loads(r["judgement"]))
                if r["judgement"]
                else None,
                compare_score=r["compare_score"],
            )
            for r in rows
        ]


class SqliteClipRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def save(self, clip: Clip) -> None:
        now = _ts(utcnow())
        self._db.conn.execute(
            """INSERT INTO clips (id, project_id, asset_id, candidate_id, title, edl, version,
                 created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET title = excluded.title, edl = excluded.edl,
                 version = excluded.version, updated_at = excluded.updated_at""",
            (
                clip.id,
                clip.project_id,
                clip.asset_id,
                clip.candidate_id,
                clip.title,
                json.dumps(clip_to_dict(clip), ensure_ascii=False),
                clip.version,
                now,
                now,
            ),
        )

    def get(self, clip_id: str) -> Clip | None:
        r = self._db.conn.execute("SELECT edl FROM clips WHERE id = ?", (clip_id,)).fetchone()
        return clip_from_dict(json.loads(r["edl"])) if r else None

    def list_for_project(self, project_id: str) -> list[Clip]:
        rows = self._db.conn.execute(
            "SELECT edl FROM clips WHERE project_id = ? ORDER BY created_at", (project_id,)
        ).fetchall()
        return [clip_from_dict(json.loads(r["edl"])) for r in rows]

    def find_by_candidate(self, candidate_id: str, style: str | None = None) -> Clip | None:
        rows = self._db.conn.execute(
            "SELECT edl FROM clips WHERE candidate_id = ? ORDER BY updated_at DESC",
            (candidate_id,),
        ).fetchall()
        for r in rows:
            clip = clip_from_dict(json.loads(r["edl"]))
            if style is None or clip.style == style:
                return clip
        return None

    def set_output(self, clip_id: str, path: str) -> None:
        self._db.conn.execute(
            "UPDATE clips SET output_path = ?, rendered_at = ? WHERE id = ?",
            (path, _ts(utcnow()), clip_id),
        )

    def delete_except(self, project_id: str, candidate_ids: set[str]) -> int:
        rows = self._db.conn.execute(
            "SELECT id, candidate_id FROM clips WHERE project_id = ?", (project_id,)
        ).fetchall()
        stale = [r["id"] for r in rows if r["candidate_id"] not in candidate_ids]
        self._db.conn.executemany("DELETE FROM clips WHERE id = ?", [(i,) for i in stale])
        return len(stale)

    def output_of(self, clip_id: str) -> str | None:
        r = self._db.conn.execute(
            "SELECT output_path FROM clips WHERE id = ?", (clip_id,)
        ).fetchone()
        return str(r["output_path"]) if r and r["output_path"] else None


class SqliteFeedbackRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def rate(self, project_id: str, rating: Rating) -> None:
        self._db.conn.execute(
            """INSERT INTO feedback (candidate_id, project_id, good, traits, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(candidate_id) DO UPDATE SET good = excluded.good,
                 traits = excluded.traits, created_at = excluded.created_at""",
            (
                rating.candidate_id,
                project_id,
                int(rating.good),
                json.dumps(list(rating.traits)),
                _ts(utcnow()),
            ),
        )

    def set_override(self, trait: str, likes: bool | None) -> None:
        if likes is None:
            self._db.conn.execute("DELETE FROM taste_overrides WHERE trait = ?", (trait,))
            return
        self._db.conn.execute(
            """INSERT INTO taste_overrides (trait, likes, created_at) VALUES (?, ?, ?)
               ON CONFLICT(trait) DO UPDATE SET likes = excluded.likes""",
            (trait, int(likes), _ts(utcnow())),
        )

    def overrides(self) -> dict[str, bool]:
        rows = self._db.conn.execute("SELECT trait, likes FROM taste_overrides").fetchall()
        return {r["trait"]: bool(r["likes"]) for r in rows}

    def all(self) -> list[Rating]:
        rows = self._db.conn.execute("SELECT * FROM feedback").fetchall()
        return [
            Rating(r["candidate_id"], bool(r["good"]), tuple(json.loads(r["traits"]))) for r in rows
        ]
