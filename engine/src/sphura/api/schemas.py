from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from sphura.domain.models import (
    Container,
    EditingMode,
    Job,
    JobStatus,
    JobType,
    MediaAsset,
    Project,
    ProjectStatus,
)


class IngestRequest(BaseModel):
    path: str = Field(description="Absolute path to a local video file (local mode only)")
    project_id: str | None = None
    project_name: str | None = None
    mode: EditingMode = EditingMode.AUTO


class ProjectOut(BaseModel):
    id: str
    name: str
    mode: EditingMode
    status: ProjectStatus
    created_at: datetime

    @classmethod
    def of(cls, p: Project) -> ProjectOut:
        return cls(id=p.id, name=p.name, mode=p.mode, status=p.status, created_at=p.created_at)


class MediaAssetOut(BaseModel):
    id: str
    project_id: str
    original_name: str
    source_uri: str
    container: Container
    size_bytes: int
    fingerprint: str
    sha256: str | None
    media_info: dict[str, Any] | None
    created_at: datetime

    @classmethod
    def of(cls, a: MediaAsset) -> MediaAssetOut:
        return cls(
            id=a.id,
            project_id=a.project_id,
            original_name=a.original_name,
            source_uri=a.source_uri,
            container=a.container,
            size_bytes=a.size_bytes,
            fingerprint=a.fingerprint,
            sha256=a.sha256,
            media_info=asdict(a.media_info) if a.media_info else None,
            created_at=a.created_at,
        )


class JobOut(BaseModel):
    id: str
    project_id: str
    type: JobType
    status: JobStatus
    stage: str | None
    progress: float
    result: dict[str, Any] | None
    error: str | None
    attempts: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, j: Job) -> JobOut:
        return cls(
            id=j.id,
            project_id=j.project_id,
            type=j.type,
            status=j.status,
            stage=j.stage,
            progress=j.progress,
            result=j.result,
            error=j.error,
            attempts=j.attempts,
            created_at=j.created_at,
            started_at=j.started_at,
            finished_at=j.finished_at,
        )


class AnalyzeRequest(BaseModel):
    language: str | None = Field(
        default=None, description="ISO code like 'en' or 'hi'; omit to auto-detect"
    )


class IngestResponse(BaseModel):
    project: ProjectOut
    asset: MediaAssetOut
    checksum_job: JobOut | None
    already_registered: bool
