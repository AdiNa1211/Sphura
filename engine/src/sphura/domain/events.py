from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sphura.domain.models import JobStatus, utcnow


@dataclass(frozen=True, slots=True)
class JobProgressEvent:
    job_id: str
    project_id: str
    status: JobStatus
    stage: str | None
    progress: float
    message: str | None = None
    at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "job.progress",
            "job_id": self.job_id,
            "project_id": self.project_id,
            "status": self.status.value,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "at": self.at.isoformat(),
        }
