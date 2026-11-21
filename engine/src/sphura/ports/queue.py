from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from sphura.domain.events import JobProgressEvent
from sphura.domain.models import Job, JobType


class JobContext(Protocol):
    """Handed to job handlers: progress reporting + cooperative cancellation."""

    @property
    def job(self) -> Job: ...

    def report(self, stage: str, progress: float, message: str | None = None) -> None: ...

    def check_cancelled(self) -> None:
        """Raise JobCancelledError if the job was cancelled."""
        ...


JobHandler = Callable[[JobContext], dict[str, Any] | None]


class JobQueue(Protocol):
    def register(self, job_type: JobType, handler: JobHandler) -> None: ...
    def enqueue(self, job: Job) -> Job: ...
    def cancel(self, job_id: str) -> bool: ...
    def start(self) -> None: ...
    def stop(self, timeout_s: float = 10.0) -> None: ...


Subscriber = Callable[[JobProgressEvent], None]


class EventBus(Protocol):
    def publish(self, event: JobProgressEvent) -> None: ...
    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """Returns an unsubscribe function."""
        ...
