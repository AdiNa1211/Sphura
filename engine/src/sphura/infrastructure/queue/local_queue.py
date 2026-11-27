from __future__ import annotations

import logging
import threading
import time

from sphura.domain.errors import JobCancelledError, SphuraError
from sphura.domain.events import JobProgressEvent
from sphura.domain.models import Job, JobStatus, JobType, utcnow
from sphura.ports.queue import EventBus, JobHandler
from sphura.ports.repositories import JobRepository

log = logging.getLogger(__name__)

_PERSIST_INTERVAL_S = 0.5
_CANCEL_POLL_INTERVAL_S = 1.0


class _Context:
    def __init__(self, job: Job, repo: JobRepository, bus: EventBus) -> None:
        self._job = job
        self._repo = repo
        self._bus = bus
        self._last_persist = 0.0
        self._last_cancel_check = 0.0

    @property
    def job(self) -> Job:
        return self._job

    def report(self, stage: str, progress: float, message: str | None = None) -> None:
        progress = min(max(progress, 0.0), 1.0)
        stage_changed = stage != self._job.stage
        self._job.stage, self._job.progress = stage, progress
        now = time.monotonic()
        # Throttle DB writes; events are cheap and always published
        if stage_changed or progress >= 1.0 or now - self._last_persist >= _PERSIST_INTERVAL_S:
            self._repo.update_progress(self._job.id, stage, progress)
            self._last_persist = now
        self._bus.publish(
            JobProgressEvent(
                self._job.id, self._job.project_id, JobStatus.RUNNING, stage, progress, message
            )
        )
        self.check_cancelled()

    def check_cancelled(self) -> None:
        now = time.monotonic()
        if now - self._last_cancel_check < _CANCEL_POLL_INTERVAL_S:
            return
        self._last_cancel_check = now
        if self._repo.status_of(self._job.id) == JobStatus.CANCELLED:
            raise JobCancelledError(self._job.id)


class LocalJobQueue:
    """SQLite-backed queue with in-process worker threads. Survives restarts."""

    def __init__(self, repo: JobRepository, bus: EventBus, workers: int = 2) -> None:
        self._repo = repo
        self._bus = bus
        self._workers = workers
        self._handlers: dict[JobType, JobHandler] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def register(self, job_type: JobType, handler: JobHandler) -> None:
        self._handlers[job_type] = handler

    def enqueue(self, job: Job) -> Job:
        if job.type not in self._handlers:
            raise ValueError(f"no handler registered for job type {job.type}")
        self._repo.add(job)
        self._publish(job)
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> bool:
        ok = self._repo.try_cancel(job_id)
        if ok and (job := self._repo.get(job_id)):
            self._publish(job)
        return ok

    def start(self) -> None:
        # The local engine is the DB's only owner, so any RUNNING job is from a dead process
        requeued = self._repo.requeue_stale(0)
        if requeued:
            log.info("requeued %d interrupted job(s)", requeued)
        self._stop.clear()
        for i in range(self._workers):
            t = threading.Thread(target=self._run, name=f"sphura-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        # One more worker kept for editor previews, so they never wait behind long jobs
        t = threading.Thread(
            target=self._run, args=(True,), name="sphura-preview-worker", daemon=True
        )
        t.start()
        self._threads.append(t)
        self._wake.set()

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout_s)
        self._threads.clear()

    def _run(self, previews_only: bool = False) -> None:
        while not self._stop.is_set():
            job = self._repo.claim_next(previews_only)
            if job is None:
                self._wake.wait(timeout=2.0)
                self._wake.clear()
                continue
            self._execute(job)

    def _execute(self, job: Job) -> None:
        handler = self._handlers.get(job.type)
        ctx = _Context(job, self._repo, self._bus)
        self._publish(job)
        try:
            if handler is None:
                raise RuntimeError(f"no handler for job type {job.type}")
            job.result = handler(ctx) or {}
            job.status, job.progress = JobStatus.SUCCEEDED, 1.0
        except JobCancelledError:
            job.status = JobStatus.CANCELLED
        except SphuraError as e:  # expected failure: message is enough, no traceback
            log.warning("job %s failed: %s", job.id, e)
            job.status, job.error = JobStatus.FAILED, f"{type(e).__name__}: {e}"
        except Exception as e:
            log.exception("job %s failed", job.id)
            job.status, job.error = JobStatus.FAILED, f"{type(e).__name__}: {e}"
        job.finished_at = utcnow()

        # A cancel that lands after the handler returned still wins
        if self._repo.status_of(job.id) == JobStatus.CANCELLED:
            job.status = JobStatus.CANCELLED
        self._repo.update(job)
        self._publish(job)

    def _publish(self, job: Job) -> None:
        self._bus.publish(
            JobProgressEvent(job.id, job.project_id, job.status, job.stage, job.progress, job.error)
        )
