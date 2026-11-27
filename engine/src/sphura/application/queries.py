from __future__ import annotations

import json

from sphura.domain.errors import NotFoundError
from sphura.domain.models import (
    Clip,
    EditingMode,
    HighlightCandidate,
    Job,
    MediaAsset,
    Project,
)
from sphura.domain.transcript import Transcript
from sphura.pipeline.preferences import TasteModel, learn
from sphura.pipeline.ranking import RankedHighlight, rank
from sphura.pipeline.stages import read_text_any
from sphura.pipeline.stories import Story, canonical_names, unfinished_fights, with_names
from sphura.pipeline.stories import link as link_stories
from sphura.pipeline.timeline import SignalTimeline
from sphura.ports.repositories import (
    CandidateRepository,
    ClipRepository,
    FeedbackRepository,
    JobRepository,
    MediaAssetRepository,
    ProjectRepository,
)
from sphura.ports.storage import StorageProvider


class QueryService:
    def __init__(
        self,
        projects: ProjectRepository,
        assets: MediaAssetRepository,
        jobs: JobRepository,
        storage: StorageProvider,
        candidates: CandidateRepository,
        clips: ClipRepository | None = None,
        feedback: FeedbackRepository | None = None,
    ) -> None:
        self._projects = projects
        self._assets = assets
        self._jobs = jobs
        self._storage = storage
        self._candidates = candidates
        self._clips = clips
        self._feedback = feedback

    def list_projects(self, limit: int = 100) -> list[Project]:
        return self._projects.list(limit)

    def project(self, project_id: str) -> Project:
        p = self._projects.get(project_id)
        if p is None:
            raise NotFoundError(f"project {project_id} not found")
        return p

    def assets(self, project_id: str) -> list[MediaAsset]:
        self.project(project_id)
        return self._assets.list_for_project(project_id)

    def jobs(self, project_id: str) -> list[Job]:
        self.project(project_id)
        return self._jobs.list_for_project(project_id)

    def job(self, job_id: str) -> Job:
        j = self._jobs.get(job_id)
        if j is None:
            raise NotFoundError(f"job {job_id} not found")
        return j

    def transcript(self, project_id: str) -> Transcript:
        self.project(project_id)
        path = self._storage.artifact_path(project_id, "analysis", "transcript.json")
        if not path.is_file():
            raise NotFoundError("no transcript yet; run analyze first")
        return Transcript.from_dict(json.loads(read_text_any(path)))

    def transcript_version(self, project_id: str) -> int:
        """Version of the transcription that made the saved transcript (0: none/unknown)."""
        path = self._storage.artifact_path(project_id, "analysis", "manifest.json")
        try:
            entry = json.loads(path.read_text(encoding="utf-8")).get("transcribe") or {}
            return int(entry.get("outputs", {}).get("version", 0))
        except (OSError, ValueError, AttributeError):
            return 0

    def signals(self, project_id: str) -> SignalTimeline:
        self.project(project_id)
        path = self._storage.artifact_path(project_id, "analysis", "signals.npz")
        if not path.is_file():
            raise NotFoundError("no signals yet; run analyze first")
        return SignalTimeline.load(path)

    def highlights(
        self, project_id: str, mode: EditingMode | None = None, limit: int | None = None
    ) -> list[RankedHighlight]:
        project = self.project(project_id)
        found = self.candidates(project_id)
        if not found:
            raise NotFoundError("no highlights yet; run analyze first")
        return rank(found, mode or project.mode, limit, self.taste())

    def candidates(self, project_id: str) -> list[HighlightCandidate]:
        """Stored moments with names spelled the same way across the whole video."""
        found = self._candidates.list_for_project(project_id)
        canon = canonical_names(found)
        found = [with_names(c, canon) for c in found]
        unfinished_fights(found)
        return found

    def stories(self, project_id: str) -> list[Story]:
        """Neighbouring moments that make one story (setup, escalation, payoff)."""
        self.project(project_id)
        return link_stories(self.candidates(project_id))

    def clips(self, project_id: str) -> list[tuple[Clip, str | None]]:
        """Clips in rank order with their rendered file path, if any."""
        self.project(project_id)
        if self._clips is None:
            return []
        current = {c.id for c in self._candidates.list_for_project(project_id)}
        found = [c for c in self._clips.list_for_project(project_id) if c.candidate_id in current]
        found.sort(key=lambda c: c.rank or 10**6)
        return [(c, self._clips.output_of(c.id)) for c in found]

    def clip_output(self, clip_id: str) -> str | None:
        return self._clips.output_of(clip_id) if self._clips else None

    def taste(self) -> TasteModel | None:
        if self._feedback is None:
            return None
        model = learn(self._feedback.all(), self._feedback.overrides())
        return model if model.weights or model.direct else None

    def taste_summary(self) -> TasteModel:
        if self._feedback is None:
            return TasteModel()
        return learn(self._feedback.all(), self._feedback.overrides())
