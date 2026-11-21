from __future__ import annotations

from typing import Any

from sphura.domain.errors import InvalidMediaError, NotFoundError
from sphura.domain.models import Job, JobStatus, JobType, MediaAsset
from sphura.pipeline.candidates import detect_candidates
from sphura.pipeline.compare import CompareStage
from sphura.pipeline.judge import JudgeStage
from sphura.pipeline.manifest import StageManifest
from sphura.pipeline.signals import SignalsStage, top_peaks
from sphura.pipeline.stages import ExtractAudioStage, StageProgress, TranscribeStage
from sphura.pipeline.tracks import choose_mix_track, choose_voice_track
from sphura.ports.queue import JobContext, JobQueue
from sphura.ports.repositories import (
    CandidateRepository,
    JobRepository,
    MediaAssetRepository,
    ProjectRepository,
)
from sphura.ports.storage import StorageProvider

# Share of overall job progress per stage (rough relative cost)
_WEIGHTS = {
    "extract_audio": 0.03,
    "transcribe": 0.22,
    "signals": 0.30,
    "judge": 0.30,
    "compare": 0.15,
}


class AnalysisService:
    def __init__(
        self,
        projects: ProjectRepository,
        assets: MediaAssetRepository,
        jobs: JobRepository,
        queue: JobQueue,
    ) -> None:
        self._projects = projects
        self._assets = assets
        self._jobs = jobs
        self._queue = queue

    def start(self, project_id: str, language: str | None = None) -> Job:
        if self._projects.get(project_id) is None:
            raise NotFoundError(f"project {project_id} not found")
        for j in self._jobs.list_for_project(project_id):
            if j.type == JobType.ANALYZE and j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                return j  # already in progress
        asset = _primary_asset(self._assets, project_id)
        return self._queue.enqueue(
            Job(
                type=JobType.ANALYZE,
                project_id=project_id,
                payload={"asset_id": asset.id, "language": language},
            )
        )


def _primary_asset(assets: MediaAssetRepository, project_id: str) -> MediaAsset:
    found = assets.list_for_project(project_id)
    if not found:
        raise NotFoundError(f"project {project_id} has no video")
    return found[0]


class AnalyzeHandler:
    """Runs the analysis stages in order. Finished stages are skipped on re-runs/resumes."""

    def __init__(
        self,
        assets: MediaAssetRepository,
        storage: StorageProvider,
        extract: ExtractAudioStage,
        transcribe: TranscribeStage,
        signals: SignalsStage,
        judge: JudgeStage,
        compare: CompareStage,
        candidates: CandidateRepository,
    ) -> None:
        self._assets = assets
        self._storage = storage
        self._extract = extract
        self._transcribe = transcribe
        self._signals = signals
        self._judge = judge
        self._compare = compare
        self._candidates = candidates

    def __call__(self, ctx: JobContext) -> dict[str, Any]:
        asset = self._assets.get(ctx.job.payload["asset_id"])
        if asset is None:
            raise NotFoundError("asset not found")
        if asset.media_info is None:
            raise InvalidMediaError("media details missing; re-run ingest")
        manifest = StageManifest(
            self._storage.artifact_path(asset.project_id, "analysis", "manifest.json")
        )
        result: dict[str, Any] = {}

        audio = self._extract.run(asset, manifest, self._progress(ctx, "extract_audio"))
        result["audio_tracks"] = {str(i): str(p) for i, p in audio.files.items()}

        voice = choose_voice_track(asset.media_info.audio)
        transcript = None
        if voice is None or voice.index not in audio.files:
            result["transcript"] = None  # silent video: later stages rely on visuals only
        else:
            t = transcript = self._transcribe.run(
                asset,
                audio.files[voice.index],
                voice.index,
                ctx.job.payload.get("language"),
                manifest,
                self._progress(ctx, "transcribe"),
            )
            result["transcript"] = {
                "language": t.language,
                "model": t.model,
                "audio_stream_index": t.audio_stream_index,
                "segments": len(t.segments),
                "words": t.word_count,
            }

        mix = choose_mix_track(asset.media_info.audio)
        tl = self._signals.run(
            asset,
            audio.files.get(mix.index) if mix else None,
            transcript,
            manifest,
            self._progress(ctx, "signals"),
        )
        result["signals"] = {
            "hops": tl.length,
            "scene_cuts": int(tl.features["scene_cut"].sum()),
            "top_peaks": [round(p.t_s, 1) for p in top_peaks(tl, 5)],
        }

        windows = detect_candidates(tl, transcript)
        found, report = self._judge.run(asset, windows, self._progress(ctx, "judge"))
        cmp = self._compare.run(asset, found, self._progress(ctx, "compare"))
        self._candidates.replace_for_asset(asset.id, found)
        result["highlights"] = {
            "candidates": len(found),
            "judge": report.provider,
            "cloud": report.sends_data_off_device,
            "judged": report.judged,
            "cached": report.cached,
            "failed": report.failed,
            "input_tokens": report.input_tokens + cmp.input_tokens,
            "output_tokens": report.output_tokens + cmp.output_tokens,
            "comparisons": cmp.calls + cmp.cached,
        }
        ctx.report("done", 1.0)
        return result

    @staticmethod
    def _progress(ctx: JobContext, stage: str) -> StageProgress:
        base = 0.0
        for name, w in _WEIGHTS.items():
            if name == stage:
                break
            base += w
        weight = _WEIGHTS[stage]
        return lambda frac, msg: ctx.report(stage, base + weight * frac, msg)
