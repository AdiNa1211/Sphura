"""AI highlight reels: many videos in, one edited compilation out.

Import validates each file and registers it as a project the sidebar doesn't list. Each
video is then analysed by the normal pipeline on the job queue (so several run at once).
pipeline/reel.py scores, deduplicates, picks and orders the moments; the user reviews and
arranges them; generating renders every moment at the target format with light effects,
snaps the cuts to the music's beat, joins them, mixes and normalises the audio, burns in
hook/captions/branding, and exports with a title, description, hashtags and thumbnail.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sphura.application.clips import ClipService, slugify
from sphura.application.ingest import IngestService
from sphura.application.queries import QueryService
from sphura.domain.errors import NotFoundError, SphuraError
from sphura.domain.models import (
    AIJudgement,
    AspectRatio,
    Clip,
    JobStatus,
    JobType,
    TextOverlay,
    TimeRange,
    new_id,
    utcnow,
)
from sphura.infrastructure.persistence.sqlite.reels import SqliteReelRepository
from sphura.pipeline import reframe
from sphura.pipeline.clips import remove_dead_time
from sphura.pipeline.effects import plan_effects
from sphura.pipeline.fx_context import classify, filter_off, key_time
from sphura.pipeline.reel import (
    Moment,
    Pick,
    beat_trims,
    clip_cap,
    mark_duplicates,
    moment_from,
    select,
    story_order,
    write_reel_copy,
)
from sphura.pipeline.reframe import plan_reframe
from sphura.pipeline.styles import geometry
from sphura.pipeline.tracks import choose_mix_track
from sphura.ports.ai import LLMProvider
from sphura.ports.media import (
    MediaDecoder,
    MediaProbe,
    ReelCompiler,
    ReelMusic,
    ReelPart,
    VideoRenderer,
)
from sphura.ports.queue import JobQueue
from sphura.ports.repositories import MediaAssetRepository

log = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, Any]] = {
    "shorts": {"label": "YouTube Shorts", "aspect": "9:16", "size": (1080, 1920)},
    "reels": {"label": "Instagram Reels", "aspect": "9:16", "size": (1080, 1920)},
    "tiktok": {"label": "TikTok", "aspect": "9:16", "size": (1080, 1920)},
    "landscape": {"label": "Landscape compilation", "aspect": "16:9", "size": (1920, 1080)},
    "square": {"label": "Square", "aspect": "1:1", "size": (1080, 1080)},
}
DEFAULTS: dict[str, Any] = {
    "target_s": 60,
    "preset": "shorts",
    "music": True,
    "captions": True,
    "hook": True,
    "effects": True,
    "transitions": "smart",
    "game": "",
    "fps": 30,
    "branding": True,
}
BRANDING: dict[str, str] = {"watermark_text": "", "watermark_logo": "", "intro": "", "outro": ""}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".flv"}
# Reels keep effects light: gentle zooms and the slow-mo on the hit, nothing louder
REEL_FX_OFF = ["cold_open", "shake", "flash", "rgb", "trail", "letterbox", "sfx", "vignette"]
FADE_S, CUT_S = 0.35, 0.04
HOOK_S = 2.8


class Cancelled(Exception):
    pass


class ReelService:
    def __init__(
        self,
        repo: SqliteReelRepository,
        ingest: IngestService,
        analysis: Any,
        queries: QueryService,
        queue: JobQueue,
        assets: MediaAssetRepository,
        clips: ClipService,
        renderer: VideoRenderer,
        compiler: ReelCompiler,
        decoder: MediaDecoder,
        probe: MediaProbe,
        sample: Callable[[str, list[float], int], list[bytes]],
        llm: LLMProvider | None,
        data_dir: Path,
    ) -> None:
        self._repo = repo
        self._ingest = ingest
        self._analysis = analysis
        self._queries = queries
        self._queue = queue
        self._assets = assets
        self._clips = clips
        self._renderer = renderer
        self._compiler = compiler
        self._decoder = decoder
        self._probe = probe
        self._sample = sample
        self._llm = llm
        self._root = data_dir / "reels"
        self._branding_file = data_dir / "reel_branding.json"
        self._lock = threading.RLock()
        self._busy: dict[str, threading.Event] = {}  # reel id -> cancel flag of its worker

    def start(self) -> None:
        """A reel that was being picked or made when SPHURA closed goes back to review."""
        for r in self._repo.all():
            if r["status"] in ("rendering", "picking"):
                was = r["status"]
                r["status"] = "review" if r.get("moments") else "importing"
                if was == "rendering":
                    r["render"] = {
                        "error": "SPHURA was closed while making this reel. Press Generate again."
                    }
                self._repo.save(r)

    def stop(self) -> None:
        for flag in list(self._busy.values()):
            flag.set()

    # ---------- reels ----------

    def create(self, name: str = "") -> dict[str, Any]:
        reel = {
            "id": new_id(),
            "name": name.strip() or f"Highlights {datetime.now():%b %d}",
            "status": "importing",
            "settings": dict(DEFAULTS),
            "sources": [],
            "moments": [],
            "picks": [],
            "picks_edited": False,
            "seed": 0,
            "story": "",
            "render": {},
            "output": None,
        }
        self._repo.save(reel)
        return reel

    def all(self) -> list[dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "status": r["status"],
                "sources": len(r["sources"]),
                "created_at": r.get("created_at"),
                "output": (r.get("output") or {}).get("path"),
            }
            for r in self._repo.all()
        ]

    def get(self, reel_id: str) -> dict[str, Any]:
        with self._lock:
            reel = self._load(reel_id)
            if reel["status"] == "analysing" and self._refresh(reel):
                self._repo.save(reel)
            return self._view(reel)

    def delete(self, reel_id: str) -> None:
        with self._lock:
            reel = self._load(reel_id)
            if flag := self._busy.get(reel_id):
                flag.set()
            self._cancel_jobs(reel)
            shutil.rmtree(self._root / reel_id, ignore_errors=True)
            self._repo.delete(reel_id)

    def hidden_projects(self) -> list[str]:
        """Projects made only for reels: the stream list leaves them out."""
        return [
            s["project_id"]
            for r in self._repo.all()
            for s in r["sources"]
            if s.get("project_id") and s.get("own_project")
        ]

    # ---------- sources ----------

    def add_paths(self, reel_id: str, paths: list[str], uploaded: bool = False) -> dict[str, Any]:
        with self._lock:
            reel = self._load(reel_id)
            self._editable(reel)
            for p in paths:
                reel["sources"].append(self._source(reel, Path(p), uploaded))
            reel["status"] = "importing"
            self._repo.save(reel)
            return self._view(reel)

    def upload_target(self, reel_id: str, name: str) -> Path:
        """Where an uploaded file is written (it is then added like a picked file)."""
        self._load(reel_id)
        safe = re.sub(r'[<>:"/\\|?*]', "_", Path(name).name).strip() or "video.mp4"
        if Path(safe).suffix.lower() not in VIDEO_EXT:
            raise SphuraError(f"{safe}: not a supported video ({', '.join(sorted(VIDEO_EXT))})")
        folder = self._root / reel_id / "files"
        folder.mkdir(parents=True, exist_ok=True)
        dest, k = folder / safe, 1
        while dest.exists():
            dest, k = folder / f"{Path(safe).stem} ({k}){Path(safe).suffix}", k + 1
        return dest

    def remove_source(self, reel_id: str, source_id: str) -> dict[str, Any]:
        with self._lock:
            reel = self._load(reel_id)
            self._editable(reel)
            src = self._src(reel, source_id)
            reel["sources"].remove(src)
            if src.get("uploaded"):
                with contextlib.suppress(OSError):
                    Path(src["path"]).unlink()
            reel["moments"] = [m for m in reel["moments"] if m["source_id"] != source_id]
            ids = {m["id"] for m in reel["moments"]}
            reel["picks"] = [p for p in reel["picks"] if p["moment_id"] in ids]
            self._repo.save(reel)
            return self._view(reel)

    def move_source(self, reel_id: str, source_id: str, index: int) -> dict[str, Any]:
        with self._lock:
            reel = self._load(reel_id)
            src = self._src(reel, source_id)
            reel["sources"].remove(src)
            reel["sources"].insert(max(0, min(index, len(reel["sources"]))), src)
            self._repo.save(reel)
            return self._view(reel)

    def source_file(self, reel_id: str, source_id: str) -> Path:
        path = Path(self._src(self._load(reel_id), source_id)["path"])
        if not path.is_file():
            raise NotFoundError("that video was moved or deleted")
        return path

    def _source(self, reel: dict[str, Any], path: Path, uploaded: bool) -> dict[str, Any]:
        src: dict[str, Any] = {
            "id": new_id(),
            "path": str(path),
            "name": path.name,
            "uploaded": uploaded,
            "status": "ready",
            "error": None,
            "progress": 0.0,
            "stage": None,
            "project_id": None,
            "asset_id": None,
            "own_project": False,
            "size": None,
            "duration": None,
            "width": None,
            "height": None,
            "fps": None,
        }
        try:
            if path.suffix.lower() not in VIDEO_EXT:
                raise SphuraError(f"not a supported video ({', '.join(sorted(VIDEO_EXT))})")
            r = self._ingest.ingest_local_file(path, project_name=f"Reel · {path.name}")
            info = r.asset.media_info
            if info is None or not info.video:
                raise SphuraError("no video stream found (is the file damaged?)")
            if any(s.get("asset_id") == r.asset.id for s in reel["sources"]):
                raise SphuraError("already in this reel")
            v = info.primary_video
            src.update(
                project_id=r.project.id,
                asset_id=r.asset.id,
                own_project=not r.already_registered,
                size=r.asset.size_bytes,
                duration=round(info.duration_s, 2),
                width=v.display_width,
                height=v.display_height,
                fps=round(v.fps, 2),
            )
        except SphuraError as e:
            src.update(status="failed", error=str(e))
        except OSError as e:
            src.update(status="failed", error=f"can't read the file: {e}")
        if uploaded and src["status"] == "failed":  # a rejected upload leaves no copy behind
            with contextlib.suppress(OSError):
                path.unlink()
        return src

    # ---------- analysis ----------

    def analyse(self, reel_id: str) -> dict[str, Any]:
        """Analyse every video that needs it; ones analysed before are reused as they are."""
        with self._lock:
            reel = self._load(reel_id)
            self._editable(reel)
            usable = [s for s in reel["sources"] if s.get("project_id")]  # failed ones retry
            if not usable:
                raise SphuraError("Add at least one video that SPHURA can read")
            for s in usable:
                latest = self._latest_analysis(s["project_id"])
                if latest is None or latest.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                    self._analysis.start(s["project_id"])
                s.update(status="queued", error=None, progress=0.0)
            reel.update(
                status="analysing", moments=[], picks=[], picks_edited=False, output=None, render={}
            )
            self._refresh(reel)
            self._repo.save(reel)
            return self._view(reel)

    def cancel(self, reel_id: str) -> dict[str, Any]:
        """Stop the analysis or the render, whichever is running."""
        with self._lock:
            reel = self._load(reel_id)
            if flag := self._busy.get(reel_id):
                flag.set()
            if reel["status"] == "analysing":
                self._cancel_jobs(reel)
                reel["status"] = "importing"
                for s in reel["sources"]:
                    if s["status"] in ("queued", "analysing"):
                        s.update(status="ready", progress=0.0, stage=None)
                self._repo.save(reel)
            return self._view(reel)

    def _cancel_jobs(self, reel: dict[str, Any]) -> None:
        for s in reel["sources"]:
            if not s.get("own_project") or not s.get("project_id"):
                continue  # never stop the analysis of a stream the user opened themselves
            with contextlib.suppress(NotFoundError):
                for j in self._queries.jobs(s["project_id"]):
                    if j.type == JobType.ANALYZE and not j.status.is_terminal:
                        self._queue.cancel(j.id)

    def _latest_analysis(self, project_id: str) -> Any:
        try:
            jobs = [j for j in self._queries.jobs(project_id) if j.type == JobType.ANALYZE]
        except NotFoundError:
            return None
        return max(jobs, key=lambda j: j.created_at, default=None)

    def _refresh(self, reel: dict[str, Any]) -> bool:
        """Follow the analysis jobs; once every video is done, start picking moments."""
        pending = False
        for s in reel["sources"]:
            if not s.get("project_id"):  # couldn't be imported: nothing to follow
                continue
            j = self._latest_analysis(s["project_id"])
            if j is None or j.status == JobStatus.QUEUED:
                s.update(status="queued", progress=0.0)
                pending = True
            elif j.status == JobStatus.RUNNING:
                s.update(status="analysing", progress=round(j.progress, 3), stage=j.stage)
                pending = True
            elif j.status == JobStatus.SUCCEEDED:
                s.update(status="done", progress=1.0, stage=None)
            else:
                s.update(
                    status="failed",
                    job_failed=True,
                    progress=0.0,
                    error=j.error or "analysis was cancelled",
                )
        if not pending:
            done = [s for s in reel["sources"] if s["status"] == "done"]
            if done:
                reel["status"] = "picking"
                self._repo.save(reel)  # the picker reads the finished statuses from here
                self._spawn(reel["id"], self._pick_worker)
            else:
                reel["status"] = "importing"
                reel["render"] = {"error": "None of the videos could be analysed."}
        return True

    # ---------- picking ----------

    def _pick_worker(self, reel_id: str, cancel: threading.Event) -> None:
        reel = self._load(reel_id)
        moments = self._pool(reel)
        if cancel.is_set():
            return
        with self._lock:
            reel = self._load(reel_id)
            reel["moments"] = [m.to_dict() for m in moments]
            self._choose(reel, moments)
            reel["status"] = "review"
            self._repo.save(reel)

    def _pool(self, reel: dict[str, Any]) -> list[Moment]:
        moments: list[Moment] = []
        for s in reel["sources"]:
            if s["status"] != "done":
                continue
            try:
                found = self._queries.highlights(s["project_id"], None, 40)
            except NotFoundError:
                found = []
            ms = [moment_from(r, s["id"], s["name"]) for r in found]
            moments += ms or self._fallback_moment(s)
        return mark_duplicates(moments)

    def _fallback_moment(self, s: dict[str, Any]) -> list[Moment]:
        """A short clip can be one moment too short to make a candidate: use its liveliest
        stretch (from the audio/motion curve) so it can still be picked."""
        try:
            tl = self._queries.signals(s["project_id"])
        except NotFoundError:
            return []
        curve = tl.features.get("interest")
        dur = float(s.get("duration") or tl.duration_s)
        if curve is None or dur < 2:
            return []
        peak = float(curve.argmax()) * tl.hop_s
        a, b = max(0.0, peak - 6), min(dur, peak + 4)
        return [
            Moment(
                id=f"clip-{s['id']}",
                source_id=s["id"],
                project_id=s["project_id"],
                asset_id=s["asset_id"],
                source_name=s["name"],
                start_s=round(a, 2),
                end_s=round(b, 2),
                peak_s=round(peak, 2),
                score=30,
                confidence=0.2,
                title=f"{Path(s['name']).stem}",
                what="The liveliest stretch of this clip",
                why="",
                hook="",
            )
        ]

    def _choose(self, reel: dict[str, Any], moments: list[Moment], again: bool = False) -> None:
        target = float(reel["settings"]["target_s"])
        prev = [p["moment_id"] for p in reel["picks"]] if again else None
        story = story_order(self._llm, moments, target, prev)
        if story is not None:
            picks, note = story
            reel["story"] = note or "Arranged by the AI editor."
        else:
            picks = select(moments, target, seed=reel["seed"], jitter=12 if again else 0)
            reel["story"] = (
                "Picked by score across all videos: no repeats of the same event, "
                "a hook first, variety in the middle, the best moment last."
            )
        reel["picks"] = [p.to_dict() for p in picks]
        reel["picks_edited"] = False

    def regenerate(self, reel_id: str) -> dict[str, Any]:
        with self._lock:
            reel = self._load(reel_id)
            if not reel["moments"]:
                raise SphuraError("Analyse the videos first")
            reel["seed"] += 1
            reel["status"] = "picking"
            self._repo.save(reel)
            self._spawn(reel_id, self._regenerate_worker)
            return self._view(reel)

    def _regenerate_worker(self, reel_id: str, cancel: threading.Event) -> None:
        reel = self._load(reel_id)
        moments = [Moment.from_dict(m) for m in reel["moments"]]
        with self._lock:
            reel = self._load(reel_id)
            self._choose(reel, moments, again=True)
            reel["status"] = "review"
            self._repo.save(reel)

    def update(self, reel_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Name, settings and the arranged picks (the user's edits are kept as they are)."""
        with self._lock:
            reel = self._load(reel_id)
            if reel["status"] in ("rendering", "picking"):
                raise SphuraError("Wait until SPHURA has finished, or cancel it first")
            if name := str(patch.get("name") or "").strip():
                reel["name"] = name[:80]
            repick = False
            if isinstance(patch.get("settings"), dict):
                old_target = reel["settings"]["target_s"]
                reel["settings"] = _clean_settings({**reel["settings"], **patch["settings"]})
                repick = reel["settings"]["target_s"] != old_target and not reel["picks_edited"]
            if isinstance(patch.get("picks"), list):
                reel["picks"] = self._clean_picks(reel, patch["picks"])
                reel["picks_edited"] = True
                repick = False
            if repick and reel["moments"]:  # a new length re-picks until the user edits picks
                self._choose(reel, [Moment.from_dict(m) for m in reel["moments"]])
            self._repo.save(reel)
            return self._view(reel)

    def _clean_picks(self, reel: dict[str, Any], picks: list[Any]) -> list[dict[str, Any]]:
        by_id = {m["id"]: m for m in reel["moments"]}
        dur = {s["id"]: float(s.get("duration") or 0) for s in reel["sources"]}
        out, seen = [], set()
        for p in picks:
            m = by_id.get(str(p.get("moment_id")))
            if m is None or m["id"] in seen:
                continue
            limit = dur.get(m["source_id"]) or m["end_s"]
            a = max(0.0, min(float(p.get("start_s", m["start_s"])), limit - 1))
            b = max(a + 1.0, min(float(p.get("end_s", m["end_s"])), limit))
            out.append(
                Pick(m["id"], round(a, 2), round(b, 2), bool(p.get("include", True))).to_dict()
            )
            seen.add(m["id"])
        return out

    # ---------- generating ----------

    def generate(self, reel_id: str) -> dict[str, Any]:
        with self._lock:
            reel = self._load(reel_id)
            if reel["status"] in ("rendering", "picking", "analysing"):
                raise SphuraError("SPHURA is still working on this reel")
            if not any(p["include"] for p in reel["picks"]):
                raise SphuraError("Include at least one highlight")
            reel["status"] = "rendering"
            reel["render"] = {
                "progress": 0.0,
                "step": "Starting",
                "error": None,
                "started_at": utcnow().isoformat(),
            }
            self._repo.save(reel)
            self._spawn(reel_id, self._render_worker)
            return self._view(reel)

    def _report(self, reel_id: str, cancel: threading.Event, frac: float, step: str) -> None:
        if cancel.is_set():
            raise Cancelled()
        with self._lock:
            reel = self._load(reel_id)
            reel["render"].update(progress=round(max(0.0, min(1.0, frac)), 3), step=step)
            self._repo.save(reel)

    def _render_worker(self, reel_id: str, cancel: threading.Event) -> None:
        work = self._root / reel_id / "work"
        try:
            out = self._make(reel_id, cancel, work)
            with self._lock:
                reel = self._load(reel_id)
                reel.update(status="done", output=out)
                reel["render"].update(
                    progress=1.0, step="Done", error=None, finished_at=utcnow().isoformat()
                )
                self._repo.save(reel)
        except Cancelled:
            self._fail(reel_id, "Cancelled")
        except Exception as e:
            log.exception("highlight reel %s failed", reel_id)
            self._fail(reel_id, str(e) or type(e).__name__)
        finally:
            shutil.rmtree(work, ignore_errors=True)  # temp renders only live as long as needed

    def _fail(self, reel_id: str, message: str) -> None:
        with self._lock, contextlib.suppress(NotFoundError):
            reel = self._load(reel_id)
            reel["status"] = "review"
            reel["render"] = {**reel.get("render", {}), "error": message}
            self._repo.save(reel)

    def _make(self, reel_id: str, cancel: threading.Event, work: Path) -> dict[str, Any]:
        reel = self._load(reel_id)
        st = reel["settings"]
        preset = PRESETS[st["preset"]]
        aspect = AspectRatio(preset["aspect"])
        by_id = {m["id"]: Moment.from_dict(m) for m in reel["moments"]}
        picks = [Pick(**p) for p in reel["picks"] if p["include"] and p["moment_id"] in by_id]
        moments = [by_id[p.moment_id] for p in picks]
        judged = self._judgements({m.project_id for m in moments})
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        def report(frac: float, step: str) -> None:
            self._report(reel_id, cancel, frac, step)

        report(0.02, "Writing the title and hook")
        copy = write_reel_copy(self._llm, moments, st["game"], st["preset"])
        best = max(moments, key=lambda m: m.score)
        seed = sum(map(ord, reel_id)) + reel["seed"]
        mood, bpm = self._clips.reel_mood(judged.get(best.id), seed)
        beat = 60 / bpm if st["music"] else 0.0

        report(0.05, "Planning the cuts")
        joins = [
            self._join(st["transitions"], moments[i], moments[i + 1])
            for i in range(len(moments) - 1)
        ]
        hook = copy["hook"] if st["hook"] else ""
        clips = [
            self._item(m, p, aspect, st, judged.get(m.id), hook if i == 0 else "")
            for i, (m, p) in enumerate(zip(moments, picks, strict=True))
        ]
        if beat:  # every cut on a beat: trim each clip's tail by less than one beat
            cuts = beat_trims([c.output_duration_s for c in clips], joins, beat)
            for i, cut in enumerate(cuts):
                if cut > 0.01:
                    p = picks[i]
                    picks[i] = Pick(p.moment_id, p.start_s, round(p.end_s - cut, 3))
                    clips[i] = self._item(
                        moments[i],
                        picks[i],
                        aspect,
                        st,
                        judged.get(moments[i].id),
                        hook if i == 0 else "",
                    )

        parts: list[ReelPart] = []
        brand = self.branding() if st["branding"] else dict(BRANDING)
        intro = self._extra(brand["intro"])
        if intro:
            parts.append(intro)
        total_items = sum(c.output_duration_s for c in clips) or 1.0
        done_s = 0.0
        for i, clip in enumerate(clips):
            asset = self._assets.get(clip.asset_id)
            if asset is None or asset.media_info is None:
                raise SphuraError(f"{moments[i].source_name}: the video details are missing")
            label = f"Rendering highlight {i + 1} of {len(clips)}"
            base, share = (
                0.08 + 0.67 * done_s / total_items,
                0.67 * clip.output_duration_s / total_items,
            )
            report(base, label)
            self._reframe(clip, asset.source_uri, asset.media_info, aspect)
            mix = choose_mix_track(asset.media_info.audio)
            path = work / f"item_{i:02d}.mp4"

            def item_progress(
                f: float, b: float = base, w: float = share, lab: str = label
            ) -> None:
                report(b + w * f, lab)

            result = self._renderer.render(
                clip,
                asset.source_uri,
                asset.media_info,
                mix.index if mix else None,
                path,
                item_progress,
            )
            parts.append(ReelPart(result.path, self._probe.probe(str(result.path)).duration_s))
            done_s += clip.output_duration_s
        outro = self._extra(brand["outro"])
        if outro:
            parts.append(outro)

        all_joins = ([CUT_S] if intro else []) + joins + ([CUT_S] if outro else [])
        total = sum(p.duration_s for p in parts) - sum(all_joins)
        music = None
        if st["music"]:
            report(0.76, "Making the music")
            start = intro.duration_s - CUT_S if intro else 0.0
            end = total - (outro.duration_s - CUT_S if outro else 0.0)
            closer = clips[-1]
            hit = key_time(closer.effects)
            accent = (
                end
                - closer.output_duration_s
                + (
                    closer.to_output_time(hit)
                    if hit is not None
                    else closer.output_duration_s * 0.6
                )
            )
            accent = round((accent - start) / (60 / bpm)) * (60 / bpm)  # hit on a beat
            info = self._clips.reel_music(
                work / "music.wav",
                end - start,
                mood,
                bpm,
                seed,
                accent,
                judged.get(best.id),
                st["game"] or None,
            )
            b = 60 / float(info["bpm"]) if info.get("bpm") else 60 / bpm
            music = ReelMusic(Path(info["path"]), float(info.get("beat0_s", 0.0)) % b, start, end)
            copy["music"] = f"{info['label']} ({'AI music' if info['ai'] else 'SPHURA score'})"

        report(0.8, "Putting it all together")
        out_dir = (
            Path.home()
            / "Videos"
            / "SPHURA"
            / f"{datetime.now():%Y-%m-%d} Highlight reel - {slugify(reel['name'], 50)}"
        )
        out = out_dir / f"{slugify(reel['name'], 40)}_{st['preset']}.mp4"
        logo = Path(brand["watermark_logo"]) if brand["watermark_logo"] else None
        done = self._compiler.compile(
            parts,
            all_joins,
            preset["size"],
            int(st["fps"]),
            music,
            brand["watermark_text"],
            logo if logo and logo.is_file() else None,
            out,
            lambda f: report(0.8 + 0.17 * f, "Putting it all together"),
        )

        report(0.98, "Thumbnail and description")
        chapters, t = [], (intro.duration_s - CUT_S) if intro else 0.0
        if intro:  # chapter lists start at 0:00
            chapters.append({"t": 0.0, "title": "Intro", "source": ""})
        for i, c in enumerate(clips):
            chapters.append(
                {"t": round(t, 2), "title": moments[i].title, "source": moments[i].source_name}
            )
            t += c.output_duration_s - (joins[i] if i < len(joins) else 0.0)
        thumb_m = by_id.get(copy["thumbnail_moment"], best)
        thumb = self._thumbnail(thumb_m, out.with_name(out.stem + "_thumbnail.jpg"))
        out.with_suffix(".txt").write_text(_post_text(copy, chapters, thumb_m), encoding="utf-8")
        return {
            "path": str(done.path),
            "folder": str(out_dir),
            "thumbnail": thumb,
            "duration": round(done.duration_s, 2),
            "preset": st["preset"],
            "title": copy["youtube_title"],
            "description": copy["description"],
            "hashtags": copy["hashtags"],
            "hook": copy["hook"],
            "thumbnail_text": copy.get("thumbnail_text", ""),
            "thumbnail_moment": thumb_m.title,
            "music": copy.get("music", ""),
            "chapters": chapters,
            "encoder": done.encoder,
        }

    def _join(self, mode: str, a: Moment, b: Moment) -> float:
        """Smart: a hard cut inside one video, a short dissolve when the video changes."""
        if mode == "cuts":
            return CUT_S
        if mode == "fades":
            return FADE_S
        return CUT_S if a.source_id == b.source_id else FADE_S

    def _item(
        self,
        m: Moment,
        p: Pick,
        aspect: AspectRatio,
        st: dict[str, Any],
        judgement: AIJudgement | None,
        hook: str,
    ) -> Clip:
        """One highlight as a Short-style clip: dead time cut out, light effects, captions."""
        tl = self._optional(self._queries.signals, m.project_id)
        transcript = self._optional(self._queries.transcript, m.project_id)
        span = TimeRange(p.start_s, max(p.end_s, p.start_s + 0.5))
        segs = remove_dead_time(span, tl, transcript) if tl else [span]
        clip = Clip(
            project_id=m.project_id,
            asset_id=m.asset_id,
            title=m.title,
            segments=segs,
            aspect=aspect,
            style="hype",
            candidate_id=m.id,
        )
        if st["effects"]:
            wav = self._clips.wav_for(m.project_id, m.asset_id)
            fx = filter_off(plan_effects(clip, m.peak_s, tl, wav, classify(judgement)), REEL_FX_OFF)
            for e in fx:
                if e["type"] == "punch":  # gentler zooms than a single Short
                    e["zoom"] = round(1 + (e["zoom"] - 1) * 0.6, 3)
                    e["shake"] = False
            clip.effects = fx
        dur = clip.output_duration_s
        if hook:
            clip.overlays.append(TextOverlay(hook, 0.1, min(HOOK_S, dur - 0.2), "hook"))
        if st["captions"] and transcript is not None:
            clip.overlays += _captions(clip, transcript, HOOK_S if hook else 0.0)
        return clip

    def _reframe(self, clip: Clip, uri: str, info: Any, aspect: AspectRatio) -> None:
        if aspect == AspectRatio.LANDSCAPE:
            return
        v = info.primary_video
        g = geometry(clip, v.display_width, v.display_height)
        plan = plan_reframe(
            clip, uri, self._decoder, v.display_width / v.display_height, g.video_aspect
        )
        clip.crop_path, clip.layout, clip.crop_version = (
            plan.keyframes,
            plan.layout,
            reframe.VERSION,
        )

    def _extra(self, path: str) -> ReelPart | None:
        """An intro/outro clip, if one is set and readable."""
        if not path or not Path(path).is_file():
            return None
        try:
            info = self._probe.probe(path)
        except SphuraError as e:
            log.warning("skipping intro/outro %s: %s", path, e)
            return None
        return ReelPart(Path(path), info.duration_s, bool(info.audio)) if info.video else None

    def moment_thumb(self, reel_id: str, moment_id: str) -> Path:
        """A frame at the moment's peak, cached next to the reel."""
        reel = self._load(reel_id)
        m = next((x for x in reel["moments"] if x["id"] == moment_id), None)
        if m is None:
            raise NotFoundError("moment not found")
        dest = self._root / reel_id / "thumbs" / f"{re.sub(r'[^A-Za-z0-9_-]', '_', moment_id)}.jpg"
        if not dest.is_file():
            asset = self._assets.get(m["asset_id"])
            if asset is None:
                raise NotFoundError("video not found")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self._sample(asset.source_uri, [float(m["peak_s"])], 480)[0])
        return dest

    def _thumbnail(self, m: Moment, dest: Path) -> str | None:
        asset = self._assets.get(m.asset_id)
        if asset is None:
            return None
        try:
            dest.write_bytes(self._sample(asset.source_uri, [m.peak_s], 1280)[0])
            return str(dest)
        except Exception as e:
            log.warning("no thumbnail for the reel: %s", e)
            return None

    def _judgements(self, project_ids: set[str]) -> dict[str, AIJudgement]:
        out: dict[str, AIJudgement] = {}
        for pid in project_ids:
            with contextlib.suppress(NotFoundError):
                for r in self._queries.highlights(pid, None):
                    if r.candidate.judgement is not None:
                        out[r.candidate.id] = r.candidate.judgement
        return out

    # ---------- branding ----------

    def branding(self) -> dict[str, str]:
        try:
            saved = json.loads(self._branding_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        return {k: str(saved.get(k, v)) for k, v in BRANDING.items()}

    def set_branding(self, values: dict[str, Any]) -> dict[str, str]:
        b = self.branding()
        for k in BRANDING:
            if k in values:
                v = str(values[k] or "").strip()
                if k != "watermark_text" and v and not Path(v).is_file():
                    raise SphuraError(f"file not found: {v}")
                b[k] = v[:200] if k == "watermark_text" else v
        self._branding_file.parent.mkdir(parents=True, exist_ok=True)
        self._branding_file.write_text(json.dumps(b, indent=2), encoding="utf-8")
        return b

    # ---------- helpers ----------

    def _spawn(self, reel_id: str, fn: Callable[[str, threading.Event], None]) -> None:
        flag = threading.Event()
        self._busy[reel_id] = flag

        def run() -> None:
            try:
                fn(reel_id, flag)
            except Exception as e:
                log.exception("highlight reel %s: background step failed", reel_id)
                self._fail(reel_id, str(e) or type(e).__name__)
            finally:
                if self._busy.get(reel_id) is flag:
                    self._busy.pop(reel_id, None)

        threading.Thread(target=run, daemon=True, name=f"reel-{reel_id[:8]}").start()

    def _load(self, reel_id: str) -> dict[str, Any]:
        reel = self._repo.get(reel_id)
        if reel is None:
            raise NotFoundError("highlight reel not found")
        return reel

    @staticmethod
    def _src(reel: dict[str, Any], source_id: str) -> dict[str, Any]:
        src: dict[str, Any] | None = next(
            (s for s in reel["sources"] if s["id"] == source_id), None
        )
        if src is None:
            raise NotFoundError("video not found in this reel")
        return src

    @staticmethod
    def _editable(reel: dict[str, Any]) -> None:
        if reel["status"] in ("analysing", "rendering", "picking"):
            raise SphuraError("Wait until SPHURA has finished, or cancel it first")

    @staticmethod
    def _optional(fn: Callable[[str], Any], project_id: str) -> Any:
        try:
            return fn(project_id)
        except NotFoundError:
            return None

    def _view(self, reel: dict[str, Any]) -> dict[str, Any]:
        """The reel plus figures the UI shows: lengths and how full the target is."""
        by_id = {m["id"]: m for m in reel["moments"]}
        chosen = [p for p in reel["picks"] if p["include"] and p["moment_id"] in by_id]
        return {
            **reel,
            "selected_s": round(sum(p["end_s"] - p["start_s"] for p in chosen), 1),
            "clip_cap_s": clip_cap(float(reel["settings"]["target_s"])),
            "presets": {k: v["label"] for k, v in PRESETS.items()},
            "ai": self._llm.info.name if self._llm else None,
        }


def _clean_settings(s: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULTS)
    out.update({k: s[k] for k in DEFAULTS if k in s})
    out["target_s"] = max(10, min(600, int(float(out["target_s"] or 60))))
    if out["preset"] not in PRESETS:
        out["preset"] = DEFAULTS["preset"]
    if out["transitions"] not in ("smart", "cuts", "fades"):
        out["transitions"] = "smart"
    out["fps"] = 60 if int(out["fps"] or 30) == 60 else 30
    out["game"] = str(out["game"] or "").strip()[:60]
    for k in ("music", "captions", "hook", "effects", "branding"):
        out[k] = bool(out[k])
    return out


def _captions(clip: Clip, transcript: Any, after_s: float) -> list[TextOverlay]:
    """Spoken lines as short subtitles (a few words at a time), on the reel's timeline."""
    words = [
        w
        for seg in transcript.segments
        if seg.confidence >= 0.45
        for w in seg.words
        if any(s.start_s <= w.start_s < s.end_s for s in clip.segments)
    ]
    out: list[TextOverlay] = []
    chunk: list[Any] = []

    def flush() -> None:
        if not chunk:
            return
        a, b = _out_time(clip, chunk[0].start_s), _out_time(clip, chunk[-1].end_s)
        text = " ".join(w.text.strip() for w in chunk).strip()
        if a is not None and b is not None and b > a and text and a >= after_s:
            out.append(TextOverlay(text, round(a, 3), round(max(b, a + 0.6), 3), "plain"))
        chunk.clear()

    for w in words:
        if chunk and (
            len(chunk) >= 5 or w.start_s - chunk[-1].end_s > 0.6 or w.end_s - chunk[0].start_s > 2.4
        ):
            flush()
        chunk.append(w)
    flush()
    return out


def _out_time(clip: Clip, src: float) -> float | None:
    """Source time -> time in the finished clip (jump cuts and slow motion included)."""
    pre = 0.0
    for s in clip.segments:
        if s.start_s <= src <= s.end_s + 0.05:
            return clip.to_output_time(pre + min(src, s.end_s) - s.start_s)
        pre += s.duration_s
    return None


def _post_text(copy: dict[str, Any], chapters: list[dict[str, Any]], thumb: Moment) -> str:
    lines = [f"TITLE\n{copy['youtube_title']}\n", f"DESCRIPTION\n{copy['description']}\n"]
    if len(chapters) > 1:
        lines.append("IN THIS VIDEO")
        lines += [f"{_mmss(c['t'])} {c['title']}" for c in chapters]
        lines.append("")
    if copy.get("music"):
        lines.append(f"Music: {copy['music']}\n")
    lines.append(copy["hashtags"] + "\n")
    lines.append(
        f"THUMBNAIL IDEA\n{copy.get('thumbnail_text') or thumb.title} "
        f"(frame from {thumb.source_name} at {_mmss(thumb.peak_s)})\n"
    )
    lines.append(f"ON-SCREEN HOOK\n{copy['hook']}\n")
    return "\n".join(lines)


def _mmss(t: float) -> str:
    t = int(max(0, t))
    return f"{t // 60}:{t % 60:02d}"
