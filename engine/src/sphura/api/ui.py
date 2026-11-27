"""Endpoints behind the browser UI (sphura ui). Local-only: the app refuses requests
whose Host isn't this machine, so web pages can't drive it via DNS rebinding."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from sphura.application.clips import shorts_folder
from sphura.application.library import recent_videos
from sphura.container import LocalEngine
from sphura.domain.errors import SphuraError
from sphura.domain.models import AspectRatio, JobType
from sphura.infrastructure.ai.ollama import OllamaProvider
from sphura.pipeline.styles import STYLES

log = logging.getLogger(__name__)

WEB = Path(__file__).resolve().parent.parent / "web"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def engine_of(request: Request) -> LocalEngine:
    engine: LocalEngine = request.app.state.engine
    return engine


Engine = Annotated[LocalEngine, Depends(engine_of)]


class LoreIndexRequest(BaseModel):
    keys: list[str] = Field(default_factory=list)  # "p:<project id>" or "d:<download id>"


class LoreCancel(BaseModel):
    project_id: str


class LoreDiscover(BaseModel):
    asset_ids: list[str] | None = None


class ShortsRequest(BaseModel):
    styles: list[str] = Field(default_factory=lambda: ["meme", "cinematic"])
    top: int = Field(default=4, ge=1, le=20)
    game: str | None = None
    include_maybe: bool = False
    plain: bool = False
    length_s: float | None = Field(default=None, ge=5, le=180)  # None = each style's own
    lengths: dict[str, float | None] = Field(default_factory=dict)  # per style; wins over length_s
    caption_bar: bool | None = None  # True: white caption bar on hype/cinematic too


class RateRequest(BaseModel):
    good: bool


class MusicChoice(BaseModel):
    choice: str = Field(description='"auto", "none", or the path of a track in the music folder')
    volume_db: float | None = Field(default=None, ge=-40, le=0)
    render: bool = True


class TrackMeta(BaseModel):
    file: str
    source: str = Field(default="", max_length=80)
    license: str = Field(default="", max_length=200)
    credit: str = Field(default="", max_length=300)
    tags: str = Field(default="", max_length=120)


MUSIC_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
MAX_UPLOAD = 60_000_000


class PreviewRequest(BaseModel):
    url: str = Field(min_length=8, max_length=500)
    playlist: bool = False


class AddDownloads(BaseModel):
    items: list[dict[str, Any]] = Field(min_length=1, max_length=500)
    quality: str | None = None
    playlist_title: str | None = None
    confirm_rights: bool = False


class EditorSave(BaseModel):
    segments: list[list[float]] | None = None
    overlays: list[dict[str, Any]] | None = None
    fx_off: list[str] | None = None
    caption_bar: bool | None = None
    render: str = "draft"  # draft (preview) | final (export the Short)


class EditorAudio(BaseModel):
    segments: list[list[float]] | None = None  # unsaved cuts; None = as saved


class ReelNew(BaseModel):
    name: str = ""


class ReelPaths(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=100)


class ReelMove(BaseModel):
    index: int


class ReelPatch(BaseModel):
    name: str | None = None
    settings: dict[str, Any] | None = None
    picks: list[dict[str, Any]] | None = None


class TextEdit(BaseModel):
    text: str | None = Field(default=None, max_length=120, description="on-screen text; '' = none")
    title: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=4000)
    render: bool = True


def build_router(dependencies: list[Any]) -> APIRouter:
    r = APIRouter()
    batches: dict[str, dict[str, Any]] = {}  # project -> state of the running shorts batch
    # clip -> music change in progress, or its error; AI music can take minutes to make
    music_work: dict[str, dict[str, Any]] = {}
    music_lock = threading.Lock()

    @r.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))

    @r.get("/v1/library/recent", dependencies=dependencies)
    def recent() -> list[dict[str, Any]]:
        return recent_videos()

    @r.get("/v1/system/status", dependencies=dependencies)
    def status(engine: Engine) -> dict[str, Any]:
        llm = engine.llm
        problem = llm.problem() if isinstance(llm, OllamaProvider) else None
        tracks = engine.music.tracks()
        profile = engine.toolchain.profile()
        return {
            "ai": {
                "name": llm.info.name if llm else None,
                "local": llm.info.runs_locally if llm else None,
                "problem": problem
                or (None if llm else "No AI judge: start Ollama, then restart SPHURA"),
            },
            "music": {
                "folder": str(engine.settings.music_dir),
                "tracks": len(tracks),
                "scores": True,
                "ai": engine.ai_music.ready(),
            },
            "encoders": list(profile.h264_encoders),
            "data_dir": str(engine.settings.data_dir),
        }

    @r.get("/v1/projects/{project_id}/moments/{candidate_id}/thumb.jpg", dependencies=dependencies)
    def thumb(project_id: str, candidate_id: str, engine: Engine) -> Response:
        path = engine.storage.artifact_path(project_id, "thumbs", f"{candidate_id}.jpg")
        if not path.is_file():
            moment = next(
                (
                    c
                    for c in engine.queries.highlights(project_id, None)
                    if c.candidate.id == candidate_id
                ),
                None,
            )
            assets = engine.queries.assets(project_id)
            if moment is None or not assets:
                raise HTTPException(404, "moment not found")
            jpeg = engine.sampler.sample(assets[0].source_uri, [moment.candidate.peak_s], 480)[0]
            path.write_bytes(jpeg)
        return FileResponse(
            path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"}
        )

    @r.post("/v1/projects/{project_id}/moments/{candidate_id}/rate", dependencies=dependencies)
    def rate(
        project_id: str, candidate_id: str, body: RateRequest, engine: Engine
    ) -> dict[str, bool]:
        engine.feedback.rate_moment(project_id, candidate_id, body.good)
        return {"ok": True}

    @r.get("/v1/projects/{project_id}/ratings", dependencies=dependencies)
    def ratings(project_id: str, engine: Engine) -> dict[str, bool]:
        return engine.feedback.ratings(project_id)

    @r.post("/v1/projects/{project_id}/shorts", dependencies=dependencies)
    def make_shorts(project_id: str, body: ShortsRequest, engine: Engine) -> dict[str, Any]:
        styles = [s for s in body.styles if s in STYLES]
        if not styles:
            raise HTTPException(400, f"choose at least one style: {', '.join(STYLES)}")
        if batches.get(project_id, {}).get("running"):
            raise HTTPException(409, "already preparing Shorts for this stream")
        lens = {st: _length(body.lengths.get(st, body.length_s)) for st in styles}
        project = engine.queries.project(project_id)
        out = shorts_folder(project.name)
        state: dict[str, Any] = {"running": True, "step": "starting", "error": None,
                                 "folder": str(out), "jobs": 0}  # fmt: skip
        batches[project_id] = state

        def work() -> None:
            # Copy writing can take a few seconds per clip, so this runs off the request
            try:
                for style in styles:
                    state["step"] = f"writing titles and planning edits ({style})"
                    clips = engine.clips.generate(
                        project_id, body.top, None, body.include_maybe, AspectRatio.VERTICAL,
                        None, not body.plain, style, body.game or None, lens[style],
                        body.caption_bar,
                    )  # fmt: skip
                    jobs = engine.clips.render(clips, out / style)
                    state["jobs"] += len(jobs)
                state["step"] = "rendering"
            except SphuraError as e:
                state["error"] = str(e)
            except Exception as e:
                state["error"] = f"{type(e).__name__}: {e}"
            finally:
                state["running"] = False

        threading.Thread(target=work, daemon=True, name=f"shorts-{project_id[:8]}").start()
        return state

    @r.get("/v1/projects/{project_id}/trending", dependencies=dependencies)
    def trending_plan(project_id: str, engine: Engine, game: str | None = None) -> dict[str, Any]:
        return engine.trending.plan(project_id, game)

    @r.post("/v1/projects/{project_id}/trending", dependencies=dependencies)
    def trending_make(project_id: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        picks = [p for p in body.get("picks", []) if isinstance(p, dict)]
        game = str(body.get("game") or "")
        if not picks:
            raise HTTPException(400, "choose at least one moment")
        if batches.get(project_id, {}).get("running"):
            raise HTTPException(409, "already preparing Shorts for this stream")
        out = shorts_folder(engine.queries.project(project_id).name) / "trending"
        state: dict[str, Any] = {"running": True, "step": "starting", "error": None,
                                 "folder": str(out), "jobs": 0}  # fmt: skip
        batches[project_id] = state

        def work() -> None:
            try:
                state["step"] = "writing titles and planning edits (trending style)"
                voices = {p["candidate_id"]: p.get("voiceover") for p in picks}
                for fmt, clips in engine.trending.make(project_id, game, picks):
                    for c in clips:
                        vo = voices.get(c.candidate_id or "")
                        if vo and vo.get("lines"):
                            state["step"] = f"recording the voice-over ({fmt})"
                            engine.voiceover.apply(c, str(vo.get("voice") or ""), vo["lines"])
                    state["jobs"] += len(engine.clips.render(clips, out / fmt))
                state["step"] = "rendering"
            except SphuraError as e:
                state["error"] = str(e)
            except Exception as e:
                state["error"] = f"{type(e).__name__}: {e}"
            finally:
                state["running"] = False

        threading.Thread(target=work, daemon=True, name=f"trend-{project_id[:8]}").start()
        return state

    @r.get("/v1/projects/{project_id}/plans", dependencies=dependencies)
    def short_plans(project_id: str, engine: Engine) -> dict[str, Any]:
        from sphura.pipeline.planner import FORMATS, VARIANTS
        from sphura.pipeline.research import appeal
        from sphura.pipeline.trending import detect_game

        name = engine.queries.project(project_id).name
        game = detect_game(name) or ""
        moments = {r.candidate.id: r.candidate for r in engine.queries.highlights(project_id)}
        made: dict[str, dict[str, str]] = {}
        for c, _path in engine.queries.clips(project_id):
            p = c.publish.get("plan") or {}
            if p.get("story"):
                made.setdefault(p["story"], {})[p["variant"]] = c.id
        out: list[dict[str, Any]] = []
        vids, hot = _research(engine, game)
        for p in engine.clips.plans(project_id, game):
            f = FORMATS[p.format]
            words = [w.capitalize() for w in re.findall(r"[A-Za-z]{4,}", p.title)]
            a = appeal(p.names, words, vids)
            trending = sorted({n for n in p.names if n.split()[0].lower() in hot})
            out.append({
                "appeal": a, "trending": trending,
                "rank": p.score + 2.0 * float(a["lift"]) + (1.0 if trending else 0.0),
                "key": p.key, "format": p.format, "format_label": f.label,
                "structure": f.structure, "title": p.title, "reason": p.reason,
                "score": p.score, "names": p.names,
                "variants": [{"id": v, "label": VARIANTS[v].label, "about": VARIANTS[v].about}
                             for v in p.variants],
                "moments": [{"id": m, "start_s": moments[m].range.start_s,
                             "title": _title(moments[m])} for m in p.moments if m in moments],
                "made": made.get(p.key, {}),
            })  # fmt: skip
        out.sort(key=lambda x: -x["rank"])
        status = engine.research.status()
        return {
            "game": game,
            "plans": out,
            "research_updated": status.get("last_updated"),
            "research_refreshing": _keep_fresh(engine, status),
        }

    @r.post("/v1/projects/{project_id}/plans", dependencies=dependencies)
    def make_plans(project_id: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        from sphura.pipeline.trending import detect_game

        picks = [p for p in body.get("picks", []) if isinstance(p, dict) and p.get("key")]
        if not picks:
            raise HTTPException(400, "choose at least one Short")
        if batches.get(project_id, {}).get("running"):
            raise HTTPException(409, "already preparing Shorts for this stream")
        name = engine.queries.project(project_id).name
        game = detect_game(name) or ""
        game_name = str(body.get("game") or "") or None
        out = shorts_folder(name) / "stories"
        state: dict[str, Any] = {"running": True, "step": "starting", "error": None,
                                 "folder": str(out), "jobs": 0}  # fmt: skip
        batches[project_id] = state

        titles = _top_titles(engine, game)

        def work() -> None:
            try:
                plans = {p.key: p for p in engine.clips.plans(project_id, game)}
                for i, pick in enumerate(picks, 1):
                    plan = plans.get(str(pick["key"]))
                    if plan is None:
                        continue
                    for v in [x for x in pick.get("variants", []) if x in plan.variants]:
                        state["step"] = f"cutting Short {i}/{len(picks)}, version {v}"
                        clip = engine.clips.make_plan(project_id, plan, v, game_name, titles)
                        state["jobs"] += len(engine.clips.render([clip], out / plan.format))
                state["step"] = "rendering"
            except SphuraError as e:
                state["error"] = str(e)
            except Exception as e:
                state["error"] = f"{type(e).__name__}: {e}"
            finally:
                state["running"] = False

        threading.Thread(target=work, daemon=True, name=f"plan-{project_id[:8]}").start()
        return state

    @r.post("/v1/clips/{clip_id}/choose", dependencies=dependencies)
    def choose_variant(clip_id: str, engine: Engine) -> dict[str, Any]:
        """Marks this version as the one to post; its siblings stay, unmarked."""
        clip = engine.clips.choose(clip_id)
        return {"chosen": clip.id}

    @r.get("/v1/voice", dependencies=dependencies)
    def voice_status(engine: Engine) -> dict[str, Any]:
        return engine.voiceover.status()

    @r.post("/v1/voice/install", dependencies=dependencies)
    def voice_install(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voiceover.install(str(body.get("engine") or "kokoro"))

    @r.post("/v1/voice/preview", dependencies=dependencies)
    def voice_preview(body: dict[str, Any], engine: Engine) -> FileResponse:
        text = str(body.get("text") or "").strip()[:600]
        if not text:
            raise HTTPException(400, "nothing to say")
        out = engine.voiceover.cached(text, str(body.get("voice") or ""), *_voice_args(body))
        return FileResponse(out, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @r.post("/v1/voice/plan", dependencies=dependencies)
    def voice_plan(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        """What the voice would read, for debugging and the preview panel."""
        text = str(body.get("text") or "").strip()[:600]
        return engine.voiceover.plan(text, str(body.get("voice") or ""), *_voice_args(body))

    @r.post("/v1/voice/warm", dependencies=dependencies)
    def voice_warm(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        items = [t if isinstance(t, dict) else str(t)[:400] for t in body.get("texts") or []][:20]
        game = str(body.get("game") or "") or None
        return {"queued": engine.voiceover.warm(items, str(body.get("voice") or ""), game)}

    @r.get("/v1/voice/intel", dependencies=dependencies)
    def voice_intel(engine: Engine) -> dict[str, Any]:
        from sphura.pipeline import voice_intel as VI

        return {
            "settings": engine.voice_intel.settings(),
            "games": engine.voice_intel.games(),
            "personalities": [asdict(p) for p in VI.PERSONALITIES.values()],
            "moods": list(VI.MOODS),
            "scenes": VI.SCENES,
            "engines": VI.ENGINES,
            "kinds": list(VI.KINDS),
        }

    @r.put("/v1/voice/intel/settings", dependencies=dependencies)
    def voice_intel_settings(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voice_intel.save_settings(body)

    @r.post("/v1/voice/intel/games", dependencies=dependencies)
    def voice_intel_new_game(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voice_intel.create_game(str(body.get("name") or ""))

    @r.get("/v1/voice/intel/{game}", dependencies=dependencies)
    def voice_intel_game(game: str, engine: Engine) -> dict[str, Any]:
        return {
            "profile": engine.voice_intel.profile(game),
            "terms": engine.voice_intel.terms(game),
        }

    @r.put("/v1/voice/intel/{game}", dependencies=dependencies)
    def voice_intel_profile(game: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voice_intel.save_profile(game, body)

    @r.post("/v1/voice/intel/{game}/scan", dependencies=dependencies)
    def voice_intel_scan(game: str, engine: Engine) -> dict[str, Any]:
        return engine.voice_intel.scan(game)

    @r.post("/v1/voice/intel/{game}/terms", dependencies=dependencies)
    def voice_intel_term(game: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voice_intel.upsert_term(game, body)

    @r.post("/v1/voice/intel/{game}/terms/{term_id}/verify", dependencies=dependencies)
    def voice_intel_verify(game: str, term_id: str, engine: Engine) -> dict[str, Any]:
        return engine.voice_intel.verify(game, term_id)

    @r.delete("/v1/voice/intel/{game}/terms/{term_id}", dependencies=dependencies)
    def voice_intel_delete(game: str, term_id: str, engine: Engine) -> dict[str, bool]:
        engine.voice_intel.delete_term(game, term_id)
        return {"ok": True}

    @r.post("/v1/projects/{project_id}/trending/scripts", dependencies=dependencies)
    def trending_scripts(project_id: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        from sphura.pipeline.trending import formats_for

        game, voice = str(body.get("game") or ""), str(body.get("voice") or "")
        fmts = {f.id: f for f in formats_for(game)}
        out = {}
        for p in body.get("picks", []):
            f = fmts.get(p.get("format"))
            if f is None:
                continue
            out[p["candidate_id"]] = engine.voiceover.script_for_moment(
                project_id, p["candidate_id"], game, voice, f.length_s, f.label
            )
        return out

    @r.get("/v1/clips/{clip_id}/voiceover", dependencies=dependencies)
    def clip_voiceover(clip_id: str, engine: Engine) -> dict[str, Any]:
        clip = engine.voiceover.get(clip_id)
        return next((e for e in clip.effects if e["type"] == "voiceover"), {})

    @r.post("/v1/clips/{clip_id}/voiceover/script", dependencies=dependencies)
    def clip_voiceover_script(clip_id: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voiceover.script_for_clip(clip_id, str(body.get("voice") or ""))

    @r.put("/v1/clips/{clip_id}/voiceover", dependencies=dependencies)
    def clip_voiceover_apply(clip_id: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        clip = engine.voiceover.get(clip_id)
        effect = engine.voiceover.apply(clip, str(body.get("voice") or ""), body.get("lines") or [])
        job = engine.clips.rerender(clip_id)
        _report_later(engine, clip_id)
        return {"voiceover": effect, "job_id": job.id if job else None}

    @r.get("/v1/clips/{clip_id}/voiceover/report", dependencies=dependencies)
    def clip_voiceover_report(
        clip_id: str, engine: Engine, refresh: bool = False
    ) -> dict[str, Any]:
        return engine.voice_qa.clip_report(clip_id, refresh)

    @r.post("/v1/clips/{clip_id}/voiceover/lines/{index}/regenerate", dependencies=dependencies)
    def clip_voiceover_regenerate(clip_id: str, index: int, engine: Engine) -> dict[str, Any]:
        """A new take of one line; the other lines keep their audio."""
        effect = engine.voiceover.regenerate_line(engine.voiceover.get(clip_id), index)
        job = engine.clips.rerender(clip_id)
        return {"voiceover": effect, "job_id": job.id if job else None,
                "report": engine.voice_qa.clip_report(clip_id, refresh=True)}  # fmt: skip

    @r.get("/v1/voice/bench", dependencies=dependencies)
    def voice_bench(engine: Engine) -> dict[str, Any]:
        from sphura.pipeline.voice_bench import SCENES

        return {"engines": engine.voice_bench.available(), "runs": engine.voice_bench.runs(),
                "scenes": SCENES}  # fmt: skip

    @r.post("/v1/voice/bench", dependencies=dependencies)
    def voice_bench_start(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voice_bench.start([str(e) for e in body.get("engines") or []])

    @r.get("/v1/voice/bench/{run_id}/{file}", dependencies=dependencies)
    def voice_bench_audio(run_id: str, file: str, engine: Engine) -> FileResponse:
        return FileResponse(engine.voice_bench.audio(run_id, file), media_type="audio/wav")

    @r.post("/v1/voice/bench/{run_id}/rate", dependencies=dependencies)
    def voice_bench_rate(run_id: str, body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.voice_bench.rate(
            run_id, str(body.get("file") or ""), int(body.get("score") or 3)
        )

    @r.get("/v1/voice/qa", dependencies=dependencies)
    def voice_qa_status(engine: Engine) -> dict[str, Any]:
        return engine.voice_qa.status()

    @r.post("/v1/voice/qa/line", dependencies=dependencies)
    def voice_qa_line(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        game, direction, overrides, _ = _voice_args(body)
        text, voice = str(body.get("text") or "")[:600], str(body.get("voice") or "")
        return engine.voice_qa.line_report(text, voice, game, direction, overrides)

    @r.post("/v1/voice/intel/{game}/terms/{term_id}/heard", dependencies=dependencies)
    def voice_intel_heard(
        game: str, term_id: str, body: dict[str, Any], engine: Engine
    ) -> dict[str, Any]:
        """False positive: recognition hears this when the name is said right."""
        return engine.voice_qa.accept_heard(game, term_id, str(body.get("heard") or ""))

    @r.get("/v1/voice/intel/{game}/variants", dependencies=dependencies)
    def voice_intel_variants(game: str, name: str, engine: Engine) -> list[str]:
        return engine.voice_qa.variants(game, name)

    @r.delete("/v1/clips/{clip_id}/voiceover", dependencies=dependencies)
    def clip_voiceover_remove(clip_id: str, engine: Engine) -> dict[str, Any]:
        engine.voiceover.remove(engine.voiceover.get(clip_id))
        job = engine.clips.rerender(clip_id)
        return {"job_id": job.id if job else None}

    @r.get("/v1/projects/{project_id}/shorts", dependencies=dependencies)
    def shorts(project_id: str, engine: Engine) -> dict[str, Any]:
        items = []
        for clip, path in engine.queries.clips(project_id):
            if not path or not Path(path).is_file():
                continue
            items.append(
                {
                    "id": clip.id,
                    "candidate_id": clip.candidate_id,
                    "rank": clip.rank,
                    "style": clip.style,
                    "aspect": clip.aspect.value,
                    "title": clip.title,
                    "publish": clip.publish,
                    "trend": clip.publish.get("trend"),
                    "plan": clip.publish.get("plan"),
                    "chosen": bool(clip.publish.get("chosen")),
                    "voiceover": bool(
                        next((e for e in clip.effects if e["type"] == "voiceover"), None)
                    ),
                    "text": _overlay_text(clip),
                    "music": {**_music_info(clip), "work": music_work.get(clip.id)},
                    "duration_s": round(clip.output_duration_s, 1),
                    "video_url": f"/v1/clips/{clip.id}/video",
                    "modified": Path(path).stat().st_mtime,
                }
            )
        items.sort(key=lambda c: (c["rank"] or 99, c["style"]))
        renders = [j for j in engine.queries.jobs(project_id) if j.type == JobType.RENDER]
        active = [j for j in renders if not j.status.is_terminal]
        return {
            "batch": batches.get(project_id),
            "rendering": len(active),
            "progress": [{"id": j.id, "stage": j.stage, "progress": j.progress} for j in active],
            "shorts": items,
        }

    @r.patch("/v1/clips/{clip_id}/text", dependencies=dependencies)
    def edit_text(clip_id: str, body: TextEdit, engine: Engine) -> dict[str, Any]:
        clip = engine.clips.edit_text(clip_id, body.text, body.title, body.description)
        job = engine.clips.rerender(clip_id) if body.render else None
        return {"text": _overlay_text(clip), "job_id": job.id if job else None}

    @r.get("/v1/downloads", dependencies=dependencies)
    def downloads(engine: Engine) -> dict[str, Any]:
        dm = engine.downloads
        return {
            "ready": dm.tools_ready(), "setup": dm.setup, "items": dm.items(),
            "settings": asdict(dm.settings()), "storage": dm.storage(),
        }  # fmt: skip

    @r.post("/v1/downloads/tools", dependencies=dependencies)
    def download_tools(engine: Engine) -> dict[str, bool]:
        engine.downloads.install_tools()
        return {"ok": True}

    @r.post("/v1/downloads/preview", dependencies=dependencies)
    def download_preview(body: PreviewRequest, engine: Engine) -> dict[str, Any]:
        return engine.downloads.preview(_youtube_url(body.url), body.playlist)

    @r.post("/v1/downloads", dependencies=dependencies)
    def add_downloads(body: AddDownloads, engine: Engine) -> dict[str, Any]:
        if not body.confirm_rights:
            raise HTTPException(400, "confirm that these are your videos or you have permission")
        return engine.downloads.add(body.items, body.quality, body.playlist_title)

    @r.post("/v1/downloads/settings/choose-folder", dependencies=dependencies)
    def choose_folder(engine: Engine) -> dict[str, Any]:
        return {"folder": _choose_folder(engine.downloads.settings().folder)}

    @r.post("/v1/downloads/{download_id}/{action}", dependencies=dependencies)
    def download_action(download_id: str, action: str, engine: Engine) -> dict[str, Any]:
        dm = engine.downloads
        actions = {"pause": dm.pause, "resume": dm.resume, "cancel": dm.cancel,
                   "retry": dm.retry, "remove": dm.remove}  # fmt: skip
        if action == "edit":
            return {"project_id": dm.edit(download_id)}
        if action == "reveal":
            _reveal(dm.file_of(download_id))
            return {"ok": True}
        if action not in actions:
            raise HTTPException(404, f"unknown action {action}")
        actions[action](download_id)
        return {"ok": True}

    @r.delete("/v1/downloads/{download_id}/file", dependencies=dependencies)
    def delete_download(download_id: str, engine: Engine) -> dict[str, bool]:
        engine.downloads.delete_file(download_id)
        return {"ok": True}

    @r.get("/v1/downloads/{download_id}/thumb.jpg", dependencies=dependencies)
    def download_thumb(download_id: str, engine: Engine) -> Response:
        path, url = engine.downloads.thumbnail(download_id)
        if path:
            return FileResponse(path, media_type="image/jpeg")
        if url.startswith("https://"):
            return RedirectResponse(url)
        raise HTTPException(404, "no thumbnail")

    @r.get("/v1/downloads/{download_id}/video", dependencies=dependencies)
    def download_video(download_id: str, engine: Engine) -> FileResponse:
        return FileResponse(engine.downloads.file_of(download_id), media_type="video/mp4")

    @r.put("/v1/downloads/settings", dependencies=dependencies)
    def download_settings(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return asdict(engine.downloads.save_settings(body))

    @r.post("/v1/library/choose-file", dependencies=dependencies)
    def choose_file() -> dict[str, Any]:
        videos = Path.home() / "Videos"
        return {"path": _choose_video(str(videos if videos.is_dir() else Path.home()))}

    @r.get("/v1/music", dependencies=dependencies)
    def music(engine: Engine) -> dict[str, Any]:
        tracks = engine.music.tracks()
        return {
            "folder": str(engine.settings.music_dir),
            "tracks": [
                {
                    "path": t.path,
                    "file": Path(t.path).name,
                    "name": t.name,
                    "length_s": t.length_s,
                    "bpm": round(60 / t.beat_s) if t.beat_s else None,
                    "drop_s": t.drop_s,
                    "tone": "dark"
                    if t.brightness < 0.25
                    else "bright"
                    if t.brightness > 0.5
                    else "balanced",
                    "builds": t.dynamics_db > 8,
                    "source": t.source,
                    "license": t.license,
                    "credit": t.credit,
                    "tags": t.tags,
                }
                for t in sorted(tracks, key=lambda t: Path(t.path).stem.lower())
            ],
        }

    @r.get("/v1/music/ai", dependencies=dependencies)
    def ai_music(engine: Engine) -> dict[str, Any]:
        ai = engine.ai_music
        return {"ready": ai.ready(), "install": ai.status, "folder": str(ai.root)}

    @r.post("/v1/music/ai/install", dependencies=dependencies)
    def ai_music_install(engine: Engine) -> dict[str, bool]:
        engine.ai_music.install()
        return {"ok": True}

    @r.post("/v1/music/upload", dependencies=dependencies)
    async def upload_music(request: Request, name: str, engine: Engine) -> dict[str, Any]:
        """Raw file body (no multipart dependency); saved into the music folder."""
        safe = Path(name).name.strip()
        if Path(safe).suffix.lower() not in MUSIC_EXT:
            raise HTTPException(400, f"music must be one of {', '.join(sorted(MUSIC_EXT))}")
        if int(request.headers.get("content-length") or 0) > MAX_UPLOAD:
            raise HTTPException(413, "file too large (max 60 MB)")
        data = await request.body()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file too large (max 60 MB)")
        folder = engine.settings.music_dir
        folder.mkdir(parents=True, exist_ok=True)
        (folder / safe).write_bytes(data)
        return {"saved": str(folder / safe)}

    @r.patch("/v1/music/meta", dependencies=dependencies)
    def music_meta(body: TrackMeta, engine: Engine) -> dict[str, bool]:
        try:
            engine.music.set_meta(body.file, body.model_dump(exclude={"file"}))
        except ValueError as e:
            raise HTTPException(404, str(e)) from e
        return {"ok": True}

    @r.post("/v1/music/open-folder", dependencies=dependencies)
    def open_music_folder(engine: Engine) -> dict[str, bool]:
        engine.settings.music_dir.mkdir(parents=True, exist_ok=True)
        _reveal(engine.settings.music_dir)
        return {"ok": True}

    @r.patch("/v1/clips/{clip_id}/music", dependencies=dependencies)
    def set_music(clip_id: str, body: MusicChoice, engine: Engine) -> dict[str, Any]:
        if music_work.get(clip_id, {}).get("busy"):
            raise HTTPException(409, "Music for this Short is already being made")
        music_work[clip_id] = {"busy": True}

        def work() -> None:
            try:
                with music_lock:  # one at a time: they share the AI music model
                    engine.clips.set_music(clip_id, body.choice, body.volume_db)
                if body.render:
                    engine.clips.rerender(clip_id)
                music_work.pop(clip_id, None)
            except Exception as e:  # shown on the Short instead of failing silently
                log.exception("music change failed for %s", clip_id)
                music_work[clip_id] = {"busy": False, "error": str(e) or type(e).__name__}

        threading.Thread(target=work, daemon=True, name=f"music-{clip_id[:8]}").start()
        return {"started": True}

    @r.get("/v1/clips/{clip_id}/editor", dependencies=dependencies)
    def editor(clip_id: str, engine: Engine) -> dict[str, Any]:
        state = engine.clips.editor_state(clip_id)
        state["has_draft"] = engine.clips.draft_path(clip_id).is_file()
        return state

    @r.post("/v1/clips/{clip_id}/editor/audio", dependencies=dependencies)
    def editor_audio(clip_id: str, body: EditorAudio, engine: Engine) -> dict[str, Any]:
        return engine.clips.editor_audio(clip_id, body.segments)

    @r.put("/v1/clips/{clip_id}/editor", dependencies=dependencies)
    def save_editor(clip_id: str, body: EditorSave, engine: Engine) -> dict[str, Any]:
        clips = engine.clips
        clip = clips.apply_edit(
            clip_id, body.segments, body.overlays, body.fx_off, body.caption_bar
        )
        job = clips.draft(clip_id) if body.render == "draft" else clips.rerender(clip_id)
        return {"job_id": job.id, "duration": round(clip.output_duration_s, 3)}

    @r.get("/v1/clips/{clip_id}/draft.mp4", dependencies=dependencies)
    def draft_video(clip_id: str, engine: Engine) -> FileResponse:
        path = engine.clips.draft_path(clip_id)
        if not path.is_file():
            raise HTTPException(404, "no preview yet")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @r.get("/v1/clips/{clip_id}/frame.jpg", dependencies=dependencies)
    def strip_frame(clip_id: str, t: float, engine: Engine) -> FileResponse:
        path, uri, at = engine.clips.frame_slot(clip_id, t)
        if not path.is_file():
            try:
                jpeg = engine.sampler.sample(uri, [at], 160)[0]
            except Exception as e:  # e.g. past the end of the recording
                raise HTTPException(404, "no frame there") from e
            path.write_bytes(jpeg)
        return FileResponse(
            path, media_type="image/jpeg", headers={"Cache-Control": "max-age=604800"}
        )

    @r.get("/v1/clips/{clip_id}/source", dependencies=dependencies)
    def source_video(clip_id: str, engine: Engine) -> FileResponse:
        path = engine.clips.source_path(clip_id)
        if not path.is_file():
            raise HTTPException(404, "the original recording was moved or deleted")
        return FileResponse(path, media_type="video/mp4")

    @r.get("/v1/clips/{clip_id}/video", dependencies=dependencies)
    def video(clip_id: str, engine: Engine) -> FileResponse:
        path = engine.queries.clip_output(clip_id)
        if not path or not Path(path).is_file():
            raise HTTPException(404, "not rendered yet")
        return FileResponse(path, media_type="video/mp4")

    @r.post("/v1/clips/{clip_id}/reveal", dependencies=dependencies)
    def reveal(clip_id: str, engine: Engine) -> dict[str, bool]:
        path = engine.queries.clip_output(clip_id)
        if not path or not Path(path).is_file():
            raise HTTPException(404, "not rendered yet")
        _reveal(Path(path))
        return {"ok": True}

    @r.post("/v1/open-folder", dependencies=dependencies)
    def open_folder(body: dict[str, str], engine: Engine) -> dict[str, bool]:
        folder = Path(body.get("path", ""))
        allowed = [(Path.home() / "Videos" / "SPHURA").resolve(),
                   Path(engine.downloads.settings().folder).resolve()]  # fmt: skip
        if not any(folder.resolve().is_relative_to(a) for a in allowed) or not folder.is_dir():
            raise HTTPException(400, "can only open SPHURA output folders")
        _reveal(folder)
        return {"ok": True}

    # ---------- AI highlight reels ----------

    @r.get("/v1/reels", dependencies=dependencies)
    def reels(engine: Engine) -> list[dict[str, Any]]:
        return engine.reels.all()

    @r.post("/v1/reels", dependencies=dependencies)
    def new_reel(body: ReelNew, engine: Engine) -> dict[str, Any]:
        return engine.reels.create(body.name)

    @r.get("/v1/reels/hidden-projects", dependencies=dependencies)
    def reel_projects(engine: Engine) -> list[str]:
        return engine.reels.hidden_projects()

    @r.get("/v1/lore", dependencies=dependencies)
    def lore_overview(engine: Engine) -> dict[str, Any]:
        return engine.lore.overview()

    @r.get("/v1/lore/library", dependencies=dependencies)
    def lore_library(engine: Engine) -> list[dict[str, Any]]:
        return engine.lore.library()

    @r.post("/v1/lore/index", dependencies=dependencies)
    def lore_index(body: LoreIndexRequest, engine: Engine) -> list[dict[str, Any]]:
        return engine.lore.index(body.keys)

    @r.post("/v1/lore/index/cancel", dependencies=dependencies)
    def lore_cancel(body: LoreCancel, engine: Engine) -> dict[str, bool]:
        engine.lore.cancel(body.project_id)
        return {"ok": True}

    @r.get("/v1/lore/videos/{asset_id}/scenes", dependencies=dependencies)
    def lore_scenes(asset_id: str, engine: Engine) -> list[dict[str, Any]]:
        return engine.lore.scenes(asset_id)

    @r.get("/v1/lore/videos/{asset_id}/source", dependencies=dependencies)
    def lore_source(asset_id: str, engine: Engine) -> FileResponse:
        return FileResponse(engine.lore.source(asset_id), media_type="video/mp4")

    @r.get("/v1/lore/frame.jpg", dependencies=dependencies)
    def lore_frame(asset: str, t: float, engine: Engine) -> FileResponse:
        try:
            path = engine.lore.frame(asset, round(t, 1))
        except SphuraError:
            raise
        except Exception as e:  # past the end of the recording, unreadable file
            raise HTTPException(404, "no frame there") from e
        return FileResponse(
            path, media_type="image/jpeg", headers={"Cache-Control": "max-age=604800"}
        )

    @r.get("/v1/lore/{game}/ideas", dependencies=dependencies)
    def lore_ideas(game: str, engine: Engine) -> dict[str, Any]:
        return engine.lore.ideas(game)

    @r.post("/v1/lore/{game}/discover", dependencies=dependencies)
    def lore_discover(game: str, body: LoreDiscover, engine: Engine) -> dict[str, Any]:
        return engine.lore.discover(game, body.asset_ids)

    @r.get("/v1/lore/{game}/sources", dependencies=dependencies)
    def lore_sources(game: str, engine: Engine) -> dict[str, Any]:
        return engine.lore.sources(game)

    @r.get("/v1/research", dependencies=dependencies)
    def research_overview(engine: Engine) -> dict[str, Any]:
        return engine.research.overview()

    @r.get("/v1/research/status", dependencies=dependencies)
    def research_status(engine: Engine) -> dict[str, Any]:
        return engine.research.status()

    @r.put("/v1/research/key", dependencies=dependencies)
    def research_key(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.research.set_key(str(body.get("key") or ""))

    @r.post("/v1/research/refresh", dependencies=dependencies)
    def research_refresh(body: dict[str, Any], engine: Engine) -> dict[str, Any]:
        return engine.research.refresh(body.get("games") or None)

    @r.get("/v1/research/videos", dependencies=dependencies)
    def research_videos(
        engine: Engine, game: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        return engine.research.videos(game, category)

    @r.get("/v1/research/videos/{video_id}/history", dependencies=dependencies)
    def research_history(video_id: str, engine: Engine) -> list[dict[str, Any]]:
        return engine.research.history(video_id)

    @r.get("/v1/reels/branding", dependencies=dependencies)
    def reel_branding(engine: Engine) -> dict[str, str]:
        return engine.reels.branding()

    @r.put("/v1/reels/branding", dependencies=dependencies)
    def set_reel_branding(body: dict[str, Any], engine: Engine) -> dict[str, str]:
        return engine.reels.set_branding(body)

    @r.post("/v1/reels/branding/choose", dependencies=dependencies)
    def choose_branding_file(kind: str) -> dict[str, Any]:
        if kind == "logo":
            types = "[('Images', '*.png *.jpg *.jpeg *.webp'), ('All files', '*.*')]"
        else:
            types = "[('Videos', '*.mp4 *.mov *.mkv *.webm *.avi'), ('All files', '*.*')]"
        dialog = (
            f"askopenfilename(initialdir=sys.argv[1], title='Choose a file', filetypes={types})"
        )
        return {"path": _pick(dialog, str(Path.home()))}

    @r.get("/v1/reels/{reel_id}", dependencies=dependencies)
    def reel(reel_id: str, engine: Engine) -> dict[str, Any]:
        return engine.reels.get(reel_id)

    @r.patch("/v1/reels/{reel_id}", dependencies=dependencies)
    def update_reel(reel_id: str, body: ReelPatch, engine: Engine) -> dict[str, Any]:
        return engine.reels.update(reel_id, body.model_dump(exclude_none=True))

    @r.delete("/v1/reels/{reel_id}", dependencies=dependencies)
    def delete_reel(reel_id: str, engine: Engine) -> dict[str, bool]:
        engine.reels.delete(reel_id)
        return {"ok": True}

    @r.post("/v1/reels/{reel_id}/sources", dependencies=dependencies)
    def add_reel_paths(reel_id: str, body: ReelPaths, engine: Engine) -> dict[str, Any]:
        return engine.reels.add_paths(reel_id, [p.strip().strip('"') for p in body.paths])

    @r.post("/v1/reels/{reel_id}/choose-files", dependencies=dependencies)
    def choose_reel_files(reel_id: str, engine: Engine) -> dict[str, Any]:
        videos = Path.home() / "Videos"
        paths = _pick_many(str(videos if videos.is_dir() else Path.home()))
        return engine.reels.add_paths(reel_id, paths) if paths else engine.reels.get(reel_id)

    @r.post("/v1/reels/{reel_id}/upload", dependencies=dependencies)
    async def upload_reel_file(
        reel_id: str, name: str, request: Request, engine: Engine
    ) -> dict[str, Any]:
        """Raw file body, streamed to disk (recordings can be several GB)."""
        dest = engine.reels.upload_target(reel_id, name)
        tmp = dest.with_name(dest.name + ".part")
        try:
            with tmp.open("wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)
            tmp.replace(dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return engine.reels.add_paths(reel_id, [str(dest)], uploaded=True)

    @r.delete("/v1/reels/{reel_id}/sources/{source_id}", dependencies=dependencies)
    def remove_reel_source(reel_id: str, source_id: str, engine: Engine) -> dict[str, Any]:
        return engine.reels.remove_source(reel_id, source_id)

    @r.post("/v1/reels/{reel_id}/sources/{source_id}/move", dependencies=dependencies)
    def move_reel_source(
        reel_id: str, source_id: str, body: ReelMove, engine: Engine
    ) -> dict[str, Any]:
        return engine.reels.move_source(reel_id, source_id, body.index)

    @r.get("/v1/reels/{reel_id}/sources/{source_id}/video", dependencies=dependencies)
    def reel_source_video(reel_id: str, source_id: str, engine: Engine) -> FileResponse:
        return FileResponse(engine.reels.source_file(reel_id, source_id), media_type="video/mp4")

    @r.get("/v1/reels/{reel_id}/moments/{moment_id}/thumb.jpg", dependencies=dependencies)
    def reel_moment_thumb(reel_id: str, moment_id: str, engine: Engine) -> FileResponse:
        path = engine.reels.moment_thumb(reel_id, moment_id)
        return FileResponse(
            path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"}
        )

    @r.post("/v1/reels/{reel_id}/analyse", dependencies=dependencies)
    def analyse_reel(reel_id: str, engine: Engine) -> dict[str, Any]:
        return engine.reels.analyse(reel_id)

    @r.post("/v1/reels/{reel_id}/cancel", dependencies=dependencies)
    def cancel_reel(reel_id: str, engine: Engine) -> dict[str, Any]:
        return engine.reels.cancel(reel_id)

    @r.post("/v1/reels/{reel_id}/regenerate", dependencies=dependencies)
    def regenerate_reel(reel_id: str, engine: Engine) -> dict[str, Any]:
        return engine.reels.regenerate(reel_id)

    @r.post("/v1/reels/{reel_id}/generate", dependencies=dependencies)
    def generate_reel(reel_id: str, engine: Engine) -> dict[str, Any]:
        return engine.reels.generate(reel_id)

    @r.get("/v1/reels/{reel_id}/output.mp4", dependencies=dependencies)
    def reel_output(reel_id: str, engine: Engine) -> FileResponse:
        out = engine.reels.get(reel_id).get("output") or {}
        if not out.get("path") or not Path(out["path"]).is_file():
            raise HTTPException(404, "the reel hasn't been made yet, or the file was moved")
        return FileResponse(
            out["path"], media_type="video/mp4", headers={"Cache-Control": "no-store"}
        )

    @r.get("/v1/reels/{reel_id}/thumbnail.jpg", dependencies=dependencies)
    def reel_thumbnail(reel_id: str, engine: Engine) -> FileResponse:
        out = engine.reels.get(reel_id).get("output") or {}
        if not out.get("thumbnail") or not Path(out["thumbnail"]).is_file():
            raise HTTPException(404, "no thumbnail")
        return FileResponse(
            out["thumbnail"], media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @r.post("/v1/reels/{reel_id}/reveal", dependencies=dependencies)
    def reveal_reel(reel_id: str, engine: Engine) -> dict[str, bool]:
        out = engine.reels.get(reel_id).get("output") or {}
        if not out.get("path"):
            raise HTTPException(404, "the reel hasn't been made yet")
        _reveal(Path(out["path"]))
        return {"ok": True}

    return r


def _length(v: float | None) -> float | None:
    if v is None:
        return None
    if not 5 <= v <= 180:
        raise HTTPException(400, "Short length must be between 5 and 180 seconds")
    return float(v)


def _report_later(engine: Any, clip_id: str) -> None:
    """The quality report needs the new audio and speech recognition: build it off-request."""

    def run() -> None:
        try:
            engine.voice_qa.clip_report(clip_id, refresh=True)
        except Exception:
            log.warning("voice-over report for %s failed", clip_id, exc_info=True)

    threading.Thread(target=run, daemon=True, name=f"voice-report-{clip_id[:8]}").start()


def _voice_args(body: dict[str, Any]) -> tuple[Any, ...]:
    """(game, direction, overrides, respell) from a preview request."""
    game = str(body.get("game") or "") or None
    direction = body.get("direction") if isinstance(body.get("direction"), dict) else None
    keys = ("personality", "speed", "intensity", "pauses", "emotion", "pronunciation", "optimize",
            "take")  # fmt: skip
    overrides = {k: body[k] for k in keys if body.get(k) is not None} or None
    respell = body.get("respell") if isinstance(body.get("respell"), dict) else None
    return game, direction, overrides, respell


def _youtube_url(url: str) -> str:
    u = url.strip()
    if not re.match(r"^https?://([\w-]+\.)?(youtube\.com|youtu\.be)/", u, re.IGNORECASE):
        raise HTTPException(400, "paste a youtube.com or youtu.be link")
    return u


def _music_info(clip: Any) -> dict[str, Any]:
    m = next((e for e in clip.effects if e["type"] == "music"), None)
    choice = (clip.publish.get("music") or {}).get("choice", "auto")
    ai = clip.publish.get("music_ai") or None
    if ai:
        ai = {
            "path": ai["path"],
            "name": Path(ai["path"]).stem,
            "reason": ai.get("reason", ""),
            "fits_well": ai.get("fits_well", True),
            "search": ai.get("search", ""),
            "alternatives": [{"path": p, "name": Path(p).stem} for p in ai.get("alternatives", [])],
        }
    info = {"choice": choice, "ai": ai, "name": None, "gain_db": None, "credit": "", "score": None}
    if m is not None:
        info.update(name=Path(m["path"]).stem, gain_db=m.get("gain_db"), credit=m.get("credit", ""))
        if m.get("generated"):
            g = m["generated"]
            info.update(
                name="SPHURA score",
                score={
                    "mood": g.get("mood"),
                    "label": g.get("label"),
                    "bpm": g.get("bpm"),
                    "ai": bool(g.get("ai")),
                    "prompt": g.get("prompt"),
                },
            )
    return info


def _overlay_text(clip: Any) -> str:
    return str(clip.overlays[0].text) if clip.overlays else ""


def _choose_folder(start: str) -> str | None:
    return _pick("askdirectory(initialdir=sys.argv[1], title='Choose download folder')", start)


def _choose_video(start: str) -> str | None:
    return _pick(
        "askopenfilename(initialdir=sys.argv[1], title='Choose a recording', filetypes=["
        "('Videos', '*.mp4 *.mkv *.mov *.webm *.avi *.flv *.ts *.m4v'), ('All files', '*.*')])",
        start,
    )


def _pick(dialog: str, start: str) -> str | None:
    """Native file/folder picker in a child process (Tk must own its own main thread)."""
    code = (
        "import sys, tkinter as tk; from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        f"print(filedialog.{dialog})"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code, start], capture_output=True,
                             text=True, timeout=600)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        raise HTTPException(500, "couldn't open the picker; type the path instead") from e
    if out.returncode != 0:
        raise HTTPException(500, "couldn't open the picker; type the path instead")
    return str(Path(out.stdout.strip())) if out.stdout.strip() else None


def _pick_many(start: str) -> list[str]:
    """Native multi-file picker for videos (child process, like _pick)."""
    code = (
        "import sys, json, tkinter as tk; from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "print(json.dumps(list(filedialog.askopenfilenames(initialdir=sys.argv[1], "
        "title='Choose videos for the highlight reel', filetypes=[('Videos', "
        "'*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.ts *.flv'), ('All files', '*.*')]))))"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code, start], capture_output=True,
                             text=True, timeout=600)  # fmt: skip
        return [str(Path(p)) for p in json.loads(out.stdout.strip() or "[]")]
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        raise HTTPException(500, "couldn't open the picker; drop the files in instead") from e


def _reveal(path: Path) -> None:
    if sys.platform == "win32":
        if path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(path)  # type: ignore[attr-defined]
        folder = path.parent if path.is_file() else path
        threading.Thread(target=_raise_explorer, args=(folder,), daemon=True).start()
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R" if path.is_file() else "", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent if path.is_file() else path)])


def _raise_explorer(folder: Path, wait_s: float = 4.0) -> None:
    """Windows opens Explorer behind the browser (SPHURA runs in the background), so find
    that window and bring it to the front. A tap of Alt lifts Windows' focus-stealing lock."""
    import ctypes
    import time
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    names = {folder.name.lower(), str(folder).lower().rstrip("\\")}
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)  # type: ignore[attr-defined]

    def find() -> int | None:
        found: list[int] = []

        def check(hwnd: int, _: int) -> bool:
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            if cls.value in ("CabinetWClass", "ExploreWClass") and user32.IsWindowVisible(hwnd):
                title = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd, title, 512)
                if title.value.lower().rstrip("\\") in names:
                    found.append(hwnd)
                    return False
            return True

        user32.EnumWindows(enum_proc(check), 0)
        return found[0] if found else None

    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        hwnd = find()
        if hwnd:
            try:
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.keybd_event(0x12, 0, 0, 0)  # Alt down/up
                user32.keybd_event(0x12, 0, 2, 0)
                user32.SetForegroundWindow(hwnd)
                user32.BringWindowToTop(hwnd)
            except Exception:
                log.debug("couldn't bring Explorer to the front", exc_info=True)
            return
        time.sleep(0.15)


def _title(c: Any) -> str:
    return str(c.judgement.title) if c.judgement else ""


def _top_titles(engine: Any, game: str, n: int = 15) -> str:
    """Titles of the most-viewed researched Shorts of this game, as examples for ours."""
    try:
        vids = engine.research.videos(game) if game else []
    except Exception:
        return ""
    top = sorted(vids, key=lambda v: -(v.get("metrics") or {}).get("views", 0))[:n]
    if not top:
        return ""
    return "Titles of the most-viewed Shorts of this game (match their voice, not words):\n" + (
        "\n".join(f"- {v['title']}" for v in top)
    )


def _research(engine: Any, game: str) -> tuple[list[dict[str, Any]], set[str]]:
    """Researched Shorts of the game, and the words in titles trending right now."""
    try:
        vids = engine.research.videos(game) if game else []
    except Exception:
        return [], set()
    hot = {w.lower() for v in vids if "trending" in (v.get("categories") or [])
           for w in re.findall(r"[A-Za-z]{4,}", v.get("title", ""))}  # fmt: skip
    return vids, hot


RESEARCH_MAX_AGE_DAYS = 3  # what's trending moves fast; a refresh costs ~2K of 10K daily units


def _keep_fresh(engine: Any, status: dict[str, Any]) -> bool:
    """Starts a background research refresh when the data is a few days old; True while one runs."""
    last = status.get("last_updated")
    if status.get("running"):
        return True
    if not (status.get("key") or {}).get("set"):
        return False
    try:
        age = (datetime.now(UTC) - datetime.fromisoformat(str(last))).days if last else 99
        if age >= RESEARCH_MAX_AGE_DAYS:
            engine.research.refresh()
            return True
    except Exception:  # never block planning on research
        log.warning("couldn't refresh the Shorts research", exc_info=True)
    return False
