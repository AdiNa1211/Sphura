"""Lore Studio: index whole videos into scenes, research lore with attribution, and discover
story ideas that the creator's own footage can show."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from sphura.domain.errors import NotFoundError, SphuraError
from sphura.domain.models import Job, JobStatus, JobType, MediaAsset
from sphura.infrastructure.persistence.sqlite.lore import SqliteLoreRepository
from sphura.infrastructure.research.wiki import WikiClient, WikiError
from sphura.pipeline import lore as L
from sphura.pipeline.manifest import StageManifest
from sphura.pipeline.research import appeal
from sphura.pipeline.stages import ExtractAudioStage, TranscribeStage
from sphura.pipeline.tracks import choose_voice_track
from sphura.ports.ai import ImageInput, LLMProvider, LLMRequest
from sphura.ports.queue import JobContext, JobQueue
from sphura.ports.repositories import MediaAssetRepository
from sphura.ports.storage import StorageProvider

log = logging.getLogger(__name__)

_WEIGHTS = {
    "waiting": 0.0,
    "extract_audio": 0.02,
    "transcribe": 0.23,
    "frames": 0.72,
    "scenes": 0.03,
}
PAGE_MAX_AGE = timedelta(days=30)
_NOT_MYTH = re.compile(r"god of war|video game|\(character\)|\(film\)|marvel", re.I)
MAX_SUBJECTS = 10


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {}
    return data if isinstance(data, dict) else {}


# ---------- indexing (a LocalJobQueue job; frame tags are saved as it goes) ----------


class LoreIndexHandler:
    def __init__(
        self,
        assets: MediaAssetRepository,
        storage: StorageProvider,
        extract: ExtractAudioStage,
        transcribe: TranscribeStage,
        sample: Any,  # (uri, times, width) -> list[jpeg]
        llm: LLMProvider | None,
        repo: SqliteLoreRepository,
        queries: Any,
    ) -> None:
        self._assets = assets
        self._storage = storage
        self._extract = extract
        self._transcribe = transcribe
        self._sample = sample
        self._llm = llm
        self._repo = repo
        self._queries = queries

    def __call__(self, ctx: JobContext) -> dict[str, Any]:
        asset = self._assets.get(ctx.job.payload["asset_id"])
        if asset is None or asset.media_info is None:
            raise NotFoundError("video not found; add it to SPHURA again")
        if self._llm is None:
            raise SphuraError("Lore indexing needs the local AI: start Ollama, then restart SPHURA")
        _wait_for_file(ctx, Path(asset.source_uri))
        game = ctx.job.payload.get("game")
        fp, ver = asset.fingerprint, L.INDEX_VERSION
        self._repo.drop_old_frames(asset.id, fp, ver)
        self._wait_for_analysis(ctx, asset.project_id)
        segs = self._speech(ctx, asset)
        frames = self._frames(ctx, asset, game, segs)
        ctx.report("scenes", 0.97, "Building scenes")
        scenes = L.build_scenes(
            frames, segs, self._moments(asset.project_id), game, asset.id, asset.project_id
        )
        self._repo.replace_scenes(asset.id, scenes)
        chars = Counter(c for s in scenes for c in s["seen"])
        places = Counter(s["location"] for s in scenes if s["location"])
        summary = {
            "scenes": len(scenes),
            "cinematic": sum(s["kind"] in ("cinematic", "dialogue") for s in scenes),
            "characters": [c for c, _ in chars.most_common(8)],
            "places": [p for p, _ in places.most_common(6)],
            "frames": len(frames),
            "unread": sum(1 for f in frames.values() if f.get("missing")),
        }
        self._repo.save_index(asset.id, asset.project_id, game, fp, ver, "done", summary, _now())
        ctx.report("done", 1.0)
        return summary

    def _wait_for_analysis(self, ctx: JobContext, project_id: str) -> None:
        """A highlight analysis already running makes the same transcript: let it finish
        instead of transcribing twice on the same GPU."""
        while any(
            j.type == JobType.ANALYZE and j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            for j in self._queries.jobs(project_id)
        ):
            ctx.report("waiting", 0.0, "Waiting for the highlight analysis of this video")
            time.sleep(3)

    def _speech(self, ctx: JobContext, asset: MediaAsset) -> list[dict[str, Any]]:
        info = asset.media_info
        assert info is not None
        manifest = StageManifest(
            self._storage.artifact_path(asset.project_id, "analysis", "manifest.json")
        )
        audio = self._extract.run(asset, manifest, self._progress(ctx, "extract_audio"))
        voice = choose_voice_track(info.audio)
        if voice is None or voice.index not in audio.files:
            return []
        t = self._transcribe.run(
            asset,
            audio.files[voice.index],
            voice.index,
            None,
            manifest,
            self._progress(ctx, "transcribe"),
        )
        return [{"start": s.start_s, "end": s.end_s, "text": s.text} for s in t.segments]

    def _frames(
        self, ctx: JobContext, asset: MediaAsset, game: str | None, segs: list[dict[str, Any]]
    ) -> dict[float, dict[str, Any]]:
        assert asset.media_info is not None and self._llm is not None
        fp, ver = asset.fingerprint, L.INDEX_VERSION
        times = L.frame_times(asset.media_info.duration_s)
        done = self._repo.frames(asset.id, fp, ver)
        todo = [t for t in times if round(t, 2) not in done]
        progress = self._progress(ctx, "frames")
        failed = 0
        for i in range(0, len(todo), L.FRAMES_PER_CALL):
            batch = todo[i : i + L.FRAMES_PER_CALL]
            progress(
                (len(times) - len(todo) + i) / max(1, len(times)),
                f"Looking at {L._mmss(batch[0])} of {L._mmss(times[-1])}",
            )
            tags = self._tag(asset, game, batch, segs)
            failed += all(f.get("missing") for f in tags.values())
            self._repo.add_frames(asset.id, fp, ver, tags)
            if failed >= 5 and failed * L.FRAMES_PER_CALL > len(todo) // 2:
                raise SphuraError("The AI couldn't read most frames. Is Ollama running?")
        return self._repo.frames(asset.id, fp, ver)

    def _tag(
        self, asset: MediaAsset, game: str | None, batch: list[float], segs: list[dict[str, Any]]
    ) -> dict[float, dict[str, Any]]:
        assert self._llm is not None
        try:
            jpegs = self._sample(asset.source_uri, batch, L.FRAME_WIDTH)
        except Exception as e:
            log.warning("lore: couldn't grab frames at %s: %s", batch[0], e)
            return L.parse_frames({}, batch)
        lines = [
            f'{L._mmss(s["start"])} "{s["text"].strip()}"'
            for s in segs
            if batch[0] - 6 <= s["start"] <= batch[-1] + 6
        ][:14]
        req = L.FRAME_SYSTEM, L.frame_prompt(game, batch, lines)
        images = [
            ImageInput(t, j, f"Frame {k + 1} at {L._mmss(t)}")
            for k, (t, j) in enumerate(zip(batch, jpegs, strict=False))
        ]
        for attempt in range(2):
            try:
                resp = self._llm.complete(LLMRequest(req[0], req[1], images, L.FRAME_SCHEMA, 1200))
                return L.parse_frames(_json(resp.text), batch)
            except Exception as e:  # one retry, then the frames are marked unread
                log.warning("lore: frame tagging failed (try %d): %s", attempt + 1, e)
        return L.parse_frames({}, batch)

    def _moments(self, project_id: str) -> list[dict[str, Any]]:
        try:
            found = self._queries.candidates(project_id)
        except Exception:
            return []
        return [
            {
                "id": c.id,
                "start_s": c.range.start_s,
                "end_s": c.range.end_s,
                "title": c.judgement.title if c.judgement else "",
            }
            for c in found
        ]

    @staticmethod
    def _progress(ctx: JobContext, stage: str) -> Any:
        base = 0.0
        for name, w in _WEIGHTS.items():
            if name == stage:
                break
            base += w
        weight = _WEIGHTS[stage]
        return lambda frac, msg=None: ctx.report(stage, base + weight * min(max(frac, 0), 1), msg)


# ---------- the studio ----------


class LoreService:
    def __init__(
        self,
        repo: SqliteLoreRepository,
        queries: Any,
        assets: MediaAssetRepository,
        queue: JobQueue,
        downloads: Any,
        research: Any,
        llm: LLMProvider | None,
        sample: Any,
        data_dir: Path,
        wiki: WikiClient | None = None,
    ) -> None:
        self._repo = repo
        self._queries = queries
        self._assets = assets
        self._queue = queue
        self._downloads = downloads
        self._research = research
        self._llm = llm
        self._sample = sample
        self._wiki = wiki or WikiClient()
        self._thumbs = data_dir / "lore" / "thumbs"
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}  # game -> discovery run

    # ---------- library ----------

    def library(self) -> list[dict[str, Any]]:
        """Every video SPHURA has (streams and finished downloads), with its index status."""
        states = self._repo.index_states()
        out: list[dict[str, Any]] = []
        linked: set[str] = set()
        downloads = {d.project_id: d for d in self._downloads.completed() if d.project_id}
        for p in self._queries.list_projects(500):
            if p.name.startswith("Reel · "):  # clips imported into a highlight reel
                continue
            assets = self._assets.list_for_project(p.id)
            if not assets or assets[0].media_info is None:
                continue
            a = assets[0]
            d = downloads.get(p.id)
            if d:
                linked.add(d.id)
            out.append(self._item(p.name, p.id, a, states.get(a.id), d.id if d else None))
        for d in self._downloads.completed():
            if d.id in linked or d.project_id:
                continue
            out.append(
                {
                    "key": f"d:{d.id}",
                    "download_id": d.id,
                    "project_id": None,
                    "asset_id": None,
                    "title": d.title,
                    "game": L.game_of(d.title),
                    "duration_s": d.duration_s,
                    "thumb": f"/v1/downloads/{d.id}/thumb.jpg",
                    "status": "not_indexed",
                }
            )
        return out

    def _item(
        self,
        name: str,
        project_id: str,
        a: MediaAsset,
        state: dict[str, Any] | None,
        download_id: str | None,
    ) -> dict[str, Any]:
        assert a.media_info is not None
        game = L.game_of(name)
        item: dict[str, Any] = {
            "key": f"p:{project_id}",
            "project_id": project_id,
            "asset_id": a.id,
            "download_id": download_id,
            "title": name,
            "game": game,
            "duration_s": round(a.media_info.duration_s, 1),
            "thumb": f"/v1/downloads/{download_id}/thumb.jpg"
            if download_id
            else f"/v1/lore/frame.jpg?asset={a.id}&t={round(a.media_info.duration_s * 0.3)}",
            "status": "not_indexed",
        }
        job = self._latest_job(project_id)
        current = (
            state
            and state["status"] == "done"
            and state["fingerprint"] == a.fingerprint
            and state["version"] == L.INDEX_VERSION
        )
        if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            item.update(
                status="queued" if job.status == JobStatus.QUEUED else "indexing",
                progress=round(job.progress, 3),
                stage=job.stage,
                job_id=job.id,
            )
        elif current:
            item["status"] = "indexed"
        elif job and job.status == JobStatus.FAILED:
            item.update(status="failed", error=job.error)
        elif job and job.status == JobStatus.CANCELLED:
            item["status"] = "paused"
        elif state and state["status"] == "done":
            item["status"] = "stale"  # file changed or a newer index version
        if state:
            item["summary"] = state["data"]
            item["indexed_at"] = state["updated_at"]
        return item

    def _latest_job(self, project_id: str) -> Job | None:
        try:
            jobs = [j for j in self._queries.jobs(project_id) if j.type == JobType.LORE_INDEX]
        except NotFoundError:
            return None
        return max(jobs, key=lambda j: j.created_at, default=None)

    def index(self, keys: list[str]) -> list[dict[str, Any]]:
        """Queue indexing for each video that isn't indexed yet; done ones are reused."""
        if self._llm is None:
            raise SphuraError("Lore indexing needs the local AI: start Ollama, then restart SPHURA")
        started = []
        for key in keys:
            kind, _, ident = key.partition(":")
            project_id = self._downloads.link(ident) if kind == "d" else ident
            assets = self._assets.list_for_project(project_id)
            if not assets:
                continue
            a = assets[0]
            state = self._repo.index_state(a.id)
            if (
                state
                and state["status"] == "done"
                and state["fingerprint"] == a.fingerprint
                and state["version"] == L.INDEX_VERSION
            ):
                continue
            job = self._latest_job(project_id)
            if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                continue
            name = self._queries.project(project_id).name
            self._queue.enqueue(
                Job(
                    type=JobType.LORE_INDEX,
                    project_id=project_id,
                    payload={"asset_id": a.id, "game": L.game_of(name)},
                )
            )
            started.append(project_id)
        return [{"project_id": p} for p in started]

    def cancel(self, project_id: str) -> None:
        job = self._latest_job(project_id)
        if job and not job.status.is_terminal:
            self._queue.cancel(job.id)

    def scenes(self, asset_id: str) -> list[dict[str, Any]]:
        return self._repo.scenes(asset_id=asset_id)

    def frame(self, asset_id: str, t: float) -> Path:
        path = self._thumbs / f"{asset_id}_{t:.1f}.jpg"
        if not path.is_file():
            a = self._assets.get(asset_id)
            if a is None:
                raise NotFoundError("video not found")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._sample(a.source_uri, [max(0.0, t)], 320)[0])
        return path

    def source(self, asset_id: str) -> Path:
        a = self._assets.get(asset_id)
        if a is None or not Path(a.source_uri).is_file():
            raise NotFoundError("the recording was moved or deleted")
        return Path(a.source_uri)

    # ---------- overview ----------

    def overview(self) -> dict[str, Any]:
        lib = self.library()
        games = []
        for key, g in L.GAMES.items():
            vids = [v for v in lib if v["game"] == key]
            if not vids:
                continue
            ideas = self._repo.ideas(key)
            games.append(
                {
                    "key": key,
                    "name": g["name"],
                    "videos": len(vids),
                    "indexed": sum(v["status"] == "indexed" for v in vids),
                    "scenes": sum(
                        (v.get("summary") or {}).get("scenes", 0)
                        for v in vids
                        if v["status"] == "indexed"
                    ),
                    "ideas": len(ideas),
                }
            )
        other = [v for v in lib if v["game"] is None]
        return {
            "games": games,
            "other_videos": len(other),
            "ai": None if self._llm else "Start Ollama, then restart SPHURA, to index footage",
            "ai_local": bool(self._llm and self._llm.info.runs_locally),
        }

    # ---------- lore sources ----------

    def sources(self, game: str) -> dict[str, Any]:
        pages = self._repo.pages(game)
        kinds = Counter(p["kind"] for p in self._repo.passages(game))
        per: dict[tuple[str, str], int] = Counter(
            (p["entity"], p["source"]) for p in self._repo.passages(game)
        )
        return {
            "passages": sum(kinds.values()),
            "kinds": dict(kinds),
            "pages": [
                {
                    "entity": p["entity"],
                    "source": p["source"],
                    "status": p["status"],
                    "fetched_at": p["fetched_at"],
                    "passages": per.get((p["entity"], p["source"]), 0),
                    **p["data"],
                }
                for p in pages
            ],
        }

    def _research_entity(self, game: str, entity: str, source: str, force: bool = False) -> None:
        """Fetch one wiki page into cited passages; cached for 30 days, misses too."""
        key = L.plain(entity)
        old = self._repo.page(game, key, source)
        fresh = old and datetime.fromisoformat(old["fetched_at"]) > datetime.now(UTC) - PAGE_MAX_AGE
        cfg = L.GAMES[game]
        ctx = f"{cfg['name']} {cfg.get('myth', '')}"
        wrong = old and not _fits_context(str(old["data"].get("page") or ""), ctx)
        if old and fresh and not force and old["status"] != "failed" and not wrong:
            return
        if source == "wiki":
            if not cfg["wiki"]:
                return
            api, site, lic = cfg["wiki"]
            query = entity
        elif source == "myth":
            if not cfg["myth"]:
                return
            api, site, lic = L.WIKIPEDIA
            query = f"{entity} {cfg['myth']}"
        else:  # the game's own article: development, reception, publicly documented facts
            api, site, lic = L.WIKIPEDIA
            query = cfg["name"] + " video game"
        try:
            titles = self._wiki.search(api, query, 6)
            title = self._best_title(
                titles, entity if source != "game" else cfg["name"], source, ctx
            )
            if title is None:
                self._repo.save_page(
                    game,
                    key,
                    source,
                    "missing",
                    {"name": entity, "query": query, "site": site},
                    [],
                    _now(),
                )
                return
            resolved, url, secs = self._wiki.page(api, title)
            meta = {"page": resolved, "url": url, "site": site, "license": lic}
            ps = L.passages_from(secs, game, key, source, meta)[:220]
            topic = cfg["myth"].split()[0].lower() if source == "myth" else ""
            if topic and not any(topic in p["text"].lower() for p in ps[:6]):
                # e.g. Wikipedia's "Atreus" is the Greek king, not the game's Norse story
                self._repo.save_page(
                    game,
                    key,
                    source,
                    "missing",
                    {"name": entity, "query": query, "site": site, "rejected": resolved},
                    [],
                    _now(),
                )
                return
            self._repo.save_page(
                game, key, source, "ok", {**meta, "name": entity, "query": query}, ps, _now()
            )
        except WikiError as e:
            log.warning("lore research: %s / %s: %s", entity, source, e)
            self._repo.save_page(
                game,
                key,
                source,
                "failed",
                {"name": entity, "query": query, "site": site, "error": str(e)},
                [],
                _now(),
            )

    @staticmethod
    def _best_title(titles: list[str], entity: str, source: str, context: str = "") -> str | None:
        """Exact title first. A bracketed one ("Kratos (Soulcalibur)") only when the bracket
        is about this game or its myths: crossover pages describe another game's version."""
        want = L.plain(entity)
        ok = [
            t
            for t in titles
            if (source != "myth" or not _NOT_MYTH.search(t)) and _fits_context(t, context)
        ]
        for t in ok:
            if L.plain(t) == want:
                return t
        for t in ok:
            if L.plain(re.sub(r"\(.*?\)", "", t)) == want:
                return t
        for t in ok:
            if L.plain(t).startswith(want) or want in L.plain(t).split():
                return t
        return None

    # ---------- discovery ----------

    def run(self, game: str) -> dict[str, Any] | None:
        return self._runs.get(game)

    def discover(self, game: str, asset_ids: list[str] | None = None) -> dict[str, Any]:
        if game not in L.GAMES:
            raise SphuraError("Lore Studio doesn't know this game yet")
        if self._llm is None:
            raise SphuraError("Discovery needs the local AI: start Ollama, then restart SPHURA")
        with self._lock:
            cur = self._runs.get(game)
            if cur and cur["status"] == "running":
                return cur
            scenes = [
                s
                for s in self._repo.scenes(game=game)
                if not asset_ids or s["asset_id"] in asset_ids
            ]
            if not scenes:
                raise SphuraError("Index some footage of this game first")
            run = {
                "status": "running",
                "step": "Starting",
                "progress": 0.0,
                "error": None,
                "started_at": _now(),
                "ideas": 0,
            }
            self._runs[game] = run
        threading.Thread(
            target=self._discover,
            args=(game, scenes, run),
            daemon=True,
            name=f"lore-discover-{game}",
        ).start()
        return run

    def _discover(self, game: str, scenes: list[dict[str, Any]], run: dict[str, Any]) -> None:
        try:
            ideas = self._discover_ideas(game, scenes, run)
            self._repo.replace_ideas(game, ideas, _now())
            run.update(status="done", step="Done", progress=1.0, ideas=len(ideas))
        except Exception as e:
            log.exception("lore discovery failed")
            run.update(status="failed", error=str(e) or type(e).__name__)

    def _discover_ideas(
        self, game: str, scenes: list[dict[str, Any]], run: dict[str, Any]
    ) -> list[dict[str, Any]]:
        stats = L.entity_stats(scenes)
        chars = sorted(
            (n for n, e in stats.items() if e["kind"] == "character" and e["scenes"] >= 2),
            key=lambda n: -stats[n]["scenes"],
        )[:8]
        places = sorted(
            (n for n, e in stats.items() if e["kind"] == "place" and e["scenes"] >= 3),
            key=lambda n: -stats[n]["scenes"],
        )[:3]
        # Research: every subject's wiki page and real-world myth, plus the game's own article
        jobs = [(e, s) for e in chars for s in ("wiki", "myth")] + [(p, "wiki") for p in places]
        jobs.append((L.GAMES[game]["name"], "game"))
        for i, (entity, source) in enumerate(jobs):
            run.update(step=f"Researching {entity}", progress=0.35 * i / len(jobs))
            self._research_entity(game, entity if source != "game" else "game", source)
        # Subjects: main characters, pairs that share scenes, then places
        together: Counter[tuple[str, str]] = Counter()
        for s in scenes:
            for a, b in combinations(sorted(set(s["characters"]) & set(chars)), 2):
                together[(a, b)] += 1
        subjects: list[list[str]] = [[c] for c in chars[:6]]
        subjects += [list(p) for p, n in together.most_common(3) if n >= 2]
        subjects += [[p] for p in places[:1]]
        subjects = subjects[:MAX_SUBJECTS]
        game_passages = self._repo.passages(game, "game")
        used = self._used_ranges(scenes)
        vids = self._research_videos(game)
        ideas: dict[str, dict[str, Any]] = {}
        for i, subject in enumerate(subjects):
            run.update(
                step=f"Finding stories about {' & '.join(subject)}",
                progress=0.35 + 0.63 * i / len(subjects),
            )
            for idea in self._ideas_for(game, subject, scenes, game_passages):
                idea["fresh"] = self._fresh(idea, used)
                a = appeal(subject, [], vids) if vids else {"lift": 0.0, "n": 0, "basis": ""}
                idea["appeal"] = a
                idea["score"] = round(
                    0.6 * idea["confidence"]
                    + 0.25 * min(max(float(a["lift"]), -1.0), 1.5) / 1.5
                    + 0.15 * idea["fresh"],
                    4,
                )
                old = ideas.get(idea["id"])
                if old is None or old["score"] < idea["score"]:
                    ideas[idea["id"]] = idea
        return sorted(ideas.values(), key=lambda x: -x["score"])

    def _ideas_for(
        self,
        game: str,
        subject: list[str],
        scenes: list[dict[str, Any]],
        game_passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        assert self._llm is not None
        want = {L.plain(n) for n in subject}

        def has(s: dict[str, Any]) -> bool:
            return bool(want & ({L.plain(c) for c in s["characters"]} | {L.plain(s["location"])}))

        mine = [s for s in scenes if has(s)]
        if not mine:
            return []
        # Cinematics and spoken scenes first, spread across videos and the timeline
        mine.sort(key=lambda s: (s["kind"] not in ("cinematic", "dialogue"), -len(s["dialogue"])))
        picked: list[dict[str, Any]] = []
        per_video: Counter[str] = Counter()
        for s in mine:
            if per_video[s["asset_id"]] < 6:
                picked.append(s)
                per_video[s["asset_id"]] += 1
            if len(picked) >= 12:
                break
        words = {w for s in picked for d in s["dialogue"] for w in L.plain(d["text"]).split()}
        pool: list[dict[str, Any]] = []
        for n in subject:
            pool += L.pick_passages(
                self._repo.passages(game, L.plain(n)), words, 14 if len(subject) == 1 else 8
            )
        pool += L.pick_passages(game_passages, words, 3)
        pool = list({p["id"]: p for p in pool}.values())
        if not pool:
            return []  # nothing to ground a story in: no ideas rather than made-up ones
        names = {s["id"]: f"F{i + 1}" for i, s in enumerate(picked)}
        names |= {p["id"]: f"S{i + 1}" for i, p in enumerate(pool)}
        back = {v: k for k, v in names.items()}
        prompt = L.idea_prompt(game, subject, picked, pool, names)
        try:
            resp = self._llm.complete(LLMRequest(L.IDEA_SYSTEM, prompt, [], L.IDEA_SCHEMA, 2600))
            raw = _json(resp.text).get("ideas") or []
        except Exception as e:
            log.warning("lore: ideas for %s failed: %s", subject, e)
            return []
        by_scene = {s["id"]: s for s in scenes}
        by_passage = {p["id"]: p for p in pool}
        out = []
        for r in raw[:3]:
            idea = L.validate_idea(r, back, by_scene, by_passage, subject)
            if idea and idea["title"]:
                idea["id"] = L.idea_id(game, idea["title"])
                idea["game"] = game
                out.append(idea)
        return out

    def _used_ranges(self, scenes: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
        """Footage already cut into Shorts, per video, so fresh scenes can win."""
        used: dict[str, list[tuple[float, float]]] = {}
        for pid in {s["project_id"] for s in scenes}:
            try:
                for clip, _ in self._queries.clips(pid):
                    used.setdefault(clip.asset_id, []).extend(
                        (r.start_s, r.end_s) for r in clip.segments
                    )
            except Exception:
                continue
        return used

    def _fresh(self, idea: dict[str, Any], used: dict[str, list[tuple[float, float]]]) -> float:
        sc = self._repo.scenes_by_id(idea["scenes"])
        if not sc:
            return 0.0
        fresh = [
            s
            for s in sc.values()
            if not any(a < s["end_s"] and b > s["start_s"] for a, b in used.get(s["asset_id"], []))
        ]
        return round(len(fresh) / len(sc), 2)

    def _research_videos(self, game: str) -> list[dict[str, Any]]:
        try:
            return list(self._research.videos(game))
        except Exception:
            return []

    def ideas(self, game: str) -> dict[str, Any]:
        """Ideas with the scenes and passages they rest on, ready to show as evidence."""
        ideas = self._repo.ideas(game)
        scenes = self._repo.scenes_by_id(list({s for i in ideas for s in i["scenes"]}))
        passages = self._repo.passages_by_id(list({p for i in ideas for p in i["passages"]}))
        titles = {
            a.id: self._queries.project(a.project_id).name
            for a in (self._assets.get(s["asset_id"]) for s in scenes.values())
            if a
        }
        return {
            "ideas": ideas,
            "scenes": {k: {**v, "video": titles.get(v["asset_id"], "")} for k, v in scenes.items()},
            "passages": passages,
            "run": self._runs.get(game),
            "sources": {k: v for k, v in self.sources(game).items() if k != "pages"},
        }


def _wait_for_file(ctx: JobContext, path: Path, wait_s: float = 60) -> None:
    """External drives can drop out for a moment (sleep, loose cable): wait a little
    before failing, and say clearly what's missing instead of failing later in ffmpeg."""
    end = time.monotonic() + wait_s
    while not path.is_file():
        if time.monotonic() > end:
            drive = Path(path.anchor) if path.anchor else None
            if drive and not drive.exists():
                raise SphuraError(
                    f"Drive {path.anchor} isn't connected. Plug it in and index again."
                )
            raise SphuraError(f"The video isn't at {path} any more. It was moved or deleted; "
                              "add it to SPHURA again from its new place.")  # fmt: skip
        ctx.report("waiting", 0.0, f"Waiting for {path.anchor or 'the drive'} to respond")
        time.sleep(3)


def _fits_context(title: str, context: str) -> bool:
    """A bracketed title must name this game or its myths: "Kratos (Soulcalibur)" doesn't."""
    q = re.findall(r"\((.*?)\)", title)
    if not q:
        return True
    good = {w for w in L.plain(f"{context} norse mythology character").split() if len(w) > 2}
    return any(w in good for w in L.plain(" ".join(q)).split())
