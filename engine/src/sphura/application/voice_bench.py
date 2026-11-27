"""Runs the voice benchmark: every scene through every chosen engine, timed and measured
(loudness, clipping, speaker consistency, names heard back). Naturalness is the user's
rating while listening; SPHURA doesn't pretend to measure it."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sphura.application.voice_intel import VoiceIntelService
from sphura.application.voice_qa import VoiceQAService, read_wav
from sphura.application.voiceover import VoiceoverService
from sphura.domain.errors import NotFoundError, SphuraError
from sphura.infrastructure.ai.voice import VOICES
from sphura.pipeline import voice_audio as VA
from sphura.pipeline import voice_bench as B
from sphura.pipeline import voice_intel as V

log = logging.getLogger(__name__)

ENGINES = {
    "kokoro": {"label": "Kokoro (fast)", "voice": "en_michael", "engine": "kokoro"},
    "qwen_clone": {"label": "Qwen3-TTS Base clone (SPHURA default)", "voice": "qwen_story",
                   "engine": "qwen"},
    "qwen_design": {"label": "Qwen3-TTS VoiceDesign (direct)", "voice": "qwen_story",
                    "engine": "qwen"},
}  # fmt: skip


class VoiceBenchService:
    def __init__(self, voiceover: VoiceoverService, intel: VoiceIntelService, qa: VoiceQAService,
                 engines: dict[str, Any], root: Path) -> None:  # fmt: skip
        self._vo = voiceover
        self._intel = intel
        self._qa = qa
        self._engines = engines  # "kokoro" / "qwen" engine objects
        self._root = root
        self._run: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def available(self) -> dict[str, Any]:
        return {k: {**v, "ready": bool(self._engines[v["engine"]].ready())}
                for k, v in ENGINES.items()}  # fmt: skip

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for f in sorted(self._root.glob("*/results.json"), reverse=True)[:10]:
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        live = self._run
        if live and live["status"] == "running" and not any(r["id"] == live["id"] for r in out):
            out.insert(0, live)
        return sorted(out, key=lambda r: r["started_at"], reverse=True)

    def start(self, engines: list[str]) -> dict[str, Any]:
        engines = [e for e in engines if e in ENGINES]
        ready = self.available()
        missing = [ENGINES[e]["label"] for e in engines if not ready[e]["ready"]]
        if not engines:
            raise SphuraError("Pick at least one engine")
        if missing:
            raise SphuraError(f"Not installed: {', '.join(missing)}")
        with self._lock:
            if self._run and self._run["status"] == "running":
                return self._run
            rid = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
            now = datetime.now(UTC).isoformat()
            self._run = {"id": rid, "status": "running", "started_at": now, "engines": engines,
                         "step": "Starting", "progress": 0.0, "samples": [], "summary": {},
                         "notes": B.NOTES, "error": None}  # fmt: skip
        threading.Thread(
            target=self._work, args=(self._run,), daemon=True, name="voice-bench"
        ).start()
        return self._run

    def _work(self, run: dict[str, Any]) -> None:
        folder = self._root / run["id"]
        folder.mkdir(parents=True, exist_ok=True)
        try:
            total = len(run["engines"]) * len(B.SCENES)
            done = 0
            for eng in run["engines"]:
                for sc, out, secs in self._generate(eng, folder, run):
                    done += 1
                    run.update(
                        progress=round(done / total, 3),
                        step=f"{ENGINES[eng]['label']}: {sc['kind']}",
                    )
                    run["samples"].append(self._measure(eng, sc, out, secs, run["id"]))
            run["summary"] = summarise(run["samples"])
            run.update(status="done", step="Done", progress=1.0)
        except Exception as e:
            log.exception("voice benchmark failed")
            run.update(status="failed", error=str(e) or type(e).__name__)
        finally:
            run["finished_at"] = datetime.now(UTC).isoformat()
            (folder / "results.json").write_text(json.dumps(run, default=_plain), encoding="utf-8")

    def _generate(self, eng: str, folder: Path, run: dict[str, Any]) -> Any:
        cfg = ENGINES[eng]
        voice = cfg["voice"]
        if eng == "qwen_design":  # the design model says the lines itself, one batch per mood
            qwen = self._engines["qwen"]
            p = self._intel.personality_for(voice)
            groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
            for sc in B.SCENES:
                pl = self._vo.plan(sc["text"], voice, sc["game"], None, {"take": run["id"]})
                groups.setdefault(V.design_for(p, pl["mood"])[0], []).append((sc, pl))
            for instruct, items in groups.items():
                outs = [folder / f"{eng}_{sc['id']}.wav" for sc, _ in items]
                t = time.perf_counter()
                qwen.design_many([pl["tts"] for _, pl in items], outs, instruct)
                each = (time.perf_counter() - t) / len(items)  # includes loading the model
                for (sc, _), out in zip(items, outs, strict=True):
                    yield sc, out, each
            return
        for sc in B.SCENES:
            out = folder / f"{eng}_{sc['id']}.wav"
            t = time.perf_counter()
            path = self._vo.cached(sc["text"], voice, sc["game"], None, {"take": run["id"]})
            secs = time.perf_counter() - t
            out.write_bytes(path.read_bytes())
            yield sc, out, secs

    def _measure(
        self, eng: str, sc: dict[str, Any], out: Path, secs: float, rid: str
    ) -> dict[str, Any]:
        a, sr = read_wav(out)
        m = VA.measure(a, sr)
        voice = ENGINES[eng]["voice"]
        said = " ".join(t for t, _ in V.parse_chunks(sc["text"]))
        names = (
            self._qa._names(a, sr, said, sc["game"], voice) if VOICES[voice][3] == "en" else None
        )
        emb = self._qa._embed(a, sr)
        return {"engine": eng, "scene": sc["id"], "kind": sc["kind"], "game": sc["game"],
                "file": out.name, "latency_s": round(secs, 2), "audio_s": m["seconds"],
                "rtf": round(secs / max(m["seconds"], 0.01), 2), "lufs": m["lufs"],
                "peak_db": m["peak_db"], "clipping": any("clipping" in x for x in m["artifacts"]),
                "artifacts": m["artifacts"], "pitch_hz": m["pitch_hz"], "rate_sps": m["rate_sps"],
                "names": names, "embedding": emb, "rating": None}  # fmt: skip

    def rate(self, run_id: str, file: str, score: int) -> dict[str, Any]:
        path = self._root / run_id / "results.json"
        if not path.is_file():
            raise NotFoundError("benchmark run not found")
        run = json.loads(path.read_text(encoding="utf-8"))
        for s in run["samples"]:
            if s["file"] == file:
                s["rating"] = max(1, min(5, int(score)))
        run["summary"] = summarise(run["samples"])
        path.write_text(json.dumps(run), encoding="utf-8")
        return dict(run)

    def audio(self, run_id: str, file: str) -> Path:
        p = (self._root / run_id / file).resolve()
        if p.parent != (self._root / run_id).resolve() or not p.is_file():
            raise NotFoundError("no such sample")
        return p


def summarise(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Per engine: speed, names heard right, loudness spread, clipping, speaker consistency
    across its scenes, and the user's average rating."""
    out: dict[str, Any] = {}
    for eng in dict.fromkeys(s["engine"] for s in samples):
        ss = [s for s in samples if s["engine"] == eng]
        names = [n for s in ss for n in (s["names"] or [])]
        embs = [np.asarray(s["embedding"]) for s in ss if s.get("embedding") is not None]
        sims = [
            VA.cosine(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))
        ]
        loud = [s["lufs"] for s in ss if s["lufs"] > -60]
        ratings = [s["rating"] for s in ss if s.get("rating")]
        out[eng] = {
            "samples": len(ss),
            "latency_s": round(float(np.mean([s["latency_s"] for s in ss])), 2),
            "rtf": round(float(np.mean([s["rtf"] for s in ss])), 2),
            "names_ok": round(sum(n["status"] == "ok" for n in names) / len(names), 2)
            if names
            else None,
            "names_checked": len(names),
            "consistency": round(float(np.mean(sims)), 3) if sims else None,
            "loudness_spread_lu": round(max(loud) - min(loud), 2) if len(loud) > 1 else None,
            "clipping": sum(bool(s["clipping"]) for s in ss),
            "rating": round(float(np.mean(ratings)), 2) if ratings else None,
            "rated": len(ratings),
        }
    return out


def _plain(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
