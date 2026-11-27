"""Voice-over for Shorts: scripts grounded in the fact bank, written for a moment before its
Short exists (so the user approves them first) or for a finished Short, then spoken by
the local narrator voice and laid into the gaps between the characters' lines."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sphura.application.queries import QueryService
from sphura.application.voice_intel import VoiceIntelService
from sphura.domain.errors import NotFoundError, SphuraError
from sphura.domain.models import Clip
from sphura.infrastructure.ai.qwen_voice import DESIGNS, QwenVoice
from sphura.infrastructure.ai.voice import DEFAULT_VOICE, VOICES, KokoroVoice
from sphura.pipeline import narration as N
from sphura.pipeline import voice_audio as VA
from sphura.pipeline import voice_intel as VI
from sphura.pipeline.trending import detect_game
from sphura.ports.ai import ImageInput, LLMProvider
from sphura.ports.repositories import ClipRepository
from sphura.ports.storage import StorageProvider

log = logging.getLogger(__name__)


class VoiceoverService:
    def __init__(
        self,
        clips: ClipRepository,
        queries: QueryService,
        storage: StorageProvider,
        llm: LLMProvider | None,
        voice: KokoroVoice,
        qwen: QwenVoice,
        cache: Path,
        frames: Callable[[str, float], list[ImageInput]] | None = None,
        intel: VoiceIntelService | None = None,
        tempo: Callable[[Path, float], None] | None = None,
        polish: Callable[[Path, str], None] | None = None,
    ) -> None:
        self._clips = clips
        self._queries = queries
        self._storage = storage
        self._llm = llm
        self._engines: dict[str, Any] = {"kokoro": voice, "qwen": qwen}
        self._frames = frames
        self._intel = intel
        self._tempo = tempo
        self._polish = polish
        self._cache = cache  # (voice, text) -> wav, shared by previews and Shorts
        self._warm_lock = threading.Lock()
        self._warm_queue: list[tuple[str, str, dict[str, Any]]] = []  # (voice, key, plan)

    def status(self) -> dict[str, Any]:
        engines = {k: {"ready": e.ready(), "setup": e.setup} for k, e in self._engines.items()}
        return {
            "engines": engines,
            "voices": [
                {
                    "id": k,
                    "label": v[2],
                    "lang": v[3],
                    "engine": v[4],
                    "ready": engines[v[4]]["ready"],
                }
                for k, v in VOICES.items()
            ],
            "default": "qwen_gamer" if engines["qwen"]["ready"] else DEFAULT_VOICE,
            "ai": self._llm.info.name if self._llm else None,
        }

    def install(self, engine: str) -> dict[str, Any]:
        if engine not in self._engines:
            raise SphuraError("unknown voice engine")
        result: dict[str, Any] = self._engines[engine].install()
        return result

    def plan(self, text: str, voice: str, game: str | None = None,
             direction: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None,
             respell: dict[str, str] | None = None,
             prev_group: str | None = None) -> dict[str, Any]:  # fmt: skip
        """What the voice will actually read for this line (dictionary, pauses, mood)."""
        voice = voice if voice in VOICES else DEFAULT_VOICE
        engine = VOICES[voice][4]
        if self._intel is None:
            return {"text": text, "tts": text,
                    "segments": [{"text": text, "pause_after_ms": 0, "speed": 1.0}],
                    "pause_before_ms": 0, "lexicon": {}, "changes": [], "speed": 1.0,
                    "engine": engine, "reference": None, "design": None,
                    "key": hashlib.sha1(f"{voice}|{text.strip()}".encode()).hexdigest()[:20],
                    }  # fmt: skip
        return self._intel.plan(text, voice, engine, _lang(voice), game, direction, overrides,
                                respell, prev_group)  # fmt: skip

    def cached(self, text: str, voice: str, game: str | None = None,
               direction: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None,
               respell: dict[str, str] | None = None,
               prev_group: str | None = None) -> Path:  # fmt: skip
        """The line's audio, from the cache or spoken now (and kept)."""
        voice = voice if voice in VOICES else DEFAULT_VOICE
        plan = self.plan(text, voice, game, direction, overrides, respell, prev_group)
        return self._spoken(plan, voice)

    def _spoken(self, plan: dict[str, Any], voice: str) -> Path:
        path = self._cache_path(plan["key"], voice)
        for _ in range(600):  # being warmed already: wait for it rather than speak it twice
            with self._warm_lock:
                pending = any(k == plan["key"] for _, k, _ in self._warm_queue)
            if not pending or path.is_file():
                break
            time.sleep(0.2)
        if not path.is_file():
            self._render([plan], voice, [path])
        return path

    def warm(self, texts: list[Any], voice: str, game: str | None = None) -> int:
        """Speaks lines in the background so pressing play later is instant.
        Each item is a line's text or {text, direction}."""
        if voice not in VOICES:
            raise SphuraError("unknown voice")
        plans = []
        for t in texts:
            text, d = (t.get("text", ""), t.get("direction")) if isinstance(t, dict) else (t, None)
            if str(text).strip():
                plans.append(self.plan(str(text).strip(), voice, game, d))
        todo = [p for p in {p["key"]: p for p in plans}.values()
                if not self._cache_path(p["key"], voice).is_file()]  # fmt: skip
        with self._warm_lock:
            start = not self._warm_queue
            queued = {k for _, k, _ in self._warm_queue}
            self._warm_queue += [(voice, p["key"], p) for p in todo if p["key"] not in queued]
        if start and todo:
            threading.Thread(target=self._warm_run, daemon=True, name="voice-warm").start()
        return len(todo)

    def _warm_run(self) -> None:
        while True:
            with self._warm_lock:
                if not self._warm_queue:
                    return
                voice = self._warm_queue[0][0]
                batch = [(k, p) for v, k, p in self._warm_queue if v == voice][:6]
            try:
                todo = [p for k, p in batch if not self._cache_path(k, voice).is_file()]
                if todo and self._engines[VOICES[voice][4]].ready():
                    self._render(todo, voice, [self._cache_path(p["key"], voice) for p in todo])
            except Exception:
                log.exception("voice warm-up failed")
            keys = {k for k, _ in batch}
            with self._warm_lock:
                self._warm_queue = [x for x in self._warm_queue
                                    if not (x[0] == voice and x[1] in keys)]  # fmt: skip

    def _render(self, plans: list[dict[str, Any]], voice: str, outs: list[Path]) -> None:
        """Says each plan's pieces, joins them with the planned pauses, sets the speed."""
        engine_id = VOICES[voice][4]
        engine = self._engines[engine_id]
        if not engine.ready():
            raise SphuraError("This voice isn't installed yet")
        work = self._cache / "_parts"
        pieces: list[list[Path]] = []
        for p in plans:
            pieces.append([work / f"{p['key']}_{i}.wav" for i in range(len(p["segments"]))])
        if engine_id == "qwen":
            groups: dict[str, list[tuple[str, Path, tuple[str, str]]]] = {}
            for p, paths in zip(plans, pieces, strict=True):
                for seg, path in zip(p["segments"], paths, strict=True):
                    ref_id, design = _reference(voice, p, seg)
                    groups.setdefault(ref_id, []).append((seg["text"], path, design))
            for ref_id, items in groups.items():  # one GPU batch per reference voice
                instruct, ref_text = items[0][2]
                engine.speak_ref([t for t, _, _ in items], [x for _, x, _ in items], ref_id,
                                 instruct, ref_text)  # fmt: skip
            for p, paths in zip(plans, pieces, strict=True):
                for seg, path in zip(p["segments"], paths, strict=True):
                    speed = float(seg.get("speed", p["speed"]))
                    if abs(speed - 1.0) > 0.02 and self._tempo:
                        self._tempo(path, speed)  # Qwen has no speed control: stretch the audio
        else:
            for p, paths in zip(plans, pieces, strict=True):
                for seg, path in zip(p["segments"], paths, strict=True):
                    speed = float(seg.get("speed", p["speed"]))
                    engine.speak(seg["text"], voice, path, speed=speed, lexicon=p["lexicon"])
        for p, paths, out in zip(plans, pieces, outs, strict=True):
            parts = [_read(x) for x in paths]
            sr = parts[0][1]
            audio = [a for a, _ in parts]
            enhance = p.get("enhance", "off")
            if enhance != "off":  # stray long silences, matched levels, soft edges
                audio = VA.level_match([VA.cap_silences(a, sr) for a in audio], sr)
            gaps = [seg["pause_after_ms"] for seg in p["segments"]]
            track = [np.zeros(int(sr * p.get("pause_before_ms", 0) / 1000), np.float32)]
            spans, t = [], len(track[0]) / sr
            for a, gap, seg in zip(audio, gaps, p["segments"], strict=True):
                spans.append({"start": round(t, 3), "end": round(t + len(a) / sr, 3),
                              "text": seg["text"], "group": seg.get("group"),
                              "speed": seg.get("speed", 1.0)})  # fmt: skip
                track += [a, np.zeros(int(sr * gap / 1000), np.float32)]
                t += (len(a) + int(sr * gap / 1000)) / sr
            tmp = out.with_suffix(".part.wav")
            _write_mono(tmp, np.concatenate(track), sr)
            if enhance != "off" and self._polish:
                try:
                    self._polish(tmp, enhance)
                except Exception:
                    log.warning("voice polish failed; keeping the raw take", exc_info=True)
            tmp.replace(out)
            out.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "segments": spans,
                        "plan": {
                            k: p.get(k)
                            for k in ("text", "tts", "changes", "transitions", "personality")
                        },
                    }
                )
            )
            for x in paths:
                x.unlink(missing_ok=True)

    def _cache_path(self, key: str, voice: str) -> Path:
        return self._cache / voice / f"{key}.wav"

    def stop(self) -> None:
        self._engines["qwen"].stop()

    # --- scripts -----------------------------------------------------------------------

    def script_for_moment(
        self, project_id: str, candidate_id: str, game: str, voice: str, length_s: float,
        format_label: str = "",
    ) -> dict[str, Any]:  # fmt: skip
        r = self._moment(project_id, candidate_id)
        c = r.candidate
        lines = self._dialogue(project_id, c.range.start_s, c.range.end_s)
        seconds = min(length_s, c.range.duration_s)
        ctx = self._context(project_id, c)
        return N.write_script(
            self._llm, game, c.judgement, lines, _lang(voice), seconds, format_label, **ctx,
            voice_ctx=self._voice_ctx(game, voice, c.judgement, format_label, seconds, lines, ctx),
        )  # fmt: skip

    def script_for_clip(self, clip_id: str, voice: str) -> dict[str, Any]:
        clip = self._clips.get(clip_id)
        if clip is None:
            raise NotFoundError("Short not found")
        game = self.game_of(clip)
        if not game:
            raise SphuraError("Voice-over facts exist for God of War Ragnarök and Phantom "
                              "Blade Zero so far")  # fmt: skip
        r = self._moment(clip.project_id, clip.candidate_id or "")
        spoken: list[str] = []
        for seg in clip.segments:
            spoken += self._dialogue(clip.project_id, seg.start_s, seg.end_s)
        label = (clip.publish.get("trend") or {}).get("label", "")
        ctx = self._context(clip.project_id, r.candidate)
        return N.write_script(
            self._llm, game, r.candidate.judgement, spoken, _lang(voice),
            clip.output_duration_s, label, **ctx,
            voice_ctx=self._voice_ctx(game, voice, r.candidate.judgement, label,
                                      clip.output_duration_s, spoken, ctx),
        )  # fmt: skip

    def line_path(self, ln: dict[str, Any], voice: str, game: str | None) -> Path:
        """Where an applied line's audio is (it's in the cache under its plan key)."""
        if ln.get("key"):
            return self._cache_path(str(ln["key"]), voice if voice in VOICES else DEFAULT_VOICE)
        return self.cached(ln["text"], voice, game, ln.get("direction"))

    def regenerate_line(self, clip: Clip, index: int) -> dict[str, Any]:
        """A new take of one line only; every other line keeps its audio."""
        effect = next((e for e in clip.effects if e["type"] == "voiceover"), None)
        if not effect or not 0 <= index < len(effect["lines"]):
            raise NotFoundError("no such voice-over line")
        lines = [dict(x) for x in effect["lines"]]
        lines[index]["take"] = int(lines[index].get("take") or 0) + 1
        return self.apply(clip, effect["voice"], lines)

    def game_of(self, clip: Clip) -> str | None:
        game = (clip.publish.get("trend") or {}).get("game")
        if not game:
            p = self._optional(self._queries.project, clip.project_id)
            game = detect_game(p.name) if p else None
        return game or None

    def _voice_ctx(self, game: str, voice: str, j: Any, label: str, seconds: float,
                   lines: list[str], ctx: dict[str, Any]) -> dict[str, Any] | None:  # fmt: skip
        if self._intel is None:
            return None
        text = " ".join([*lines, *ctx.get("story", []), getattr(j, "what_happens", "") or "",
                         " ".join(getattr(j, "names", ()) or ())])  # fmt: skip
        return self._intel.voice_context(game, voice, VI.scene_type(j, label), seconds, text)

    def _context(self, project_id: str, c: Any) -> dict[str, Any]:
        """What the AI needs to get the moment right: the dialogue leading up to it,
        the video's name and three frames of the moment itself."""
        a = c.range.start_s
        frames: list[ImageInput] = []
        if self._frames is not None:
            try:
                frames = self._frames(c.asset_id, c.peak_s)
            except Exception:
                log.warning("couldn't grab frames for the voice-over script", exc_info=True)
        project = self._optional(self._queries.project, project_id)
        return {
            "story": self._dialogue(project_id, max(0.0, a - 90), a),
            "video": project.name if project else "",
            "frames": frames,
        }

    # --- voicing -----------------------------------------------------------------------

    def apply(self, clip: Clip, voice: str, lines: list[dict[str, Any]]) -> dict[str, Any]:
        """Speaks the approved lines, places them and stores the track on the clip.
        Returns the effect (with lines that didn't fit marked as dropped)."""
        if voice not in VOICES:
            raise SphuraError("unknown voice")
        lines = [N.line(str(x.get("kind", "reaction")), str(x.get("text", "")).strip(),
                        str(x.get("fact_id", "")), str(x.get("source", "")),
                        x.get("direction") if isinstance(x.get("direction"), dict) else None)
                 | ({"take": int(x["take"])} if x.get("take") else {})
                 for x in lines if str(x.get("text", "")).strip()]  # fmt: skip
        game = self.game_of(clip)
        clip.effects = [e for e in clip.effects if e["type"] != "voiceover"]
        if not lines:
            self._clips.save(clip)
            return {}
        # One narration: each line starts in the voice the one before ended on, so moods
        # only switch reference when they really change (see voice_intel.smooth_moods)
        plans, prev = [], None
        for ln in lines:
            take = {"take": ln["take"]} if ln.get("take") else None
            pl = self.plan(ln["text"], voice, game, ln.get("direction"), take, None, prev)
            prev = pl.get("last_group")
            plans.append(pl)
            ln["key"] = pl["key"]
        audio = [_read(self._spoken(pl, voice)) for pl in plans]
        total = clip.output_duration_s
        transcript = self._optional(self._queries.transcript, clip.project_id)
        spans = N.speech_spans(clip, transcript)
        starts = N.place([len(a) / sr for a, sr in audio], spans, total)
        placed, parts = [], []
        for ln, (a, sr), t in zip(lines, audio, starts, strict=True):
            if t is None:
                placed.append({**ln, "dropped": True})
                continue
            parts.append((t, a, sr))
            placed.append({**ln, "start": t, "end": round(t + len(a) / sr, 2)})
        track = N.mix_track(parts, total)
        out = self._storage.artifact_path(clip.project_id, "voice", f"{clip.id}.wav")
        _write(out, track, 48000)
        effect = {"type": "voiceover", "path": str(out), "voice": voice, "lines": placed,
                  "version": 1}  # fmt: skip
        clip.effects.append(effect)
        self._clips.save(clip)
        return effect

    def remove(self, clip: Clip) -> None:
        clip.effects = [e for e in clip.effects if e["type"] != "voiceover"]
        self._clips.save(clip)

    def get(self, clip_id: str) -> Clip:
        clip = self._clips.get(clip_id)
        if clip is None:
            raise NotFoundError("Short not found")
        return clip

    # --- helpers -----------------------------------------------------------------------

    def _moment(self, project_id: str, candidate_id: str) -> Any:
        r = next((h for h in self._queries.highlights(project_id, None)
                  if h.candidate.id == candidate_id), None)  # fmt: skip
        if r is None:
            raise NotFoundError("moment not found")
        return r

    def _dialogue(self, project_id: str, a: float, b: float) -> list[str]:
        t = self._optional(self._queries.transcript, project_id)
        if t is None:
            return []
        return [s.text.strip() for s in t.segments if s.end_s > a and s.start_s < b
                and s.text.strip()]  # fmt: skip

    @staticmethod
    def _optional(fn: Any, project_id: str) -> Any:
        try:
            return fn(project_id)
        except Exception:
            return None


def _lang(voice: str) -> str:
    return VOICES.get(voice, VOICES[DEFAULT_VOICE])[3]


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return a, sr


def _write(path: Path, stereo: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(stereo, -1, 1) * 32767).astype(np.int16).tobytes())


def _write_mono(path: Path, a: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def _reference(
    voice: str, plan: dict[str, Any], seg: dict[str, Any]
) -> tuple[str, tuple[str, str]]:
    """Which designed reference a Qwen piece is cloned from. The two original voices keep
    their recordings for the neutral mood, so nothing users already liked changes."""
    group = seg.get("reference") or "neutral"
    pid = plan.get("personality") or VI.VOICE_PERSONALITY.get(voice, "hype")
    legacy = {v: k for k, v in VI.VOICE_PERSONALITY.items()}.get(pid)
    if group == "neutral" and legacy:
        return legacy, DESIGNS[legacy]
    design = seg.get("design") or VI.design_for(VI.personality(pid), "neutral")
    tag = hashlib.sha1(design[0].encode()).hexdigest()[:6]  # a new description, a new voice
    return f"p_{pid}_{group}_{tag}", (design[0], design[1])
