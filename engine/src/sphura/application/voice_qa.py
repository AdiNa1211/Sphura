"""Voice quality checks after generation: measured audio (loudness, clipping, pauses,
pitch, timbre, speaker identity), names checked by listening back with speech recognition,
and estimates (mood switches, joins) kept apart from measurements. Nothing is rejected
or changed automatically: the report flags, the user decides."""

from __future__ import annotations

import json
import logging
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from sphura.application.voice_intel import VoiceIntelService
from sphura.application.voiceover import VoiceoverService
from sphura.domain.errors import NotFoundError, SphuraError
from sphura.infrastructure.ai.speaker import SpeakerEmbedder
from sphura.infrastructure.ai.voice import VOICES
from sphura.pipeline import voice_audio as VA
from sphura.pipeline import voice_intel as V

log = logging.getLogger(__name__)


def read_wav(path: Path) -> tuple[VA.F32, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return (a.reshape(-1, ch).mean(axis=1) if ch > 1 else a).astype(np.float32), sr


class VoiceQAService:
    def __init__(
        self,
        voiceover: VoiceoverService,
        intel: VoiceIntelService,
        asr: Any,
        speaker: SpeakerEmbedder,
        reports: Path,
    ) -> None:
        self._vo = voiceover
        self._intel = intel
        self._asr = asr  # faster-whisper provider; optional at runtime (model may be missing)
        self._speaker = speaker
        self._reports = reports
        self._asr_error: str | None = None
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "asr": self._asr is not None and self._asr_error is None,
            "asr_error": self._asr_error,
            "speaker": self._speaker.status(),
            "qc_defaults": VA.QC_DEFAULTS,
        }

    # ---------- one narration ----------

    def report(
        self,
        lines: list[dict[str, Any]],
        game: str | None,
        voice: str,
        check_names: bool | None = None,
    ) -> dict[str, Any]:
        """lines: [{label, path, text}] in playback order (text: what the script says)."""
        s = self._intel.settings()
        check_names = s["verify_pronunciation"] if check_names is None else check_names
        segs: list[dict[str, Any]] = []
        per_line: list[dict[str, Any]] = []
        estimates: list[dict[str, Any]] = []
        for li, ln in enumerate(lines):
            path = Path(ln["path"])
            if not path.is_file():
                per_line.append({"index": ln.get("index", li), "label": ln.get("label", ""),
                                 "missing": True})  # fmt: skip
                continue
            a, sr = read_wav(path)
            meta = _sidecar(path)
            spans = meta.get("segments") or [
                {"start": 0.0, "end": len(a) / sr, "text": ln["text"], "speed": 1.0}
            ]
            for si, sp in enumerate(spans):
                piece = a[int(sp["start"] * sr) : int(sp["end"] * sr)]
                if len(piece) < int(0.2 * sr):
                    continue
                segs.append(
                    {
                        "id": f"{li}.{si}",
                        "line": li,
                        "label": _label(ln, sp),
                        "metrics": VA.measure(piece, sr),
                        "embedding": self._embed(piece, sr),
                        "speed": sp.get("speed", 1.0),
                        "group": sp.get("group"),
                    }
                )
            whole = VA.measure(a, sr)
            names = self._names(a, sr, ln["text"], game, voice) if check_names else None
            per_line.append(
                {
                    "index": ln.get("index", li),  # position in the voice-over's own lines
                    "label": ln.get("label", ""),
                    "text": ln["text"],
                    "lufs": whole["lufs"],
                    "peak_db": whole["peak_db"],
                    "seconds": whole["seconds"],
                    "segments": len(spans),
                    "names": names,
                }
            )
            for n in (meta.get("plan") or {}).get("transitions") or []:
                if n.get("abrupt"):
                    estimates.append(
                        {
                            "line": li,
                            "kind": "mood",
                            "message": f"Big mood change ({n['from']} to {n['to']}): Qwen can't "
                            "blend voices, so it's a switch with a pause. Listen to "
                            "check it doesn't sound like a new narrator.",
                        }
                    )
            for sp in spans[:-1]:
                if len(sp["text"].split()) < 3 and not sp.get("intentional"):
                    estimates.append(
                        {
                            "line": li,
                            "kind": "boundary",
                            "message": f'Very short piece "{sp["text"]}" said on its own; it may '
                            "sound clipped.",
                        }
                    )
        cons = VA.consistency(segs, s.get("qc") or {})
        names_flat = [n for ln in per_line for n in (ln.get("names") or [])]
        clipping = [
            f for f in cons["flags"] if f["kind"] == "artifact" and "clipping" in f["message"]
        ]
        report = {
            "measured": {  # objective: computed from the audio
                "consistency_score": cons["score"],
                "identity": cons["identity"],
                "identity_model": cons["identity_model"],
                "loudness_spread_lu": cons["loudness_spread_lu"],
                "clipping": len(clipping),
                "flags": [f for f in cons["flags"] if f["kind"] != "artifact" or f not in clipping]
                + clipping,
                "segments": cons["segments"],
                "thresholds": cons["thresholds"],
            },
            "recognised": {  # automatic, and recognition can be wrong about fictional names
                "checked": check_names and self._asr_error is None,
                "error": self._asr_error,
                "names": names_flat,
                "warnings": [
                    n for n in names_flat if n["status"] in ("check", "suspect", "unclear")
                ],
            },
            "estimates": estimates,  # rules of thumb, not measurements
            "lines": per_line,
            "game": game,
            "voice": voice,
        }
        return report

    def _embed(self, a: VA.F32, sr: int) -> Any:
        try:
            e = self._speaker.embed(a, sr)
            return None if e is None else e.tolist()
        except Exception:
            log.warning("speaker embedding failed", exc_info=True)
            return None

    def _names(
        self, a: VA.F32, sr: int, text: str, game: str | None, voice: str
    ) -> list[dict[str, Any]] | None:
        if not game or VOICES.get(voice, ("", "", "", "en"))[3] != "en" or self._asr is None:
            return None
        terms = [V.Term.from_dict(t) for t in self._intel.terms(game)]
        if not V.find_terms(text, terms):
            return []
        try:
            with self._lock:
                segs = self._asr.transcribe(
                    VA.resample(a, sr, 16000), 16000, "en", 0.0, lambda f: None
                )
            self._asr_error = None
        except Exception as e:  # model not downloaded, offline, GPU busy...
            self._asr_error = f"Speech recognition unavailable: {e}"
            log.warning("pronunciation check: %s", e)
            return None
        heard = [(w.text, w.probability) for s in segs for w in s.words]
        return V.align_names(text, heard, terms)

    # ---------- a Short's voice-over ----------

    def clip_report(self, clip_id: str, refresh: bool = False) -> dict[str, Any]:
        path = self._reports / f"{clip_id}.json"
        if path.is_file() and not refresh:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        clip = self._vo.get(clip_id)
        effect = next((e for e in clip.effects if e["type"] == "voiceover"), None)
        if not effect:
            raise NotFoundError("This Short has no voice-over")
        voice, game = effect.get("voice", ""), self._vo.game_of(clip)
        lines = [
            {
                "index": i,
                "label": f"{ln.get('kind', 'line').capitalize()} {i + 1}",
                "text": ln["text"],
                "path": str(self._vo.line_path(ln, voice, game)),
            }
            for i, ln in enumerate(effect.get("lines", []))
            if not ln.get("dropped")
        ]
        report = self.report(lines, game, voice)
        report["clip_id"] = clip_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
        return report

    def line_report(
        self,
        text: str,
        voice: str,
        game: str | None,
        direction: dict[str, Any] | None,
        overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Speak one line (as previewed) and check it."""
        if not text.strip():
            raise SphuraError("nothing to say")
        path = self._vo.cached(text, voice, game, direction, overrides)
        said = " ".join(t for t, _ in V.parse_chunks(text))
        return self.report([{"label": "Preview", "text": said, "path": str(path)}], game, voice)

    # ---------- false positives and corrections (always the user's call) ----------

    def accept_heard(self, game: str, term_id: str, heard: str) -> dict[str, Any]:
        """'It sounds right': recognition hears this when the name is said correctly."""
        t = next((x for x in self._intel.terms(game) if x["id"] == term_id), None)
        if t is None:
            raise NotFoundError("not in the dictionary")
        heard = heard.strip()[:60]
        if heard and heard not in t["heard"]:
            t["heard"] = [*t["heard"], heard][-8:]
        term = V.Term.from_dict(t)  # keep its source and verification as they are
        self._intel.save_term(game, term)
        return {**term.to_dict(), "needs_review": term.needs_review}

    def variants(self, game: str, name: str) -> list[str]:
        t = next(
            (
                V.Term.from_dict(x)
                for x in self._intel.terms(game)
                if V.key_of(x["canonical"]) == V.key_of(name)
            ),
            None,
        )
        return V.respell_variants(t or V.Term(name))


def _label(ln: dict[str, Any], sp: dict[str, Any]) -> str:
    return f"{ln.get('label', '')} · {sp['text'][:40]}"


def _sidecar(path: Path) -> dict[str, Any]:
    side = path.with_suffix(".json")
    try:
        return dict(json.loads(side.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}
