"""Voice Intelligence: per-game pronunciation dictionaries (built from the footage, wikis and
AI suggestions, corrected by the user), voice settings, and the plan for each spoken line:
what the engine reads, where it pauses, how fast, in which mood."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sphura.domain.errors import NotFoundError, SphuraError
from sphura.infrastructure.persistence.sqlite.voice import SqliteVoiceRepository
from sphura.pipeline import voice_intel as V
from sphura.pipeline.judge import is_name
from sphura.pipeline.trending import detect_game
from sphura.ports.ai import LLMProvider, LLMRequest

log = logging.getLogger(__name__)

GAME_NAMES = {"gow": "God of War Ragnarök", "pbz": "Phantom Blade Zero"}
DEFAULTS: dict[str, Any] = {
    "personality": "auto",  # auto: from the voice (Gamer -> Gaming Hype, Storyteller -> Cinematic)
    "speed": 1.0,
    "intensity": 0.6,
    "pauses": 1.0,
    "emotion": True,  # mood per line
    "pronunciation": True,  # dictionary applied to what the voice reads
    "optimize": True,  # speech-friendly scripts and numbers
    "scene_aware": True,  # delivery follows the kind of scene
    "glossary": True,  # names given to the script writer
    "auto_save": "save",  # save | confident | review: what a scan stores by itself
    "transition_sensitivity": 0.5,  # 0: one voice throughout .. 1: follow every mood change
    "enhance": "light",  # off | light | standard: levelling, compression, loudness
    "verify_pronunciation": True,  # listen back with speech recognition after generating
    "qc": {},  # thresholds for the consistency report (see voice_audio.QC_DEFAULTS)
    "custom": {},
}
AUTO_SAVE = ("save", "confident", "review")
MAX_PROMPT_TERMS = 20

PRON_SYSTEM = (
    "You help a text-to-speech voice say video game names the way the game's own voice "
    "actors say them. Only answer from what you know about this game. When you don't know "
    "how the game says a name, say so: set unsure and a low confidence. Never invent."
)
PRON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(V.KINDS)},
                    "respell": {"type": "string"},
                    "phonetic": {"type": "string"},
                    "origin": {"type": "string"},
                    "confidence": {"type": "number"},
                    "unsure": {"type": "boolean"},
                },
                "required": [
                    "name",
                    "kind",
                    "respell",
                    "phonetic",
                    "origin",
                    "confidence",
                    "unsure",
                ],
            },
        }
    },
    "required": ["names"],
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", V.key_of(name) or name.lower()).strip("-")[:40] or "game"


class VoiceIntelService:
    def __init__(
        self,
        repo: SqliteVoiceRepository,
        queries: Any,
        lore_repo: Any | None,
        llm: LLMProvider | None,
    ) -> None:
        self._repo = repo
        self._queries = queries
        self._lore = lore_repo
        self._llm = llm
        self._lock = threading.Lock()
        self._scans: dict[str, dict[str, Any]] = {}

    # ---------- settings ----------

    def settings(self) -> dict[str, Any]:
        return {**DEFAULTS, **self._repo.settings()}

    def save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        s = self.settings()
        for k, v in patch.items():
            if k not in DEFAULTS:
                continue
            if k == "personality" and v != "auto" and v not in V.PERSONALITIES:
                raise SphuraError("unknown voice personality")
            if k == "auto_save" and v not in AUTO_SAVE:
                raise SphuraError("unknown auto-save policy")
            if k == "enhance" and v not in ("off", "light", "standard"):
                raise SphuraError("unknown enhancement level")
            if k in ("speed", "intensity", "pauses", "transition_sensitivity"):
                lo, hi = {"speed": (0.7, 1.4), "intensity": (0.0, 1.0), "pauses": (0.3, 2.5),
                          "transition_sensitivity": (0.0, 1.0)}[k]  # fmt: skip
                v = round(min(max(float(v), lo), hi), 2)
            elif isinstance(DEFAULTS[k], bool):
                v = bool(v)
            elif k == "qc":
                from sphura.pipeline.voice_audio import QC_DEFAULTS

                v = {kk: float(vv) for kk, vv in dict(v).items() if kk in QC_DEFAULTS}
            elif k == "custom":
                v = {
                    kk: vv
                    for kk, vv in dict(v).items()
                    if kk in ("voice", "pace", "intensity", "pitch", "pauses", "emphasis", "writer")
                }
            s[k] = v
        self._repo.save_settings({k: s[k] for k in DEFAULTS}, _now())
        return s

    def personality_for(
        self, voice: str, s: dict[str, Any] | None = None, game: str | None = None
    ) -> V.Personality:
        s = s or self.settings()
        pid = s["personality"]
        if pid == "auto" and game:
            pid = (self._repo.profile(game) or {}).get("personality") or "auto"
        if pid == "auto":
            pid = V.VOICE_PERSONALITY.get(voice, "casual")
        return V.personality(pid, s.get("custom"))

    # ---------- games ----------

    def games(self) -> list[dict[str, Any]]:
        """Every game SPHURA has footage or a profile for; new ones start a profile."""
        found: dict[str, str] = dict(GAME_NAMES)
        for p in self._safe(lambda: self._queries.list_projects(500)) or []:
            g = detect_game(p.name)
            if g:
                found.setdefault(g, GAME_NAMES.get(g, g))
        for g, prof in self._repo.profiles().items():
            found[g] = prof.get("name") or found.get(g, g)
        return [self.profile(g, name) for g, name in found.items()]

    def ensure_profile(self, game: str, name: str | None = None) -> dict[str, Any]:
        prof = self._repo.profile(game)
        if prof is None:
            prof = {
                "game": game,
                "name": name or GAME_NAMES.get(game, game),
                "personality": "auto",
                "accent": "neutral English",
                "language": "en",
                "version": V.PROFILE_VERSION,
                "created_at": _now(),
                "scanned_at": None,
                "sources": [],
            }
            self._repo.save_profile(game, prof, _now())
            seeds = [V.Term(**d, source="builtin") for d in V.SEEDS.get(game, [])]
            if seeds:
                self._repo.save_terms(game, seeds, _now())
        return prof

    def create_game(self, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise SphuraError("Give the game a name")
        game = detect_game(name) or slug(name)
        self.ensure_profile(game, name)
        return self.profile(game)

    def profile(self, game: str, name: str | None = None) -> dict[str, Any]:
        prof = self.ensure_profile(game, name)
        terms = self._repo.terms(game)
        kinds = Counter(t.kind for t in terms)
        return {
            **prof,
            "stats": {
                "terms": len(terms),
                "characters": kinds.get("character", 0) + kinds.get("boss", 0),
                "places": kinds.get("place", 0),
                "other": len(terms)
                - kinds.get("character", 0)
                - kinds.get("boss", 0)
                - kinds.get("place", 0),
                "verified": sum(t.verified for t in terms),
                "review": sum(t.needs_review for t in terms),
            },
            "scan": self._scans.get(game),
        }

    def save_profile(self, game: str, patch: dict[str, Any]) -> dict[str, Any]:
        prof = self.ensure_profile(game)
        for k in ("name", "personality", "accent", "language"):
            if k in patch:
                v = str(patch[k])[:80]
                if k == "personality" and v != "auto" and v not in V.PERSONALITIES:
                    raise SphuraError("unknown voice personality")
                prof[k] = v
        self._repo.save_profile(game, prof, _now())
        return self.profile(game)

    # ---------- dictionary ----------

    def terms(self, game: str) -> list[dict[str, Any]]:
        self.ensure_profile(game)
        return [{**t.to_dict(), "needs_review": t.needs_review} for t in self._repo.terms(game)]

    def upsert_term(self, game: str, data: dict[str, Any]) -> dict[str, Any]:
        """A user's entry: verified, and it wins over any scan from now on."""
        self.ensure_profile(game)
        canonical = str(data.get("canonical") or "").strip()
        if not canonical:
            raise SphuraError("Enter the name as it's written")
        old_id = str(data.get("id") or "")
        old = self._repo.term(game, old_id) if old_id else None
        t = V.Term(
            canonical=canonical[:60],
            kind=str(data.get("kind") or (old.kind if old else "term")),
            aliases=_list(data.get("aliases")),
            nicknames=_list(data.get("nicknames")),
            respell=str(data.get("respell") or "").strip()[:80],
            phonetic=str(data.get("phonetic") or "").strip()[:80],
            ipa=str(data.get("ipa") or "").strip()[:80],
            notes=str(data.get("notes") or "").strip()[:200],
            origin=str(data.get("origin") or "").strip()[:40],
            confidence=1.0,
            source="user",
            verified=bool(data.get("verified", True)),
        )
        if t.kind not in V.KINDS:
            t.kind = "term"
        if old and old.id != t.id:
            self._repo.delete_term(game, old.id)  # renamed
        self._repo.save_terms(game, [t], _now())
        return {**t.to_dict(), "needs_review": t.needs_review}

    def save_term(self, game: str, term: V.Term) -> None:
        """Stores an entry exactly as given (source and verification untouched)."""
        self.ensure_profile(game)
        self._repo.save_terms(game, [term], _now())

    def verify(self, game: str, term_id: str) -> dict[str, Any]:
        t = self._repo.term(game, term_id)
        if t is None:
            raise NotFoundError("not in the dictionary")
        t.verified, t.confidence = True, max(t.confidence, 0.9)
        self._repo.save_terms(game, [t], _now())
        return {**t.to_dict(), "needs_review": t.needs_review}

    def delete_term(self, game: str, term_id: str) -> None:
        if not self._repo.delete_term(game, term_id):
            raise NotFoundError("not in the dictionary")

    def relevant(self, game: str | None, text: str) -> list[V.Term]:
        """Dictionary entries this text mentions: the only ones worth sending to the AI."""
        if not game:
            return []
        self.ensure_profile(game)
        terms = self._repo.terms(game)
        hit = V.find_terms(text, terms)
        hit.sort(key=lambda t: (not t.verified, -t.confidence))
        return hit[:MAX_PROMPT_TERMS]

    # ---------- glossary scan ----------

    def scan(self, game: str) -> dict[str, Any]:
        self.ensure_profile(game)
        with self._lock:
            cur = self._scans.get(game)
            if cur and cur["status"] == "running":
                return cur
            run = {
                "status": "running",
                "step": "Reading your footage",
                "progress": 0.0,
                "error": None,
                "added": 0,
                "updated": 0,
                "review": 0,
                "suggestions": [],
            }
            self._scans[game] = run
        threading.Thread(
            target=self._scan, args=(game, run), daemon=True, name=f"voice-scan-{game}"
        ).start()
        return run

    def _scan(self, game: str, run: dict[str, Any]) -> None:
        try:
            self._scan_run(game, run)
            run.update(status="done", step="Done", progress=1.0)
        except Exception as e:
            log.exception("glossary scan failed")
            run.update(status="failed", error=str(e) or type(e).__name__)

    def _scan_run(self, game: str, run: dict[str, Any]) -> None:
        s = self.settings()
        prof = self.ensure_profile(game)
        existing = {t.id: t for t in self._repo.terms(game)}
        found: dict[str, V.Term] = {}
        sources: Counter[str] = Counter()

        def add(t: V.Term, src: str) -> None:
            sources[src] += 1
            found[t.id] = V.merge(found.get(t.id), t)

        for d in V.SEEDS.get(game, []):
            add(V.Term(**d, source="builtin"), "built-in list")
        names = self._names_in_footage(game)
        run.update(step="Reading wiki pronunciations", progress=0.3)
        wiki = self._wiki_pronunciations(game)
        for name, (kind, n) in names.items():
            add(
                V.Term(
                    name,
                    kind=kind,
                    confidence=0.3,
                    source="scan",
                    notes=f"Seen in your footage {n} times",
                ),
                "your footage",
            )
        for name, phonetic, url in wiki:
            add(
                V.Term(
                    name,
                    kind="character",
                    phonetic=phonetic,
                    respell=_respell_from(phonetic),
                    confidence=0.85,
                    source="wiki",
                    notes=f"From {url}",
                ),
                "wiki",
            )
        # Names nobody has said how to pronounce: a rule-based guess, then the AI's opinion
        unknown = [
            t
            for t in found.values()
            if not t.respell
            and not t.ipa
            and not (existing.get(t.id) and (existing[t.id].respell or existing[t.id].ipa))
        ]
        for t in unknown:
            t.respell, t.confidence = V.guess_respell(t.canonical, t.origin)
            if t.respell == t.canonical:
                t.respell = ""  # nothing to change: English reading is probably fine
        if self._llm is not None and unknown:
            run.update(step=f"Asking the AI about {len(unknown)} names", progress=0.55)
            for t in self._ai_suggest(game, prof["name"], unknown):
                add(t, "AI suggestion")
        run.update(step="Saving the dictionary", progress=0.9)
        policy = s["auto_save"]
        save, review = [], []
        for t in found.values():
            old = existing.get(t.id)
            merged = V.merge(old, t) if old else t
            if old and merged is old and merged.to_dict() == old.to_dict():
                continue
            keep = (
                old is not None
                or policy == "save"
                or (policy == "confident" and t.confidence >= 0.6)
            )
            if not keep:
                review.append({**t.to_dict(), "needs_review": True})
                continue
            save.append(merged)
            run["updated" if old else "added"] += 1
        self._repo.save_terms(game, save, _now())
        prof.update(scanned_at=_now(), sources=sorted(sources), version=V.PROFILE_VERSION)
        self._repo.save_profile(game, prof, _now())
        run.update(review=sum(t.needs_review for t in self._repo.terms(game)), suggestions=review)

    def _names_in_footage(self, game: str) -> dict[str, tuple[str, int]]:
        """Names the judge read on screen (HUD, subtitles) and the lore index saw, with how
        often: names said once are often misreadings, so they need two sightings."""
        seen: Counter[str] = Counter()
        kinds: dict[str, str] = {}
        for p in self._safe(lambda: self._queries.list_projects(500)) or []:
            if detect_game(p.name) != game:
                continue
            for c in self._safe(lambda p=p: self._queries.candidates(p.id)) or []:
                j = c.judgement
                if j is None:
                    continue
                for n in {V.title_name(x) for x in (*j.names, *j.characters)}:
                    if _usable(n):
                        seen[n] += 1
                        kinds.setdefault(n, "character")
        if self._lore is not None:
            for sc in self._safe(lambda: self._lore.scenes(game=game)) or []:
                for n in (V.title_name(x) for x in sc.get("seen", [])):
                    if _usable(n):
                        seen[n] += 1
                        kinds.setdefault(n, "character")
                loc = sc.get("location", "")
                if loc and loc[:1].isupper() and _usable(loc) and len(loc.split()) <= 3:
                    seen[loc] += 1
                    kinds.setdefault(loc, "place")
        # one spelling per name: the most common one wins
        best: dict[str, tuple[str, int]] = {}
        for n, k in seen.most_common():
            key = V.key_of(n)
            if key and key not in best and k >= 2:
                best[key] = (n, k)
        return {n: (kinds[n], k) for n, k in best.values()}

    def _wiki_pronunciations(self, game: str) -> list[tuple[str, str, str]]:
        """'(pronounced ay-TREE-us)' lines in cached wiki passages, per subject."""
        out: list[tuple[str, str, str]] = []
        if self._lore is None:
            return out
        for p in self._safe(lambda: self._lore.passages(game)) or []:
            m = re.search(
                r"pronounced[:\s]+[\"“'/(]?([A-Za-z][A-Za-z' -]{1,40}?)[\"”'/),;.]",
                p.get("text", ""),
                re.I,
            )
            if m and p.get("entity") not in ("game", ""):
                name = p.get("page") or p["entity"].title()
                out.append((re.sub(r"\s*\(.*\)", "", name), m.group(1).strip(), p.get("url", "")))
        return out

    def _ai_suggest(self, game: str, name: str, terms: list[V.Term]) -> list[V.Term]:
        assert self._llm is not None
        out: list[V.Term] = []
        for i in range(0, len(terms), 25):
            batch = terms[i : i + 25]
            prompt = (
                f"Game: {name}.\nNames seen in the creator's footage:\n"
                + "\n".join(f"- {t.canonical}" for t in batch)
                + "\n\nFor each: kind, a respelling an English voice will read correctly "
                "(e.g. 'Ay-tree-us'), a phonetic with the stressed syllable in capitals "
                "(e.g. 'ay-TREE-us'), the language it comes from, and how sure you are "
                "(0 to 1) that this is how the game says it."
            )
            try:
                resp = self._llm.complete(LLMRequest(PRON_SYSTEM, prompt, [], PRON_SCHEMA, 2000))
                rows = json.loads(resp.text).get("names") or []
            except Exception as e:
                log.warning("pronunciation suggestions failed: %s", e)
                continue
            for r in rows:
                n = str(r.get("name", "")).strip()
                if V.key_of(n) not in {t.id for t in batch}:
                    continue  # only names we asked about
                conf = min(float(r.get("confidence") or 0), 0.6)  # never as good as a check
                if r.get("unsure"):
                    conf = min(conf, 0.35)
                kind = str(r.get("kind") or "term")
                out.append(
                    V.Term(
                        n,
                        kind=kind if kind in V.KINDS else "term",
                        respell=str(r.get("respell") or "").strip()[:80] if conf >= 0.35 else "",
                        phonetic=str(r.get("phonetic") or "").strip()[:80],
                        origin=str(r.get("origin") or "").strip()[:40],
                        confidence=round(conf, 2),
                        source="ai",
                        notes="AI suggestion: check it"
                        + (" (the AI wasn't sure)" if r.get("unsure") else ""),
                    )
                )
        return out

    # ---------- what the voice reads ----------

    def plan(
        self,
        text: str,
        voice: str,
        engine: str,
        lang: str,
        game: str | None = None,
        direction: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
        respell: dict[str, str] | None = None,
        prev_group: str | None = None,
    ) -> dict[str, Any]:
        """Everything an engine needs to say one line, and why (for preview and debugging).
        prev_group: the reference the line before ended on, so a narration stays one voice.
        respell: one-off spellings to try (pronunciation test), never saved."""
        s = {**self.settings(), **(overrides or {})}
        caps = V.ENGINES.get(engine, V.ENGINES["kokoro"])
        p = self.personality_for(voice, s, game)
        if game:
            self.ensure_profile(game)  # a game seen for the first time starts its profile
        terms = self._repo.terms(game) if game and s["pronunciation"] else []
        for name, sp in (respell or {}).items():
            t = next((x for x in terms if V.key_of(x.canonical) == V.key_of(name)), None)
            t = V.Term(name, respell=sp, source="user", confidence=1.0, verified=True,
                       aliases=t.aliases if t else [])  # fmt: skip
            terms = [x for x in terms if x.id != t.id] + [t]
        keep = {f for t in terms for f in [t.canonical, *t.aliases, *t.nicknames]}
        items: list[tuple[str, V.Direction, str, bool]] = []
        lexicon: dict[str, str] = {}
        changes: list[dict[str, str]] = []
        for spoken, d in V.parse_chunks(text):
            explicit = d != V.Direction()  # the writer tagged this piece
            if direction and not explicit:  # the line's delivery from the editor
                d = V.Direction.from_dict(direction)
            if not s["emotion"]:
                d.mood = "neutral"
            if p.intensity * float(s["intensity"]) < 0.25 and V.MOOD_GROUP[d.mood] == "intense":
                d.mood = "neutral"  # a calm personality at low intensity doesn't push hype
            text_in = V.speakable(spoken, keep) if s["optimize"] and lang == "en" else spoken
            prep = V.apply_pronunciation(text_in, terms, bool(caps["phonemes"]))
            lexicon.update(prep.lexicon)
            changes += prep.changes
            items.append((spoken, d, prep.tts, explicit or bool(direction)))
        refs = caps["emotion"] == "reference"
        if refs:  # one voice unless the mood really changes (Qwen can't blend references)
            groups, notes = V.smooth_moods(
                [(d.mood, len(t.split()), ex) for _, d, t, ex in items],
                float(s["transition_sensitivity"]),
                prev_group,
            )
        else:
            groups, notes = ["neutral"] * len(items), []
        abrupt = {n["at"] for n in notes if n.get("abrupt")}
        pieces = []
        for i, ((_, d, tts, ex), g) in enumerate(zip(items, groups, strict=True)):
            gap = max(d.pause_before_ms if i else 0, 500 if i in abrupt else 0)
            pieces.append(V.Piece(tts, g, d.mood, d.pace, gap, ex))
        segs = V.group_segments(pieces, float(s["pauses"]) * p.pauses)
        for sg in segs:
            sg["speed"] = round(float(s["speed"]) * p.pace * V.PACES[sg["pace"]], 3)
            sg["reference"] = sg["group"] if refs else None
            sg["design"] = V.design_for(p, sg["group"]) if refs else None
        d0 = items[0][1] if items else V.Direction()
        if segs:
            segs[-1]["pause_after_ms"] = d0.pause_after_ms
        plan = {
            "text": " ".join(x[0] for x in items), "tts": " ".join(x[2] for x in items),
            "segments": segs,
            "pause_before_ms": d0.pause_before_ms if not direction and not prev_group else 0,
            "lexicon": lexicon,
            "changes": list({(c["term"], c["said"]): c for c in changes}.values()),
            "mood": d0.mood, "pace": d0.pace, "speed": segs[0]["speed"] if segs else 1.0,
            "engine": engine, "personality": p.id,
            "reference": segs[0]["reference"] if segs else None,
            "last_group": segs[-1]["group"] if segs else prev_group,
            "transitions": notes,
            "enhance": s["enhance"],
            "caps": {k: caps[k] for k in ("emotion", "phonemes", "speed", "pauses")},
        }  # fmt: skip
        plan["key"] = V.plan_hash(
            voice,
            {k: v for k, v in plan.items() if k not in ("text", "changes", "caps", "transitions")},
            (overrides or {}).get("take", 0),
        )
        return plan

    def voice_context(
        self, game: str | None, voice: str, scene: str, seconds: float, text: str
    ) -> dict[str, Any]:
        """What the script writer gets: scene, personality, the names it must spell right."""
        s = self.settings()
        p = self.personality_for(voice, s, game)
        scene = scene if s["scene_aware"] else "calm"
        terms = self.relevant(game, text) if s["glossary"] else []
        name = (self._repo.profile(game) or {}).get("name") if game else None
        return {
            "brief": V.writer_brief(
                name or GAME_NAMES.get(game or "", game or "the game"),
                scene,
                p,
                terms,
                bool(s["emotion"]),
                seconds,
            ),
            "emotion": bool(s["emotion"]),
            "optimize": bool(s["optimize"]),
            "default": V.SCENES.get(scene, V.SCENES["calm"]),
            "scene": scene,
            "personality": p.id,
        }

    @staticmethod
    def _safe(fn: Any) -> Any:
        try:
            return fn()
        except Exception:
            return None


def _list(v: Any) -> list[str]:
    if isinstance(v, str):
        v = v.split(",")
    return [str(x).strip()[:60] for x in (v or []) if str(x).strip()][:10]


def _usable(name: str) -> bool:
    n = name.strip()
    return 2 < len(n) <= 40 and not re.search(r"\d", n) and is_name(n) and n[:1].isupper()


def _respell_from(phonetic: str) -> str:
    """'ay-TREE-us' -> 'Ay-tree-us' (a respelling a voice reads naturally)."""
    p = phonetic.strip().lower()
    return p[:1].upper() + p[1:] if p else ""
