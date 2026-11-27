"""Voice Intelligence, no I/O: game pronunciation dictionaries, a voice-direction layer that
keeps delivery notes out of the spoken text, speech-friendly rewriting and voice
personalities. Engines get only what they can use (IPA, respelling, a mood reference)."""

# ruff: noqa: RUF001  (IPA symbols look like Latin letters on purpose)
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

PROFILE_VERSION = 1
KINDS = ("character", "place", "boss", "item", "faction", "term", "event")

# ---------- pronunciation dictionary ----------


@dataclass(slots=True)
class Term:
    canonical: str
    kind: str = "term"
    aliases: list[str] = field(default_factory=list)  # other spellings: said the same way
    nicknames: list[str] = field(default_factory=list)  # other names: recognised, not respelled
    heard: list[str] = field(default_factory=list)  # what speech recognition hears when it's right
    respell: str = ""  # TTS-friendly spelling, e.g. "Ay-tree-us"
    phonetic: str = ""  # for people, e.g. "ay-TREE-us"
    ipa: str = ""
    notes: str = ""
    origin: str = ""  # e.g. Old Norse, Greek, Chinese
    confidence: float = 0.5
    source: str = "auto"  # builtin, wiki, transcript, ai, user
    verified: bool = False
    id: str = ""

    def __post_init__(self) -> None:
        self.id = self.id or key_of(self.canonical)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Term:
        fields = cls.__slots__
        return cls(**{k: v for k, v in d.items() if k in fields})

    @property
    def needs_review(self) -> bool:
        return not self.verified and self.confidence < 0.6

    def rank(self) -> tuple[int, int, float]:
        """User fixes first, then anything verified, then the most confident guess."""
        return (self.source == "user", self.verified, self.confidence)


def key_of(name: str) -> str:
    s = name.lower().replace("æ", "ae").replace("ø", "o").replace("ð", "d").replace("þ", "th")
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in s if not unicodedata.combining(c)))


def merge(old: Term | None, new: Term) -> Term:
    """A rescan never overwrites what the user said or verified; it only fills gaps."""
    if old is None:
        return new
    if old.source == "user" or old.verified or old.rank() >= new.rank():
        for a in new.aliases:
            if a not in old.aliases and a != old.canonical:
                old.aliases.append(a)
        for a in new.nicknames:
            if a not in old.nicknames:
                old.nicknames.append(a)
        for a in new.heard:
            if a not in old.heard:
                old.heard.append(a)
        if not old.ipa and new.ipa and not (old.source == "user" and old.respell):
            old.ipa = new.ipa
        return old
    new.aliases = list(dict.fromkeys([*new.aliases, *old.aliases]))
    return new


# How the games say these names. Seeds, not gospel: unverified until the user confirms them
SEEDS: dict[str, list[dict[str, Any]]] = {
    "gow": [
        {
            "canonical": "Kratos",
            "kind": "character",
            "respell": "Kraytos",
            "phonetic": "KRAY-toss",
            "ipa": "kɹˈeɪtoʊs",
            "origin": "Greek",
            "confidence": 0.85,
        },
        {
            "canonical": "Atreus",
            "kind": "character",
            "respell": "Ay-tree-us",
            "phonetic": "ay-TREE-us",
            "ipa": "eɪtɹˈiːəs",
            "origin": "Greek",
            "confidence": 0.85,
            "nicknames": ["Loki"],
        },
        {
            "canonical": "Mímir",
            "kind": "character",
            "aliases": ["Mimir"],
            "respell": "Mee-meer",
            "phonetic": "MEE-meer",
            "ipa": "mˈiːmɪɹ",
            "origin": "Old Norse",
            "confidence": 0.85,
        },
        {
            "canonical": "Freya",
            "kind": "character",
            "respell": "Fray-uh",
            "phonetic": "FRAY-uh",
            "origin": "Old Norse",
            "confidence": 0.75,
        },
        {
            "canonical": "Heimdall",
            "kind": "character",
            "respell": "Haym-dahl",
            "phonetic": "HAYM-dahl",
            "ipa": "hˈeɪmdɑːl",
            "origin": "Old Norse",
            "confidence": 0.8,
        },
        {
            "canonical": "Jörmungandr",
            "kind": "character",
            "aliases": ["Jormungandr"],
            "respell": "Yor-mun-gand",
            "phonetic": "YOR-mun-gand",
            "ipa": "jˈɔːɹmʊŋɡˌænd",
            "origin": "Old Norse",
            "confidence": 0.8,
        },
        {
            "canonical": "Mjölnir",
            "kind": "item",
            "aliases": ["Mjolnir"],
            "respell": "Myol-neer",
            "phonetic": "MYOL-neer",
            "ipa": "mjˈoʊlnɪɹ",
            "origin": "Old Norse",
            "confidence": 0.8,
        },
        {
            "canonical": "Baldur",
            "kind": "character",
            "respell": "Bahl-der",
            "phonetic": "BAHL-der",
            "ipa": "bˈɑːldɚ",
            "origin": "Old Norse",
            "confidence": 0.8,
        },
        {
            "canonical": "Angrboda",
            "kind": "character",
            "respell": "Ang-er-boda",
            "phonetic": "ANG-er-boh-da",
            "ipa": "ˈæŋɡɚbˌoʊdə",
            "origin": "Old Norse",
            "confidence": 0.75,
        },
        {
            "canonical": "Týr",
            "kind": "character",
            "aliases": ["Tyr"],
            "respell": "Teer",
            "phonetic": "TEER",
            "origin": "Old Norse",
            "confidence": 0.7,
        },
        {
            "canonical": "Sindri",
            "kind": "character",
            "respell": "Sin-dree",
            "phonetic": "SIN-dree",
            "origin": "Old Norse",
            "confidence": 0.7,
        },
        {
            "canonical": "Thrúd",
            "kind": "character",
            "aliases": ["Thrud"],
            "respell": "Throod",
            "phonetic": "THROOD",
            "origin": "Old Norse",
            "confidence": 0.65,
        },
        {
            "canonical": "Svartalfheim",
            "kind": "place",
            "respell": "Svart-alf-haym",
            "phonetic": "SVART-alf-haym",
            "origin": "Old Norse",
            "confidence": 0.65,
        },
        {
            "canonical": "Vanaheim",
            "kind": "place",
            "respell": "Vah-nah-haym",
            "phonetic": "VAH-nah-haym",
            "origin": "Old Norse",
            "confidence": 0.65,
        },
        {
            "canonical": "Jötunheim",
            "kind": "place",
            "aliases": ["Jotunheim"],
            "respell": "Yo-tun-haym",
            "phonetic": "YO-tun-haym",
            "origin": "Old Norse",
            "confidence": 0.65,
        },
        {
            "canonical": "Ragnarök",
            "kind": "event",
            "aliases": ["Ragnarok"],
            "respell": "Rag-nuh-rock",
            "phonetic": "RAG-nuh-rock",
            "origin": "Old Norse",
            "confidence": 0.75,
        },
    ],
}

# Surface form -> (term id, possessive suffix). Matches whole words only, any of the names
_POSS = r"(?P<poss>['’]s|['’](?=\s|$|[.,!?;:]))?"


def _forms(t: Term) -> list[str]:
    return sorted({t.canonical, *t.aliases} - {""}, key=len, reverse=True)


def term_pattern(terms: list[Term]) -> re.Pattern[str] | None:
    forms = {f: t.id for t in terms for f in _forms(t)}
    if not forms:
        return None
    alt = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<![\w'’-])(?P<name>{alt})(?![\w-]){_POSS}", re.IGNORECASE)


def find_terms(text: str, terms: list[Term]) -> list[Term]:
    """Terms the text mentions (any spelling or nickname), as names, not ordinary words."""
    alls = [Term(t.canonical, aliases=[*t.aliases, *t.nicknames], id=t.id) for t in terms]
    pat = term_pattern(alls)
    if pat is None:
        return []
    real = {t.id: t for t in terms}
    by_form = {f.lower(): real[t.id] for t in alls for f in _forms(t)}
    seen: dict[str, Term] = {}
    for m in pat.finditer(text):
        if _is_name(m.group("name")):
            t = by_form[m.group("name").lower()]
            seen.setdefault(t.id, t)
    return list(seen.values())


def _is_name(token: str) -> bool:
    # "Soul" the boss, not "soul" the word; all-caps titles count as names too
    return token[:1].isupper()


@dataclass(slots=True)
class Prepared:
    tts: str  # what the engine reads
    lexicon: dict[str, str]  # surface word -> IPA, for engines that take phonemes
    changes: list[dict[str, str]]


def apply_pronunciation(text: str, terms: list[Term], phonemes: bool) -> Prepared:
    """Swaps names for how they should be said, in the engine's input only. One term is
    said one way for the whole text (its best entry); possessives and case are kept."""
    best: dict[str, Term] = {}
    for t in terms:
        if t.id not in best or t.rank() > best[t.id].rank():
            best[t.id] = t
    pat = term_pattern(list(best.values()))
    if pat is None:
        return Prepared(text, {}, [])
    by_form = {f.lower(): t for t in best.values() for f in _forms(t)}
    lexicon: dict[str, str] = {}
    changes: list[dict[str, str]] = []

    def sub(m: re.Match[str]) -> str:
        name, poss = m.group("name"), m.group("poss") or ""
        if not _is_name(name):
            return m.group(0)
        t = by_form[name.lower()]
        if phonemes and t.ipa:
            word = name + poss
            lexicon[word] = t.ipa + (_possessive_ipa(t.canonical) if poss.endswith("s") else "")
            changes.append({"term": t.canonical, "said": f"/{lexicon[word]}/", "how": "ipa"})
            return word
        if t.respell and t.respell.lower() != name.lower():
            said = t.respell.upper() if name.isupper() and len(name) > 1 else t.respell
            changes.append({"term": t.canonical, "said": said, "how": "respell"})
            return said + poss
        return m.group(0)

    out = pat.sub(sub, text)
    uniq = list({(c["term"], c["said"]): c for c in changes}.values())
    return Prepared(out, lexicon, uniq)


def _possessive_ipa(name: str) -> str:
    return "ɪz" if re.search(r"[sxz]$|ch$|sh$", name.lower()) else "z"


# Old Norse spelling -> English sounds, for names nobody has checked yet. A guess: flagged
_NORSE = [
    (r"j", "y"),
    (r"ö|ø", "o"),
    (r"ð|þ", "th"),
    (r"æ", "eye"),
    (r"ei", "ay"),
    (r"í", "ee"),
    (r"ú", "oo"),
    (r"á", "ah"),
    (r"é", "ay"),
    (r"ý", "ee"),
    (r"gr$", "gur"),
    (r"ndr$", "nd"),
    (r"heim$", "haym"),
]


def guess_respell(name: str, origin: str = "") -> tuple[str, float]:
    """A first guess for a name without an entry; low confidence on purpose."""
    norse = origin.lower().startswith(("old norse", "norse")) or bool(
        re.search(r"[öøðþæíúáý]|heim\b", name, re.I)
    )
    if not norse:
        return name, 0.3
    words = []
    for w in name.split():
        low = w.lower()
        if low in _SMALL:
            words.append(low)
            continue
        for a, b in _NORSE:
            low = re.sub(a, b, low)
        words.append(low.capitalize())
    return " ".join(words), 0.45


_SMALL = {"the", "of", "and", "a", "an", "in", "on", "to"}


def title_name(name: str) -> str:
    """'FRÆKNI THE ZEALOUS' (as the HUD shows it) -> 'Frækni the Zealous'."""
    if not name.isupper():
        return name.strip()
    return " ".join(w.lower() if i and w.lower() in _SMALL else w.capitalize()
                    for i, w in enumerate(name.split()))  # fmt: skip


# ---------- voice direction ----------

MOODS = (
    "neutral",
    "calm",
    "warm",
    "reflective",
    "sad",
    "tense",
    "mysterious",
    "excited",
    "epic",
    "angry",
    "playful",
    "urgent",
)
PACES = {"slow": 0.9, "normal": 1.0, "fast": 1.08}
# Qwen clones a reference recording: one per mood group carries the delivery
MOOD_GROUP = {
    "neutral": "neutral",
    "warm": "neutral",
    "calm": "soft",
    "reflective": "soft",
    "sad": "soft",
    "tense": "tense",
    "mysterious": "tense",
    "excited": "intense",
    "epic": "intense",
    "angry": "intense",
    "urgent": "intense",
    "playful": "playful",
}
GROUP_DELIVERY = {
    "neutral": ("natural and conversational", "So here's what's actually going on in this scene."),
    "soft": (
        "quiet, restrained and emotional, slower, letting pauses land",
        "He didn't say anything. He didn't have to. That silence said everything.",
    ),
    "tense": (
        "low, tense and mysterious, almost a whisper, building suspense",
        "Something was wrong here. And he knew it, long before anyone else did.",
    ),
    "intense": (
        "powerful and intense, building energy, emphatic but not shouting",
        "And then he stopped holding back. Every single swing landed like thunder.",
    ),
    "playful": (
        "light, amused and playful, like sharing a joke with a friend",
        "Honestly? Nobody asked him to do that. He just did it anyway.",
    ),
}

_TAG = re.compile(r"^\s*\[([^\]]{1,80})\]\s*")
_STRAY = re.compile(
    r"\[[^\]]{0,80}\]|\((?:whisper\w*|pause|beat|laughs?|sighs?|softly|"
    r"slowly|angrily|sadly|excited\w*|dramatic\w*)\)|\*([^*]+)\*",
    re.I,
)


@dataclass(slots=True)
class Direction:
    mood: str = "neutral"
    pace: str = "normal"
    intensity: float = 0.5
    pause_before_ms: int = 0
    pause_after_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Direction:
        d = d or {}
        mood = str(d.get("mood", "neutral")).lower()
        pace = str(d.get("pace", "normal")).lower()
        return cls(
            mood if mood in MOODS else "neutral",
            pace if pace in PACES else "normal",
            float(min(max(float(d.get("intensity", 0.5) or 0.5), 0.0), 1.0)),
            int(d.get("pause_before_ms") or 0),
            int(d.get("pause_after_ms") or 0),
        )


_WORDS = {
    "whisper": ("tense", None),
    "whispering": ("tense", None),
    "tense": ("tense", None),
    "mysterious": ("mysterious", None),
    "sad": ("sad", None),
    "reflective": ("reflective", None),
    "calm": ("calm", None),
    "warm": ("warm", None),
    "excited": ("excited", None),
    "epic": ("epic", None),
    "angry": ("angry", None),
    "anger": ("angry", None),
    "playful": ("playful", None),
    "funny": ("playful", None),
    "urgent": ("urgent", None),
    "dramatic": ("epic", None),
    "emotional": ("sad", None),
    "slow": (None, "slow"),
    "slow pace": (None, "slow"),
    "fast": (None, "fast"),
    "quick": (None, "fast"),
}


def parse_direction(line: str) -> tuple[str, Direction]:
    """'[whispering, tense, slow pace] Something was wrong.' -> text + direction."""
    d = Direction()
    m = _TAG.match(line)
    if m:
        for part in (p.strip().lower() for p in m.group(1).split(",")):
            if "pause" in part:
                d.pause_before_ms = 450 if "short" in part else 800
            for w, (mood, pace) in _WORDS.items():
                if re.search(rf"\b{w}\b", part):
                    d.mood = mood or d.mood
                    d.pace = pace or d.pace
            if "increasing" in part or "intens" in part:
                d.intensity = 0.8
        line = line[m.end() :]
    return clean_spoken(line), d


def parse_chunks(text: str) -> list[tuple[str, Direction]]:
    """Text with several [direction] tags -> pieces, each with its own delivery."""
    starts = [m.start() for m in re.finditer(r"\[[^\]]{1,80}\]", text)]
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    out = []
    for a, b in zip(starts, [*starts[1:], len(text)], strict=True):
        piece, d = parse_direction(text[a:b])
        if piece:
            out.append((piece, d))
    return out or [("", Direction())]


def clean_spoken(text: str) -> str:
    """Removes stage directions so they're never read aloud; keeps *emphasis* words."""
    text = _STRAY.sub(lambda m: m.group(1) or "", text)
    return re.sub(r"\s+", " ", text).strip().strip('"“”').strip()


# What each engine can do with direction. Neither takes SSML or inline emotion tags
ENGINES: dict[str, dict[str, Any]] = {
    "qwen": {
        "name": "Qwen3-TTS",
        "emotion": "reference",
        "ssml": False,
        "tags": False,
        "phonemes": False,
        "respell": True,
        "speed": "post",
        "pauses": "silence",
    },
    "kokoro": {
        "name": "Kokoro",
        "emotion": None,
        "ssml": False,
        "tags": False,
        "phonemes": True,
        "respell": True,
        "speed": "native",
        "pauses": "silence",
    },
}


@dataclass(slots=True)
class Segment:
    text: str
    pause_after_ms: int


_PAUSE_MS = {"...": 550, "…": 550, "—": 350, " - ": 300, ".": 280, "?": 300, "!": 260, ";": 220}


def segments(text: str, d: Direction, pause_scale: float = 1.0) -> list[Segment]:
    """Splits a line where a pause belongs; the engine says each piece and SPHURA adds the
    silence, so pauses work on engines that ignore '...' or tags."""
    scale = pause_scale * (
        1.35
        if d.mood in ("sad", "reflective", "tense", "mysterious")
        else 0.8
        if d.mood in ("urgent", "excited")
        else 1.0
    )
    parts = re.split(r"(\.\.\.|…|—|\s-\s|(?<=[.?!;])\s+)", text)
    out: list[Segment] = []
    buf = ""
    for p in parts:
        if p is None:
            continue
        if p in _PAUSE_MS or (p.strip() == "" and buf.endswith((".", "?", "!", ";"))):
            key = p if p in _PAUSE_MS else buf[-1]
            if buf.strip():
                out.append(
                    Segment(
                        _end(buf, "," if key in ("—", " - ") else "."), int(_PAUSE_MS[key] * scale)
                    )
                )
            buf = ""
        else:
            buf += p
    if buf.strip():
        out.append(Segment(_end(buf), 0))
    if out:
        out[-1].pause_after_ms = d.pause_after_ms
    return out or [Segment(text, 0)]


def _end(s: str, mark: str = ".") -> str:
    s = s.strip()
    return s if s.endswith((".", "?", "!", ",")) else s + mark


# ---------- emotion transitions (whole narration) ----------

GROUPS = ("neutral", "soft", "tense", "intense", "playful")
# How different two reference deliveries sound (0 same .. 1 opposite). Qwen can't blend two
# references, so a switch is a cut from one recording style to another: worth it only when
# the change is real, and softened with a breath when it's big.
_DIST = {
    ("neutral", "soft"): 0.3, ("neutral", "playful"): 0.35, ("neutral", "tense"): 0.45,
    ("neutral", "intense"): 0.5, ("soft", "tense"): 0.4, ("tense", "intense"): 0.45,
    ("playful", "intense"): 0.55, ("soft", "playful"): 0.6, ("tense", "playful"): 0.7,
    ("soft", "intense"): 0.8,
}  # fmt: skip
ABRUPT = 0.7


def mood_distance(a: str, b: str) -> float:
    ga, gb = MOOD_GROUP.get(a, a), MOOD_GROUP.get(b, b)
    if ga == gb:
        return 0.0
    return _DIST.get((ga, gb)) or _DIST.get((gb, ga)) or 0.5


def smooth_moods(
    beats: list[tuple[str, int, bool]], sensitivity: float = 0.5, start: str | None = None
) -> tuple[list[str], list[dict[str, Any]]]:
    """Reference group per beat (mood, word count, explicitly tagged) across a narration.
    Small changes and brief detours keep the current reference, so the narrator sounds like
    one person; real changes switch. Sensitivity 1 follows every change, 0 almost none."""
    sens = min(max(sensitivity, 0.0), 1.0)
    threshold = 0.2 + 0.5 * (1 - sens)
    min_words = round(3 + 6 * (1 - sens))
    groups: list[str] = []
    notes: list[dict[str, Any]] = []
    cur = MOOD_GROUP.get(start or "", start) if start else None
    for i, (mood, words, explicit) in enumerate(beats):
        g = MOOD_GROUP.get(mood, "neutral")
        if cur is None or g == cur:
            cur = cur or g
            groups.append(cur)
            continue
        d = mood_distance(cur, g)
        nxt = MOOD_GROUP.get(beats[i + 1][0], "neutral") if i + 1 < len(beats) else None
        if d < (threshold * 0.6 if explicit else threshold):
            notes.append({"at": i, "from": cur, "to": g, "action": "kept",
                          "reason": "small change: same voice kept"})  # fmt: skip
        elif words < min_words and nxt == cur and not explicit:
            notes.append({"at": i, "from": cur, "to": g, "action": "kept",
                          "reason": "brief detour: same voice kept"})  # fmt: skip
        else:
            notes.append({"at": i, "from": cur, "to": g, "action": "switched",
                          "abrupt": d >= ABRUPT, "distance": round(d, 2),
                          "reason": "big change: a pause separates the two deliveries"
                          if d >= ABRUPT else "mood change"})  # fmt: skip
            cur = g
        groups.append(cur)
    return groups, notes


# ---------- grouping text into natural segments ----------

MAX_SEGMENT_WORDS = 32
MIN_SEGMENT_WORDS = 3
_INTENT = re.compile(r"(\.\.\.|…|\n\s*\n)")  # pauses the writer asked for


@dataclass(slots=True)
class Piece:
    text: str
    group: str
    mood: str
    pace: str
    pause_before_ms: int = 0  # an intentional pause before this piece
    explicit: bool = False


def group_segments(pieces: list[Piece], pause_scale: float = 1.0,
                   max_words: int = MAX_SEGMENT_WORDS) -> list[dict[str, Any]]:  # fmt: skip
    """Whole sentences that belong together are said in one go (one TTS call), so the
    engine keeps its own phrasing. New segments start only at a pause the writer asked for,
    a change of reference voice or pace, or when a run gets too long; never mid-sentence."""
    phrases: list[tuple[str, Piece, int]] = []  # (text, piece, intentional pause before)
    pause = 0  # a trailing "..." carries over to the next piece
    for pc in pieces:
        parts = _INTENT.split(pc.text)
        pause = max(pause, pc.pause_before_ms)
        for part in parts:
            if _INTENT.fullmatch(part or ""):
                pause = max(pause, int(_PAUSE_MS["..."] * pause_scale * _mood_scale(pc.mood)))
                continue
            if part and part.strip():
                phrases.append((part.strip(), pc, pause))
                pause = 0
    segs: list[dict[str, Any]] = []
    for text, pc, pause in phrases:
        prev = segs[-1] if segs else None
        same = prev and prev["group"] == pc.group and prev["pace"] == pc.pace
        if prev and same and not pause and _words(prev["text"]) + _words(text) <= max_words:
            prev["text"] = f"{prev['text']} {text}"
            continue
        for sentence_run in _runs(text, max_words):
            if segs:
                gap = pause or (int(_PAUSE_MS["."] * pause_scale) if segs[-1]["group"] == pc.group
                                else 250)  # fmt: skip
                segs[-1]["pause_after_ms"] = max(segs[-1]["pause_after_ms"], gap)
            segs.append(
                {
                    "text": sentence_run,
                    "group": pc.group,
                    "mood": pc.mood,
                    "pace": pc.pace,
                    "pause_after_ms": 0,
                    "intentional": bool(pause),
                }
            )
            pause = 0
    _merge_tiny(segs)
    for sg in segs:
        sg["text"] = _end(sg["text"])
    return segs


def _mood_scale(mood: str) -> float:
    if mood in ("sad", "reflective", "tense", "mysterious"):
        return 1.35
    return 0.8 if mood in ("urgent", "excited") else 1.0


def _words(t: str) -> int:
    return len(t.split())


def _runs(text: str, max_words: int) -> list[str]:
    """Sentences packed into runs of at most max_words; a longer sentence stays whole."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if out and _words(out[-1]) + _words(sent) <= max_words:
            out[-1] = f"{out[-1]} {sent}"
        else:
            out.append(sent)
    return [o for o in out if o.strip()]


def _merge_tiny(segs: list[dict[str, Any]]) -> None:
    """A two-word piece alone sounds clipped: join it to a neighbour with the same voice,
    unless the writer asked for the pause that isolates it."""
    i = 0
    while i < len(segs):
        sg = segs[i]
        if _words(sg["text"]) < MIN_SEGMENT_WORDS and not sg["intentional"]:
            prev = segs[i - 1] if i else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            if prev and prev["group"] == sg["group"] and prev["pace"] == sg["pace"]:
                prev["text"] = f"{prev['text']} {sg['text']}"
                prev["pause_after_ms"] = sg["pause_after_ms"]
                segs.pop(i)
                continue
            if (
                nxt
                and nxt["group"] == sg["group"]
                and nxt["pace"] == sg["pace"]
                and not nxt["intentional"]
            ):
                nxt["text"] = f"{sg['text']} {nxt['text']}"
                segs.pop(i)
                continue
        i += 1


# ---------- listening back: did the voice say the names right? ----------

_SOUND = [(r"ph", "f"), (r"ck|q|c(?=[aou])|c$", "k"), (r"c", "s"), (r"x", "ks"), (r"[bp]", "p"),
          (r"[dt]", "t"), (r"[gk]", "k"), (r"[vfw]", "f"), (r"[sz]", "s"), (r"a", "A"),
          (r"[eiy]", "I"), (r"[ou]", "U"), (r"h", ""), (r"(.)\1+", r"\1")]  # fmt: skip


def sound_key(text: str) -> str:
    """A rough sound skeleton: 'Atreus' and 'a tray us' come out alike."""
    k = key_of(text)
    for a, b in _SOUND:
        k = re.sub(a, b, k)
    return k


def name_similarity(heard: str, t: Term) -> float:
    """How close what recognition heard is to any way the name should sound (0..1)."""
    from difflib import SequenceMatcher

    if not heard.strip():
        return 0.0
    h, hk = sound_key(heard), key_of(heard)
    refs = [t.canonical, *t.aliases, t.respell, t.phonetic, *t.heard]
    if any(hk == key_of(r) for r in t.heard if r):
        return 1.0  # the user said this is how recognition hears it when it's right

    def score(r: str) -> float:  # sound and spelling both have to agree
        return (SequenceMatcher(None, h, sound_key(r)).ratio()
                + SequenceMatcher(None, hk, key_of(r)).ratio()) / 2  # fmt: skip

    return max((score(r) for r in refs if r), default=0.0)


def align_names(
    text: str, heard: list[tuple[str, float]], terms: list[Term]
) -> list[dict[str, Any]]:
    """For each name in the script: what recognition heard in its place, and how close it
    is. heard: [(word, probability)]. Recognition also mishears correct fictional names,
    so a low score is 'check this', never 'wrong'."""
    from difflib import SequenceMatcher

    found = find_terms(text, terms)
    if not found:
        return []
    exp = re.findall(r"[\w'’-]+", text)
    exp_k = [key_of(w) for w in exp]
    heard_k = [key_of(w) for w, _ in heard]
    ops = SequenceMatcher(None, exp_k, heard_k, autojunk=False).get_opcodes()
    out = []
    for t in found:
        forms = sorted(
            {key_of(f) for f in [t.canonical, *t.aliases, *t.nicknames]}, key=len, reverse=True
        )
        for i, w in enumerate(exp_k):
            if not any(w == f or (w.startswith(f) and w[len(f) :] in ("s", "")) for f in forms):
                continue
            span = _heard_span(ops, i, heard)
            words = [heard[j][0] for j in span]
            prob = min((heard[j][1] for j in span), default=0.0)
            said = " ".join(words)
            score = round(name_similarity(said, t), 2) if said else 0.0
            status = ("ok" if score >= 0.8 else "unclear" if not words or prob < 0.35
                      else "check" if score >= 0.6 else "suspect")  # fmt: skip
            out.append({"term": t.canonical, "id": t.id, "expected": exp[i], "heard": said,
                        "probability": round(prob, 2), "score": score, "status": status,
                        "respell": t.respell, "verified": t.verified})  # fmt: skip
    return out


def _heard_span(ops: Sequence[tuple[str, int, int, int, int]], i: int,
                heard: list[tuple[str, float]]) -> list[int]:  # fmt: skip
    for tag, a0, a1, b0, b1 in ops:
        if a0 <= i < a1:
            if tag == "equal":
                return [b0 + i - a0]
            if tag == "replace":
                # a name heard as several words ("a tray us"): give it the whole replaced run
                if a1 - a0 == 1:
                    return list(range(b0, b1))
                k = b0 + round((i - a0) * (b1 - b0) / (a1 - a0))
                return [min(k, b1 - 1)] if b1 > b0 else []
            return []
    return []


def respell_variants(t: Term, n: int = 4) -> list[str]:
    """Other ways to spell how a name is said, to audition (never saved by themselves)."""
    cands = []
    if t.phonetic:
        p = t.phonetic.strip()
        cands += [p[:1].upper() + p[1:].lower(), p.replace("-", " ").title()]
    guess, _ = guess_respell(t.canonical, t.origin or "Old Norse")
    cands.append(guess)
    v = "aeiouyáéíóúöæø"
    syl = re.sub(rf"([{v}])([^{v}\s-]*?)([^{v}\s-])(?=[{v}])", r"\1\2-\3", t.canonical.lower())
    cands.append(syl.capitalize())
    if t.respell:
        cands.append(t.respell.replace("-", ""))
        cands.append(t.respell.replace("ee", "i").replace("ay", "ai"))
    seen = {key_of(t.respell)} if t.respell else set()
    out = []
    for c in cands:
        if c and key_of(c) not in seen:
            seen.add(key_of(c))
            out.append(c)
    return out[:n]


# ---------- speech-friendly text ----------

CLICHES = [
    r"little did (?:he|she|they|we|anyone) know,?",
    r"this is where things get interesting,?",
    r"what happens next will shock you,?",
    r"you won'?t believe (?:what|this),?",
    r"buckle up,?",
    r"let'?s dive (?:in|into it)[.,!]?",
    r"without further ado,?",
    r"in a world where",
    r"it'?s worth noting that",
    r"needless to say,?",
    r"at the end of the day,?",
    r"the plot thickens[.!]?",
    r"get ready,?",
]
_CONTRACT = [
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bcannot\b", "can't"),
    (r"\bcould not\b", "couldn't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bwill not\b", "won't"),
    (r"\bhave not\b", "haven't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bI am\b", "I'm"),
    (r"\blet us\b", "let's"),
    (r"\b(it|that|he|she|there|what|who) is\b(?=\s+[a-z])", r"\1's"),
    (r"\b(they|we|you) are\b(?=\s+[a-z])", r"\1're"),
]
_STOP = set(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "is",
        "was",
        "it",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "this",
        "that",
        "with",
        "for",
        "from",
        "his",
        "her",
        "their",
        "its",
        "be",
        "are",
        "were",
    ]
)


def optimize_script(text: str, lang: str = "en") -> tuple[str, list[str]]:
    """Rewrites a line for the ear: no clichés, contractions, no run-on sentences."""
    notes: list[str] = []
    if lang != "en":
        return clean_spoken(text), notes
    t = clean_spoken(text)
    for c in CLICHES:
        if re.search(c, t, re.I):
            t = re.sub(rf"\s*{c}\s*", " ", t, flags=re.I).strip()
            notes.append("removed a cliché")
    for a, b in _CONTRACT:
        t2 = re.sub(a, b, t, flags=re.I if not a.startswith(r"\b(") else 0)
        if t2 != t:
            t = t2
            if "contractions" not in notes:
                notes.append("contractions")
    sentences = re.split(r"(?<=[.!?])\s+", t)
    out = []
    for s in sentences:
        if len(s.split()) > 22:
            s2 = re.sub(
                r",\s+(and|but|so|because|while|which)\s+",
                lambda m: f". {m.group(1).capitalize()} ",
                s,
                count=1,
            )
            if s2 != s:
                notes.append("split a long sentence")
            s = s2
        out.append(s)
    t = re.sub(r"\s+", " ", " ".join(out)).strip()
    t = t[:1].upper() + t[1:] if t else t
    if t and t[-1] not in ".!?…":
        t += "."
    words = [w for w in re.findall(r"[a-z']{5,}", t.lower()) if w not in _STOP]
    rep = [w for w, n in ((w, words.count(w)) for w in set(words)) if n >= 3]
    if rep:
        notes.append(f"repeats: {', '.join(sorted(rep))}")
    return t, notes


_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORD = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_ABBR = {
    r"\bvs\.?(?=\s)": "versus",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\betc\.": "and so on",
    r"&": " and ",
    r"%": " percent",
    r"\bft\.": "featuring",
    r"\bHP\b": "health",
    r"\bw/": "with ",
    r"\bmins?\b": "minutes",
    r"\bhrs?\b": "hours",
}


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + number_words(rest))
    if n < 1_000_000:
        rest = n % 1000
        return (
            number_words(n // 1000)
            + " thousand"
            + ("" if rest == 0 else (" and " if rest < 100 else " ") + number_words(rest))
        )
    return str(n)


def speakable(text: str, keep: set[str] | None = None) -> str:
    """TTS-only rewrite of what engines misread: numbers, years, ordinals, symbols and
    acronyms. `keep`: names from the dictionary that must stay as they are."""
    keep = keep or set()
    t = text
    for a, b in _ABBR.items():
        t = re.sub(a, b, t)

    def year(m: re.Match[str]) -> str:
        y = int(m.group(0))
        hi, lo = divmod(y, 100)
        if lo == 0:
            return number_words(hi) + " hundred"
        return number_words(hi) + " " + ("oh " + _ONES[lo] if lo < 10 else number_words(lo))

    t = re.sub(
        r"\b([A-Z]{2,4})(\d?)\b",
        lambda m: (
            m.group(0)
            if m.group(1) in keep or m.group(1).capitalize() in keep
            else " ".join(m.group(1)) + (" " + m.group(2) if m.group(2) else "")
        ),
        t,
    )
    t = re.sub(r"(?<![\d.,])(1[1-9]\d\d|20[0-9]\d)(?![\d%]|[.,]\d)", year, t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: _ordinal(int(m.group(1))), t)
    t = re.sub(
        r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d.])",
        lambda m: number_words(int(m.group(0).replace(",", ""))),
        t,
    )
    t = re.sub(r"(?<![\d.])\d+(?![\d.])", lambda m: number_words(int(m.group(0))), t)
    return re.sub(r"\s+", " ", t).strip()


def _ordinal(n: int) -> str:
    w = number_words(n)
    last = re.split(r"[ -]", w)[-1]
    if last in _ORD:
        return w[: -len(last)] + _ORD[last]
    if last.endswith("y"):
        return w[:-1] + "ieth"
    return w + "th"


# ---------- scenes and personalities ----------

SCENES: dict[str, dict[str, Any]] = {
    "emotional": {
        "mood": "sad",
        "pace": "slow",
        "guide": "Restrained and slower. Let pauses land. No hype.",
    },
    "epic": {
        "mood": "epic",
        "pace": "normal",
        "guide": "Build intensity gradually; emphasise one or two words, never every sentence.",
    },
    "lore": {
        "mood": "warm",
        "pace": "normal",
        "guide": "Curious, confident storytelling. Explain like a friend who "
        "knows the lore, not a textbook.",
    },
    "suspense": {
        "mood": "tense",
        "pace": "slow",
        "guide": "Build tension step by step, use pauses, don't give the payoff away early.",
    },
    "funny": {
        "mood": "playful",
        "pace": "normal",
        "guide": "Conversational, leave room for the punchline, never force a joke.",
    },
    "action": {"mood": "urgent", "pace": "fast", "guide": "Short sentences, urgency, stay clear."},
    "reveal": {
        "mood": "mysterious",
        "pace": "slow",
        "guide": "Set it up, pause, then the reveal in a short sentence.",
    },
    "calm": {
        "mood": "calm",
        "pace": "normal",
        "guide": "Relaxed and natural, like exploring with a friend.",
    },
}


def scene_type(j: Any, format_label: str = "") -> str:
    """The kind of scene a moment is, from what the judge saw."""
    f = format_label.lower()
    if any(w in f for w in ("lore", "fact", "myth")):
        return "lore"
    if j is None:
        return "calm"
    text = " ".join(
        [
            str(getattr(j, "what_happens", "")),
            " ".join(getattr(j, "tags", ()) or ()),
            " ".join(getattr(j, "formats", ()) or ()),
        ]
    ).lower()
    emotion, intensity = int(getattr(j, "emotion", 0) or 0), int(getattr(j, "intensity", 0) or 0)
    if re.search(r"funny|meme|joke|roast|awkward|banter", text):
        return "funny"
    if re.search(r"reveal|twist|betray|unexpected|surpris", text):
        return "reveal"
    if re.search(r"boss|combat|fight|battle|finisher", text) and intensity >= 6:
        return "epic"
    if emotion >= 7 and intensity < 7:
        return "emotional"
    if re.search(r"mystery|tense|suspense|horror|dark", text):
        return "suspense"
    if intensity >= 6:
        return "action"
    return "calm"


@dataclass(frozen=True, slots=True)
class Personality:
    id: str
    name: str
    voice: str  # who speaks: the base the Qwen voice is designed from
    pace: float  # 1.0 = normal speed
    intensity: float  # 0..1, how far moods are pushed
    pitch: str  # low, mid, high (designed voices only)
    pauses: float  # pause length multiplier
    emphasis: str
    suits: str
    writer: str  # how the script should read


PERSONALITIES: dict[str, Personality] = {
    p.id: p
    for p in [
        Personality(
            "cinematic",
            "Cinematic Storyteller",
            "A deep, calm male storyteller, warm and cinematic, measured pace, natural "
            "breathing, quietly intense",
            0.95,
            0.6,
            "low",
            1.2,
            "selective",
            "story recaps, cinematics, lore",
            "Measured and cinematic. Short sentences with weight. Let moments breathe.",
        ),
        Personality(
            "hype",
            "Gaming Hype",
            "A young male gaming YouTuber, deep confident voice, natural and conversational, "
            "energetic, with dramatic pauses",
            1.05,
            0.8,
            "mid",
            0.85,
            "strong",
            "boss fights, highlights, Shorts",
            "Energetic and confident, like a creator reacting live. Punchy, never shouty.",
        ),
        Personality(
            "lore",
            "Lore Expert",
            "A curious, intelligent male narrator, conversational and engaging, clear diction, a "
            "hint of wonder",
            1.0,
            0.5,
            "mid",
            1.0,
            "selective",
            "lore explained, facts, myth vs game",
            "Conversational expert. Hook with a question, explain naturally, end on a reveal.",
        ),
        Personality(
            "dark",
            "Dark Narrator",
            "A low, atmospheric male voice, mysterious and restrained, slow and deliberate, "
            "close to the microphone",
            0.9,
            0.55,
            "low",
            1.4,
            "restrained",
            "mysteries, foreshadowing, horror",
            "Atmospheric and restrained. Fewer words, longer silences, tension over volume.",
        ),
        Personality(
            "casual",
            "Casual Creator",
            "A relaxed young male creator talking to friends, natural, friendly, slightly amused",
            1.03,
            0.45,
            "mid",
            0.9,
            "light",
            "gameplay commentary, funny moments",
            "Relaxed and human. Contractions, asides, the way you'd talk on a stream.",
        ),
        Personality(
            "emotional",
            "Emotional Storyteller",
            "A gentle, expressive male narrator, sensitive and emotionally nuanced, soft and "
            "sincere",
            0.93,
            0.65,
            "mid",
            1.3,
            "gentle",
            "tragic moments, character arcs",
            "Sincere and gentle. Name the feeling once, then let the scene carry it.",
        ),
        Personality(
            "documentary",
            "Documentary",
            "A clear, measured male documentary narrator, neutral and informative, steady pace",
            0.98,
            0.35,
            "mid",
            1.0,
            "minimal",
            "explainers, history, comparisons",
            "Clear and informative. Facts in plain words, one idea per sentence.",
        ),
        Personality(
            "custom",
            "Custom",
            "A natural male narrator with a clear voice",
            1.0,
            0.5,
            "mid",
            1.0,
            "selective",
            "anything",
            "Natural and engaging.",
        ),
    ]
}
VOICE_PERSONALITY = {"qwen_gamer": "hype", "qwen_story": "cinematic"}


def personality(pid: str, custom: dict[str, Any] | None = None) -> Personality:
    p = PERSONALITIES.get(pid) or PERSONALITIES["hype"]
    if p.id == "custom" and custom:
        return Personality(
            "custom",
            "Custom",
            str(custom.get("voice") or p.voice)[:300],
            float(custom.get("pace", p.pace)),
            float(custom.get("intensity", p.intensity)),
            str(custom.get("pitch") or p.pitch),
            float(custom.get("pauses", p.pauses)),
            str(custom.get("emphasis") or p.emphasis),
            p.suits,
            str(custom.get("writer") or p.writer)[:300],
        )
    return p


def design_for(p: Personality, mood: str) -> tuple[str, str]:
    """(voice description, reference line) for a Qwen reference voice in this mood group."""
    group = mood if mood in GROUP_DELIVERY else MOOD_GROUP.get(mood, "neutral")
    delivery, line = GROUP_DELIVERY[group]
    pitch = {"low": "low-pitched", "high": "higher-pitched"}.get(p.pitch, "")
    return f"{p.voice}{', ' + pitch if pitch else ''}. Speaking {delivery}.", line


# ---------- the writer's brief ----------

BANNED = [
    "insane",
    "epic",
    "legendary",
    "ultimate",
    "unleash",
    "you won't believe",
    "showdown",
    "chaos",
    "little did",
    "buckle up",
    "what happens next",
]


def writer_brief(
    game_name: str,
    scene: str,
    p: Personality,
    terms: list[Term],
    emotion_tags: bool,
    seconds: float,
) -> str:
    """Extra rules for the script writer: scene, personality, names, speech style."""
    s = SCENES.get(scene, SCENES["calm"])
    names = ", ".join(
        f"{t.canonical}" + (f" (also {', '.join(t.aliases[:2])})" if t.aliases else "")
        for t in terms[:20]
    )
    parts = [
        f"Voice: {p.name}. {p.writer}",
        f"Scene type: {scene}. Delivery: {s['guide']}",
        f"Spell these {game_name} names exactly like this: {names}." if names else "",
        "Write for the ear: contractions, varied sentence length, mostly short sentences, "
        "no lists, no stage directions in the text, no hashtags or emoji. Read it aloud in "
        "your head; rewrite anything that sounds written.",
        f"Never use: {', '.join(BANNED)}.",
        f"About {round(seconds)} seconds of speech in total; don't describe every visual.",
        "If unsure of a name, say who they are instead of guessing a name.",
        f"For each line give a mood ({', '.join(MOODS)}) and a pace (slow, normal, fast) "
        "that fit the scene; vary them, don't make everything excited."
        if emotion_tags
        else "",
    ]
    return "\n".join(x for x in parts if x)


def plan_hash(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]
