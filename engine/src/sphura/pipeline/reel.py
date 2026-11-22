"""AI highlight reel: the best moments of many videos, trimmed, deduplicated and ordered
into one edit.

Each source video is analysed on its own by the normal pipeline (signals, transcript, AI
judge). This module only works on the resulting moments: it scores them on a shared scale,
drops near-duplicate events, picks the strongest set that fits the target length (quality
first: a shorter reel beats a padded one) and orders it for pacing: a strong hook first,
variety in the middle, the best payoff last. An LLM, when available, may re-arrange the
picks for storytelling; its answer is validated and the rules above are the fallback.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from sphura.pipeline.candidates import FINALE
from sphura.pipeline.copywriter import is_generic
from sphura.pipeline.judge import guard_claims
from sphura.pipeline.ranking import RankedHighlight
from sphura.ports.ai import LLMProvider, LLMRequest

log = logging.getLogger(__name__)

MIN_CLIP_S = 3.0
QUALITY_FLOOR = 40  # moments under this only fill in when there are almost none
HOOK_KINDS = ("action", "clutch", "unexpected", "funny")
_TIER_W = {"top": 1.0, "strong": 0.75, "maybe": 0.4}
_STOP = set(
    [
        "the",
        "and",
        "with",
        "that",
        "this",
        "from",
        "into",
        "onto",
        "over",
        "under",
        "then",
        "they",
        "them",
        "their",
        "there",
        "what",
        "when",
        "where",
        "while",
        "which",
        "who",
        "will",
        "would",
        "could",
        "should",
        "about",
        "after",
        "before",
        "again",
        "against",
        "being",
        "character",
        "player",
        "players",
        "scene",
        "moment",
        "shows",
        "shown",
        "screen",
        "game",
        "gameplay",
        "video",
        "clip",
        "during",
        "between",
        "some",
        "very",
        "more",
        "most",
        "just",
        "also",
    ]
)


@dataclass(slots=True)
class Moment:
    id: str
    source_id: str
    project_id: str
    asset_id: str
    source_name: str
    start_s: float
    end_s: float
    peak_s: float
    score: int  # engagement potential, 0..100, comparable across videos
    confidence: float  # 0..1: how sure the judge was
    title: str
    what: str  # what happens
    why: str  # why it is interesting
    hook: str  # suggested opening line / caption
    category: str = ""
    tags: list[str] = field(default_factory=list)
    tier: str | None = None
    judged: bool = False
    finale: bool = False
    dup_of: str | None = None  # id of the stronger moment showing the same event

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Moment:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def moment_from(r: RankedHighlight, source_id: str, source_name: str) -> Moment:
    c, j = r.candidate, r.candidate.judgement
    span = c.clip_range
    score, conf = engagement(r)
    title = (j.title if j and j.title else "") or f"Moment at {_hms(c.peak_s)}"
    return Moment(
        id=c.id, source_id=source_id, project_id=c.project_id, asset_id=c.asset_id,
        source_name=source_name, start_s=round(span.start_s, 2), end_s=round(span.end_s, 2),
        peak_s=round(c.peak_s, 2), score=score, confidence=conf, title=title,
        what=j.what_happens if j else "Loud, busy stretch picked from the audio and motion",
        why=j.reason if j else "", hook=j.hook if j else "",
        category=j.category.value if j and j.category else "",
        tags=list(j.tags) if j else [], tier=r.tier.value if r.tier else None,
        judged=j is not None, finale=FINALE in c.drivers,
    )  # fmt: skip


def engagement(r: RankedHighlight) -> tuple[int, float]:
    """Engagement potential on one scale for every video: the AI's quality call weighs
    most, then the moment's tier, its signal peak and how it fared in side-by-sides."""
    c, j = r.candidate, r.candidate.judgement
    sig = float(c.signal_score)
    cmp_ = float(c.compare_score) if c.compare_score is not None else sig
    tier = _TIER_W.get(r.tier.value, 0.3) if r.tier else 0.3
    if j is not None:
        s = 0.45 * j.quality / 10 + 0.25 * tier + 0.15 * sig + 0.15 * cmp_
        s += 0.05 if j.standalone else 0.0
        s *= 1.0 if j.is_highlight else 0.55
        conf = float(j.confidence)
    else:  # not reviewed by the AI: signals only, and it shows in the score
        s = (0.5 * sig + 0.3 * cmp_ + 0.2 * tier) * 0.7
        conf = 0.15 + 0.35 * sig
    if FINALE in c.drivers:
        s += 0.06
    return round(100 * max(0.0, min(1.0, s))), round(max(0.0, min(1.0, conf)), 2)


# ---------- duplicates ----------


def _words(m: Moment) -> set[str]:
    text = f"{m.title} {m.what}".lower()
    return {w for w in re.findall(r"[a-z0-9']+", text) if len(w) > 3 and w not in _STOP}


def same_event(a: Moment, b: Moment) -> bool:
    if a.source_id == b.source_id:
        overlap = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
        return bool(overlap > 0.3 * min(a.length_s, b.length_s) or abs(a.peak_s - b.peak_s) < 6)
    if not (a.judged and b.judged):  # only the AI's own descriptions say what happened
        return False
    if a.category and b.category and a.category != b.category:
        return False
    wa, wb = _words(a), _words(b)
    shared = len(wa & wb)
    return shared >= 4 and shared / len(wa | wb) >= 0.55  # one event described alike twice


def mark_duplicates(moments: list[Moment]) -> list[Moment]:
    """Strongest first; a weaker moment showing the same event points at the stronger one."""
    ordered = sorted(moments, key=lambda m: (-m.score, -m.confidence))
    kept: list[Moment] = []
    for m in ordered:
        m.dup_of = next((k.id for k in kept if same_event(k, m)), None)
        if m.dup_of is None:
            kept.append(m)
    return ordered


# ---------- trimming ----------


def clip_cap(target_s: float) -> float:
    """Longest a single moment may run in a reel of this length: short reels need
    snappy clips, long ones can let a moment breathe."""
    for limit, cap in ((30, 7.0), (60, 11.0), (90, 14.0), (180, 20.0)):
        if target_s <= limit:
            return cap
    return 25.0


def trim(m: Moment, cap: float) -> tuple[float, float]:
    """The strongest stretch of the moment, at most cap long: around the peak with a
    little more build-up than tail, and always room for the payoff to land. Fight
    finales keep their ending."""
    if m.length_s <= cap:
        return m.start_s, m.end_s
    if m.finale:
        return round(m.end_s - cap, 2), m.end_s
    a = m.peak_s - 0.6 * cap
    a = min(a, m.peak_s + 1.5 - cap)  # at least 1.5 s after the peak
    a = max(m.start_s, min(a, m.end_s - cap))
    return round(a, 2), round(a + cap, 2)


# ---------- selection and order ----------


@dataclass(slots=True)
class Pick:
    moment_id: str
    start_s: float
    end_s: float
    include: bool = True

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select(
    moments: list[Moment], target_s: float, seed: int = 0, jitter: float = 0.0
) -> list[Pick]:
    """Best set of distinct moments that fits target_s, favouring variety across videos
    and kinds of moment. jitter > 0 (regenerate) explores other good combinations."""
    rng = random.Random(seed)
    noise = {m.id: rng.uniform(-jitter, jitter) for m in moments}
    pool = [m for m in moments if m.dup_of is None]
    cap = clip_cap(target_s)
    chosen: list[Moment] = []
    picks: dict[str, Pick] = {}
    total = 0.0
    skipped: set[str] = set()

    def value(m: Moment) -> float:
        same_src = sum(c.source_id == m.source_id for c in chosen)
        same_kind = sum(bool(m.category) and c.category == m.category for c in chosen)
        return m.score + noise[m.id] - 7 * same_src - 4 * same_kind

    while target_s - total >= MIN_CLIP_S:
        floor = QUALITY_FLOOR if len(chosen) >= 2 else 0  # quality over quantity
        options = [
            m for m in pool if m.id not in picks and m.id not in skipped and m.score >= floor
        ]
        if not options:
            break
        m = max(options, key=value)
        a, b = trim(m, cap)
        room = target_s - total
        if b - a > room:  # shorten around the peak to fill the last gap
            a = max(m.start_s, min(m.peak_s - 0.6 * room, m.end_s - room))
            b = a + room
        if b - a < MIN_CLIP_S:
            skipped.add(m.id)
            continue
        chosen.append(m)
        picks[m.id] = Pick(m.id, round(a, 2), round(b, 2))
        total += b - a
    # The same variation reshapes the order, so regenerating a small pool still changes
    by_id = {m.id: replace(m, score=round(m.score + noise[m.id])) for m in moments}
    return order([picks[m.id] for m in chosen], by_id)


def order(picks: list[Pick], by_id: dict[str, Moment]) -> list[Pick]:
    """Pacing: open on a punchy hook (the second-best moment), build up through varied
    moments, end on the best payoff. Neighbours avoid the same video and the same kind."""
    if len(picks) <= 1:
        return picks
    score = {p.moment_id: by_id[p.moment_id].score for p in picks}
    closer = max(picks, key=lambda p: score[p.moment_id])
    rest = [p for p in picks if p is not closer]

    def hook_value(p: Pick) -> float:
        m = by_id[p.moment_id]
        return m.score + (6 if m.category in HOOK_KINDS else 0) + (4 if p.length_s <= 8 else 0)

    opener = max(rest, key=hook_value)
    middle = sorted((p for p in rest if p is not opener), key=lambda p: score[p.moment_id])
    seq = [opener]
    while middle:
        prev = by_id[seq[-1].moment_id]

        def clash(p: Pick, prev: Moment = prev) -> int:
            m = by_id[p.moment_id]
            return (m.source_id == prev.source_id) * 2 + (
                bool(m.category) and m.category == prev.category
            )

        nxt = min(middle, key=lambda p: (clash(p), score[p.moment_id]))
        middle.remove(nxt)
        seq.append(nxt)
    return [*seq, closer]


STORY_SYSTEM = """You are the editor of a gaming highlight reel. From a list of detected
moments, choose the ones to use and the order they play in.
Rules:
- Open with a strong, instantly readable hook moment.
- Build momentum; end on the most satisfying payoff.
- Keep variety: avoid the same kind of moment or the same video back to back.
- Never use two moments that show essentially the same event.
- Quality over quantity: leave weak moments out even if there is time left.
- The total length must not go over the target."""

STORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "string"},
                  "description": "moment ids (like m3) in playing order"},
        "note": {"type": "string", "description": "one sentence on the arc you built"},
    },
    "required": ["order", "note"],
}  # fmt: skip


def story_order(
    llm: LLMProvider | None, moments: list[Moment], target_s: float, avoid: list[str] | None = None
) -> tuple[list[Pick], str] | None:
    """Let the AI pick and arrange the reel. None when it's unavailable or its answer
    doesn't hold up (unknown ids, duplicates, too long, too few)."""
    pool = [m for m in moments if m.dup_of is None][:24]
    if llm is None or len(pool) < 3:
        return None
    cap = clip_cap(target_s)
    keys = {f"m{i + 1}": m for i, m in enumerate(pool)}
    spans = {k: trim(m, cap) for k, m in keys.items()}
    lines = [
        f"{k}: {spans[k][1] - spans[k][0]:.0f}s, {m.category or 'moment'}, score {m.score}, "
        f'video "{m.source_name}": {m.title}. {m.what}'
        for k, m in keys.items()
    ]
    prompt = f"Target length: about {target_s:.0f} seconds.\n\nMoments:\n" + "\n".join(lines)
    if avoid:
        prompt += "\n\nGive a DIFFERENT selection or order than last time, which was: " + ", ".join(
            k for k, m in keys.items() if m.id in avoid
        )
    try:
        resp = llm.complete(
            LLMRequest(system=STORY_SYSTEM, prompt=prompt, json_schema=STORY_SCHEMA, max_tokens=400)
        )
        raw = json.loads(resp.text)
        ids = [str(x).strip() for x in raw.get("order", [])]
    except Exception as e:
        log.warning("AI could not arrange the reel, using the built-in order: %s", e)
        return None
    seen: list[str] = []
    for k in ids:
        if k in keys and k not in seen and not any(same_event(keys[k], keys[s]) for s in seen):
            seen.append(k)
    total = sum(spans[k][1] - spans[k][0] for k in seen)
    while seen and total > target_s * 1.1:  # trim the weakest until it fits
        weakest = min(seen[1:-1] or seen, key=lambda k: keys[k].score)
        seen.remove(weakest)
        total -= spans[weakest][1] - spans[weakest][0]
    if len(seen) < 2 or total < min(target_s * 0.4, 12):
        return None
    note = str(raw.get("note", "")).strip()[:200]
    return [Pick(keys[k].id, *spans[k]) for k in seen], note


# ---------- beat-snapped cut points ----------


def beat_trims(
    lengths: list[float], overlaps: list[float], beat: float, keep_s: float = 2.5
) -> list[float]:
    """Seconds to cut from the end of each clip so every cut lands on a beat (never
    more than one beat, never below keep_s). The last clip ends with the music fade."""
    out: list[float] = []
    t = 0.0
    for i, length in enumerate(lengths):
        end = t + length - (overlaps[i] if i < len(overlaps) else 0.0)
        cut = 0.0
        if i < len(lengths) - 1 and beat > 0.2:
            cut = end - (end // beat) * beat
            if length - cut < keep_s:
                cut = 0.0
        out.append(round(cut, 3))
        t = end - cut
    return out


# ---------- copy ----------

COPY_SYSTEM = """You write the post copy for a gaming highlight compilation.
You get the moments in the order they play. Write about THESE moments specifically: the
enemies, moves, places and reactions listed. Never invent events, numbers or names.
Banned: generic hype like "epic", "insane", "legendary", "you won't believe", "ultimate"."""

COPY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "<=70 chars, specific, no hashtags"},
        "hook": {"type": "string", "description": "<=8 words shown on screen at the start, "
                 "makes people stay"},
        "description": {"type": "string", "description": "2-3 sentences naming the actual "
                        "highlights, ending with a question to the viewer"},
        "thumbnail_text": {"type": "string", "description": "<=4 words for the thumbnail"},
        "thumbnail_moment": {"type": "string", "description": "id of the moment whose frame "
                             "makes the best thumbnail"},
        "tags": {"type": "array", "items": {"type": "string"},
                 "description": "4-6 topic hashtags without '#', lowercase, no spaces"},
    },
    "required": ["title", "hook", "description", "thumbnail_text", "thumbnail_moment", "tags"],
}  # fmt: skip

PLATFORM_TAGS = {
    "shorts": "shorts",
    "reels": "reels",
    "tiktok": "fyp",
    "landscape": "",
    "square": "",
}


def write_reel_copy(
    llm: LLMProvider | None, moments: list[Moment], game: str, platform: str
) -> dict[str, Any]:
    """Title, description, hashtags, on-screen hook and thumbnail pick for the reel."""
    if not moments:
        raise ValueError("no moments")
    best = max(moments, key=lambda m: m.score)
    copy: dict[str, Any] = {
        "title": (f"{len(moments)} best moments" + (f": {game}" if game else ""))[:70],
        "hook": best.hook or "Wait for the last one",
        "description": "In this one: "
        + "; ".join(m.title for m in moments[:5])
        + ". Which moment was the best?",
        "thumbnail_text": "",
        "thumbnail_moment": best.id,
        "tags": [],
    }
    if llm is not None:
        keys = {f"m{i + 1}": m for i, m in enumerate(moments)}
        lines = [f"{k}: {m.title}. {m.what}" + (f" Why: {m.why}" if m.why else "")
                 for k, m in keys.items()]  # fmt: skip
        prompt = (f"Game: {game or 'unknown (do not name it)'}\n"
                  "Moments in playing order:\n" + "\n".join(lines))  # fmt: skip
        try:
            resp = llm.complete(LLMRequest(system=COPY_SYSTEM, prompt=prompt,
                                           json_schema=COPY_SCHEMA, max_tokens=500))  # fmt: skip
            raw = json.loads(resp.text)
            for k in ("title", "hook", "description", "thumbnail_text"):
                text = re.sub(r"#\w+", "", str(raw.get(k, ""))).strip().strip('"')
                if text and guard_claims(text) and not (k != "description" and is_generic(text)):
                    copy[k] = text
            if str(raw.get("thumbnail_moment")) in keys:
                copy["thumbnail_moment"] = keys[str(raw["thumbnail_moment"])].id
            copy["tags"] = [re.sub(r"[^a-z0-9]", "", str(t).lower()) for t in raw.get("tags", [])]
        except Exception as e:
            log.warning("could not write reel copy with %s: %s", llm.info.name, e)
    game_tag = re.sub(r"[^a-z0-9]", "", game.lower()) if game else ""
    kinds = [m.category for m in moments if m.category]
    tags = [t for t in [game_tag, *copy["tags"], *dict.fromkeys(kinds), "highlights", "gaming",
                        PLATFORM_TAGS.get(platform, "")] if t]  # fmt: skip
    copy["tags"] = list(dict.fromkeys(tags))[:10]
    copy["hashtags"] = " ".join(f"#{t}" for t in copy["tags"])
    copy["youtube_title"] = (copy["title"].rstrip(" .") + (f" - {game}" if game and game.lower()
                             not in copy["title"].lower() else ""))[:100]  # fmt: skip
    return copy


def _hms(t: float) -> str:
    t = int(max(0, t))
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
