"""Edit planner: turns a story (or one strong moment) into a Short of a fitting format,
built from what actually happens in the footage, in up to three genuinely different cuts:
A cinematic story, B high-impact, C original audio. Every plan says why it was made."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from sphura.domain.models import AIJudgement, HighlightCandidate, TimeRange
from sphura.domain.transcript import Transcript, TranscriptSegment
from sphura.pipeline.clips import remove_dead_time
from sphura.pipeline.stories import Story
from sphura.pipeline.timeline import SignalTimeline


@dataclass(frozen=True, slots=True)
class Format:
    id: str
    label: str
    structure: str  # how the Short is built, shown to the user
    variants: tuple[str, ...]  # best first


FORMATS: dict[str, Format] = {
    f.id: f
    for f in (
        Format("story", "Cinematic story", "strong opening, brief context, the original "
               "dialogue, escalation, emotional payoff", ("A", "C", "B")),
        Format("combat", "High-impact combat", "opens on the best hit, enough context to "
               "follow the fight, builds, ends on the strongest moment", ("B", "A", "C")),
        Format("boss", "Boss fight story", "introduces the threat, shows the danger, the "
               "fight's progress, then the outcome", ("B", "A", "C")),
        Format("unexpected", "Unexpected moment", "sets up the situation, builds curiosity, "
               "keeps the game's own audio, then the payoff", ("C", "B")),
        Format("lore", "Lore and story context", "the meaningful scene and its dialogue, only "
               "the context that adds value, ends on the implication", ("A", "C")),
        Format("discovery", "Gameplay discovery", "shows the result or problem, then how it "
               "works, ends on the useful takeaway", ("C", "A")),
    )
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Variant:
    id: str
    label: str
    style: str  # the existing effect style it starts from
    music: str  # auto | none
    max_s: float
    about: str


VARIANTS: dict[str, Variant] = {
    "A": Variant("A", "Cinematic", "cinematic", "auto", 60.0,
                 "the same beat with a soft score low under the voices; no effects"),
    "B": Variant("B", "Edit", "hype", "auto", 25.0,
                 "opens on the strongest instant, music drives it, a few impact accents"),
    "C": Variant("C", "Clean", "hype", "none", 60.0,
                 "one continuous beat with the game's own sound: no music, no effects, "
                 "like the top Shorts of this game"),
}  # fmt: skip

# Game priorities from the brief: GoW leans on story and dialogue, Phantom Blade on combat
GAME_BIAS = {
    "gow": {"story": 0.6, "lore": 0.5, "boss": 0.4, "unexpected": 0.4, "combat": 0.1},
    "pbz": {"combat": 0.6, "boss": 0.5, "unexpected": 0.3, "discovery": 0.2, "story": 0.1},
}
PIECE_S = {"A": (10.0, 8.0, 14.0), "B": (5.0, 5.0, 9.0), "C": (10.0, 0.0, 14.0)}
PLAN_VERSION = 1
MAX_PER_FORMAT = 2
CONTEXT_S = 4.0  # cinematic cuts start a little earlier, so viewers know where they are
LINE_GRACE_S = 4.0  # stretch a piece this far to avoid cutting someone off mid-sentence


@dataclass(slots=True)
class Plan:
    key: str  # story id or moment id
    format: str
    moments: list[str]
    title: str
    reason: str
    variants: list[str]
    score: float
    names: list[str] = field(default_factory=list)
    beat: TimeRange | None = None  # one continuous range: no stitching
    kind: str = ""  # dialogue | combat


@dataclass(slots=True)
class Cut:
    segments: list[TimeRange]
    key_s: float  # source time of the payoff, for effects and music
    structure: list[dict[str, Any]]  # what each piece is and why


DIALOGUE_GAP_S = 3.0  # lines closer than this belong to one exchange
REACTION_S = 2.0  # hold after the last line so the reaction lands
BEAT_MIN_S, BEAT_MAX_S = 8.0, 60.0
COMBAT_S = {"pbz": (10.0, 22.0), "gow": (14.0, 28.0)}
REACH_S = 20.0  # a beat may run past its moment's window to finish the exchange
PUNCHLINE_GAP_S = 8.0
LONG_LINE_S = 9.0  # longer than this is a transcription merge, not one spoken line


def beat_plans(
    moments: list[HighlightCandidate],
    transcript: Transcript | None,
    tl: SignalTimeline | None,
    game: str,
    limit: int = 5,
) -> list[Plan]:
    """What the top Shorts of these games are: ONE complete beat. For a scene, from the
    line that opens the exchange to its punchline and the reaction; for a fight, the
    densest stretch ending on the finisher. Never pieces glued together."""
    lines = [s for s in (transcript.segments if transcript else []) if s.text.strip()]
    found: list[Plan] = []
    for c in moments:
        j = c.judgement
        if j is None or not j.is_highlight or "routine" in j.tags:
            continue
        talky = j.dialogue_matters or bool({"story", "lore", "unexpected"} & set(j.formats))
        if talky and j.intensity >= 6 and "cutscene" not in j.tags and "funny" not in j.tags:
            talky = False  # chatter during a fight: the fight is the beat, if anything
        beat = _dialogue_beat(c, lines) if talky else None
        kind = "dialogue"
        if beat is None:
            beat, kind = _combat_beat(c, tl, game), "combat"
        if beat is None:
            continue
        fmt = pick_format(list(j.formats), game, [j])
        if kind == "combat" and fmt in ("story", "lore"):
            fmt = "boss" if "boss" in j.formats else "combat"
        score = _beat_score(j, kind, game)
        variants = ["C", "A"] if kind == "dialogue" else ["C", "B"]
        why = f"{j.what_happens} One continuous {kind} beat, {beat.duration_s:.0f}s."
        found.append(Plan(c.id, fmt, [c.id], j.title, why, variants, score, list(j.names),
                          beat, kind))  # fmt: skip
    found.sort(key=lambda p: -p.score)
    # One fight per boss: its last moment, where the finisher is; earlier ones are retries
    last: dict[str, float] = {}
    for p in found:
        if p.kind == "combat" and p.beat:
            for n in p.names:
                last[n] = max(last.get(n, 0.0), p.beat.start_s)
    found = [p for p in found if not (p.kind == "combat" and p.beat and any(
        last.get(n, 0.0) > p.beat.start_s for n in p.names))]  # fmt: skip
    out: list[Plan] = []
    for p in found:  # the same beat found from two neighbouring moments: keep one
        if p.beat and any(q.beat and _overlap(p.beat, q.beat) > 0.3 for q in out):
            continue
        out.append(p)
    return _diverse(out)[:limit]


def _beat_score(j: AIJudgement, kind: str, game: str) -> float:
    s = float(j.quality) + (1.0 if j.payoff_seen else 0.0)
    if kind == "dialogue":
        s += 0.4 * j.emotion + (1.5 if "funny" in j.tags or "unexpected" in j.formats else 0.0)
        s += 0.8 if game == "gow" else 0.0  # GoW's top Shorts are cutscene beats
    else:
        s += 0.3 * j.intensity + (0.8 if game == "pbz" else 0.0)
    return s


def _overlap(a: TimeRange, b: TimeRange) -> float:
    inter = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    return max(0.0, inter) / max(1e-6, min(a.duration_s, b.duration_s))


def _dialogue_beat(c: HighlightCandidate, lines: list[TranscriptSegment]) -> TimeRange | None:
    lo, hi = c.range.start_s - REACH_S, c.range.end_s + REACH_S
    near = [s for s in lines if lo <= s.start_s <= hi]
    if not near:
        return None
    exchanges: list[list[TranscriptSegment]] = [[near[0]]]
    for s in near[1:]:
        if s.start_s - exchanges[-1][-1].end_s <= DIALOGUE_GAP_S:
            exchanges[-1].append(s)
        else:
            exchanges.append([s])
    j = c.judgement
    anchor = j.payoff_s if j and j.payoff_s is not None else c.peak_s

    def fit(ex: list[TranscriptSegment]) -> float:
        inside = min(ex[-1].end_s, c.range.end_s) - max(ex[0].start_s, c.range.start_s)
        return max(0.0, inside) - 0.2 * min(abs(ex[-1].end_s - anchor), abs(ex[0].start_s - anchor))

    k = max(range(len(exchanges)), key=lambda i: fit(exchanges[i]))
    ex = list(exchanges[k])
    if fit(ex) <= 0:
        return None
    # A punchline often comes after the action, a few seconds past the exchange
    nxt = exchanges[k + 1] if k + 1 < len(exchanges) else []
    if nxt and nxt[0].start_s - ex[-1].end_s <= PUNCHLINE_GAP_S:
        ex.append(nxt[0])
    # Open on a line the game actually shows: misheard or merged lines (music, crowd)
    # make a dull, wrongly captioned opening
    read = [t for _, t in (j.dialogue if j else ())]
    if read:
        shown = [i for i, s in enumerate(ex) if _match(s.text, [("", t) for t in read])[0] != ""
                 or any(SequenceMatcher(None, s.text.lower(), t.lower()).ratio() >= 0.55
                        for t in read)]  # fmt: skip
        if shown:
            ex = ex[shown[0] :]
    ex = [s for s in ex if s.end_s - s.start_s <= LONG_LINE_S] or ex
    start = ex[0].start_s - 0.8
    end = max(ex[-1].end_s, anchor if ex[0].start_s <= anchor <= ex[-1].end_s + 6 else 0.0)
    end += REACTION_S
    if end - start > BEAT_MAX_S:  # keep the punchline: drop opening lines instead
        start = next(
            (s.start_s - 0.5 for s in ex if end - s.start_s <= BEAT_MAX_S), end - BEAT_MAX_S
        )
    if end - start < BEAT_MIN_S:
        start = max(lo, end - BEAT_MIN_S)
    return TimeRange(max(0.0, start), end)


def _combat_beat(c: HighlightCandidate, tl: SignalTimeline | None, game: str) -> TimeRange | None:
    """The stretch with the most hits, ending just after the strongest one."""
    short, long_ = COMBAT_S.get(game, (12.0, 25.0))
    if tl is None or "z_flux" not in tl.features:
        return TimeRange(c.range.start_s, min(c.range.end_s, c.range.start_s + long_))
    hop = tl.hop_s
    hits = np.nan_to_num(tl.features["z_flux"]) > 2.0
    still = tl.features.get("still")
    lo = max(0.0, c.range.start_s - 5.0)
    hi = min(tl.duration_s, c.range.end_s + 8.0)
    j = c.judgement
    pay = j.payoff_s if j and j.payoff_s is not None else c.peak_s
    best, best_s = None, -1e9
    for length in (short, (short + long_) / 2, long_):
        t = lo
        while t + length <= hi:
            a, b = int(t / hop), int((t + length) / hop)
            n = float(hits[a:b].sum())
            dead = float(still[a:b].mean()) if still is not None else 0.0
            ends_on_payoff = 3.0 if t + length - 4.0 <= pay <= t + length - 1.0 else 0.0
            score = n / length * 10 + ends_on_payoff - 6.0 * dead
            if score > best_s:
                best, best_s = TimeRange(t, t + length), score
            t += 0.5
    return best


def plans(
    stories: list[Story], moments: list[HighlightCandidate], game: str, limit: int = 6
) -> list[Plan]:
    """Fewer, better: the strongest stories, then standout single moments not already in
    one, each with the format that best fits what was judged to happen."""
    by_id = {c.id: c for c in moments}
    out: list[Plan] = []
    used: set[str] = set()
    for s in stories:
        js = [j for m in s.moments if m in by_id and (j := by_id[m].judgement) is not None]
        if not js or s.score < 9.5:
            continue
        fmt = pick_format([f for j in js for f in j.formats] + [s.kind], game, js)
        title = next((j.title for j in js[::-1] if j.payoff_seen), js[-1].title)
        out.append(Plan(s.id, fmt, s.moments, title, s.why, _variants(fmt, js), s.score, s.names))
        used.update(s.moments)
    singles = [
        c for c in moments
        if c.id not in used and c.judgement and c.judgement.is_highlight
        and c.judgement.quality >= 7 and c.judgement.standalone
    ]  # fmt: skip
    for c in sorted(singles, key=lambda c: -_single_score(c)):
        j = c.judgement
        assert j is not None
        fmt = pick_format(list(j.formats), game, [j])
        why = f"{j.what_happens} {j.reason}".strip()
        out.append(Plan(c.id, fmt, [c.id], j.title, why, _variants(fmt, [j]), _single_score(c),
                        list(j.names)))  # fmt: skip
    out.sort(key=lambda p: -p.score)
    return _diverse(out)[:limit]


def _single_score(c: HighlightCandidate) -> float:
    j = c.judgement
    if j is None:
        return 0.0
    return j.quality + 0.3 * j.emotion + (0.5 if j.payoff_seen else 0.0)


def _diverse(ps: list[Plan]) -> list[Plan]:
    """At most two Shorts per format, so ten near-identical fights don't crowd out the
    one great conversation. Fewer, better: the rest aren't proposed at all."""
    seen: dict[str, int] = {}
    keep: list[Plan] = []
    for p in ps:
        seen[p.format] = seen.get(p.format, 0) + 1
        if seen[p.format] <= MAX_PER_FORMAT:
            keep.append(p)
    return keep


def pick_format(votes: list[str], game: str, js: list[AIJudgement]) -> str:
    score = dict.fromkeys(FORMATS, 0.0)
    for v in votes:
        if v in score:
            score[v] += 1.0
    for f, b in GAME_BIAS.get(game, {}).items():
        score[f] += b
    emotion = max(j.emotion for j in js)
    if emotion >= 7 and any(j.dialogue_matters for j in js):
        score["story"] += 1.0
    if any("funny" in j.tags for j in js):
        score["unexpected"] += 1.0
    return max(score, key=lambda f: score[f])


def _variants(fmt: str, js: list[AIJudgement]) -> list[str]:
    vs = list(FORMATS[fmt].variants)
    if "B" in vs and max(j.intensity for j in js) < 5:
        vs.remove("B")  # a high-impact cut of a quiet scene would be fake energy
    return vs


# --- cutting ---------------------------------------------------------------------------


def cut(
    plan: Plan,
    variant: str,
    moments: dict[str, HighlightCandidate],
    tl: SignalTimeline | None,
    transcript: Transcript | None,
) -> Cut:
    """Source ranges for one variant. One moment: its own dead-air-free footage, framed
    for the variant. A story: setup, the strongest escalation, the payoff, in order."""
    v = VARIANTS[variant]
    ms = [moments[m] for m in plan.moments if m in moments]
    lines = transcript.segments if transcript else []
    if plan.beat is not None:
        b = plan.beat
        if variant == "B" and b.duration_s > v.max_s:
            b = TimeRange(b.end_s - v.max_s, b.end_s)  # keep the finish
        key = _payoff(ms[0]) if ms else b.end_s - REACTION_S
        key = min(max(key, b.start_s), b.end_s)
        why = f"one continuous {plan.kind} beat: {plan.title}"
        return Cut([b], key, [{"role": plan.kind, "start_s": round(b.start_s, 2),
                               "end_s": round(b.end_s, 2), "why": why}])  # fmt: skip
    if len(ms) == 1:
        return _cut_one(ms[0], v, tl, transcript, lines)
    setup, rest = ms[0], ms[1:]
    payoff = next((c for c in rest[::-1] if c.judgement and c.judgement.payoff_seen), rest[-1])
    middle = [c for c in rest if c is not payoff]
    escalation = max(middle, key=_intensity) if middle else None
    lens = PIECE_S[variant]
    pieces: list[tuple[str, TimeRange, str]] = []
    pieces.append(
        ("setup", _around(setup, _hook(setup), lens[0], 0.3, lines), _why(setup, "setup"))
    )
    if escalation is not None and lens[1] > 0:
        pieces.append(("escalation", _around(escalation, escalation.peak_s, lens[1], 0.5, lines),
                       _why(escalation, "escalation")))  # fmt: skip
    key = _payoff(payoff)
    pieces.append(("payoff", _around(payoff, key, lens[2], 0.65, lines), _why(payoff, "payoff")))
    pieces.sort(key=lambda p: p[1].start_s)
    segs = _no_overlap([p[1] for p in pieces])
    structure = [
        {"role": role, "start_s": round(r.start_s, 2), "end_s": round(r.end_s, 2), "why": why}
        for (role, _, why), r in zip(pieces, segs, strict=True)
    ]
    return Cut(segs, key, structure)


def _cut_one(
    c: HighlightCandidate,
    v: Variant,
    tl: SignalTimeline | None,
    transcript: Transcript | None,
    lines: list[TranscriptSegment],
) -> Cut:
    span = c.clip_range
    if v.id == "A":
        span = TimeRange(_line_start(max(0.0, span.start_s - CONTEXT_S), lines), span.end_s)
    segs = remove_dead_time(span, tl, transcript) if tl else [span]
    key = _payoff(c)
    total = sum(s.duration_s for s in segs)
    if total > v.max_s:
        before = 0.65 if v.id == "B" else 0.75  # high-impact ends sooner after the hit
        a = max(segs[0].start_s, key - v.max_s * before)
        a = _line_start(a, lines)
        b = _line_end(min(segs[-1].end_s, a + v.max_s), lines)
        segs = [TimeRange(max(s.start_s, a), min(s.end_s, b)) for s in segs
                if s.end_s > a and s.start_s < b]  # fmt: skip
    why = _why(c, "the moment")
    structure = [{"role": "moment", "start_s": round(segs[0].start_s, 2),
                  "end_s": round(segs[-1].end_s, 2), "why": why}]  # fmt: skip
    return Cut(segs, key, structure)


def _around(
    c: HighlightCandidate, t: float, length: float, before: float,
    lines: list[TranscriptSegment],
) -> TimeRange:  # fmt: skip
    """A piece of a moment around t, kept inside the moment, not cutting anyone off."""
    lo, hi = c.range.start_s, c.range.end_s
    a = max(lo, min(t - length * before, hi - length))
    b = min(hi, a + length)
    return TimeRange(max(lo, _line_start(a, lines)), min(hi + LINE_GRACE_S, _line_end(b, lines)))


def _line_start(t: float, lines: list[TranscriptSegment]) -> float:
    for s in lines:
        if s.start_s < t < s.end_s and t - s.start_s <= LINE_GRACE_S:
            return s.start_s
    return t


def _line_end(t: float, lines: list[TranscriptSegment]) -> float:
    for s in lines:
        if s.start_s < t < s.end_s and s.end_s - t <= LINE_GRACE_S:
            return s.end_s + 0.2
    return t


def _no_overlap(rs: list[TimeRange]) -> list[TimeRange]:
    out: list[TimeRange] = []
    for r in rs:
        if out and r.start_s < out[-1].end_s:
            r = TimeRange(out[-1].end_s, max(out[-1].end_s + 0.5, r.end_s))
        out.append(r)
    return out


def _intensity(c: HighlightCandidate) -> int:
    return c.judgement.intensity if c.judgement else 0


def _hook(c: HighlightCandidate) -> float:
    j = c.judgement
    return j.hook_s if j and j.hook_s is not None else c.peak_s


def _payoff(c: HighlightCandidate) -> float:
    j = c.judgement
    return j.payoff_s if j and j.payoff_s is not None else c.peak_s


def _why(c: HighlightCandidate, role: str) -> str:
    j = c.judgement
    if j is None:
        return role
    said = next((f'{sp}: "{t}"' if sp else f'"{t}"' for sp, t in j.dialogue), "")
    return f"{role}: {j.title}" + (f" ({said})" if said else "")


# --- dialogue captions -----------------------------------------------------------------


def subtitle_lines(
    segments: list[TimeRange],
    transcript: Transcript | None,
    read: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Spoken lines inside the cut, on the cut's own timeline (before speed changes),
    with the speaker and exact words read from the game's subtitles when they match."""
    if transcript is None:
        return []
    out: list[dict[str, Any]] = []
    offset = 0.0
    for seg in segments:
        for s in transcript.segments:
            a, b = max(s.start_s, seg.start_s), min(s.end_s, seg.end_s)
            if b - a < 0.3 or not s.text.strip():
                continue
            if s.end_s - s.start_s > LONG_LINE_S or s.confidence < 0.5:
                continue  # a merged or unsure transcription: a wrong caption is worse than none
            speaker, text = _match(s.text, read)
            out.append({"start": round(offset + a - seg.start_s, 2),
                        "end": round(offset + b - seg.start_s, 2),
                        "speaker": speaker, "text": text})  # fmt: skip
        offset += seg.duration_s
    return out


def _match(heard: str, read: list[tuple[str, str]]) -> tuple[str, str]:
    h = heard.lower().strip()
    best, score = None, 0.0
    for sp, t in read:
        r = SequenceMatcher(None, h, t.lower()).ratio()
        if r > score:
            best, score = (sp, t), r
    if best and score >= 0.55:
        return best
    return "", heard.strip()


# --- effects and sound per variant -----------------------------------------------------

GRADE_FILM = {"type": "grade", "contrast": 1.08, "saturation": 0.95, "gamma": 0.97}
_NEVER = {"A": {"rgb", "glitch", "pixel", "flash", "ghost", "letterbox", "cold_open"},
          "B": {"pixel", "letterbox"}}  # fmt: skip
_MAX = {"A": {"punch": 1, "sfx": 1, "speed": 3}, "B": {"punch": 3, "sfx": 3, "flash": 1,
        "rgb": 1, "glitch": 1, "speed": 3}}  # fmt: skip


def budget(fx: list[dict[str, Any]], variant: str, j: AIJudgement | None) -> list[dict[str, Any]]:
    """Effects only where they add emphasis: A is nearly clean, B keeps a few accents
    around the hit, C has none. Slow motion only for genuinely intense moments."""
    keep_types = {"style", "voiceover", "subs"}
    if variant == "C":
        return [e for e in fx if e["type"] in keep_types]
    never = _NEVER.get(variant, set())
    limit = _MAX.get(variant, {})
    intense = j is not None and j.intensity >= 7
    out: list[dict[str, Any]] = []
    count: dict[str, int] = {}
    for e in fx:
        t = e["type"]
        if t in never or (t in ("speed", "slowmo") and not intense):
            continue
        if t == "grade":
            e = dict(GRADE_FILM) if variant == "A" else e
            if j is not None and j.visual == "dark":
                e = {**e, "gamma": 1.12, "saturation": max(1.0, float(e.get("saturation", 1)))}
        count[t] = count.get(t, 0) + 1
        if t in limit and count[t] > limit[t]:
            continue
        if t == "punch" and variant == "A":
            e = {**e, "shake": False, "zoom": min(float(e.get("zoom", 1.1)), 1.08)}
        out.append(e)
    return out


def tune_music(m: dict[str, Any], variant: str, dialogue: bool) -> dict[str, Any]:
    """Music never covers the characters: always ducked under dialogue, and low when the
    lines carry the moment."""
    gain = {"A": -14.0 if dialogue else -9.0, "B": -12.0 if dialogue else -8.0}.get(variant, -12.0)
    return {**m, "gain_db": gain, "duck": True}
