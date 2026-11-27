"""Links neighbouring moments into stories (a boss intro, the fight, its finish; a
conversation and what it leads to), so a Short can be built from a setup, escalation and
payoff that really happen in the footage instead of one slice around a loud peak."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any

from sphura.domain.models import HighlightCandidate
from sphura.pipeline.judge import CAST, is_name

CLOSE_GAP_S = 45.0  # moments this close follow on from each other
NAMED_GAP_S = 180.0  # further apart, they need a shared name (speaker) to link
BOSS_GAP_S = 600.0  # the same boss bar: retries and phases of one fight
COMMON_SHARE = 0.4  # names in this share of moments (the protagonist) don't link anything
MAX_MOMENTS = 8  # a long boss fight; the edit planner picks the best of them


@dataclass(slots=True)
class Story:
    id: str
    moments: list[str]  # candidate ids in time order
    roles: dict[str, str]  # candidate id -> setup | escalation | payoff
    start_s: float
    end_s: float
    kind: str  # boss | combat | story | unexpected
    names: list[str]
    score: float
    why: str
    formats: list[str] = field(default_factory=list)


def link(candidates: list[HighlightCandidate]) -> list[Story]:
    """Stories of 2+ judged moments, best first."""
    judged = sorted((c for c in candidates if c.judgement), key=lambda c: c.range.start_s)
    if len(judged) < 2:
        return []
    common = _common_names(judged)
    groups: list[list[HighlightCandidate]] = [[judged[0]]]
    for c in judged[1:]:
        prev = groups[-1][-1]
        gap = c.range.start_s - prev.range.end_s
        shared = (_names(prev) & _names(c)) - common
        boss = _bars(prev) & _bars(c)
        if len(groups[-1]) < MAX_MOMENTS and (
            gap <= CLOSE_GAP_S
            or (gap <= NAMED_GAP_S and shared)
            or (gap <= BOSS_GAP_S and boss - common)
        ):
            groups[-1].append(c)
        else:
            groups.append([c])
    stories = [_story(g, common) for g in groups if len(g) >= 2]
    return sorted(stories, key=lambda s: -s.score)


def _names(c: HighlightCandidate) -> set[str]:
    j = c.judgement
    if j is None:
        return set()
    found = {n.lower() for n in j.names} | {sp.lower() for sp, _ in j.dialogue if sp}
    found |= {ch.lower() for ch in j.characters if is_name(ch)}
    return {n for n in found if n not in ("the player", "player")}


def _bars(c: HighlightCandidate) -> set[str]:
    return {n.lower() for n in c.judgement.names} if c.judgement else set()


def _weight(c: HighlightCandidate) -> int:
    return c.judgement.intensity + c.judgement.emotion if c.judgement else 0


def _common_names(moments: list[HighlightCandidate]) -> set[str]:
    counts = Counter(n for c in moments for n in _names(c))
    return {n for n, k in counts.items() if k / len(moments) > COMMON_SHARE}


def _story(g: list[HighlightCandidate], common: set[str]) -> Story:
    js = [c.judgement for c in g if c.judgement]
    formats = Counter(f for j in js for f in j.formats)
    emotion = sum(j.emotion for j in js) / len(js)
    if formats.get("boss"):
        kind = "boss"
    elif formats.get("unexpected") and formats["unexpected"] >= len(js) / 2:
        kind = "unexpected"
    elif emotion >= 6 or formats.get("story", 0) + formats.get("lore", 0) >= len(js) / 2:
        kind = "story"
    else:
        kind = "combat"
    # The payoff: the last moment whose outcome is visible, else the most intense one
    later = g[1:]  # the payoff comes after the setup
    seen = [c for c in later if c.judgement and c.judgement.payoff_seen]
    payoff = (
        seen[-1]
        if seen
        else max(later, key=lambda c: (_weight(c), c.range.start_s))  # ties: the later one
    )
    roles = {c.id: "escalation" for c in g}
    roles[g[0].id] = "setup"
    roles[payoff.id] = "payoff"
    names = sorted({n for c in g for n in _names(c)} - common)
    best = max(j.quality for j in js)
    score = best + 0.3 * emotion + (0.5 if seen else 0.0) + 0.2 * len(g)
    span = g[-1].range.end_s - g[0].range.start_s
    about = f" with {', '.join(n.title() for n in names[:3])}" if names else ""
    end = "the outcome is on screen" if seen else "no clear outcome on screen"
    why = (
        f"{len(g)} moments over {span / 60:.1f} min{about}: it starts at "
        f"{_hms(g[0].range.start_s)} and builds to {_hms(payoff.range.start_s)}; {end}."
    )
    return Story(
        id=f"s-{g[0].id[:8]}",
        moments=[c.id for c in g],
        roles=roles,
        start_s=g[0].range.start_s,
        end_s=g[-1].range.end_s,
        kind=kind,
        names=names,
        score=round(score, 2),
        why=why,
        formats=[f for f, _ in formats.most_common()],
    )


def _hms(t: float) -> str:
    t = int(t)
    return (
        f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
    )


def to_dict(s: Story) -> dict[str, object]:
    return {
        "id": s.id,
        "moments": s.moments,
        "roles": s.roles,
        "start_s": s.start_s,
        "end_s": s.end_s,
        "kind": s.kind,
        "names": s.names,
        "score": s.score,
        "why": s.why,
        "formats": s.formats,
    }


def speech_stats(segments: list[Any], a: float, b: float) -> dict[str, float]:
    """How much of a span is speech and how clearly it was heard (transcript confidence)."""
    span = max(b - a, 1e-6)
    inside = [s for s in segments if s.end_s > a and s.start_s < b and s.text.strip()]
    talk = sum(min(s.end_s, b) - max(s.start_s, a) for s in inside)
    clarity = sum(s.confidence for s in inside) / len(inside) if inside else 0.0
    return {"speech_share": round(min(talk / span, 1.0), 2), "clarity": round(clarity, 2)}


def canonical_names(candidates: list[HighlightCandidate]) -> dict[str, str]:
    """Spelling fixes across a video: the judge reads the same boss bar as FRÆKNI,
    FRÆKKNI and FRËKNI on different moments; the most common reading wins."""
    counts: Counter[str] = Counter()
    for c in candidates:
        j = c.judgement
        if j:
            counts.update([*j.names, *j.characters, *(sp for sp, _ in j.dialogue if sp)])
    canon: dict[str, str] = {}  # every spelling -> the most common one
    # Known characters first: a frequent misreading ("Areus") must not beat "Atreus"
    ordered = sorted(counts.items(), key=lambda kv: (_fold(kv[0]) not in CAST, -kv[1]))
    for name, _ in ordered:
        if name in canon or not is_name(name):
            continue  # descriptions ("a man in armor") aren't spellings of a name
        canon[name] = name
        key = _fold(name)
        for other in counts:
            o = _fold(other)
            # "Fräkni" is the first word of "FRÆKNI THE ZEALOUS"
            same = SequenceMatcher(None, key, o).ratio() >= 0.8 or (
                " " in key
                and " " not in o
                and SequenceMatcher(None, key.split()[0], o).ratio() >= 0.8
            )
            if other not in canon and same:
                canon[other] = name
    return {k: v for k, v in canon.items() if k != v}


def nice_name(name: str) -> str:
    joiners = {"the", "of", "de", "von"}
    words = name.lower().split()
    return " ".join(w if i and w in joiners else w[:1].upper() + w[1:] for i, w in enumerate(words))


def _fold(s: str) -> str:
    s = s.lower().replace("æ", "ae").replace("ä", "ae").replace("ø", "o").replace("ë", "e")
    return re.sub(r"[^a-z ]", "", s.replace("ö", "o"))


def with_names(c: HighlightCandidate, canon: dict[str, str]) -> HighlightCandidate:
    j = c.judgement
    if j is None:
        return c

    def fix(text: str) -> str:
        for wrong, right in canon.items():
            if wrong in text:
                text = text.replace(wrong, right)
        for name in {*canon.values(), *j.names}:
            if name.isupper() and name in text:  # "FRÆKNI THE ZEALOUS" from a boss bar
                text = text.replace(name, nice_name(name))
        return text

    c.judgement = replace(
        j,
        names=tuple(dict.fromkeys(canon.get(n, n) for n in j.names)),
        characters=tuple(dict.fromkeys(canon.get(n, n) for n in j.characters)),
        dialogue=tuple((canon.get(sp, sp), t) for sp, t in j.dialogue),
        title=fix(j.title),
        what_happens=fix(j.what_happens),
        hook=fix(j.hook),
        reason=fix(j.reason),
    )
    return c


_DEFEAT = re.compile(r"\b(Defeat(?:s|ing|ed)?|Takes? Down|Taking Down|Kills?|Killing|Slays?)\b")


def unfinished_fights(candidates: list[HighlightCandidate]) -> None:
    """A boss bar that shows up again later means the earlier moment wasn't the end of
    that fight, whatever the judge claimed: drop its payoff and its "defeats" title."""
    bosses: dict[str, float] = {}
    for c in candidates:
        for n in c.judgement.names if c.judgement else ():
            bosses[n] = max(bosses.get(n, 0.0), c.range.start_s)
    for c in candidates:
        j = c.judgement
        if j is None or not (j.payoff_seen or _DEFEAT.search(j.title)):
            continue
        if any(bosses.get(n, 0.0) > c.range.end_s for n in j.names):
            c.judgement = replace(
                j,
                payoff_seen=False,
                payoff_s=None,
                title=_DEFEAT.sub("Fighting", j.title),
                what_happens=j.what_happens,
            )
