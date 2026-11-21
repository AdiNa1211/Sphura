"""Learns what the user likes from keep/reject ratings and nudges the ranking toward it.

Each rating stores a snapshot of the moment's traits (AI tags, category, signal drivers,
length), so taste carries across videos and re-analysis. A trait's weight is its log-odds
lift among good vs bad ratings, shrunk toward zero until enough ratings back it up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sphura.domain.models import HighlightCandidate

SHRINK = 3.0  # ratings needed before a trait's weight counts for half its raw value
MAX_SHIFT = 0.2  # largest change taste can make to a ranking score
OVERRIDE_WEIGHT = 0.6  # what the user states outright beats anything inferred


def traits_of(c: HighlightCandidate) -> tuple[str, ...]:
    out: list[str] = []
    j = c.judgement
    if j is not None:
        out += [f"tag:{t}" for t in j.tags]
        if j.category is not None:
            out.append(f"category:{j.category.value}")
        if j.standalone:
            out.append("standalone")
    out += [f"signal:{d}" for d in c.drivers]
    dur = c.clip_range.duration_s
    out.append("length:short" if dur < 20 else "length:long" if dur > 40 else "length:medium")
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True, slots=True)
class Rating:
    candidate_id: str
    good: bool
    traits: tuple[str, ...]


@dataclass(slots=True)
class TasteModel:
    weights: dict[str, float] = field(default_factory=dict)
    overrides: dict[str, bool] = field(default_factory=dict)  # trait -> user says likes?
    direct: dict[str, bool] = field(default_factory=dict)  # candidate id -> rated good?
    good: int = 0
    bad: int = 0

    @property
    def ratings(self) -> int:
        return self.good + self.bad

    def score(self, c: HighlightCandidate) -> float:
        """-1..1: how well this moment matches what the user kept before."""
        if not self.weights:
            return 0.0
        return math.tanh(sum(self.weights.get(t, 0.0) for t in traits_of(c)) / 2)

    def shift(self, c: HighlightCandidate) -> float:
        if self.direct.get(c.id):
            return 0.3  # the user kept this exact moment
        return MAX_SHIFT * self.score(c)

    def rejected(self, c: HighlightCandidate) -> bool:
        return self.direct.get(c.id) is False


def learn(ratings: list[Rating], overrides: dict[str, bool] | None = None) -> TasteModel:
    good = sum(r.good for r in ratings)
    bad = len(ratings) - good
    model = TasteModel(
        good=good,
        bad=bad,
        direct={r.candidate_id: r.good for r in ratings},
        overrides=dict(overrides or {}),
    )
    if good and bad:
        base = math.log((good + 1) / (bad + 1))
        counts: dict[str, list[int]] = {}
        for r in ratings:
            for t in r.traits:
                g_b = counts.setdefault(t, [0, 0])
                g_b[0 if r.good else 1] += 1
        total = len(ratings)
        for t, (g, b) in counts.items():
            lift = math.log((g + 1) / (b + 1)) - base
            n = g + b
            # A trait on nearly every rated clip (e.g. "action" in an action game) can't
            # separate good from bad: discount by prevalence to avoid blaming it
            rarity = 1.0 - n / total if n / total > 0.5 else 1.0
            model.weights[t] = lift * n / (n + SHRINK) * rarity
    for t, likes in model.overrides.items():
        model.weights[t] = OVERRIDE_WEIGHT if likes else -OVERRIDE_WEIGHT
    return model


def describe(model: TasteModel, top: int = 5) -> tuple[list[str], list[str]]:
    """Human-readable leanings, strongest first; user-stated ones are marked."""
    ranked = sorted(model.weights.items(), key=lambda kv: -kv[1])

    def show(t: str) -> str:
        return _label(t) + (" (you said)" if t in model.overrides else "")

    likes = [show(t) for t, w in ranked if w >= 0.15][:top]
    dislikes = [show(t) for t, w in reversed(ranked) if w <= -0.15][:top]
    return likes, dislikes


KNOWN_TRAITS = {
    **{c: f"category:{c}" for c in ("action", "clutch", "funny", "cinematic", "unexpected")},
    **{
        t: f"tag:{t}"
        for t in (
            "finisher",
            "player dies",
            "boss/giant",
            "many enemies",
            "cutscene",
            "rare",
            "routine",
            "high spectacle",
        )
    },
    **{
        s: f"signal:{s}"
        for s in (
            "loud spike",
            "impacts",
            "shouting",
            "fast motion",
            "visual change",
            "talking fast",
            "laughter",
            "exclamation",
        )
    },
    **{f"{n} clips": f"length:{n}" for n in ("short", "medium", "long")},
    "boss": "tag:boss/giant",
    "story": "tag:cutscene",
    "death": "tag:player dies",
}


def resolve_trait(word: str) -> str | None:
    """Map what the user types ("action", "boss", "short clips") to a trait key."""
    w = word.strip().lower().removesuffix(" moments")
    if w in KNOWN_TRAITS:
        return KNOWN_TRAITS[w]
    if w in ("short", "medium", "long"):
        return f"length:{w}"
    return None


def _label(trait: str) -> str:
    kind, _, value = trait.partition(":")
    return {
        "tag": value,
        "category": f"{value} moments",
        "signal": value,
        "length": f"{value} clips",
    }.get(kind, trait)
