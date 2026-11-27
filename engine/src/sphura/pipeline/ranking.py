"""Phase 5c: mode-aware ranking with diversity. Pure function of stored candidates, so
switching modes re-ranks instantly without new AI calls."""

from __future__ import annotations

from dataclasses import dataclass

from sphura.domain.models import (
    EditingMode,
    HighlightCandidate,
    HighlightCategory,
    HighlightTier,
)
from sphura.pipeline.preferences import TasteModel

_C = HighlightCategory
MODE_BONUS: dict[EditingMode, dict[HighlightCategory, float]] = {
    EditingMode.AUTO: {},
    EditingMode.HIGHLIGHT: {_C.ACTION: 0.12, _C.CLUTCH: 0.15},
    EditingMode.FUNNY: {_C.FUNNY: 0.2, _C.UNEXPECTED: 0.1},
    EditingMode.CINEMATIC: {_C.CINEMATIC: 0.2},
    EditingMode.VIRAL: {_C.UNEXPECTED: 0.15, _C.FUNNY: 0.12, _C.CLUTCH: 0.12},
    EditingMode.MINIMAL: {},
}
_NEARBY_S = 90.0


_TOP_SHARE = 0.15  # fewer, better: only the clear standouts are recommended
_STRONG_SHARE = 0.25


@dataclass(frozen=True, slots=True)
class RankedHighlight:
    rank: int
    candidate: HighlightCandidate
    score: float  # internal ordering value; not shown to users as a precision number
    tier: HighlightTier | None  # None = not AI-reviewed; never invented from signals alone


def assign_tiers(
    ordered: list[HighlightCandidate], taste: TasteModel | None = None
) -> dict[str, HighlightTier]:
    """Tiers follow the final list order and are relative to this video: AI judges
    (especially small local models) rate most moments 8-9, so absolute scores alone
    can't separate good from great."""
    eligible = [
        c
        for c in ordered
        if c.judgement and c.judgement.is_highlight and not (taste and taste.rejected(c))
    ]
    n = len(eligible)
    n_top = max(1, round(n * _TOP_SHARE)) if n else 0
    n_strong = round(n * _STRONG_SHARE)
    tiers = {c.id: HighlightTier.MAYBE for c in ordered if c.judgement}
    tops = strongs = 0
    for c in eligible:
        j = c.judgement
        assert j is not None
        kept = bool(taste and taste.direct.get(c.id))
        if "routine" in j.tags and not kept:
            continue  # the AI itself called it ordinary: never recommended
        # TOP needs a payoff a viewer can see, or a scene that carries weight on its own
        if tops < n_top and j.quality >= 7 and j.standalone and (j.payoff_seen or j.emotion >= 7):
            tiers[c.id] = HighlightTier.TOP
            tops += 1
        elif strongs < n_strong and j.quality >= 6:
            tiers[c.id] = HighlightTier.STRONG
            strongs += 1
    return tiers


# Same weights the judge's rubric gives these tags (pipeline.judge.rubric_quality)
_TAG_WEIGHT = {
    "finisher": 1.0,
    "player dies": 1.0,
    "boss/giant": 1.0,
    "many enemies": 0.5,
    "cutscene": 1.0,
    "rare": 2.0,
    "funny": 2.0,
    "high spectacle": 1.0,
}


def tag_frequencies(candidates: list[HighlightCandidate]) -> dict[str, float]:
    judged = [c.judgement for c in candidates if c.judgement]
    if not judged:
        return {}
    return {t: sum(t in j.tags for j in judged) / len(judged) for t in _TAG_WEIGHT}


def adjusted_quality(c: HighlightCandidate, freq: dict[str, float]) -> float:
    """A tag most moments share (e.g. "boss/giant" on 90% of them) can't tell moments
    apart, so its bonus is discounted by how common it is in this video."""
    j = c.judgement
    if j is None:
        return 0.0
    return j.quality - sum(_TAG_WEIGHT[t] * freq.get(t, 0.0) for t in j.tags if t in _TAG_WEIGHT)


def candidate_score(
    c: HighlightCandidate,
    mode: EditingMode,
    freq: dict[str, float] | None = None,
    taste: TasteModel | None = None,
) -> float:
    # Candidates are already the top ~10% of moments: spread that band over 0..1
    sig = min(max((c.signal_score - 0.9) / 0.1, 0.0), 1.0)
    j = c.judgement
    if j is None:
        return 0.5 * sig
    q = adjusted_quality(c, freq) if freq else float(j.quality)
    ai = min(max(q, 0.0), 10.0) / 10 * (0.75 + 0.25 * j.confidence)
    if c.compare_score is not None:
        # Side-by-side placings are what small models judge best: let them lead
        ai = 0.6 * c.compare_score + 0.4 * ai
    s = 0.75 * ai + 0.25 * sig
    if j.category is not None:
        s += MODE_BONUS[mode].get(j.category, 0.0)
    if not j.is_highlight:
        s *= 0.5
    if not j.standalone:
        s -= 0.08
    # Moment database v2: a scene that matters can beat a louder routine fight
    if j.emotion > 5:
        s += 0.04 * (j.emotion - 5) / 5
    if j.payoff_seen:
        s += 0.04
    if j.visual in ("dark", "blurry"):
        s -= 0.05
    if taste is not None:
        s += taste.shift(c)  # learned from the user's own keep/reject ratings
    return s


def rank(
    candidates: list[HighlightCandidate],
    mode: EditingMode,
    limit: int | None = None,
    taste: TasteModel | None = None,
) -> list[RankedHighlight]:
    """Greedy pick with penalties for moments right next to, or of the same kind as, ones
    already picked, so the top of the list isn't five slices of one fight."""
    freq = tag_frequencies(candidates)
    # Moments the AI rejected go after every moment it accepted, whatever their signals
    rejected = [
        c
        for c in candidates
        if (c.judgement and not c.judgement.is_highlight) or (taste and taste.rejected(c))
    ]
    rejected_ids = {c.id for c in rejected}
    pool = {
        c.id: (c, candidate_score(c, mode, freq, taste))
        for c in candidates
        if c.id not in rejected_ids
    }
    picked: list[tuple[HighlightCandidate, float]] = []
    while pool:

        def adjusted(item: tuple[HighlightCandidate, float]) -> float:
            c, s = item
            near = sum(abs(c.peak_s - p.peak_s) < _NEARBY_S for p, _ in picked)
            cat = c.judgement.category if c.judgement else None
            same = sum(1 for p, _ in picked if cat and p.judgement and p.judgement.category == cat)
            return s - 0.12 * near - 0.03 * same

        best_id = max(pool, key=lambda k: adjusted(pool[k]))
        picked.append(pool.pop(best_id))
    picked += sorted(
        ((c, candidate_score(c, mode, freq, taste)) for c in rejected), key=lambda cs: -cs[1]
    )
    # Rank everything first so tiers reflect the whole video, then cut to the limit
    tiers = assign_tiers([c for c, _ in picked], taste)
    order = {HighlightTier.TOP: 0, HighlightTier.STRONG: 1, HighlightTier.MAYBE: 2}
    picked.sort(key=lambda cs: order.get(tiers[cs[0].id], 3) if cs[0].id in tiers else 3)
    ranked = [RankedHighlight(i + 1, c, s, tiers.get(c.id)) for i, (c, s) in enumerate(picked)]
    return ranked[:limit] if limit is not None else ranked
