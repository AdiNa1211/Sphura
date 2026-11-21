"""Phase 5a: turn the interest curve into candidate clip windows with sensible boundaries.

A window covers build-up -> peak -> payoff/reaction, snapped to scene cuts and sentence
boundaries, 15-90 s long. Windows that are mostly menus/loading are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sphura.domain.transcript import Transcript, TranscriptSegment
from sphura.pipeline.signals import DRIVER_LABELS
from sphura.pipeline.timeline import Curve, SignalTimeline, rolling_median

MIN_LEN_S = 15.0
MAX_LEN_S = 90.0
_PRE_MIN_S, _PRE_MAX_S = 5.0, 30.0
_POST_MIN_S, _POST_MAX_S = 3.0, 25.0
_CONTEXT_S = 4.0  # extra lead-in before the build-up so viewers understand the situation
_TAIL_S = 3.0  # let the reaction land before cutting
_SNAP_CUT_S = 3.0
_SNAP_SPEECH_S = 8.0
_NMS_GAP_S = 20.0
_MAX_STILL_SHARE = 0.5
# Long action stretches (boss fights): their ending always becomes a candidate
_BLOCK_PCT = 75
_BLOCK_MIN_S = 90.0
_BLOCK_GAP_S = 30.0
_FINALE_BEFORE_S, _FINALE_AFTER_S = 20.0, 35.0
FINALE = "fight finale"


@dataclass(slots=True)
class CandidateWindow:
    start_s: float
    end_s: float
    peak_s: float
    peak_score: float
    signal_score: float  # 0..1: where the peak sits among all moments of this video
    drivers: tuple[str, ...]
    features: dict[str, float] = field(default_factory=dict)
    transcript: list[TranscriptSegment] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def max_candidates(duration_s: float) -> int:
    """Bound AI cost: ~20 per hour, at least 12 (fight finales come on top)."""
    return max(12, math.ceil(duration_s / 3600 * 20))


def detect_candidates(
    tl: SignalTimeline, transcript: Transcript | None, limit: int | None = None
) -> list[CandidateWindow]:
    f = tl.features
    hop = tl.hop_s
    interest = np.nan_to_num(f["interest"])
    n = interest.size
    if n == 0:
        return []
    baseline = rolling_median(interest, int(120 / hop) | 1)
    cuts = np.flatnonzero(f.get("scene_cut", np.zeros(n)) > 0) * hop
    still = f.get("still", np.zeros(n, np.float32))
    segments = transcript.segments if transcript else []
    limit = limit or max_candidates(tl.duration_s)

    sorted_vals = np.sort(interest)
    threshold = float(np.percentile(interest, 90))
    windows: list[CandidateWindow] = []
    for i in _peaks(interest, threshold, int(_NMS_GAP_S / hop)):
        span = _snap(
            _window_around(i, interest, baseline, hop, tl.duration_s),
            cuts,
            segments,
            tl.duration_s,
        )
        a, b = int(span[0] / hop), max(int(span[0] / hop) + 1, int(span[1] / hop))
        if float(still[a:b].mean()) > _MAX_STILL_SHARE:
            continue
        rank = float(np.searchsorted(sorted_vals, interest[i])) / n
        windows.append(
            CandidateWindow(
                start_s=span[0],
                end_s=span[1],
                peak_s=i * hop,
                peak_score=float(interest[i]),
                signal_score=rank,
                drivers=_drivers(f, a, b),
                features=_summary(f, a, b),
            )
        )
        if len(windows) >= limit * 2:  # merging below shrinks the list
            break

    merged = _merge(windows)
    merged.sort(key=lambda w: -w.peak_score)
    merged = merged[:limit]
    for fin in _finales(tl, cuts, segments, still):
        # The finale replaces ordinary windows whose peak falls inside it
        merged = [w for w in merged if not fin.start_s <= w.peak_s <= fin.end_s]
        merged.append(fin)
    for w in merged:
        w.transcript = [s for s in segments if s.end_s > w.start_s and s.start_s < w.end_s]
    return sorted(merged, key=lambda w: w.start_s)


def action_blocks(interest: Curve, hop: float) -> list[tuple[float, float]]:
    """Sustained high-interest stretches (fights) of at least _BLOCK_MIN_S, short lulls
    bridged. Uses a 30 s average so single spikes don't count."""
    k = max(1, int(30 / hop))
    smooth = np.convolve(interest, np.ones(k) / k, mode="same")
    idx = np.flatnonzero(smooth > np.percentile(smooth, _BLOCK_PCT))
    if idx.size == 0:
        return []
    blocks, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        if (i - prev) * hop > _BLOCK_GAP_S:
            blocks.append((start, prev))
            start = int(i)
        prev = int(i)
    blocks.append((start, prev))
    return [(a * hop, (b + 1) * hop) for a, b in blocks if (b - a) * hop >= _BLOCK_MIN_S]


def _finales(
    tl: SignalTimeline, cuts: Curve, segments: list[TranscriptSegment], still: Curve
) -> list[CandidateWindow]:
    """One window per fight covering its last blows and the payoff right after
    (finisher, cutscene, reactions), which peak-picking alone tends to miss."""
    f, hop = tl.features, tl.hop_s
    interest = np.nan_to_num(f["interest"])
    out = []
    for a_s, b_s in action_blocks(interest, hop):
        start, end = _snap(
            (max(0.0, b_s - _FINALE_BEFORE_S), min(tl.duration_s, b_s + _FINALE_AFTER_S)),
            cuts, segments, tl.duration_s,
        )  # fmt: skip
        a, b = int(start / hop), max(int(start / hop) + 1, int(end / hop))
        if float(still[a:b].mean()) > _MAX_STILL_SHARE:
            continue
        peak = a + int(np.argmax(interest[a:b]))
        out.append(
            CandidateWindow(
                start_s=start, end_s=end, peak_s=peak * hop,
                peak_score=float(interest[peak]),
                signal_score=float(np.mean(interest <= interest[peak])),
                drivers=(FINALE, *_drivers(f, a, b))[:4],
                features=_summary(f, a, b) | {"finale_of_s": round(b_s - a_s, 1),
                                              "fight_start_s": round(a_s, 1)},
            )
        )  # fmt: skip
    return out


def _peaks(x: Curve, threshold: float, gap: int) -> list[int]:
    """Highest-first local maxima above threshold, at least `gap` hops apart."""
    order = np.argsort(-x)
    taken: list[int] = []
    for i in order:
        if x[i] < threshold:
            break
        if all(abs(int(i) - j) >= gap for j in taken):
            taken.append(int(i))
    return taken


def _window_around(
    i: int, x: Curve, baseline: Curve, hop: float, duration: float
) -> tuple[float, float]:
    """Extend while the curve stays clearly above its local baseline (the action's
    build-up and decay), then add context before and a tail after."""
    level = baseline[i] + 0.25 * (x[i] - baseline[i])
    a = i
    while a > 0 and (x[a - 1] > level or (i - a) * hop < _PRE_MIN_S):
        if (i - a) * hop >= _PRE_MAX_S:
            break
        a -= 1
    b = i
    while b < x.size - 1 and (x[b + 1] > level or (b - i) * hop < _POST_MIN_S):
        if (b - i) * hop >= _POST_MAX_S:
            break
        b += 1
    start = max(0.0, a * hop - _CONTEXT_S)
    end = min(duration, (b + 1) * hop + _TAIL_S)
    return _fit_length(start, end, i * hop, duration)


def _fit_length(start: float, end: float, peak: float, duration: float) -> tuple[float, float]:
    if end - start < MIN_LEN_S:
        # Too short to make sense on its own: add mostly lead-in (2/3) and some tail
        need = MIN_LEN_S - (end - start)
        start = max(0.0, start - need * 2 / 3)
        end = min(duration, start + MIN_LEN_S)
        start = max(0.0, end - MIN_LEN_S)
    if end - start > MAX_LEN_S:
        # Keep the peak, favouring the build-up side
        start = max(start, peak - MAX_LEN_S * 0.6)
        end = start + MAX_LEN_S
    return start, end


def _snap(
    w: tuple[float, float], cuts: Curve, segments: list[TranscriptSegment], duration: float
) -> tuple[float, float]:
    start, end = w
    # Start on a scene cut if one is close: a clean visual entry point
    near = cuts[(cuts >= start - _SNAP_CUT_S) & (cuts <= start + _SNAP_CUT_S)]
    if near.size:
        start = float(near[np.argmin(np.abs(near - start))])
    # Never start or end mid-sentence
    for s in segments:
        if s.start_s < start < s.end_s and start - s.start_s <= _SNAP_SPEECH_S:
            start = s.start_s
        if s.start_s < end < s.end_s and s.end_s - end <= _SNAP_SPEECH_S:
            end = s.end_s
    start, end = max(0.0, start), min(duration, end)
    if end - start > MAX_LEN_S:
        end = start + MAX_LEN_S
    return start, end


def _merge(windows: list[CandidateWindow]) -> list[CandidateWindow]:
    """Overlapping windows are one moment: merge when the union stays within MAX_LEN_S,
    otherwise keep the stronger one."""
    out: list[CandidateWindow] = []
    for w in sorted(windows, key=lambda w: w.start_s):
        if out and w.start_s < out[-1].end_s:
            prev = out[-1]
            union = (min(prev.start_s, w.start_s), max(prev.end_s, w.end_s))
            stronger = prev if prev.peak_score >= w.peak_score else w
            if union[1] - union[0] <= MAX_LEN_S:
                stronger.start_s, stronger.end_s = union
                stronger.drivers = tuple(dict.fromkeys(prev.drivers + w.drivers))[:4]
            out[-1] = stronger
            continue
        out.append(w)
    return out


def _drivers(f: dict[str, Curve], a: int, b: int) -> tuple[str, ...]:
    strength = {n: float(np.nanmax(f[n][a:b])) for n in DRIVER_LABELS if n in f and b > a}
    return tuple(
        DRIVER_LABELS[n]
        for n, v in sorted(strength.items(), key=lambda kv: -kv[1])
        if v >= (1.0 if n in ("laugh_text", "exclaim") else 2.0)
    )[:3]


def _summary(f: dict[str, Curve], a: int, b: int) -> dict[str, float]:
    def peak(name: str) -> float:
        return round(float(np.nanmax(f[name][a:b])), 2) if name in f and b > a else 0.0

    return {
        "loud_spike": peak("z_loud_rel"),
        "impacts": peak("z_flux"),
        "shouting": peak("z_hf"),
        "motion": peak("z_motion"),
        "visual_change": peak("z_novelty"),
        "scene_cuts": float(np.nansum(f["scene_cut"][a:b])) if "scene_cut" in f else 0.0,
        "laughter_text": peak("laugh_text"),
    }
