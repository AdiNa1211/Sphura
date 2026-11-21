"""Phase 6a: turn a ranked highlight into an editable Clip (EDL) with dead time removed."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sphura.domain.models import (
    MIN_CLIP_S,
    AspectRatio,
    Clip,
    TextOverlay,
    TimeRange,
)
from sphura.domain.transcript import Transcript
from sphura.pipeline.ranking import RankedHighlight
from sphura.pipeline.timeline import SignalTimeline

DEAD_MIN_S = 2.0  # quieter/stiller than this long inside a clip becomes a jump cut
EDGE_DEAD_MIN_S = 1.5  # dead lead-in/tail shorter than this is kept
KEEP_MARGIN_S = 0.4  # keep a little of each dead stretch so cuts don't feel abrupt
MIN_PIECE_S = 1.0  # don't leave slivers of footage between two cuts
HOOK_S = 3.0


def build_clip(
    r: RankedHighlight, tl: SignalTimeline | None, transcript: Transcript | None
) -> Clip:
    c = r.candidate
    span = c.clip_range
    segments = remove_dead_time(span, tl, transcript) if tl else [span]
    j = c.judgement
    title = (j.title if j and j.title else "") or f"Moment at {_hms(c.peak_s)}"
    overlays = []
    if j and j.hook:
        total = sum(s.duration_s for s in segments)
        overlays.append(TextOverlay(j.hook, 0.0, min(HOOK_S, total)))
    return Clip(
        project_id=c.project_id,
        asset_id=c.asset_id,
        title=title,
        segments=segments,
        aspect=AspectRatio.LANDSCAPE,  # 9:16 reframing arrives in Phase 7
        overlays=overlays,
        candidate_id=c.id,
        rank=r.rank,
    )


def dead_mask(tl: SignalTimeline, transcript: Transcript | None) -> npt.NDArray[np.bool_]:
    f = tl.features
    n = tl.length
    still = f.get("still", np.zeros(n, np.float32)) > 0
    quiet_static = (np.nan_to_num(f.get("z_motion", np.zeros(n))) < -0.5) & (
        np.nan_to_num(f.get("z_loud_rel", np.zeros(n))) < 0.0
    )
    dead = still | quiet_static
    if transcript:
        for s in transcript.segments:  # never cut while someone is talking
            a, b = int(s.start_s / tl.hop_s), int(np.ceil(s.end_s / tl.hop_s))
            dead[max(0, a) : min(n, b)] = False
    return np.asarray(dead, dtype=bool)


def remove_dead_time(
    span: TimeRange, tl: SignalTimeline, transcript: Transcript | None
) -> list[TimeRange]:
    hop = tl.hop_s
    mask = dead_mask(tl, transcript)
    a, b = int(span.start_s / hop), min(tl.length, int(np.ceil(span.end_s / hop)))
    runs = _runs(mask[a:b])

    start, end = span.start_s, span.end_s
    cuts: list[tuple[float, float]] = []
    for i0, i1 in runs:
        t0, t1 = span.start_s + i0 * hop, span.start_s + i1 * hop
        if i0 == 0 and t1 - t0 >= EDGE_DEAD_MIN_S:
            start = max(start, t1 - KEEP_MARGIN_S)
        elif i1 == b - a and t1 - t0 >= EDGE_DEAD_MIN_S:
            end = min(end, t0 + KEEP_MARGIN_S)
        elif t1 - t0 >= DEAD_MIN_S:
            cuts.append((t0 + KEEP_MARGIN_S, t1 - KEEP_MARGIN_S))

    pieces: list[TimeRange] = []
    cursor = start
    for c0, c1 in cuts:
        if c0 - cursor >= MIN_PIECE_S:
            pieces.append(TimeRange(cursor, c0))
        cursor = max(cursor, c1)
    if end - cursor >= MIN_PIECE_S:
        pieces.append(TimeRange(cursor, end))

    # Too much removed: a jumpy 9 s clip is worse than an honest 20 s one
    if not pieces or sum(p.duration_s for p in pieces) < MIN_CLIP_S:
        return [span]
    return pieces


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _hms(t: float) -> str:
    s = int(t)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
