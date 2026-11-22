"""Phase 7a: where to put a 9:16 window inside 16:9 footage, over time.

People come first: when a detector finds them, the window keeps them in frame (bigger,
closer people count more). Otherwise column-wise motion energy says where the action is.
The camera moves like an editor's: locked shots (Viterbi over window positions, with a
cost for every move and a pull toward the centre), joined by short eased pans.
If the action is spread across the whole frame, the clip is shown whole ("fit").
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sphura.domain.models import Clip, CropKeyframe
from sphura.ports.media import MediaDecoder, PeopleDetector, PersonBox

VERSION = 3  # bump when the camera planning changes, so saved paths are re-planned
ANALYSIS_FPS = 8.0
ANALYSIS_W, ANALYSIS_H = 320, 180
TARGET_ASPECT = 9 / 16
CENTER_PRIOR_SIGMA = 0.35  # fraction of width; wide, so real action still wins
BINS = 64  # column resolution of the motion profile
POSITIONS = 41  # candidate window centres
MOVE_COST_S = 1.0  # a move must win ~this many seconds of fully better framing
CENTER_COST = 2.5  # per-sample cost of (offset from centre)^2, in widths
MIN_MOVE = 0.05  # moves smaller than this are dropped (fraction of width)
PAN_SPEED = 0.6  # widths per second during a pan
MIN_PAN_S = 0.5
# "Fit" (whole frame) only when action is clearly spread out: little of it fits in one
# window AND the best window keeps hopping. Steady movement or camera pans stay "crop"
FIT_MAX_COVERAGE = 0.7
FIT_JITTER = 0.025
PEOPLE_FPS = 3.0
PEOPLE_W, PEOPLE_H = 640, 360
PEOPLE_HOLD_S = 1.0  # keep a person through brief misses (splashes, motion blur)
MOTION_WEIGHT = 0.15  # with people in shot, motion only breaks ties
EMPTY_WEIGHT = 0.3  # moments without people (in a clip that has them) barely pull the camera


@dataclass(frozen=True, slots=True)
class ReframePlan:
    keyframes: list[CropKeyframe]  # clip-relative time, cx = window centre (0..1 of width)
    layout: str  # "crop" or "fit"
    coverage: float  # median share of (pan-corrected) motion inside the chosen window
    jitter: float = 0.0  # how much the best window hops between samples


def plan_reframe(
    clip: Clip,
    source_uri: str,
    decoder: MediaDecoder,
    display_aspect: float,
    target_aspect: float = TARGET_ASPECT,
    people: PeopleDetector | None = None,
) -> ReframePlan:
    """display_aspect = width / height of the source as displayed; target_aspect = the
    footage area of the output."""
    win = min(1.0, target_aspect / display_aspect)  # window width as fraction of frame
    if win >= 0.999:
        return ReframePlan([CropKeyframe(0.0, 0.5, 0.5)], "crop", 1.0)

    profiles: list[npt.NDArray[np.float32]] = []
    for seg in clip.render_segments():
        prev: npt.NDArray[np.float32] | None = None
        for batch in decoder.iter_proxy_frames(
            source_uri, seg.start_s, seg.duration_s, ANALYSIS_FPS, ANALYSIS_W, ANALYSIS_H, 16
        ):
            frames = np.frombuffer(batch.rgb, np.uint8).reshape(-1, ANALYSIS_H, ANALYSIS_W, 3)
            for fr in frames:
                gray = fr.astype(np.float32).mean(axis=2)
                # A segment's first frame has no motion yet: an empty profile (no preference)
                profiles.append(
                    np.zeros(BINS, np.float32) if prev is None else motion_profile(gray - prev)
                )
                prev = gray
    if not profiles:
        return ReframePlan([CropKeyframe(0.0, 0.5, 0.5)], "crop", 1.0)

    prof = np.stack(profiles)
    found = _people_profiles(clip, source_uri, decoder, people, display_aspect, len(prof), win)
    if found is not None:
        crowd, groups = found
        gain, coverage = _people_gain(prof, crowd, win)
        layout = "fit" if coverage < FIT_MAX_COVERAGE and _group(groups) else "crop"
        jitter = 0.0
    else:
        _, coverage, jitter = _best_windows(prof, win)
        layout = "fit" if coverage < FIT_MAX_COVERAGE and jitter > FIT_JITTER else "crop"
        gain = _captured(prof, win)
    path = camera_path(gain, win, 1 / ANALYSIS_FPS)
    keyframes = [
        CropKeyframe(round(i / ANALYSIS_FPS, 3), round(float(x), 4), 0.5)
        for i, x in enumerate(path)
    ]
    return ReframePlan(keyframes, layout, coverage, jitter)


def _people_profiles(
    clip: Clip,
    uri: str,
    decoder: MediaDecoder,
    people: PeopleDetector | None,
    display_aspect: float,
    n: int,
    win: float,
) -> tuple[npt.NDArray[np.float32], list[bool]] | None:
    """(n, BINS) column weight of people on the motion samples' clock, plus per detection
    whether it was a group that can't share one window. None: no detector, or nobody."""
    if people is None:
        return None
    pad = (1 - min(1.0, display_aspect / (PEOPLE_W / PEOPLE_H))) / 2  # pillarbox, if any
    found: list[tuple[float, npt.NDArray[np.float32]]] = []
    groups: list[bool] = []
    offset = 0.0
    for seg in clip.render_segments():
        for batch in decoder.iter_proxy_frames(
            uri, seg.start_s, seg.duration_s, PEOPLE_FPS, PEOPLE_W, PEOPLE_H, 8
        ):
            size = PEOPLE_W * PEOPLE_H * 3
            for k, t in enumerate(batch.timestamps_s):
                boxes = people.detect(batch.rgb[k * size : (k + 1) * size], PEOPLE_W, PEOPLE_H)
                if boxes is None:
                    return None
                found.append((offset + t, _columns(boxes, pad)))
                if boxes:
                    groups.append(_apart(boxes, pad, win))
        offset += seg.duration_s
    if not any(c.any() for _, c in found):
        return None
    times = np.array([t for t, _ in found])
    cols = np.stack([c for _, c in found])
    out = np.zeros((n, BINS), np.float32)
    last: npt.NDArray[np.float32] | None = None
    last_t = -1e9
    for i in range(n):
        t = i / ANALYSIS_FPS
        j = int(np.argmin(np.abs(times - t)))
        if abs(times[j] - t) <= 0.5 / PEOPLE_FPS + 1e-6 and cols[j].any():
            last, last_t = cols[j], times[j]
        if last is not None and t - last_t <= PEOPLE_HOLD_S:
            out[i] = last
    return out, groups


def _columns(boxes: list[PersonBox], pad: float) -> npt.NDArray[np.float32]:
    """Each person as a tent over their width (centre matters most), weighted by
    confidence and size: the closest, clearest person is the subject."""
    centres = (np.arange(BINS) + 0.5) / BINS
    out = np.zeros(BINS, np.float32)
    span = max(1 - 2 * pad, 1e-6)
    for b in boxes:
        x0, x1 = (b.x0 - pad) / span, (b.x1 - pad) / span
        w = max(x1 - x0, 1e-3)
        size = w * (b.y1 - b.y0)
        if size < 0.002:  # specks: far background people, false hits
            continue
        tent = np.clip(1 - np.abs(centres - (x0 + x1) / 2) / (w / 2), 0, None)
        if tent.sum() > 0:
            out += (tent / tent.sum() * b.score * size**0.75).astype(np.float32)
    return out


def _people_gain(
    motion: npt.NDArray[np.float32], crowd: npt.NDArray[np.float32], win: float
) -> tuple[npt.NDArray[np.float64], float]:
    """Per-sample reward: people captured, motion as a tie-breaker; samples with nobody
    count little, so splashes and flashes can't drag the camera off the subject."""
    cap_p = _captured(crowd, win)
    cap_m = _captured(motion, win)
    has = crowd.sum(axis=1) > 1e-6
    gain = np.where(has[:, None], cap_p + MOTION_WEIGHT * cap_m, EMPTY_WEIGHT * cap_m)
    coverage = float(np.median(cap_p[has].max(axis=1))) if has.any() else 1.0
    return gain, coverage


def _apart(boxes: list[PersonBox], pad: float, win: float) -> bool:
    """Two or more equally prominent people too far apart for one window. When one person
    clearly dominates (a close-up speaker), cropping to them reads better than a letterbox."""
    span = max(1 - 2 * pad, 1e-6)
    sizes = [(b.x1 - b.x0) * (b.y1 - b.y0) for b in boxes]
    big = [b for b, a in zip(boxes, sizes, strict=True) if a >= max(sizes) * 0.5]
    if len(big) < 2:
        return False
    lo = min((b.x0 - pad) / span for b in big)
    hi = max((b.x1 - pad) / span for b in big)
    return hi - lo > win * 1.15


def _group(groups: list[bool]) -> bool:
    return bool(groups) and sum(groups) > len(groups) / 2


def motion_profile(diff: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Column motion energy in BINS bins, minus the median (removes camera pans and
    flashes that move everything at once)."""
    col = np.abs(diff).mean(axis=0)
    col = (
        col.reshape(BINS, -1).mean(axis=1)
        if col.size % BINS == 0
        else np.interp(np.linspace(0, col.size - 1, BINS), np.arange(col.size), col)
    )
    return np.clip(col - np.median(col), 0, None).astype(np.float32)


def _positions(win: float) -> npt.NDArray[np.float64]:
    return np.linspace(win / 2, 1 - win / 2, POSITIONS)


def _captured(prof: npt.NDArray[np.float32], win: float) -> npt.NDArray[np.float64]:
    """(samples, POSITIONS): share of each sample's motion inside each candidate window.
    Samples without motion are 0 everywhere, so they express no preference."""
    centres = (np.arange(BINS) + 0.5) / BINS
    pos = _positions(win)
    inside = (np.abs(centres[None, :] - pos[:, None]) <= win / 2).astype(np.float64)
    total = prof.sum(axis=1, keepdims=True)
    return np.where(total > 1e-6, (prof @ inside.T) / np.maximum(total, 1e-6), 0.0)


def _best_windows(
    prof: npt.NDArray[np.float32], win: float
) -> tuple[npt.NDArray[np.float64], float, float]:
    cap = _captured(prof, win)
    moving = prof.sum(axis=1) > 1e-6
    if not moving.any():
        return np.full(len(prof), 0.5), 1.0, 0.0
    pos = _positions(win)
    prior = np.exp(-0.5 * ((pos - 0.5) / CENTER_PRIOR_SIGMA) ** 2)
    best = pos[np.argmax(cap * prior, axis=1)]
    coverage = float(np.median(cap[moving].max(axis=1)))
    b = _median(best[moving].astype(np.float32), 3)
    jitter = float(np.mean(np.abs(np.diff(b)))) if b.size > 1 else 0.0
    return best, coverage, jitter


def camera_path(
    captured: npt.NDArray[np.float64], win: float, dt: float
) -> npt.NDArray[np.float64]:
    """Locked shots chosen by Viterbi (reward: share of the subject in frame, per sample x
    position; costs: distance from centre and every move), joined with short eased pans."""
    pos = _positions(win)
    gain = captured - CENTER_COST * (pos[None, :] - 0.5) ** 2
    move = MOVE_COST_S / dt  # in samples of full reward
    n, m = gain.shape
    score = gain[0].copy()
    back = np.zeros((n, m), np.int32)
    for i in range(1, n):
        stay = score
        jump_from = int(np.argmax(score))
        jump = score[jump_from] - move
        back[i] = np.where(stay >= jump, np.arange(m), jump_from)
        score = np.maximum(stay, jump) + gain[i]
    idx = np.empty(n, np.int32)
    idx[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        idx[i - 1] = back[i, idx[i]]
    shots = pos[idx]
    for i in range(1, n):  # drop moves too small to be worth it
        if abs(shots[i] - shots[i - 1]) < MIN_MOVE:
            shots[i] = shots[i - 1]
    return _ease(shots, dt)


def _ease(shots: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    out = shots.copy()
    for i in np.flatnonzero(np.diff(shots)) + 1:
        a, b = shots[i - 1], shots[i]
        half = max(MIN_PAN_S, abs(b - a) / PAN_SPEED) / 2 / dt
        lo, hi = max(0, int(i - half)), min(len(shots), int(i + half) + 1)
        for j in range(lo, hi):
            u = min(max((j - (i - half)) / (2 * half), 0.0), 1.0)
            out[j] = a + (b - a) * u * u * (3 - 2 * u)  # smoothstep
    return out


def _median(x: npt.NDArray[np.float32], k: int) -> npt.NDArray[np.float32]:
    if x.size < k:
        return x
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.asarray(np.median(view, axis=1), dtype=np.float32)


def simplify(keyframes: list[CropKeyframe], tolerance: float = 0.004) -> list[CropKeyframe]:
    """Ramer-Douglas-Peucker on (t, cx): keeps the camera path's shape with few points,
    so it fits in a compact FFmpeg expression."""
    if len(keyframes) <= 2:
        return keyframes
    t = np.array([k.t_s for k in keyframes])
    x = np.array([k.cx for k in keyframes])
    keep = np.zeros(len(keyframes), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(keyframes) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg_t = t[a + 1 : b]
        line = x[a] + (x[b] - x[a]) * (seg_t - t[a]) / max(t[b] - t[a], 1e-9)
        err = np.abs(x[a + 1 : b] - line)
        i = int(np.argmax(err))
        if err[i] > tolerance:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return [k for k, kept in zip(keyframes, keep, strict=True) if kept]


def crop_x_expression(keyframes: list[CropKeyframe], frame_w: int, crop_w: int) -> str:
    """Piecewise-linear x(t) in source pixels: x0 + sum(slope_i * clip(t - t_i, 0, dt_i))."""
    pts = simplify(keyframes)
    tol = 0.004
    while len(pts) > 200:  # keep the command line well within OS limits
        tol *= 2
        pts = simplify(keyframes, tol)

    def px(cx: float) -> float:
        return min(max(cx * frame_w - crop_w / 2, 0.0), float(frame_w - crop_w))

    expr = f"{px(pts[0].cx):.1f}"
    for a, b in itertools.pairwise(pts):
        dt = b.t_s - a.t_s
        dx = px(b.cx) - px(a.cx)
        if dt <= 0 or abs(dx) < 0.05:
            continue
        expr += f"+{dx / dt:.4f}*clip(t-{a.t_s:.3f},0,{dt:.3f})"
    return expr
