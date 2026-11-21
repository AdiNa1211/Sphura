"""Phase 8/9: a professional short-form edit, planned from signals the analysis found.

Structure: cold open (the climax first) -> glitch flash -> build-up with a slow push-in and a
riser -> speed ramp into slow motion on the key hit with a bass drop -> back to speed as the
music drops in. Impacts get punch-ins with camera shake, a sub-hit and an RGB split; jump
cuts get a whoosh and a small zoom so the pacing never sags. Nothing is invented: every
effect is anchored to a real impact or cut in the footage.

All edits are plain dicts on Clip.effects (editable). Times are on the pre-speed timeline
(teaser included) unless named out_s.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sphura.domain.models import Clip, TimeRange
from sphura.pipeline.fx_context import apply_context, freeze
from sphura.pipeline.timeline import SignalTimeline
from sphura.pipeline.wav import WavFile

STYLE_VERSION = 4  # bump to re-plan clips made with an older style

COLD_OPEN_BEFORE_S, COLD_OPEN_AFTER_S = 0.7, 0.5
COLD_OPEN_MIN_LEAD_S = 4.0  # if the climax is this early anyway, no teaser needed
KEY_ZOOM, PUNCH_ZOOM, CUT_ZOOM = 1.16, 1.1, 1.06
PUNCH_S = 0.45
MAX_PUNCHES, PUNCH_GAP_S, PUNCH_MIN_Z = 4, 2.0, 2.0
PUSH_S, PUSH_ZOOM = 2.0, 0.07  # slow push-in that builds anticipation before the key hit
# Speed ramp around the key hit: (seconds before/after the hit, factor)
RAMP = ((-0.45, -0.25, 0.7), (-0.25, 0.55, 0.4), (0.55, 0.8, 0.7))
CALM_S = 5.0  # longest stretch without any visual accent before a pattern interrupt
GRADE = {"type": "grade", "contrast": 1.08, "saturation": 1.18, "gamma": 0.98}
GRADE_MEME = {"type": "grade", "contrast": 1.04, "saturation": 1.08, "gamma": 1.0}
# Near-monochrome, crushed blacks, cool cast: the look of the top "edit" Shorts
GRADE_MONO = {"type": "grade", "contrast": 1.28, "saturation": 0.3, "gamma": 0.92, "tint": "cool"}


@dataclass(frozen=True, slots=True)
class MusicTrack:
    path: str
    length_s: float
    drop_s: float  # biggest energy rise in the track
    energy: float
    beat_s: float = 0.0  # beat period (0 = unknown)
    beat0_s: float = 0.0  # time of a beat (phase)
    brightness: float = 0.5  # share of treble energy: low = dark/bassy, high = bright
    dynamics_db: float = 0.0  # loud-vs-quiet spread: high = builds and drops, low = steady
    source: str = ""  # where it came from (YouTube Audio Library, Pixabay, ...)
    license: str = ""
    credit: str = ""  # attribution line the license asks for ("" = none required)
    tags: str = ""  # the user's own words for the track, e.g. "dark phonk"

    @property
    def name(self) -> str:
        return Path(self.path).stem

    def profile(self) -> str:
        """One line describing the track for the AI music picker."""
        tempo = f"{round(60 / self.beat_s)} BPM" if self.beat_s else "no steady beat"
        tone = (
            "dark/bassy"
            if self.brightness < 0.25
            else "bright"
            if self.brightness > 0.5
            else "balanced"
        )
        shape = "big build-ups and drops" if self.dynamics_db > 8 else "steady"
        bits = [tempo, tone, shape, f"drop at {self.drop_s:.0f}s", f"{self.length_s:.0f}s long"]
        if self.tags:
            bits.append(f"tags: {self.tags}")
        return f"{self.name} ({', '.join(bits)})"


def key_moment(clip: Clip, peak_s: float, tl: SignalTimeline | None, wav: WavFile | None) -> float:
    """Source time of the payoff: the strong impact nearest the moment's peak."""
    impacts = _impacts(clip, tl, wav)
    return min(impacts[:3], key=lambda t: abs(t - peak_s)) if impacts else _refine(peak_s, wav)


def plan_effects(
    clip: Clip, peak_s: float, tl: SignalTimeline | None, wav: WavFile | None,
    kind: str = "action",
) -> list[dict[str, Any]]:  # fmt: skip
    """Base plan for the style, reshaped for what the moment is (fx_context)."""
    if clip.style == "meme":
        return _plan_meme(clip, peak_s, tl, wav)
    fx = (_plan_cinematic if clip.style == "cinematic" else _plan_hype)(clip, peak_s, tl, wav)
    teaser = next((e["end_s"] - e["start_s"] for e in fx if e["type"] == "cold_open"), 0.0)
    return apply_context(fx, kind, clip.style, teaser + clip.duration_s)


def _plan_meme(
    clip: Clip, peak_s: float, tl: SignalTimeline | None, wav: WavFile | None
) -> list[dict[str, Any]]:
    """Setup plays straight; the punchline freezes with a zoom, a flash and a hit."""
    fx: list[dict[str, Any]] = [{"type": "style", "version": STYLE_VERSION}, dict(GRADE_MEME)]
    k = _to_clip(clip, key_moment(clip, peak_s, tl, wav))
    if 0 < k < clip.duration_s:
        fx.append(_punch(k, 1.2, shake=False, dur=1.2))
        fx += freeze(k)
    return fx


def _plan_hype(
    clip: Clip, peak_s: float, tl: SignalTimeline | None, wav: WavFile | None
) -> list[dict[str, Any]]:
    impacts = _impacts(clip, tl, wav)  # source times, strongest first
    key = min(impacts[:3], key=lambda t: abs(t - peak_s)) if impacts else _refine(peak_s, wav)
    if not any(s.start_s <= key <= s.end_s for s in clip.segments):
        return [{"type": "style", "version": STYLE_VERSION}, dict(GRADE)]

    fx: list[dict[str, Any]] = [{"type": "style", "version": STYLE_VERSION}, dict(GRADE)]
    sfx: list[dict[str, Any]] = []
    lead = 0.0
    if _to_clip(clip, key) >= COLD_OPEN_MIN_LEAD_S:
        a, b = key - COLD_OPEN_BEFORE_S, key + COLD_OPEN_AFTER_S
        lead = b - a
        fx.append({"type": "cold_open", "start_s": _r(a), "end_s": _r(b)})
        fx.append(_punch(COLD_OPEN_BEFORE_S, KEY_ZOOM, shake=True))
        fx.append({"type": "rgb", "t_s": _r(COLD_OPEN_BEFORE_S), "dur_s": 0.12})
        sfx.append(_sfx("hit", COLD_OPEN_BEFORE_S, -4))
        # Glitch cut back to the start, sold by a whoosh
        fx.append({"type": "flash", "t_s": _r(lead), "dur_s": 0.1})
        fx.append({"type": "glitch", "t_s": _r(lead), "dur_s": 0.16})
        sfx.append(_sfx("whoosh", lead, -3))

    k = lead + _to_clip(clip, key)
    accents = [k]
    # Anticipation: slow push-in and a riser into the hit, then slow motion and a bass drop
    fx.append({"type": "push", "start_s": _r(max(lead, k - PUSH_S)), "end_s": _r(k),
               "zoom": PUSH_ZOOM})  # fmt: skip
    fx.append(_punch(k, KEY_ZOOM, shake=True))
    fx.append({"type": "rgb", "t_s": _r(k), "dur_s": 0.15})
    for da, db, f in RAMP:
        fx.append({"type": "speed", "start_s": _r(max(lead, k + da)), "end_s": _r(k + db),
                   "factor": f})  # fmt: skip
    fx.append({"type": "vignette", "start_s": _r(k + RAMP[0][0]), "end_s": _r(k + RAMP[-1][1])})
    sfx.append(_sfx("riser", k, -3))
    sfx.append(_sfx("boom", k, -2))

    for t in impacts:
        if len(accents) > MAX_PUNCHES:
            break
        tc = lead + _to_clip(clip, t)
        if all(abs(tc - x) >= PUNCH_GAP_S for x in accents):
            fx.append(_punch(tc, PUNCH_ZOOM, shake=True))
            fx.append({"type": "rgb", "t_s": _r(tc), "dur_s": 0.1})
            sfx.append(_sfx("hit", tc, -8))
            accents.append(tc)

    # Pattern interrupts: every jump cut gets a whoosh and a small zoom
    acc = lead
    for seg in clip.segments[:-1]:
        acc += seg.duration_s
        if all(abs(acc - x) >= 1.0 for x in accents):
            fx.append(_punch(acc, CUT_ZOOM, shake=False, dur=0.3))
            sfx.append(_sfx("whoosh", acc, -9))
            accents.append(acc)
    # ...and long calm stretches get a gentle zoom so the frame keeps moving
    total = lead + clip.duration_s
    marks = sorted([lead, *accents, total])
    for a, b in itertools.pairwise(marks):
        t = a + CALM_S
        while t < b - 1.5:
            fx.append(_punch(t, CUT_ZOOM, shake=False, dur=1.4))
            t += CALM_S
    fx.extend(sfx)
    return fx


def pick_track(clip: Clip, tracks: list[MusicTrack]) -> MusicTrack | None:
    """Auto choice: one of the livelier half of the folder, stable per clip."""
    if not tracks:
        return None
    lively = sorted(tracks, key=lambda m: -m.energy)[: max(1, (len(tracks) + 1) // 2)]
    return lively[int(hashlib.sha1(clip.id.encode()).hexdigest(), 16) % len(lively)]


def plan_music(
    clip: Clip, tracks: list[MusicTrack], pick: MusicTrack | None = None
) -> dict[str, Any] | None:
    """Line the track's drop up with the moment slow motion ends, so the music hits right
    as the action comes back to speed. Music sits out the teaser and dips out just before
    the key hit (the riser carries the tension)."""
    pick = pick or pick_track(clip, tracks)
    if pick is None:
        return None
    teaser = next((e for e in clip.effects if e["type"] == "cold_open"), None)
    lead = teaser["end_s"] - teaser["start_s"] if teaser else 0.0
    pieces = clip.speed_pieces()
    key_in = pieces[0][0] if pieces else None
    back = clip.to_output_time(pieces[-1][1]) if pieces else clip.to_output_time(lead)
    out_dur = clip.output_duration_s
    offset = min(max(0.0, pick.drop_s - back), max(0.0, pick.length_s - out_dur))
    return {
        "type": "music",
        "path": pick.path,
        "offset_s": _r(offset),
        "gain_db": -15.0,
        "in_s": _r(clip.to_output_time(lead)),
        "dip": [_r(clip.to_output_time(key_in) - 0.5), _r(back)] if key_in is not None else None,
    }


def _plan_cinematic(
    clip: Clip, peak_s: float, tl: SignalTimeline | None, wav: WavFile | None
) -> list[dict[str, Any]]:
    """Fewer, heavier accents: glitch transitions at every cut, a ramp into slow motion on
    the payoff with a pixel smear, zoom-and-shake on the biggest hits. No cold open: the
    format opens on a mood shot with a caption instead. Beat alignment happens once the
    music is chosen (align_to_beats)."""
    impacts = _impacts(clip, tl, wav)
    key = min(impacts[:3], key=lambda t: abs(t - peak_s)) if impacts else _refine(peak_s, wav)
    fx: list[dict[str, Any]] = [{"type": "style", "version": STYLE_VERSION}, dict(GRADE_MONO)]
    if not any(s.start_s <= key <= s.end_s for s in clip.segments):
        return fx
    k = _to_clip(clip, key)
    fx.append(_punch(k, KEY_ZOOM, shake=True))
    fx.append({"type": "pixel", "t_s": _r(k), "dur_s": 0.18})
    for da, db, f in RAMP:
        fx.append({"type": "speed", "start_s": _r(max(0.0, k + da)), "end_s": _r(k + db),
                   "factor": f})  # fmt: skip
    fx.append({"type": "ghost", "start_s": _r(k + RAMP[0][0]), "end_s": _r(k + RAMP[-1][1])})
    fx.append(_sfx("boom", k, -3))
    accents = [k]
    for t in impacts:
        tc = _to_clip(clip, t)
        if len(accents) > 3:
            break
        if all(abs(tc - x) >= PUNCH_GAP_S for x in accents):
            fx.append(_punch(tc, PUNCH_ZOOM, shake=True))
            fx.append({"type": "rgb", "t_s": _r(tc), "dur_s": 0.1})
            accents.append(tc)
    acc = 0.0
    for seg in clip.segments[:-1]:
        acc += seg.duration_s
        fx.append({"type": "pixel", "t_s": _r(acc), "dur_s": 0.22})
        fx.append(_sfx("whoosh", acc, -8))
    return fx


def align_to_beats(clip: Clip, music: dict[str, Any], track: MusicTrack) -> None:
    """Snap visual accents (punches, pixel glitches, RGB splits) onto the music's beats
    when a beat is close, so cuts and hits land on the rhythm."""
    if track.beat_s <= 0:
        return
    # First slide the music (by under half a beat) so the payoff hit lands on a beat
    anchor = _anchor(clip)
    if anchor is not None:
        tt = clip.to_output_time(anchor) + music["offset_s"]
        off = (tt - track.beat0_s + track.beat_s / 2) % track.beat_s - track.beat_s / 2
        if music["offset_s"] - off >= 0:
            music["offset_s"] = _r(music["offset_s"] - off)
    for e in clip.effects:
        if e["type"] not in ("punch", "pixel", "rgb"):
            continue
        out_t = clip.to_output_time(e["t_s"])
        track_t = out_t + music["offset_s"]
        n = round((track_t - track.beat0_s) / track.beat_s)
        snapped = track.beat0_s + n * track.beat_s - music["offset_s"]
        if abs(snapped - out_t) <= 0.12 and not _in_speed(clip, e["t_s"]):
            e["t_s"] = _r(e["t_s"] + (snapped - out_t))


_TIMED = ("t_s", "start_s", "end_s")


def beat_sync_cuts(clip: Clip, beat_s: float) -> float:
    """Make every jump cut land on the music's (half-)beat grid relative to the key hit, by
    trimming a sliver of footage between the cut and the hit (at most a quarter beat per
    cut, never below 1 s of footage). Shifts all later effects to match. Returns the total
    trimmed. Run before choosing the music offset; the offset then puts the hit on a beat,
    which puts the cuts on beats too."""
    if _anchor(clip) is None or beat_s <= 0:
        return 0.0
    grid = beat_s / 2
    trimmed = 0.0
    lead = _lead(clip)
    for i in range(len(clip.segments) - 1):
        cut = lead + sum(s.duration_s for s in clip.segments[: i + 1])  # pre-speed time
        k = _anchor(clip)
        if k is None or abs(cut - k) < 1e-6 or _in_speed(clip, cut):
            continue
        dist = abs(clip.to_output_time(k) - clip.to_output_time(cut))
        r = dist % grid
        if r < 0.02 or r > grid - 0.02:
            continue
        if cut < k:  # trim the start of the segment after the cut
            seg = clip.segments[i + 1]
            if seg.duration_s - r < 1.0:
                continue
            clip.segments[i + 1] = TimeRange(seg.start_s + r, seg.end_s)
            _shift_after(clip, cut, -r)
        else:  # trim the end of the segment before the cut
            seg = clip.segments[i]
            if seg.duration_s - r < 1.0:
                continue
            clip.segments[i] = TimeRange(seg.start_s, seg.end_s - r)
            _shift_after(clip, cut - r, -r)
        trimmed += r
    if trimmed:
        clip.crop_path = []  # the footage changed under the camera path
    return trimmed


def dialogue_ranges(clip: Clip, spoken: list[tuple[float, float]]) -> list[list[float]]:
    """Source-time speech spans -> output-time spans inside the finished Short, so music
    can make room for dialogue."""
    out: list[list[float]] = []
    acc = 0.0
    for seg in clip.render_segments():
        for a, b in spoken:
            lo, hi = max(a, seg.start_s), min(b, seg.end_s)
            if hi - lo >= 0.3:
                out.append([_r(clip.to_output_time(acc + lo - seg.start_s)),
                            _r(clip.to_output_time(acc + hi - seg.start_s))])  # fmt: skip
        acc += seg.duration_s
    out.sort()
    merged: list[list[float]] = []
    for r in out:
        if merged and r[0] - merged[-1][1] < 0.6:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return merged[:16]


def _lead(clip: Clip) -> float:
    teaser = next((e for e in clip.effects if e["type"] == "cold_open"), None)
    return float(teaser["end_s"] - teaser["start_s"]) if teaser else 0.0


def _anchor(clip: Clip) -> float | None:
    """What sits exactly on a beat: the key hit (biggest punch), else the first cut."""
    key = max((e for e in clip.effects if e["type"] == "punch"), key=lambda e: e["zoom"],
              default=None)  # fmt: skip
    if key is not None:
        return float(key["t_s"])
    if len(clip.segments) > 1:
        return _lead(clip) + clip.segments[0].duration_s
    return None


def _shift_after(clip: Clip, t0: float, delta: float) -> None:
    for e in clip.effects:
        if e["type"] in ("cold_open", "music", "style", "grade"):
            continue
        for f in _TIMED:
            if f in e and e[f] > t0:
                e[f] = _r(max(t0, e[f] + delta))


def _in_speed(clip: Clip, t: float) -> bool:
    return any(a <= t <= b for a, b, _ in clip.speed_pieces())


def _punch(t: float, zoom: float, shake: bool, dur: float = PUNCH_S) -> dict[str, Any]:
    return {"type": "punch", "t_s": _r(t), "zoom": zoom, "dur_s": dur, "shake": shake}


def _sfx(kind: str, t: float, gain_db: float) -> dict[str, Any]:
    return {"type": "sfx", "kind": kind, "t_s": _r(t), "gain_db": gain_db}


def _r(x: float) -> float:
    return round(float(x), 3)


def _impacts(clip: Clip, tl: SignalTimeline | None, wav: WavFile | None) -> list[float]:
    """Strong, well-separated audio impacts inside the clip, strongest first."""
    if tl is None or "z_flux" not in tl.features:
        return []
    flux = np.nan_to_num(tl.features["z_flux"])
    found: list[tuple[float, float]] = []
    for seg in clip.segments:
        a = int(seg.start_s / tl.hop_s)
        b = min(len(flux), int(seg.end_s / tl.hop_s))
        for i in range(a + 1, b - 1):
            if flux[i] >= PUNCH_MIN_Z and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
                t = _refine(i * tl.hop_s + tl.hop_s / 2, wav)
                if seg.start_s + 0.3 <= t <= seg.end_s - 0.3:
                    found.append((float(flux[i]), t))
    picked: list[float] = []
    for _, t in sorted(found, reverse=True):
        if all(abs(t - p) >= PUNCH_GAP_S for p in picked):
            picked.append(t)
    return picked


def _refine(t: float, wav: WavFile | None, radius: float = 0.4) -> float:
    """Snap to the sharpest energy rise near t (the 0.5 s timeline is too coarse to hit on
    the beat)."""
    if wav is None:
        return t
    frame = 0.01
    rms = wav.rms_frames(max(0.0, t - radius), t + radius, frame)
    if rms.size < 3:
        return t
    rise = np.diff(np.log(rms + 1e-4))
    return max(0.0, t - radius) + (int(np.argmax(rise)) + 1) * frame


def _to_clip(clip: Clip, source_t: float) -> float:
    """Source time -> time in the clip body (segments only, no teaser)."""
    acc = 0.0
    for s in clip.segments:
        if s.start_s <= source_t <= s.end_s:
            return acc + source_t - s.start_s
        acc += s.duration_s
    return acc
