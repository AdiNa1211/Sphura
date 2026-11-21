"""SPHURA original score: a short piece of music composed for one clip.

Built on the clip's own beat grid, so the key hit lands on a downbeat and the drop hits
exactly when the action returns to speed. Everything is synthesised here (drums, 808 bass,
cowbell melody, pads), so it is original music with nothing to license or credit.

Ten styles (MOODS) in three pools: hype (phonk, drift phonk, drill, trap, rage), cinematic
(dark, epic taiko, tension pulse) and funny (bounce, funk). Each clip gets a style from its
pool (not the one just used), its own key, riff and tempo.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:  # optional: vectorised filtering (faster-whisper's deps usually bring scipy along)
    from scipy.signal import lfilter as _lfilter  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _lfilter = None

RATE = 44100
F = npt.NDArray[np.float64]

MINOR_PENTA = (0, 3, 5, 7, 10)
MAJOR_PENTA = (0, 2, 4, 7, 9)


@dataclass(frozen=True, slots=True)
class Mood:
    """One musical style. Drum steps are 16th notes within a bar (0..15)."""

    label: str
    bpm: int
    minor: bool
    prog: tuple[int, ...]  # bar-by-bar chord roots, semitones from the key
    kick: tuple[int, ...]
    snare: tuple[int, ...]
    hats: tuple[int, ...]
    bass: str  # 808 | slide | sub | pluck | none
    bass_steps: tuple[int, ...]
    lead: str  # cowbell | bell | pluck | saw | strings | pulse | none
    lead_steps: tuple[int, ...]
    hat_rolls: bool = False
    taiko: tuple[int, ...] = ()
    pad_gain: float = 0.10


EIGHTHS = tuple(range(0, 16, 2))
SIXTEENTHS = tuple(range(16))
MOODS: dict[str, Mood] = {
    "phonk": Mood("Memphis phonk", 132, True, (0, -4, -2, 0), (0, 6, 10), (4, 12), EIGHTHS,
                  "808", (0, 10), "cowbell", EIGHTHS, hat_rolls=True),
    "drift": Mood("Drift phonk", 150, True, (0, -4, -2, -5), (0, 4, 8, 12), (4, 12), EIGHTHS,
                  "808", (0, 4, 8, 12), "cowbell", (0, 3, 6, 8, 10, 11, 14)),
    "drill": Mood("Drill", 142, True, (0, -2, -4, -5), (0, 11), (8,), (0, 3, 6, 9, 12, 14),
                  "slide", (0, 7, 11), "bell", (0, 6, 10), hat_rolls=True),
    "trap": Mood("Dark trap", 140, True, (0, -4, -7, -2), (0, 7, 10), (8,), SIXTEENTHS,
                 "808", (0, 7), "pluck", (0, 3, 6, 10, 13), hat_rolls=True),
    "rage": Mood("Rage synth", 155, True, (0, -4, -2, -7), (0, 8, 11), (4, 12), EIGHTHS,
                 "808", (0, 8), "saw", SIXTEENTHS),
    "dark": Mood("Dark cinematic", 96, True, (0, -4, -2, 0), (0,), (8,), (2, 6, 10, 14),
                 "808", (0,), "none", (), pad_gain=0.14),
    "epic": Mood("Epic taiko", 120, True, (0, -4, -5, -2), (), (), (), "sub", (0, 8),
                 "strings", EIGHTHS, taiko=(0, 3, 6, 8, 11, 14), pad_gain=0.12),
    "tension": Mood("Tension pulse", 90, True, (0, 1, 0, -2), (0, 10), (8,), (4, 12),
                    "none", (), "pulse", SIXTEENTHS, pad_gain=0.13),
    "bounce": Mood("Light bounce", 104, False, (0, 5, 7, 5), (0, 8), (4, 12), EIGHTHS,
                   "808", (0,), "pluck", (2, 6, 10, 14)),
    "funk": Mood("Funky groove", 112, False, (0, 5, 3, 7), (0, 7, 10), (4, 12), SIXTEENTHS,
                 "pluck", (0, 3, 7, 10, 14), "bell", (4, 12)),
}  # fmt: skip
POOLS = {
    "hype": ("phonk", "drift", "drill", "trap", "rage"),
    "cinematic": ("dark", "epic", "tension"),
    "funny": ("bounce", "funk"),
}
BPM = {k: m.bpm for k, m in MOODS.items()}


@dataclass(frozen=True, slots=True)
class ScorePlan:
    duration_s: float
    bpm: float
    anchor_s: float  # lands on a downbeat (the key hit)
    drop_s: float  # where the full beat comes in (snapped to a downbeat at/after this)
    mood: str
    seed: int

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm


def base_bpm(mood: str, seed: int) -> float:
    """Each clip gets its own tempo around the style's (±6 BPM)."""
    return MOODS[mood].bpm + float(np.random.default_rng(seed).integers(-6, 7))


def fit_bpm(mood: str, anchor_s: float, drop_s: float, seed: int = 0) -> float:
    """Nudge the clip's tempo (at most 10%) so the hit-to-drop gap is a whole number of
    beats: the drop then lands exactly where the action comes back to speed."""
    base = base_bpm(mood, seed)
    gap = drop_s - anchor_s
    if gap < 0.3:
        return base
    beats = max(1, round(gap / (60 / base)))
    bpm = 60 * beats / gap
    return bpm if 0.9 * base <= bpm <= 1.1 * base else base


def mood_for(
    category: str | None, tags: tuple[str, ...], style: str, seed: int = 0,
    avoid: tuple[str, ...] = (),
) -> str:  # fmt: skip
    """Pick a style from the pool that fits the clip, avoiding ones used just before."""
    if category == "funny":
        pool = POOLS["funny"]
    elif style == "cinematic" or category == "cinematic":
        pool = POOLS["cinematic"]
    else:
        pool = POOLS["hype"]
    fresh = [m for m in pool if m not in avoid] or list(pool)
    return fresh[int(np.random.default_rng(seed).integers(0, len(fresh)))]


def compose(plan: ScorePlan) -> F:
    m = MOODS[plan.mood]
    rng = np.random.default_rng(plan.seed)
    n = int((plan.duration_s + 0.5) * RATE)
    out = np.zeros(n)
    beat = plan.beat_s
    bar = 4 * beat
    # The key hit sits on the beat grid; the drop is the first beat at/after its hint and
    # becomes the "one" of a bar, so both land musically
    k = max(0.0, np.ceil((plan.drop_s - plan.anchor_s) / beat - 0.25))
    drop = plan.anchor_s + k * beat
    first = drop - np.ceil(drop / bar) * bar
    build = drop - 2 * bar  # two bars of tension before the drop
    key = 36 + int(rng.integers(0, 8))  # MIDI note of the key root (C2..G2)
    scale = MINOR_PENTA if m.minor else MAJOR_PENTA
    riff = [scale[int(i)] for i in rng.integers(0, len(scale), 16)]  # one-bar riff
    triad = (0, 3, 7) if m.minor else (0, 4, 7)

    def add(sig: F, at: float, gain: float = 1.0) -> None:
        i = int(at * RATE)
        if i >= n or i + len(sig) <= 0:
            return
        a, b = max(0, i), min(n, i + len(sig))
        out[a:b] += sig[a - i : b - i] * gain

    t_bar, bar_no = first, 0
    while t_bar < plan.duration_s:
        root = key + m.prog[bar_no % len(m.prog)]
        section = "drop" if t_bar >= drop - 1e-6 else "build" if t_bar >= build - 1e-6 else "intro"
        add(_pad([_hz(root + 24 + iv) for iv in triad], bar), t_bar,
            m.pad_gain * (0.7 if section == "drop" else 1.1))  # fmt: skip
        for step in range(16):
            t = t_bar + step * beat / 4
            if t >= plan.duration_s:
                break
            if section == "intro":
                _intro_step(add, m, step, t, root, key, riff)
            elif section == "build":
                # Roll that speeds up into the drop, then one beat of silence
                if t < drop - beat:
                    rate = 4 if t < build + bar else 2 if t < drop - bar / 2 else 1
                    if step % rate == 0:
                        lift = (t - build) / (2 * bar)
                        hit = _taiko(False) if m.taiko else _snare()
                        add(hit, t, 0.12 + 0.25 * lift)
            else:
                _drop_step(add, m, step, t, root, key, riff, rng, beat)
        t_bar += bar
        bar_no += 1

    add(_riser(min(2 * bar, max(0.5, drop - max(build, 0.0)))), max(build, 0.0), 0.14)
    add(_impact(), drop, 0.5)
    out = np.tanh(1.3 * out)  # glue: gentle saturation, like a bus compressor
    peak = float(np.max(np.abs(out))) + 1e-9
    result: F = (out / peak * 0.85)[: int(plan.duration_s * RATE)]
    return result


def _intro_step(add, m: Mood, step: int, t: float, root: int, key: int,  # type: ignore[no-untyped-def]
                riff: list[int]) -> None:  # fmt: skip
    if step in m.lead_steps and m.lead not in ("none", "saw", "strings"):
        add(_lead(m.lead, _hz(key + 36 + riff[step]), muted=True), t, 0.10)
    if m.lead == "pulse" and step % 2 == 0:
        add(_pulse(_hz(root + 12)), t, 0.12)
    if m.lead == "strings" and step % 4 == 0:
        add(_strings(_hz(root + 24)), t, 0.10)
    if m.hats and step % 4 == 2 and m.label != "Dark cinematic":
        add(_hat(), t, 0.08)


def _drop_step(add, m: Mood, step: int, t: float, root: int, key: int,  # type: ignore[no-untyped-def]
               riff: list[int], rng: np.random.Generator, beat: float) -> None:  # fmt: skip
    if step in m.kick:
        add(_kick(), t, 0.9)
    if step in m.snare:
        add(_clap() if m.bpm >= 120 and m.lead != "bell" else _snare(), t, 0.5)
    if step in m.taiko:
        add(_taiko(step in (0, 8)), t, 0.75 if step in (0, 8) else 0.45)
    if step in m.hats:
        add(_hat(), t, 0.14 if step % 4 else 0.18)
        if m.hat_rolls and step >= 12 and rng.random() < 0.35:  # trap-style 32nd rolls
            add(_hat(), t + beat / 8, 0.10)
    if step in m.bass_steps:
        nxt = m.bass_steps[(m.bass_steps.index(step) + 1) % len(m.bass_steps)]
        length = ((nxt - step) % 16 or 16) * beat / 4
        if m.bass == "808":
            add(_bass808(_hz(root), round(min(length, 1.6), 3)), t, 0.65)
        elif m.bass == "slide":  # drill: 808 gliding to the next note
            target = root + (riff[nxt] if nxt else 0) - 12
            add(_slide808(_hz(root), _hz(target), round(min(length, 1.2), 3)), t, 0.65)
        elif m.bass == "sub":
            add(_bass808(_hz(root), round(min(length, 2.0), 3)), t, 0.5)
        elif m.bass == "pluck":
            add(_pluck(_hz(root + 12 + riff[step] % 7)), t, 0.4)
    if step in m.lead_steps:
        if m.lead == "pulse":
            add(_pulse(_hz(root + 12)), t, 0.16)
            if step in (0, 6, 10):
                add(_bell(_hz(key + 48 + riff[step])), t, 0.14)
        elif m.lead == "strings":
            add(_strings(_hz(root + 24 + (7 if step % 4 else 0))), t, 0.18)
        elif m.lead == "saw":
            arp = (0, 3, 7, 12) if m.minor else (0, 4, 7, 12)
            add(_saw(_hz(root + 36 + arp[step % 4])), t, 0.12)
        elif m.lead != "none":
            add(_lead(m.lead, _hz(key + 36 + riff[step])), t, 0.24)


@lru_cache(maxsize=512)
def _lead(kind: str, f: float, muted: bool = False) -> F:
    sig = _cowbell(f) if kind == "cowbell" else _bell(f) if kind == "bell" else _pluck(f)
    return _lowpass(sig, 900) if muted else sig


# --- instruments -------------------------------------------------------------------------


def _hz(midi: float) -> float:
    return float(440.0 * 2 ** ((midi - 69) / 12))


def _t(dur: float) -> F:
    return np.arange(int(dur * RATE)) / RATE


def _lowpass(x: F, cutoff: float | F) -> F:
    """One-pole low-pass; cutoff may vary per sample (for sweeps)."""
    a = 1 - np.exp(-2 * np.pi * np.asarray(cutoff, dtype=np.float64) / RATE)
    if a.ndim == 0 and _lfilter is not None:
        return np.asarray(_lfilter([float(a)], [1, float(a) - 1], x), dtype=np.float64)
    coef = np.broadcast_to(a, x.shape)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += coef[i] * (x[i] - acc)
        y[i] = acc
    return y


@cache
def _kick() -> F:
    t = _t(0.42)
    freq = 45 + 110 * np.exp(-t * 35)
    body = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * 7)
    return np.tanh(2.2 * body)


@lru_cache(maxsize=256)
def _bass808(f: float, dur: float) -> F:
    t = _t(dur)
    glide = f * (1 + 0.25 * np.exp(-t * 30))
    x = np.sin(2 * np.pi * np.cumsum(glide) / RATE) + 0.25 * np.sin(
        4 * np.pi * np.cumsum(glide) / RATE
    )
    return np.tanh(1.8 * x * np.exp(-t * (1.4 / dur)))


def _noise(dur: float, seed: int) -> F:
    return np.random.default_rng(seed).standard_normal(int(dur * RATE))


@cache
def _snare() -> F:
    t = _t(0.2)
    n = _noise(0.2, 11)
    bright = n - _lowpass(n, 1800)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    return (0.7 * bright * np.exp(-t * 22) + 0.5 * tone) * 0.8


@cache
def _clap() -> F:
    out = np.zeros(int(0.25 * RATE))
    n = _noise(0.25, 12)
    n = n - _lowpass(n, 1200)
    t = _t(0.25)
    for k, d in enumerate((0.0, 0.011, 0.022)):
        i = int(d * RATE)
        env = np.exp(-(t[: len(out) - i]) * (90 if k < 2 else 18))
        out[i:] += n[: len(out) - i] * env
    return out * 0.6


@cache
def _hat() -> F:
    t = _t(0.05)
    n = _noise(0.05, 13)
    return (n - _lowpass(n, 6000)) * np.exp(-t * 90)


@lru_cache(maxsize=256)
def _cowbell(f: float) -> F:
    t = _t(0.22)
    sq = np.tanh(5 * np.sin(2 * np.pi * f * t)) + 0.8 * np.tanh(
        5 * np.sin(2 * np.pi * f * 1.48 * t)
    )
    return _lowpass(sq, 3500) * np.exp(-t * 16) * 0.5


@lru_cache(maxsize=256)
def _bell(f: float) -> F:
    """FM-ish bell: inharmonic partials, long ring."""
    t = _t(0.7)
    mod = np.sin(2 * np.pi * f * 3.5 * t) * 2.2 * np.exp(-t * 6)
    return np.sin(2 * np.pi * f * t + mod) * np.exp(-t * 5) * 0.45


@lru_cache(maxsize=256)
def _slide808(f0: float, f1: float, dur: float) -> F:
    t = _t(dur)
    glide_at = 0.6 * dur
    freq = np.where(t < glide_at, f0, f0 + (f1 - f0) * np.minimum(1, (t - glide_at) / 0.08))
    x = np.sin(2 * np.pi * np.cumsum(freq) / RATE)
    return np.tanh(2.2 * x * np.exp(-t * (1.0 / dur)))


@lru_cache(maxsize=256)
def _saw(f: float) -> F:
    """Detuned saw stab (rage/supersaw lead)."""
    t = _t(0.16)
    x: F = np.sum([((f * d * t) % 1.0) * 2 - 1 for d in (0.993, 1.0, 1.007)], axis=0)
    return _lowpass(x / 3, 4500) * np.exp(-t * 14) * 0.6


@lru_cache(maxsize=256)
def _strings(f: float) -> F:
    """Staccato string section: bowed saw ensemble, soft attack."""
    t = _t(0.28)
    x: F = np.sum([((f * d * t) % 1.0) * 2 - 1 for d in (0.996, 1.0, 1.004, 2.0)], axis=0)
    env = np.minimum(1, t / 0.02) * np.exp(-t * 7)
    out: F = _lowpass(x / 4, 2600) * env * 0.7
    return out


@lru_cache(maxsize=256)
def _pulse(f: float) -> F:
    t = _t(0.12)
    x = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + np.sin(2 * np.pi * f / 2 * t)
    return _lowpass(x, 700) * np.exp(-t * 25) * 0.7


@lru_cache(maxsize=4)
def _taiko(big: bool) -> F:
    t = _t(0.7 if big else 0.35)
    f0 = 70 if big else 110
    freq = f0 + 60 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * (6 if big else 12))
    n = _noise(len(t) / RATE, 21)
    slap = (n - _lowpass(n, 2500)) * np.exp(-t * 60) * 0.3
    return np.tanh(1.8 * (body + slap))


@lru_cache(maxsize=256)
def _pluck(f: float) -> F:
    t = _t(0.35)
    x = sum(np.sin(2 * np.pi * f * h * t) / h for h in (1, 2, 3, 4))
    return x * np.exp(-t * 9) * 0.5


@lru_cache(maxsize=64)
def _pad_cached(freqs: tuple[float, ...], dur: float) -> F:
    return _pad_impl(list(freqs), dur)


def _pad(freqs: list[float], dur: float) -> F:
    return _pad_cached(tuple(round(f, 3) for f in freqs), round(dur, 4))


def _pad_impl(freqs: list[float], dur: float) -> F:
    t = _t(dur)
    x = np.zeros_like(t)
    for f in freqs:
        for detune in (0.997, 1.003):  # two slightly detuned voices: width and movement
            x += sum(np.sin(2 * np.pi * f * detune * h * t) / h for h in range(1, 6))
    env = np.minimum(1, t / 0.25) * np.minimum(1, (dur - t) / 0.15)
    result: F = _lowpass(x, 1400) * env / (2 * len(freqs))
    return result


def _riser(dur: float) -> F:
    t = _t(dur)
    u = t / dur
    n = _noise(dur, 14)
    return (n - _lowpass(n, 400 + 6000 * u**2)) * u**2.5


@cache
def _impact() -> F:
    t = _t(1.2)
    freq = 35 + 60 * np.exp(-t * 5)
    return np.tanh(2 * np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * 3))


def write_wav(path: Path, mono: F) -> None:
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())
