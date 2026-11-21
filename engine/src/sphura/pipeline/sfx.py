"""Sound design, synthesised from scratch (no sample packs, nothing to license).

whoosh: swept noise for transitions; riser: tension build into a big moment;
hit: sub-bass thump that accents impacts without masking the game's own sound;
boom: long bass drop under the key moment.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

RATE = 48000
Audio = npt.NDArray[np.float32]


def _t(dur: float) -> npt.NDArray[np.float64]:
    return np.arange(int(dur * RATE)) / RATE


def _lowpass(x: npt.NDArray[np.float64], cutoff: npt.NDArray[np.float64]) -> Audio:
    """One-pole low-pass with a per-sample cutoff (for sweeps)."""
    a = 1 - np.exp(-2 * np.pi * np.clip(cutoff, 20, RATE / 2.5) / RATE)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y.astype(np.float32)


def _norm(x: Audio, peak: float = 0.7) -> Audio:
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak).astype(np.float32)


def whoosh(dur: float = 0.45, rng: np.random.Generator | None = None) -> Audio:
    rng = rng or np.random.default_rng(1)
    t = _t(dur)
    u = t / dur
    env = np.sin(np.pi * u) ** 1.5
    cutoff = 400 + 7000 * np.sin(np.pi * u) ** 2
    bright = _lowpass(rng.standard_normal(len(t)), cutoff)
    dull = _lowpass(rng.standard_normal(len(t)), cutoff * 0.25)
    return _norm(((bright - 0.6 * dull) * env).astype(np.float32), 0.5)


def riser(dur: float = 1.4, rng: np.random.Generator | None = None) -> Audio:
    rng = rng or np.random.default_rng(2)
    t = _t(dur)
    u = t / dur
    noise = _lowpass(rng.standard_normal(len(t)), 300 + 9000 * u**2)
    freq = 180 + 900 * u**2
    tone = np.sin(2 * np.pi * np.cumsum(freq) / RATE)
    x = (0.7 * noise + 0.3 * tone) * u**2.2
    x[-int(0.01 * RATE) :] *= np.linspace(1, 0, int(0.01 * RATE))  # cut off sharply, no click
    return _norm(x.astype(np.float32), 0.45)


def hit(dur: float = 0.35) -> Audio:
    t = _t(dur)
    freq = 45 + 70 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * 11)
    click = np.random.default_rng(3).standard_normal(len(t)) * np.exp(-t * 400) * 0.25
    return _norm(np.tanh(1.8 * (body + click)).astype(np.float32), 0.6)


def boom(dur: float = 1.6) -> Audio:
    t = _t(dur)
    freq = 30 + 55 * np.exp(-t * 4)
    x = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * 2.2)
    x += 0.3 * np.random.default_rng(4).standard_normal(len(t)) * np.exp(-t * 60)
    return _norm(np.tanh(2.2 * x).astype(np.float32), 0.75)


SOUNDS = {"whoosh": whoosh, "riser": riser, "hit": hit, "boom": boom}


def mix_track(events: list[dict[str, Any]], duration_s: float) -> Audio:
    """events: {"kind", "out_s" (when it peaks/starts on the output timeline), "gain_db"}.
    Risers end at out_s and whooshes are centred on it; others start there."""
    track = np.zeros(int(duration_s * RATE) + 1, np.float32)
    for ev in events:
        snd = SOUNDS[ev["kind"]]().copy()
        fade = min(len(snd), int(0.03 * RATE))
        snd[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        gain = 10 ** (ev.get("gain_db", 0.0) / 20)
        at = ev["out_s"]
        if ev["kind"] == "riser":
            at -= len(snd) / RATE
        elif ev["kind"] == "whoosh":
            at -= len(snd) / RATE / 2
        i = int(at * RATE)
        a, b = max(0, i), min(len(track), i + len(snd))
        if b > a:
            track[a:b] += snd[a - i : b - i] * gain
    return track


def write_wav(path: Path, mono: Audio) -> None:
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(stereo.tobytes())
