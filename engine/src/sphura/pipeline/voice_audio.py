"""Measured properties of generated speech, numpy only: loudness (BS.1770 gated LUFS), peaks
and clipping, pitch, timbre, speaking rate, pauses, artifacts; Kaldi-style fbank features for
the speaker model; gentle per-segment levelling; and the consistency report across segments.
Everything here is measurement. Nothing in it judges how natural a voice sounds."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
SR = 16000  # analysis rate: speech content lives below 8 kHz

QC_DEFAULTS: dict[str, float] = {
    "identity_min": 0.65,  # cosine to the narration's average voice (speaker model)
    "timbre_max": 1.8,  # timbre distance relative to the narration's typical spread
    "pitch_shift_st": 3.0,  # semitones from the narration's median pitch
    "loudness_spread_lu": 3.0,  # LU between the quietest and loudest segment
    "rate_change": 0.3,  # share faster/slower than the narration, after planned pace
    "boundary_jump_db": 6.0,  # level jump across a segment join
    "long_pause_ms": 1500.0,  # silence inside a segment nobody planned
}


def resample(a: F32, sr: int, to: int = SR) -> F32:
    if sr == to or len(a) == 0:
        return a.astype(np.float32)
    n = int(len(a) * to / sr)
    return np.interp(np.arange(n) * sr / to, np.arange(len(a)), a).astype(np.float32)


def db(x: float) -> float:
    return 20 * math.log10(max(x, 1e-9))


# ---------- loudness (ITU-R BS.1770-4, K-weighted, gated) ----------


def _biquad_response(b: list[float], a: list[float], w: npt.NDArray[np.float64]) -> Any:
    z = np.exp(-1j * w)
    return (b[0] + b[1] * z + b[2] * z**2) / (a[0] + a[1] * z + a[2] * z**2)


def _k_weighting(sr: int, n: int) -> Any:
    """Magnitude of the K filter (high shelf + high pass) on an rfft grid of length n."""

    def shelf(g: float, q: float, fc: float, kind: str) -> tuple[list[float], list[float]]:
        k = math.tan(math.pi * fc / sr)
        vh = 10 ** (g / 20)
        vb = vh**0.4996667741545416
        a0 = 1 + k / q + k * k
        den = [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
        if kind == "shelf":
            num = [
                (vh + vb * k / q + k * k) / a0,
                2 * (k * k - vh) / a0,
                (vh - vb * k / q + k * k) / a0,
            ]
            return num, den
        return [1, -2, 1], den

    w = 2 * np.pi * np.fft.rfftfreq(n, 1 / sr) / sr
    b1, a1 = shelf(4.0, 1 / math.sqrt(2), 1681.974450955533, "shelf")
    b2, a2 = shelf(0.0, 0.5003270373238773, 38.13547087602444, "hp")
    return np.abs(_biquad_response(b1, a1, w) * _biquad_response(b2, a2, w))


def lufs(a: F32, sr: int) -> float:
    """Integrated loudness; -70 when there's nothing to measure."""
    if len(a) < int(0.4 * sr):
        a = np.pad(a, (0, int(0.4 * sr) - len(a)))
    spec = np.fft.rfft(a.astype(np.float64))
    y = np.fft.irfft(spec * _k_weighting(sr, len(a)), len(a))
    block, hop = int(0.4 * sr), int(0.1 * sr)
    z = np.array([np.mean(y[i : i + block] ** 2) for i in range(0, len(y) - block + 1, hop)])
    if not len(z):
        return -70.0
    loud = -0.691 + 10 * np.log10(np.maximum(z, 1e-12))
    z = z[loud > -70]
    if not len(z):
        return -70.0
    rel = -0.691 + 10 * math.log10(float(np.mean(z))) - 10
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return round(-0.691 + 10 * math.log10(float(np.mean(z))), 2) if len(z) else -70.0


# ---------- frame-level measures ----------


def _frames(a: F32, size: int, hop: int) -> F32:
    if len(a) < size:
        a = np.pad(a, (0, size - len(a)))
    n = 1 + (len(a) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    out: F32 = a[idx]
    return out


def rms_db(a: F32, sr: int, ms: float = 20) -> F32:
    f = _frames(a, int(sr * ms / 1000), int(sr * ms / 1000))
    return (20 * np.log10(np.sqrt(np.mean(f**2, axis=1)) + 1e-9)).astype(np.float32)


def pitch(a16: F32) -> F32:
    """f0 per voiced 10 ms frame (Hz), by normalised autocorrelation in 70-400 Hz."""
    f = _frames(a16, 640, 160)
    f = f - f.mean(axis=1, keepdims=True)
    energy = np.sqrt(np.mean(f**2, axis=1))
    spec = np.fft.rfft(f * np.hanning(640), 1280, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :640]
    ac = ac / np.maximum(ac[:, :1], 1e-9)
    lo, hi = SR // 400, SR // 70
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(ac)), lag]
    voiced = (strength > 0.45) & (energy > max(1e-4, float(np.percentile(energy, 60)) * 0.3))
    return (SR / lag[voiced]).astype(np.float32)


def _mel_bank(n_fft: int, n_mels: int, sr: int, lo: float = 20.0, hi: float | None = None) -> Any:
    hi = hi or sr / 2

    def mel(f: Any) -> Any:
        return 1127.0 * np.log(1 + np.asarray(f) / 700.0)

    pts = np.linspace(mel(lo), mel(hi), n_mels + 2)
    freqs = mel(np.fft.rfftfreq(n_fft, 1 / sr))
    bank = np.zeros((n_mels, len(freqs)), np.float32)
    for m in range(n_mels):
        left, mid, right = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - left) / (mid - left)
        down = (right - freqs) / (right - mid)
        bank[m] = np.maximum(0, np.minimum(up, down))
    return bank


def fbank(a: F32, sr: int, n_mels: int = 80) -> F32:
    """Kaldi-style log mel filterbank (25 ms / 10 ms, povey window, pre-emphasis), mean-
    normalised: the input the WeSpeaker speaker model was trained on."""
    a16 = resample(a, sr) * 32768.0
    f = _frames(a16, 400, 160).astype(np.float64)
    f = f - f.mean(axis=1, keepdims=True)
    f[:, 1:] = f[:, 1:] - 0.97 * f[:, :-1]
    f[:, 0] *= 1 - 0.97
    f = f * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85
    power = np.abs(np.fft.rfft(f, 512, axis=1)) ** 2
    feats = np.log(np.maximum(power @ _mel_bank(512, n_mels, SR).T, np.finfo(np.float32).eps))
    out: F32 = (feats - feats.mean(axis=0)).astype(np.float32)
    return out


def timbre(a16: F32) -> F32:
    """13 cepstral coefficients (c1..c13) averaged over loud frames: a voice-colour print."""
    f = _frames(a16, 400, 160)
    power = np.abs(np.fft.rfft(f * np.hamming(400), 512, axis=1)) ** 2
    logmel = np.log(power @ _mel_bank(512, 40, SR).T + 1e-9)
    loud = logmel.mean(axis=1) > np.percentile(logmel.mean(axis=1), 40)
    n = np.arange(40)
    dct = np.cos(np.pi * np.outer(np.arange(1, 14), n + 0.5) / 40)
    return (
        (logmel[loud] @ dct.T).mean(axis=0).astype(np.float32)
        if loud.any()
        else np.zeros(13, np.float32)
    )


def syllable_rate(a16: F32) -> float:
    """Syllable nuclei per second of speech (energy peaks at least 100 ms apart)."""
    env = rms_db(a16, SR, 10)
    speech = env > float(np.max(env)) - 35
    if speech.sum() < 10:
        return 0.0
    smooth = np.convolve(env, np.ones(5) / 5, mode="same")
    thresh = float(np.median(smooth[speech]))
    peaks, last = 0, -100
    for i in range(1, len(smooth) - 1):
        if (
            smooth[i] > thresh
            and smooth[i] >= smooth[i - 1]
            and smooth[i] > smooth[i + 1]
            and i - last >= 10
        ):
            peaks, last = peaks + 1, i
    return round(float(peaks / (speech.sum() * 0.01)), 2)


def pauses(
    a: F32, sr: int, floor_db: float = -45.0, min_ms: float = 150
) -> list[tuple[float, float]]:
    """(start s, length ms) of silences inside the audio, not at its edges."""
    env = rms_db(a, sr, 10)
    quiet = env < max(floor_db, float(np.max(env)) - 40)
    out, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if start > 0 and (i - start) * 10 >= min_ms:
                out.append((round(start * 0.01, 2), float((i - start) * 10)))
            start = None
    return out


def artifacts(a: F32, sr: int) -> list[str]:
    """Things a listener would hear as faults: clipping, clicks, DC offset."""
    out = []
    clipped = int(np.sum(np.abs(a) >= 0.999))
    if clipped >= 3:
        out.append(f"clipping: {clipped} samples at full scale")
    jumps = np.abs(np.diff(a))
    loud = np.abs(a[1:]) + np.abs(a[:-1])
    clicks = int(np.sum((jumps > 0.6) & (jumps > 4 * np.percentile(jumps, 99.9))))
    if clicks and float(np.max(loud)) > 0:
        out.append(f"{clicks} click{'s' if clicks > 1 else ''} (sudden sample jumps)")
    if abs(float(np.mean(a))) > 0.01:
        out.append(f"DC offset {float(np.mean(a)):+.3f}")
    return out


def _iqr_st(f0: F32) -> float | None:
    if len(f0) <= 10:
        return None
    return round(float(12 * np.log2(np.percentile(f0, 75) / np.percentile(f0, 25))), 2)


def measure(a: F32, sr: int) -> dict[str, Any]:
    a16 = resample(a, sr)
    f0 = pitch(a16)
    return {
        "seconds": round(len(a) / sr, 2),
        "lufs": lufs(a, sr),
        "peak_db": round(db(float(np.max(np.abs(a))) if len(a) else 0.0), 2),
        "pitch_hz": round(float(np.median(f0)), 1) if len(f0) > 10 else None,
        "pitch_iqr_st": _iqr_st(f0),
        "rate_sps": syllable_rate(a16),
        "timbre": [round(float(x), 3) for x in timbre(a16)],
        "pauses": pauses(a, sr),
        "artifacts": artifacts(a, sr),
        # speech level of the first and last second: a join that jumps is heard there
        "head_db": round(speech_level(a[:sr], sr), 2) if len(a) else -90.0,
        "tail_db": round(speech_level(a[-sr:], sr), 2) if len(a) else -90.0,
    }


# ---------- gentle levelling for assembly ----------


def speech_level(a: F32, sr: int) -> float:
    """RMS level (dBFS) of the loud half of 20 ms frames: speech, not pauses."""
    env = rms_db(a, sr)
    return float(np.mean(env[env >= np.percentile(env, 50)])) if len(env) else -90.0


def level_match(parts: list[F32], sr: int, max_gain_db: float = 6.0) -> list[F32]:
    """Brings every segment to the median speech level (within ±max_gain_db) and fades its
    edges by 8 ms, so joins don't jump or click. Conservative on purpose."""
    levels = [speech_level(p, sr) for p in parts]
    target = float(np.median(levels)) if levels else -20.0
    fade = int(0.008 * sr)
    out = []
    for p, lv in zip(parts, levels, strict=True):
        g = 10 ** (max(-max_gain_db, min(max_gain_db, target - lv)) / 20)
        q = (p * g).astype(np.float32)
        if len(q) > 2 * fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            q[:fade] *= ramp
            q[-fade:] *= ramp[::-1]
        out.append(q)
    return out


def cap_silences(a: F32, sr: int, max_ms: float = 1200, to_ms: float = 600) -> F32:
    """Shortens accidental long silences inside a TTS segment; planned pauses are added
    between segments, after this, so they're untouched."""
    out, last = [], 0
    for start, ms in pauses(a, sr):
        if ms > max_ms:
            i = int(start * sr)
            j = i + int(ms / 1000 * sr)
            keep = int(to_ms / 1000 * sr)
            out += [a[last : i + keep // 2], a[j - keep // 2 : j]]
            last = j
    out.append(a[last:])
    return np.concatenate(out).astype(np.float32) if len(out) > 1 else a


# ---------- consistency report ----------


def cosine(a: Any, b: Any) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def consistency(segs: list[dict[str, Any]], qc: dict[str, float] | None = None) -> dict[str, Any]:
    """Compares segments of one narration with each other. Every flag is a measurement
    against a threshold the user can change; nothing is rejected automatically.
    segs: [{id, label, metrics, embedding?, speed, planned_pause_ms?}] in playback order."""
    qc = {**QC_DEFAULTS, **(qc or {})}
    flags: list[dict[str, Any]] = []
    if not segs:
        return {"score": None, "flags": [], "segments": [], "thresholds": qc}
    m = [s["metrics"] for s in segs]

    def flag(i: int, kind: str, msg: str, value: float | None = None) -> None:
        flags.append({"segment": segs[i]["id"], "label": segs[i].get("label", ""), "kind": kind,
                      "message": msg, "value": value})  # fmt: skip

    per: list[dict[str, Any]] = [{"id": s["id"]} for s in segs]
    embs = [s.get("embedding") for s in segs]
    identity = None
    if all(e is not None for e in embs) and len(embs) > 1:
        sims = []
        for i, e in enumerate(embs):
            others = np.mean([x for j, x in enumerate(embs) if j != i], axis=0)
            sims.append(cosine(e, others))
            per[i]["identity"] = round(sims[-1], 3)
            if sims[-1] < qc["identity_min"]:
                flag(
                    i,
                    "identity",
                    f"Sounds like a different speaker (similarity {sims[-1]:.2f})",
                    sims[-1],
                )
        identity = float(np.mean(sims))
    tim = np.array([x["timbre"] for x in m], np.float32)
    if len(tim) >= 4:  # the spread estimate is noise with fewer pieces
        center = np.median(tim, axis=0)
        dist = np.linalg.norm(tim - center, axis=1)
        spread = float(np.median(dist)) or 1.0
        for i, d in enumerate(dist):
            per[i]["timbre"] = round(float(d / spread), 2)
            if d / spread > qc["timbre_max"] and d > 2.0:
                flag(
                    i,
                    "timbre",
                    f"Voice colour differs from the rest ({d / spread:.1f}x the usual spread)",
                    float(d / spread),
                )
    f0 = [x["pitch_hz"] for x in m]
    known = [x for x in f0 if x]
    if len(known) > 1:
        med = float(np.median(known))
        for i, x in enumerate(f0):
            if x:
                st = 12 * math.log2(x / med)
                per[i]["pitch_st"] = round(st, 2)
                if abs(st) > qc["pitch_shift_st"]:
                    flag(
                        i,
                        "pitch",
                        f"Pitch {'higher' if st > 0 else 'lower'} than the rest by "
                        f"{abs(st):.1f} semitones",
                        st,
                    )
    loud = [x["lufs"] for x in m if x["lufs"] > -60]
    spread_lu = round(max(loud) - min(loud), 2) if len(loud) > 1 else 0.0
    if spread_lu > qc["loudness_spread_lu"]:
        med = float(np.median(loud))
        for i, x in enumerate(m):
            if abs(x["lufs"] - med) > qc["loudness_spread_lu"] / 2:
                flag(
                    i,
                    "loudness",
                    f"{'Louder' if x['lufs'] > med else 'Quieter'} than the rest by "
                    f"{abs(x['lufs'] - med):.1f} LU",
                    x["lufs"] - med,
                )
    rates = [
        x["rate_sps"] / max(float(s.get("speed", 1.0)), 0.1) for x, s in zip(m, segs, strict=True)
    ]
    good = [r for r in rates if r > 0]
    if len(good) > 1:
        med = float(np.median(good))
        for i, r in enumerate(rates):
            if r > 0 and abs(r / med - 1) > qc["rate_change"]:
                flag(
                    i,
                    "rate",
                    f"Speaks {'faster' if r > med else 'slower'} than planned "
                    f"({abs(r / med - 1):.0%})",
                    r / med - 1,
                )
    for i in range(len(segs) - 1):
        jump = m[i + 1]["head_db"] - m[i]["tail_db"]
        if abs(jump) > qc["boundary_jump_db"]:
            flag(
                i + 1,
                "boundary",
                f"Level jumps {jump:+.1f} dB at the join with the piece before",
                jump,
            )
    for i, x in enumerate(m):
        for start, ms in x["pauses"]:
            if ms > qc["long_pause_ms"]:
                flag(
                    i,
                    "pause",
                    f"{ms / 1000:.1f} s of silence at {start:.1f} s that wasn't planned",
                    ms,
                )
        for a in x["artifacts"]:
            flag(i, "artifact", a.capitalize())
    # One number for the narration: identity when the speaker model ran, timbre otherwise
    parts = []
    if identity is not None:
        parts.append(min(1.0, max(0.0, (identity - 0.3) / 0.6)))
    elif len(tim) > 1:
        parts.append(max(0.0, 1 - sum(1 for f in flags if f["kind"] == "timbre") / len(segs)))
    parts.append(max(0.0, 1 - spread_lu / 12))
    if len(known) > 1:
        parts.append(
            max(0.0, 1 - float(np.std([12 * math.log2(x / np.median(known)) for x in known])) / 6)
        )
    return {
        "score": round(100 * float(np.mean(parts))) if parts else None,
        "identity": round(identity, 3) if identity is not None else None,
        "identity_model": identity is not None,
        "loudness_spread_lu": spread_lu,
        "segments": per,
        "flags": flags,
        "thresholds": qc,
    }
