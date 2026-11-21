"""Phase 4: per-video audio/visual/speech signals on a 0.5 s grid + preliminary interest curve.

Visual features come from one low-res decode (256x144 @ 4 fps), split into segments that are
checkpointed, so an interrupted run resumes instead of re-decoding hours of video.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sphura.domain.models import MediaAsset
from sphura.domain.transcript import Transcript
from sphura.pipeline.manifest import StageManifest, stage_key
from sphura.pipeline.timeline import (
    Curve,
    SignalTimeline,
    moving_average,
    robust_z,
    rolling_median,
)
from sphura.pipeline.wav import WavFile
from sphura.ports.media import MediaDecoder
from sphura.ports.storage import StorageProvider

log = logging.getLogger(__name__)

HOP_S = 0.5
FPS = 4.0
FRAME_W, FRAME_H = 256, 144
SEGMENT_S = 300.0  # visual decode checkpoint granularity

StageProgress = Callable[[float, str | None], None]

# ---------------------------------------------------------------- audio


def audio_features(wav: WavFile, n_hops: int) -> dict[str, Curve]:
    """Loudness, spectral flux (onsets: hits, gunshots, impacts) and 1-4 kHz energy share
    (shouting/screaming sits there). Processed in 60 s blocks to keep memory flat."""
    sr = wav.sample_rate
    hop = int(HOP_S * sr)
    n_fft, stft_hop = 512, 256
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    hf_band = (freqs >= 1000) & (freqs <= 4000)

    loud = np.full(n_hops, np.nan, np.float32)
    flux = np.full(n_hops, np.nan, np.float32)
    hf = np.full(n_hops, np.nan, np.float32)
    prev_logmag: npt.NDArray[np.float32] | None = None

    block_hops = 120
    for b0 in range(0, n_hops, block_hops):
        b1 = min(b0 + block_hops, n_hops)
        x = wav.read(b0 * HOP_S, b1 * HOP_S)
        if x.size < n_fft:
            break
        m = x.size // hop
        sq = x[: m * hop].reshape(m, hop)
        loud[b0 : b0 + m] = 20 * np.log10(np.sqrt(np.mean(sq * sq, axis=1)) + 1e-5)

        n_frames = 1 + (x.size - n_fft) // stft_hop
        idx = np.arange(n_fft)[None, :] + stft_hop * np.arange(n_frames)[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * window, axis=1)).astype(np.float32)
        logmag = np.log1p(10 * spec)
        diff = np.diff(logmag, axis=0, prepend=logmag[:1] if prev_logmag is None else prev_logmag)
        frame_flux = np.maximum(diff, 0).mean(axis=1)
        power = spec * spec
        frame_hf = power[:, hf_band].sum(axis=1) / (power.sum(axis=1) + 1e-9)
        prev_logmag = logmag[-1:]

        hop_of_frame = (np.arange(n_frames) * stft_hop + n_fft / 2) // hop
        for h in range(m):
            sel = hop_of_frame == h
            if sel.any():
                # p90 not mean: a single impact inside a quiet half-second should still register
                flux[b0 + h] = np.percentile(frame_flux[sel], 90)
                hf[b0 + h] = frame_hf[sel].mean()
    return {"loudness_db": loud, "flux": flux, "hf_ratio": hf}


# ---------------------------------------------------------------- speech

_LAUGH = re.compile(r"\b(?:ha){2,}\b|\bhe(?:he)+\b|\blo+l\b|\blmao\b|laugh", re.I)


def speech_features(transcript: Transcript | None, n_hops: int) -> dict[str, Curve]:
    wps = np.zeros(n_hops, np.float32)
    exclaim = np.zeros(n_hops, np.float32)
    laugh = np.zeros(n_hops, np.float32)
    if transcript is None:
        return {"speech_wps": wps, "exclaim": exclaim, "laugh_text": laugh}
    for seg in transcript.segments:
        a = max(0, int(seg.start_s / HOP_S))
        b = min(n_hops, math.ceil(seg.end_s / HOP_S))
        if "!" in seg.text:
            exclaim[a:b] = 1.0
        if _LAUGH.search(seg.text):
            laugh[a:b] = 1.0
        for w in seg.words:
            h = int((w.start_s + w.end_s) / 2 / HOP_S)
            if 0 <= h < n_hops:
                wps[h] += seg.confidence / HOP_S
    return {"speech_wps": wps, "exclaim": exclaim, "laugh_text": laugh}


# ---------------------------------------------------------------- visual


@dataclass(slots=True)
class _VisualState:
    prev_small: npt.NDArray[np.float32] | None = None
    prev_hist: npt.NDArray[np.float32] | None = None
    ema_thumb: npt.NDArray[np.float32] | None = None


def frame_features(
    rgb: npt.NDArray[np.uint8], state: _VisualState
) -> tuple[float, float, float, float, float]:
    """(motion, novelty, cut_distance, brightness, saturation) for one frame."""
    f = rgb.astype(np.float32)
    gray = f @ np.array([0.299, 0.587, 0.114], np.float32)
    h, w = gray.shape
    small = gray.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    thumb = gray.reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))

    hist = np.concatenate(
        [np.histogram(rgb[..., c], bins=16, range=(0, 256))[0] for c in range(3)]
    ).astype(np.float32)
    hist /= hist.sum() or 1.0

    motion = (
        0.0 if state.prev_small is None else float(np.abs(small - state.prev_small).mean() / 255)
    )
    cut = 0.0 if state.prev_hist is None else float(0.5 * np.abs(hist - state.prev_hist).sum())
    if state.ema_thumb is None:
        state.ema_thumb = thumb
    novelty = float(np.abs(thumb - state.ema_thumb).mean() / 255)
    state.ema_thumb = state.ema_thumb * 0.975 + thumb * 0.025  # ~10 s memory at 4 fps

    mx, mn = f.max(axis=2), f.min(axis=2)
    saturation = float(((mx - mn) / (mx + 1e-3)).mean())
    state.prev_small, state.prev_hist = small, hist
    return motion, novelty, cut, float(gray.mean() / 255), saturation


def _decode_segment(
    decoder: MediaDecoder, uri: str, start: float, length: float, on_frames: Callable[[int], None]
) -> npt.NDArray[np.float32]:
    """Rows: (t, motion, novelty, cut, brightness, saturation)."""
    rows: list[tuple[float, float, float, float, float, float]] = []
    state = _VisualState()
    for batch in decoder.iter_proxy_frames(uri, start, length, FPS, FRAME_W, FRAME_H, 16):
        frames = np.frombuffer(batch.rgb, np.uint8).reshape(-1, FRAME_H, FRAME_W, 3)
        for t, fr in zip(batch.timestamps_s, frames, strict=True):
            rows.append((t, *frame_features(fr, state)))
        on_frames(len(frames))
    return np.array(rows, np.float32).reshape(-1, 6)


def aggregate_visual(rows: npt.NDArray[np.float32], n_hops: int) -> dict[str, Curve]:
    out = {
        k: np.full(n_hops, np.nan, np.float32)
        for k in ("motion", "novelty", "cut", "brightness", "saturation")
    }
    if rows.size == 0:
        return out
    hop_idx = np.clip((rows[:, 0] / HOP_S).astype(int), 0, n_hops - 1)
    for col, name, fn in (
        (1, "motion", np.mean),
        (2, "novelty", np.max),
        (3, "cut", np.max),
        (4, "brightness", np.mean),
        (5, "saturation", np.mean),
    ):
        sums = np.zeros(n_hops, np.float64)
        if fn is np.mean:
            cnt = np.bincount(hop_idx, minlength=n_hops)
            sums = np.bincount(hop_idx, weights=rows[:, col], minlength=n_hops)
            with np.errstate(invalid="ignore"):
                out[name] = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan).astype(np.float32)
        else:
            vals = np.full(n_hops, -np.inf)
            np.maximum.at(vals, hop_idx, rows[:, col])
            out[name] = np.where(np.isfinite(vals), vals, np.nan).astype(np.float32)
    return out


def detect_cuts(cut: Curve) -> Curve:
    """Cut where the colour-histogram jump is extreme for this video and large in absolute terms."""
    c = np.nan_to_num(cut)
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med))) * 1.4826 or 0.01
    return ((c > med + 8 * mad) & (c > 0.3)).astype(np.float32)


# ---------------------------------------------------------------- fusion


def interest_curve(f: dict[str, Curve]) -> dict[str, Curve]:
    """Preliminary hand-weighted fusion; Phase 5 replaces it with candidate scoring + AI ranking.
    Loudness is taken relative to the surrounding minute so sudden spikes count, not volume."""
    loud = np.nan_to_num(f["loudness_db"], nan=-90.0)
    loud_rel = loud - rolling_median(loud, 121)
    z = {
        "loud_rel": robust_z(loud_rel),
        "flux": robust_z(f["flux"]),
        "hf": robust_z(f["hf_ratio"]) * (loud > np.median(loud)),
        "motion": robust_z(f["motion"]),
        "novelty": robust_z(f["novelty"]),
        "speech": robust_z(f["speech_wps"]),
    }
    # Cap each signal so one maxed-out channel (e.g. intro music kicking in, no action on
    # screen) can't outrank moments where several signals agree
    c = {k: np.clip(v, -3.0, 4.0) for k, v in z.items()}
    raw = (
        0.30 * c["loud_rel"]
        + 0.20 * c["flux"]
        + 0.10 * c["hf"]
        + 0.20 * c["motion"]
        + 0.10 * c["novelty"]
        + 0.05 * c["speech"]
        + 0.6 * f["laugh_text"]
        + 0.4 * f["exclaim"]
    )
    # Menus/loading/AFK: near-still picture with quiet audio
    still = (z["motion"] < -0.8) & (z["loud_rel"] < 0)
    raw = raw - 0.8 * moving_average(still.astype(np.float32), 20)
    return {
        **{f"z_{k}": v for k, v in z.items()},
        "loud_rel_db": loud_rel.astype(np.float32),
        "still": still.astype(np.float32),
        "interest": moving_average(raw.astype(np.float32), 6),
    }


@dataclass(frozen=True, slots=True)
class Peak:
    t_s: float
    score: float
    drivers: tuple[str, ...]


DRIVER_LABELS = {
    "z_loud_rel": "loud spike",
    "z_flux": "impacts",
    "z_hf": "shouting",
    "z_motion": "fast motion",
    "z_novelty": "visual change",
    "z_speech": "talking fast",
    "laugh_text": "laughter",
    "exclaim": "exclamation",
}


def top_peaks(tl: SignalTimeline, k: int = 15, min_gap_s: float = 30.0) -> list[Peak]:
    interest = tl.features["interest"]
    order = np.argsort(-interest)
    gap = int(min_gap_s / tl.hop_s)
    taken: list[int] = []
    for i in order:
        if all(abs(int(i) - j) >= gap for j in taken):
            taken.append(int(i))
            if len(taken) == k:
                break
    peaks = []
    for i in taken:
        lo, hi = max(0, i - 6), i + 7
        strength = {
            name: float(np.nanmax(tl.features[name][lo:hi]))
            for name in DRIVER_LABELS
            if name in tl.features
        }
        drivers = tuple(
            DRIVER_LABELS[n]
            for n, v in sorted(strength.items(), key=lambda kv: -kv[1])
            if v >= (1.0 if n in ("laugh_text", "exclaim") else 2.0)
        )[:3]
        peaks.append(Peak(i * tl.hop_s, float(interest[i]), drivers))
    return peaks


# ---------------------------------------------------------------- stage


class SignalsStage:
    name = "signals"
    version = 2  # v2: capped per-signal contribution

    def __init__(self, decoder: MediaDecoder, storage: StorageProvider) -> None:
        self._decoder = decoder
        self._storage = storage

    def timeline_path(self, project_id: str) -> Path:
        return self._storage.artifact_path(project_id, "analysis", "signals.npz")

    def run(
        self,
        asset: MediaAsset,
        wav_path: Path | None,
        transcript: Transcript | None,
        manifest: StageManifest,
        progress: StageProgress,
    ) -> SignalTimeline:
        assert asset.media_info is not None
        duration = asset.media_info.duration_s
        transcript_id = (transcript.model, len(transcript.segments)) if transcript else None
        key = stage_key(self.name, self.version, asset.fingerprint, str(wav_path), transcript_id)
        out = self.timeline_path(asset.project_id)
        if manifest.completed(self.name, key) and out.is_file():
            return SignalTimeline.load(out)

        n_hops = math.ceil(duration / HOP_S)
        feats: dict[str, Curve] = {}

        progress(0.0, "audio signals")
        if wav_path is not None:
            feats.update(audio_features(WavFile(wav_path), n_hops))
        else:
            feats.update(
                {k: np.zeros(n_hops, np.float32) for k in ("loudness_db", "flux", "hf_ratio")}
            )
        feats.update(speech_features(transcript, n_hops))
        progress(0.05, "visual signals")

        # Decode checkpoints depend only on decode settings, so re-tuning the fusion
        # never forces another pass over hours of video
        visual_key = stage_key("visual", 1, asset.fingerprint, FPS, FRAME_W, FRAME_H, SEGMENT_S)
        rows = self._visual_rows(asset, visual_key[:12], duration, progress)
        visual = aggregate_visual(rows, n_hops)
        visual["scene_cut"] = detect_cuts(visual["cut"])
        feats.update(visual)
        feats.update(interest_curve(feats))

        tl = SignalTimeline(HOP_S, duration, feats)
        tl.save(out)
        manifest.mark(self.name, key, {"files": [str(out)], "hops": n_hops})
        progress(1.0, None)
        return tl

    def _visual_rows(
        self, asset: MediaAsset, key: str, duration: float, progress: StageProgress
    ) -> npt.NDArray[np.float32]:
        parts_dir = self._storage.artifact_path(asset.project_id, "analysis", f"visual_{key}")
        parts_dir.mkdir(parents=True, exist_ok=True)
        n_seg = max(1, math.ceil(duration / SEGMENT_S))
        total_frames = duration * FPS
        done_frames = 0.0
        out: list[npt.NDArray[np.float32]] = []
        for s in range(n_seg):
            start = s * SEGMENT_S
            length = min(SEGMENT_S, duration - start)
            part = parts_dir / f"seg_{s:04d}.npy"
            msg = f"decoding {_hms(start)} / {_hms(duration)}"
            if part.is_file():
                rows = np.load(part)
            else:
                seen = [0]

                def on_frames(
                    n: int, b: float = done_frames, m: str = msg, c: list[int] = seen
                ) -> None:
                    c[0] += n
                    progress(0.05 + 0.95 * (b + c[0]) / total_frames, m)

                rows = _decode_segment(self._decoder, asset.source_uri, start, length, on_frames)
                tmp = part.with_name(part.stem + ".tmp.npy")
                np.save(tmp, rows)
                tmp.replace(part)
            done_frames += length * FPS
            progress(0.05 + 0.95 * done_frames / total_frames, msg)
            out.append(rows)
        return np.concatenate(out) if out else np.zeros((0, 6), np.float32)


def _hms(t: float) -> str:
    s = int(t)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
