"""SignalTimeline: per-video feature curves on a fixed time grid, stored as .npz."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Curve = npt.NDArray[np.float32]


@dataclass(slots=True)
class SignalTimeline:
    hop_s: float
    duration_s: float
    features: dict[str, Curve] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return math.ceil(self.duration_s / self.hop_s)

    def times(self) -> Curve:
        return (np.arange(self.length, dtype=np.float32) * self.hop_s).astype(np.float32)

    def save(self, path: Path) -> None:
        tmp = path.with_name(path.name + ".tmp.npz")
        arrays: dict[str, Any] = {
            "__hop_s": np.float64(self.hop_s),
            "__duration_s": np.float64(self.duration_s),
            **self.features,
        }
        save: Any = np.savez_compressed  # numpy's stub can't type **arrays
        save(tmp, **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> SignalTimeline:
        with np.load(path) as z:
            feats = {k: z[k].astype(np.float32) for k in z.files if not k.startswith("__")}
            return cls(float(z["__hop_s"]), float(z["__duration_s"]), feats)


def robust_z(x: Curve, clip: tuple[float, float] = (-4.0, 8.0)) -> Curve:
    """Per-video normalisation: a loud streamer's baseline is not 'exciting' by itself."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * 1.4826
    scale = mad if mad > 1e-6 else (float(np.std(finite)) or 1.0)
    z = (np.nan_to_num(x, nan=med) - med) / scale
    return np.asarray(np.clip(z, clip[0], clip[1]), dtype=np.float32)


def rolling_median(x: Curve, window: int) -> Curve:
    if x.size == 0:
        return x
    half = window // 2
    padded = np.pad(np.nan_to_num(x), (half, half), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.asarray(np.median(view, axis=1), dtype=np.float32)


def moving_average(x: Curve, window: int) -> Curve:
    if window <= 1 or x.size == 0:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(
        np.pad(x, (window // 2, window - 1 - window // 2), mode="edge"), kernel, "valid"
    ).astype(np.float32)
