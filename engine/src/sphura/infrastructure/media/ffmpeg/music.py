"""The user's own music folder: finds tracks and measures where each one drops."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sphura.domain.errors import ToolchainError
from sphura.infrastructure.media.ffmpeg.process import run_bytes
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain
from sphura.pipeline.effects import MusicTrack

log = logging.getLogger(__name__)

EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
RATE = 4000  # plenty for an energy envelope
FRAME_S = 0.1
ONSET_S = 0.02  # finer envelope for beat tracking
CACHE_VERSION = 6
META_FILE = "sphura_tracks.json"  # source/license/credit/tags, kept next to the music
_META_KEYS = ("source", "license", "credit", "tags")


class MusicLibrary:
    def __init__(self, folder: Path, toolchain: FFmpegToolchain, cache_path: Path) -> None:
        self.folder = folder
        self._tc = toolchain
        self._cache_path = cache_path
        self._lock = threading.Lock()

    def tracks(self) -> list[MusicTrack]:
        if not self.folder.is_dir():
            return []
        files = sorted(p for p in self.folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
        meta = self.meta()
        with self._lock:
            cache = self._read_cache()
            out: list[MusicTrack] = []
            for f in files:
                key = f"v{CACHE_VERSION}|{f}|{f.stat().st_mtime_ns}"
                if key not in cache:
                    try:
                        cache[key] = self._analyze(f)
                    except (ToolchainError, ValueError) as e:
                        log.warning("skipping music %s: %s", f.name, e)
                        continue
                a = cache[key]
                if a["length_s"] < 15:
                    continue
                m = meta.get(f.name, {})
                out.append(
                    MusicTrack(
                        str(f),
                        a["length_s"],
                        a["drop_s"],
                        a["energy"],
                        a["beat_s"],
                        a["beat0_s"],
                        a["brightness"],
                        a["dynamics_db"],
                        **{k: str(m.get(k, "")) for k in _META_KEYS},
                    )
                )
            self._write_cache(cache)
        return out

    def analyze(self, path: Path, **meta: str) -> MusicTrack:
        """Beat grid, drop and loudness of one file (e.g. AI-generated music)."""
        a = self._analyze(path)
        return MusicTrack(
            str(path), a["length_s"], a["drop_s"], a["energy"], a["beat_s"], a["beat0_s"],
            a["brightness"], a["dynamics_db"],
            **{k: str(meta.get(k, "")) for k in _META_KEYS},
        )  # fmt: skip

    def meta(self) -> dict[str, dict[str, str]]:
        try:
            raw = json.loads((self.folder / META_FILE).read_text(encoding="utf-8"))
            return {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            return {}

    def set_meta(self, filename: str, values: dict[str, str]) -> None:
        name = Path(filename).name
        if not (self.folder / name).is_file():
            raise ValueError("no such track in the music folder")
        with self._lock:
            meta = self.meta()
            entry = meta.get(name, {})
            entry.update({k: str(v).strip()[:500] for k, v in values.items() if k in _META_KEYS})
            meta[name] = entry
            (self.folder / META_FILE).write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _analyze(self, path: Path) -> dict[str, float]:
        raw = run_bytes(
            [self._tc.ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(RATE), "-f", "s16le", "-"],
            timeout_s=120,
        )  # fmt: skip
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        n = int(RATE * FRAME_S)
        if x.size < n * 50:
            raise ValueError("too short")
        rms = np.sqrt((x[: x.size // n * n].reshape(-1, n) ** 2).mean(axis=1))
        db = 20 * np.log10(rms + 1e-4)
        # Drop: the biggest jump from a quieter 3 s into a louder 1.5 s, after the intro
        before, after = int(3 / FRAME_S), int(1.5 / FRAME_S)
        best, drop = -1e9, 0.0
        for i in range(max(before, int(6 / FRAME_S)), len(db) - after):
            rise = db[i : i + after].mean() - db[i - before : i].mean()
            if rise > best:
                best, drop = float(rise), i * FRAME_S
        beat, beat0 = _beat_grid(x)
        sec = int(1 / FRAME_S)
        db1 = 20 * np.log10(
            np.sqrt((rms[: rms.size // sec * sec] ** 2).reshape(-1, sec).mean(axis=1)) + 1e-4
        )
        loud = db1[db1 > db1.max() - 40]  # 1 s windows: build-ups and drops, not drum decays
        return {
            "length_s": round(len(db) * FRAME_S, 2),
            "drop_s": round(drop, 2),
            "energy": round(float(np.mean(rms)), 4),
            "beat_s": round(beat, 4),
            "beat0_s": round(beat0, 4),
            "brightness": round(_brightness(x), 3),
            "dynamics_db": round(float(np.percentile(loud, 95) - np.percentile(loud, 10)), 1),
        }

    def _read_cache(self) -> dict[str, dict[str, float]]:
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            return {}

    def _write_cache(self, cache: dict[str, dict[str, float]]) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass


def _brightness(x: npt.NDArray[np.floating[Any]]) -> float:
    """Share of energy above 700 Hz (of what a 4 kHz decode keeps): bassy tracks score low."""
    win = RATE  # 1 s windows
    frames = x[: x.size // win * win].reshape(-1, win)
    spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / RATE)
    total = spec[:, freqs > 30].sum()
    return float(spec[:, freqs > 700].sum() / total) if total > 0 else 0.5


def _beat_grid(x: npt.NDArray[np.floating[Any]]) -> tuple[float, float]:
    """(beat period, a beat time) from the onset envelope: tempo by autocorrelation in the
    70-180 BPM range, phase by where onsets pile up. (0, 0) when there is no clear pulse."""
    # Spectral flux: rises in any frequency band. Sustained notes give none (a loudness
    # envelope ripples on them and fakes a pulse)
    hop, win = int(RATE * ONSET_S), 256
    count = (x.size - win) // hop
    if count < 500:
        return 0.0, 0.0
    idx = np.arange(win)[None, :] + hop * np.arange(count)[:, None]
    mag = np.log1p(50 * np.abs(np.fft.rfft(x[idx] * np.hanning(win), axis=1)))
    onset = np.clip(np.diff(mag, axis=0), 0, None).sum(axis=1)
    # Remove slow swells (fades, pads) so only sharp hits count as beats
    k = int(0.5 / ONSET_S)
    onset = onset - np.convolve(onset, np.ones(k) / k, mode="same")
    if onset.size < 500 or np.std(onset) < 3.0:  # no hits worth calling a beat (pads, drones)
        return 0.0, 0.0
    ac = np.correlate(onset, onset, mode="full")[onset.size - 1 :]
    ac = ac / ac[0] if ac[0] > 0 else ac
    lo, hi = int(60 / 180 / ONSET_S), int(60 / 70 / ONSET_S)
    lag = lo + int(np.argmax(ac[lo:hi]))
    # A real pulse has a valley between beats; a smooth swell correlates with everything
    valley = float(ac[max(1, lag // 4) : max(2, 3 * lag // 4)].min())
    if ac[lag] < 0.15 or ac[lag] - valley < 0.25:
        return 0.0, 0.0
    # Tempo-halving check: if half the period is nearly as strong, the beat is twice as fast
    half = lag // 2
    if half >= int(60 / 200 / ONSET_S):
        h = half - 1 + int(np.argmax(ac[half - 1 : half + 2]))
        if ac[h] >= 0.6 * ac[lag]:
            lag = h
    # Refine: the peak 8 beats out, interpolated, pins the period to well under 1 ms
    k = 8
    c = k * lag
    if c + 4 < ac.size:
        i = c - 3 + int(np.argmax(ac[c - 3 : c + 4]))
        y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        period = (i + frac) / k * ONSET_S
    else:
        period = lag * ONSET_S
    beats = np.arange(0, (onset.size - 1) * ONSET_S, period)
    phases = np.arange(0, period, ONSET_S / 2)
    scores = [onset[((ph + beats) / ONSET_S).astype(int).clip(0, onset.size - 1)].sum()
              for ph in phases]  # fmt: skip
    return float(period), float(phases[int(np.argmax(scores))] + ONSET_S)
