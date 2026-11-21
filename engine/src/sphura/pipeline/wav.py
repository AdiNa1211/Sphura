"""Memory-mapped access to 16-bit mono PCM WAV files, so hours of audio never sit in RAM."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt


class WavFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        offset, size, self.sample_rate, channels, bits = _parse_header(path)
        if channels != 1 or bits != 16:
            raise ValueError(f"{path.name}: expected 16-bit mono PCM, got {channels}ch/{bits}bit")
        self._pcm: npt.NDArray[np.int16] = np.memmap(
            path, dtype="<i2", mode="r", offset=offset, shape=(size // 2,)
        )

    @property
    def num_samples(self) -> int:
        return int(self._pcm.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate

    def read(self, start_s: float, end_s: float) -> npt.NDArray[np.float32]:
        a = max(0, int(start_s * self.sample_rate))
        b = min(self.num_samples, int(end_s * self.sample_rate))
        return np.asarray(self._pcm[a:b], dtype=np.float32) / 32768.0

    def rms_frames(
        self, start_s: float, end_s: float, frame_s: float = 0.1
    ) -> npt.NDArray[np.float32]:
        x = self.read(start_s, end_s)
        n = int(frame_s * self.sample_rate)
        usable = (len(x) // n) * n
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        frames = x[:usable].reshape(-1, n)
        return np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)


def _parse_header(path: Path) -> tuple[int, int, int, int, int]:
    """Returns (data offset, data size, sample rate, channels, bits). Handles RIFF and RF64."""
    with path.open("rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] not in (b"RIFF", b"RF64") or riff[8:12] != b"WAVE":
            raise ValueError(f"{path.name}: not a WAV file")
        rf64_data_size: int | None = None
        fmt: tuple[int, int, int] | None = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                raise ValueError(f"{path.name}: no data chunk")
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"ds64":
                body = f.read(size)
                rf64_data_size = struct.unpack("<Q", body[8:16])[0]
                continue
            if cid == b"fmt ":
                body = f.read(size)
                _, channels, rate = struct.unpack("<HHI", body[:8])
                bits = struct.unpack("<H", body[14:16])[0]
                fmt = (rate, channels, bits)
                continue
            if cid == b"data":
                if fmt is None:
                    raise ValueError(f"{path.name}: data before fmt chunk")
                data_size = rf64_data_size if size == 0xFFFFFFFF and rf64_data_size else size
                # ffmpeg writes a placeholder size when it can't seek; trust the file length then
                remaining = path.stat().st_size - f.tell()
                return f.tell(), min(data_size, remaining), *fmt
            f.seek(size + (size & 1), 1)
