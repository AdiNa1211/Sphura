from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from sphura.infrastructure.media.ffmpeg.process import run_bytes
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain


class FFmpegFrameSampler:
    """JPEG stills via fast input seeking: each frame costs one short decode near a keyframe."""

    def __init__(self, toolchain: FFmpegToolchain, workers: int = 4) -> None:
        self._tc = toolchain
        self._workers = workers

    def sample(self, uri: str, times_s: list[float], width: int) -> list[bytes]:
        def one(t: float) -> bytes:
            return run_bytes(
                [
                    self._tc.ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
                    "-ss", f"{max(t, 0):.3f}", "-i", uri, "-frames:v", "1",
                    "-vf", f"scale={width}:-2:flags=bicubic", "-q:v", "4",
                    "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1",
                ],
                timeout_s=60,
            )  # fmt: skip

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(one, times_s))
