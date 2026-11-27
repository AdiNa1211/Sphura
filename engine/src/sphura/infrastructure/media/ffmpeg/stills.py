from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sphura.infrastructure.media.ffmpeg.process import run, run_bytes
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain


class FFmpegFrameSampler:
    """JPEG stills via fast input seeking: each frame costs one short decode near a keyframe."""

    def __init__(self, toolchain: FFmpegToolchain, workers: int = 4) -> None:
        self._tc = toolchain
        self._workers = workers

    def sample(
        self,
        uri: str,
        times_s: list[float],
        width: int,
        box: tuple[float, float, float, float] | None = None,
    ) -> list[bytes]:
        vf = f"scale={width}:-2:flags=bicubic"
        if box:
            x0, y0, x1, y1 = box
            vf = f"crop=iw*{x1 - x0:.4f}:ih*{y1 - y0:.4f}:iw*{x0:.4f}:ih*{y0:.4f}," + vf

        def one(t: float) -> bytes:
            return run_bytes(
                [
                    self._tc.ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
                    "-ss", f"{max(t, 0):.3f}", "-i", uri, "-frames:v", "1",
                    "-vf", vf, "-q:v", "4",
                    "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1",
                ],
                timeout_s=60,
            )  # fmt: skip

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(one, times_s))


def retime_wav(ffmpeg: str, path: Path, speed: float) -> None:
    """Speeds speech up or slows it down without changing its pitch, in place."""
    speed = min(max(speed, 0.5), 2.0)
    tmp = path.with_name(path.stem + ".tempo.wav")
    run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y", "-i", str(path),
         "-filter:a", f"atempo={speed:.3f}", str(tmp)], timeout_s=120)  # fmt: skip
    tmp.replace(path)


POLISH = {  # conservative: a little compression, then loudness; "off" leaves the voice alone
    "light": "acompressor=threshold=-24dB:ratio=1.5:attack=15:release=200:makeup=1,"
             "loudnorm=I=-18:TP=-2:LRA=11",
    "standard": "acompressor=threshold=-22dB:ratio=2.5:attack=10:release=150:makeup=1,"
                "loudnorm=I=-16:TP=-1.5:LRA=9",
}  # fmt: skip


def polish_wav(ffmpeg: str, path: Path, mode: str) -> None:
    """Gentle compression and loudness normalisation of a voice-over, in place."""
    chain = POLISH.get(mode)
    if not chain:
        return
    with open(path, "rb") as f:
        f.seek(24)
        sr = int.from_bytes(f.read(4), "little")  # keep the rate: loudnorm outputs 192 kHz
    tmp = path.with_name(path.stem + ".polish.wav")
    run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y", "-i", str(path),
         "-af", chain, "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
        timeout_s=120)  # fmt: skip
    tmp.replace(path)
