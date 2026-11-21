from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

from sphura.infrastructure.media.ffmpeg.process import run_with_progress
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain


class FFmpegAudioExtractor:
    """Decodes every requested audio stream to 16-bit mono WAV in a single read of the source."""

    def __init__(self, toolchain: FFmpegToolchain) -> None:
        self._tc = toolchain

    def extract(
        self,
        uri: str,
        outputs: dict[int, Path],
        sample_rate: int,
        duration_s: float,
        on_progress: Callable[[float], None],
    ) -> None:
        cmd = [self._tc.ffmpeg, "-hide_banner", "-v", "error", "-y", "-i", uri]
        parts: dict[Path, Path] = {}
        for index, out in outputs.items():
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            parts[part] = out
            cmd += [
                "-map", f"0:{index}", "-vn", "-sn", "-dn",
                "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
                "-rf64", "auto", "-f", "wav", str(part),
            ]  # fmt: skip

        try:
            run_with_progress(cmd, duration_s, lambda frac, _t: on_progress(frac))
        except BaseException:
            for part in parts:
                with contextlib.suppress(OSError):
                    part.unlink()
            raise
        # Rename only after success so a half-written WAV is never mistaken for a finished one
        for part, out in parts.items():
            part.replace(out)
