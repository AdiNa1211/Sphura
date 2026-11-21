from __future__ import annotations

import collections
import subprocess
import threading
from collections.abc import Iterator

from sphura.domain.errors import ToolchainError
from sphura.infrastructure.media.ffmpeg.process import POPEN_FLAGS
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain
from sphura.ports.media import FrameBatch


class FFmpegFrameDecoder:
    """Streams small RGB frames over a pipe. Uses hardware decoding when FFmpeg has any."""

    def __init__(self, toolchain: FFmpegToolchain) -> None:
        self._tc = toolchain

    def iter_proxy_frames(
        self,
        uri: str,
        start_s: float,
        duration_s: float,
        fps: float,
        width: int,
        height: int,
        batch: int,
    ) -> Iterator[FrameBatch]:
        hw = ["-hwaccel", "auto"] if self._tc.profile().hwaccels else []
        vf = (
            f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease"
            f":flags=area,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=rgb24"
        )
        cmd = [
            self._tc.ffmpeg, "-hide_banner", "-v", "error", "-nostdin", *hw,
            "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}", "-i", uri,
            "-map", "0:v:0", "-an", "-sn", "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]  # fmt: skip
        frame_bytes = width * height * 3
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=frame_bytes * batch,
            **POPEN_FLAGS,
        )
        err_tail: collections.deque[bytes] = collections.deque(maxlen=20)
        drain = threading.Thread(target=lambda: err_tail.extend(proc.stderr or ()), daemon=True)
        drain.start()

        index = 0
        try:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(frame_bytes * batch)
                n = len(buf) // frame_bytes
                if n:
                    ts = tuple(start_s + (index + i) / fps for i in range(n))
                    index += n
                    yield FrameBatch(ts, width, height, buf[: n * frame_bytes])
                if len(buf) < frame_bytes * batch:
                    break
            code = proc.wait()
        finally:
            if proc.poll() is None:  # consumer stopped early (cancel): don't leave ffmpeg running
                proc.kill()
                proc.wait()
            drain.join(timeout=5)
        if code != 0:
            tail = b"".join(err_tail).decode("utf-8", "replace").strip().splitlines()[-5:]
            raise ToolchainError(f"frame decode failed ({code}): {' | '.join(tail)}")
