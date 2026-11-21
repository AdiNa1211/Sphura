"""Subprocess plumbing for FFmpeg/FFprobe: errors, progress parsing, cancellation."""

from __future__ import annotations

import collections
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sphura.domain.errors import ToolchainError

# Avoid console windows flashing when the engine runs as a Windows GUI sidecar
POPEN_FLAGS: dict[str, Any] = (
    {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}
)


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def run(cmd: Sequence[str], timeout_s: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run to completion; raise ToolchainError on non-zero exit, keeping stderr for diagnosis."""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            check=False,
            **POPEN_FLAGS,
        )
    except FileNotFoundError as e:
        raise ToolchainError(f"executable not found: {cmd[0]}") from e
    except subprocess.TimeoutExpired as e:
        raise ToolchainError(f"{_name(cmd)} timed out after {timeout_s:.0f}s") from e
    if proc.returncode != 0:
        raise ToolchainError(f"{_name(cmd)} failed ({proc.returncode}): {_tail(proc.stderr)}")
    return proc


def run_bytes(cmd: Sequence[str], timeout_s: float = 60.0) -> bytes:
    """Like run(), but returns raw stdout (e.g. an encoded image)."""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            check=False,
            **POPEN_FLAGS,
        )
    except FileNotFoundError as e:
        raise ToolchainError(f"executable not found: {cmd[0]}") from e
    except subprocess.TimeoutExpired as e:
        raise ToolchainError(f"{_name(cmd)} timed out after {timeout_s:.0f}s") from e
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace")
        raise ToolchainError(f"{_name(cmd)} failed ({proc.returncode}): {_tail(err)}")
    return bytes(proc.stdout)


def run_with_progress(
    cmd: Sequence[str],
    duration_s: float | None,
    on_progress: Callable[[float, float], None],
    cwd: Path | None = None,
) -> float:
    """Run an ffmpeg command, reporting (fraction, media_seconds) from `-progress pipe:1`.

    Exceptions raised by on_progress (e.g. cancellation) kill the process and propagate.
    Returns the last media timestamp reached, in seconds.
    """
    full = [cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]]
    try:
        proc = subprocess.Popen(
            full,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            **POPEN_FLAGS,
        )
    except FileNotFoundError as e:
        raise ToolchainError(f"executable not found: {cmd[0]}") from e

    # Drain stderr concurrently so a chatty ffmpeg can't block on a full pipe
    err_tail: collections.deque[str] = collections.deque(maxlen=40)
    drain = threading.Thread(target=lambda: err_tail.extend(proc.stderr or ()), daemon=True)
    drain.start()

    last_t = 0.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit():
                last_t = int(value) / 1_000_000
                frac = min(last_t / duration_s, 1.0) if duration_s else 0.0
                on_progress(frac, last_t)
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        drain.join(timeout=5)

    if code != 0:
        raise ToolchainError(f"{_name(cmd)} failed ({code}): {_tail(''.join(err_tail))}")
    return last_t


def _name(cmd: Sequence[str]) -> str:
    return cmd[0].replace("\\", "/").rsplit("/", 1)[-1]
