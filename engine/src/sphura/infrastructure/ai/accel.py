"""Verifies GPU speech-to-text works before committing a long job to it.

A missing cuDNN/cuBLAS DLL can abort the whole process inside CTranslate2 (no Python exception),
so the check runs a tiny model in a throwaway subprocess. Run as a module, it is that subprocess.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import site
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_PROBE_MODEL = "tiny"  # ~75 MB; enough to exercise the encoder and decoder kernels
_CHECK_VERSION = 1
_EXIT_DOWNLOAD_FAILED = 3
_LIB_HINT = (
    "usually missing CUDA 12 / cuDNN 9 libraries (pip install nvidia-cublas-cu12 nvidia-cudnn-cu12)"
)


def register_nvidia_dlls() -> None:
    """pip's nvidia-cublas/cudnn wheels put DLLs in site-packages/nvidia/*/bin, which Windows
    doesn't search. Add them before CTranslate2 loads."""
    if sys.platform != "win32":
        return
    roots = [*site.getsitepackages(), site.getusersitepackages()]
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                continue
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def cuda_device_count() -> int:
    register_nvidia_dlls()
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _ct2_version() -> str:
    try:
        import ctranslate2

        return str(ctranslate2.__version__)
    except Exception:
        return "missing"


@dataclass(frozen=True, slots=True)
class AccelCheck:
    device: str  # device that was tested
    ok: bool
    detail: str


class WhisperAccelChecker:
    def __init__(self, models_dir: Path, cache_path: Path) -> None:
        self._models_dir = models_dir
        self._cache_path = cache_path
        self._lock = threading.Lock()

    def check(self, device: str, refresh: bool = False) -> AccelCheck:
        if device == "cpu":
            return AccelCheck("cpu", True, "CPU needs no check")
        key = f"{_CHECK_VERSION}|{device}|{_ct2_version()}|{sys.executable}"
        with self._lock:
            cache = self._read()
            if not refresh and key in cache:
                c = cache[key]
                return AccelCheck(str(c["device"]), bool(c["ok"]), str(c["detail"]))
            result, cacheable = self._run_probe(device)
            if not cacheable:  # network trouble says nothing about the GPU; retry next time
                return result
            cache[key] = asdict(result)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(cache, indent=2))
            return result

    def _run_probe(self, device: str) -> tuple[AccelCheck, bool]:
        log.info("testing Whisper on %s (one-time check)", device)
        cmd = [sys.executable, "-m", __name__, device, str(self._models_dir)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired:
            return AccelCheck(device, False, "GPU test timed out"), True
        if proc.returncode == 0 and "SPHURA_ACCEL_OK" in proc.stdout:
            return AccelCheck(device, True, "GPU test passed"), True
        out = f"{proc.stdout}\n{proc.stderr}"
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if proc.returncode == _EXIT_DOWNLOAD_FAILED:
            reason = lines[-1][:200] if lines else "network error"
            return AccelCheck(device, False, f"couldn't download the test model: {reason}"), False
        # Prefer a line naming the missing library (CTranslate2 prints it before aborting)
        dll = next((ln for ln in lines if ".dll" in ln.lower() or ".so" in ln.lower()), None)
        if dll:
            return AccelCheck(device, False, f"{dll[:200]} - {_LIB_HINT}"), True
        if not lines or proc.returncode < 0 or proc.returncode > 255:
            crashed = f"GPU test crashed (code {proc.returncode}), {_LIB_HINT}"
            return AccelCheck(device, False, crashed), True
        return AccelCheck(device, False, lines[-1][:300]), True

    def _read(self) -> dict[str, dict[str, object]]:
        try:
            data: dict[str, dict[str, object]] = json.loads(self._cache_path.read_text())
            return data
        except (FileNotFoundError, ValueError):
            return {}


def _probe(device: str, models_dir: str) -> None:
    register_nvidia_dlls()
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    try:
        path = download_model(_PROBE_MODEL, cache_dir=models_dir)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(_EXIT_DOWNLOAD_FAILED)
    compute = "float16" if device == "cuda" else "auto"
    model = WhisperModel(path, device=device, compute_type=compute)
    t = np.arange(16_000 * 3, dtype=np.float32) / 16_000
    audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    segments, _ = model.transcribe(audio, language="en", vad_filter=False, beam_size=1)
    list(segments)  # decoding is lazy; iterate to actually run the GPU kernels
    print("SPHURA_ACCEL_OK")


if __name__ == "__main__":
    _probe(sys.argv[1], sys.argv[2])
