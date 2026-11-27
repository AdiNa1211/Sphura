"""Natural narrator voice: Qwen3-TTS (Apache 2.0) on this PC's GPU, in its own Python
environment managed by uv (like the AI music). Each voice is designed once from a
description, then every line is cloned from that reference, so a voice never drifts."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

from sphura.domain.errors import SphuraError
from sphura.infrastructure.ai.acestep import AceStep
from sphura.infrastructure.media.ffmpeg.process import POPEN_FLAGS

log = logging.getLogger(__name__)

PORT = 47823
TORCH = ["torch==2.7.1+cu128", "torchaudio==2.7.1+cu128"]  # same as the AI music: cached
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
MODELS = ["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]
SERVER = Path(__file__).with_name("qwen_tts_server.py")
SERVER_VERSION = 3  # bump when the worker changes, so a stale one is replaced

# id -> (description the voice is designed from, the line it's designed on)
DESIGNS = {
    "qwen_gamer": (
        "A young male gaming YouTuber narrating a boss fight, deep confident voice, natural "
        "and conversational, slightly hyped, with dramatic pauses.",
        "Alright, so here's the thing. This fight looks simple at first, but watch closely, "
        "because what happens next changes everything.",
    ),
    "qwen_story": (
        "A deep, calm male storyteller telling Norse myths, warm and cinematic, measured "
        "pace, natural breathing, quietly intense.",
        "Long before this moment, the gods made a choice. And every choice, sooner or later, "
        "comes back to collect its price.",
    ),
}


class QwenVoice:
    def __init__(self, root: Path, music_root: Path) -> None:
        self.root = root
        self._music = music_root  # ACE-Step's folder: reuse its uv and download cache
        self._voices = root / "voices"
        self._venv = root / ".venv"
        self._py = self._venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self.setup: dict[str, Any] = {"running": False, "step": "", "error": None}

    # --- install -----------------------------------------------------------------------

    def ready(self) -> bool:
        return (self.root / ".installed").is_file() and self._py.is_file()

    def install(self) -> dict[str, Any]:
        with self._lock:
            if self.setup["running"]:
                return self.setup
            self.setup = {"running": True, "step": "starting", "error": None}
        threading.Thread(target=self._install, daemon=True, name="qwen-install").start()
        return self.setup

    def _install(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            uv = self._uv()
            if not self._py.is_file():  # a retry keeps what an earlier attempt installed
                self._run([uv, "venv", "--python", "3.12", str(self._venv)],
                          "creating its Python")  # fmt: skip
            pip = [uv, "pip", "install", "--python", str(self._py)]
            if sys.platform == "win32":  # PyTorch's own index, as pytorch.org documents
                self._run([*pip, *TORCH, "--index-url", TORCH_INDEX],
                          "installing PyTorch for your GPU")  # fmt: skip
            self._run([*pip, "qwen-tts", "soundfile", "huggingface_hub"],
                      "installing Qwen3-TTS")  # fmt: skip
            # Windows needs admin or Developer Mode for symlinks: make the cache copy files
            get = (
                "import huggingface_hub.file_download as f; "
                "f.are_symlinks_supported = lambda *a, **k: False; "
                "from huggingface_hub import snapshot_download as s; "
            ) + "; ".join(f"s('{m}')" for m in MODELS)
            self._run([str(self._py), "-c", get], "downloading the voice models (~9 GB)")
            (self.root / ".installed").write_text(time.strftime("%Y-%m-%d"), encoding="utf-8")
            self.setup["step"] = "done"
        except Exception as e:
            self.setup["error"] = f"couldn't set up the natural voice: {e}"
        finally:
            self.setup["running"] = False

    def _uv(self) -> str:
        shared = self._music / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")
        if shared.is_file():
            return str(shared)
        own = AceStep(self.root)  # same download code as the AI music installer
        if not own._uv().is_file():
            own._get_uv()
        return str(own._uv())

    def _env(self) -> dict[str, str]:
        cache = self._music / "uv-cache"
        env = dict(os.environ)
        env.update(
            UV_CACHE_DIR=str(cache if cache.is_dir() else self.root / "uv-cache"),
            UV_PYTHON_INSTALL_DIR=str(self.root / "python"),
            HF_HOME=str(self.root / "hf"),
            HF_HUB_DISABLE_SYMLINKS_WARNING="1",
            PYTHONUTF8="1",
        )
        return env

    def _run(self, cmd: list[str], step: str) -> None:
        self.setup["step"] = step
        with (self.root / "install.log").open("a", encoding="utf-8") as logf:
            logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            logf.flush()
            code = subprocess.run(cmd, env=self._env(), stdout=logf, stderr=subprocess.STDOUT,
                                  **POPEN_FLAGS).returncode  # fmt: skip
        if code != 0:
            raise SphuraError(f"{step} failed (see {self.root / 'install.log'})")

    # --- worker ------------------------------------------------------------------------

    def _ensure_running(self, timeout_s: float = 900) -> None:
        info = self._health()
        if info is not None and info.get("version") == SERVER_VERSION:
            return
        if info is not None:  # an older worker left from a previous run
            self._kill(int(info.get("pid") or 0))
        if self._proc is None or self._proc.poll() is not None:
            logf = (self.root / "server.log").open("a", encoding="utf-8")
            logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} starting\n")
            logf.flush()
            self._proc = subprocess.Popen(
                [str(self._py), "-P", "-u", str(SERVER), str(PORT)],  # -P: sibling http.py
                env={**self._env(), "HF_HUB_OFFLINE": "1"},  # models are in the cache
                stdout=logf,
                stderr=subprocess.STDOUT, **POPEN_FLAGS,
            )  # fmt: skip
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            info = self._health()
            if info is not None and info.get("version") == SERVER_VERSION:
                return
            if self._proc.poll() is not None:
                raise SphuraError(f"the voice server stopped (see {self.root / 'server.log'})")
            time.sleep(2)
        raise SphuraError("the voice server didn't start in time")

    def _health(self) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2) as r:
                data: dict[str, Any] = json.loads(r.read())
                return data
        except (OSError, ValueError):
            return None

    def _kill(self, pid: int) -> None:
        if pid and sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                           capture_output=True, **POPEN_FLAGS)  # fmt: skip
        elif pid:
            os.kill(pid, 9)
        else:  # the oldest worker didn't report its pid: free the port by its owner
            if sys.platform == "win32":
                out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                                     text=True, **POPEN_FLAGS).stdout  # fmt: skip
                for ln in out.splitlines():
                    parts = ln.split()
                    if (
                        len(parts) >= 5
                        and parts[1].endswith(f":{PORT}")
                        and parts[3] == "LISTENING"
                    ):
                        self._kill(int(parts[4]))
                        break
        for _ in range(20):
            if self._health() is None:
                return
            time.sleep(0.5)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data: dict[str, Any] = json.loads(r.read())
                return data
        except urllib.error.HTTPError as e:
            raise SphuraError(json.loads(e.read()).get("error", str(e))) from e

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self._proc.pid)],
                               capture_output=True, **POPEN_FLAGS)  # fmt: skip
            else:
                self._proc.kill()
        self._proc = None

    # --- speech ------------------------------------------------------------------------

    def speak(self, text: str, voice: str, out_path: Path) -> float:
        return self.speak_many([text], voice, [out_path])[0]

    def speak_many(self, texts: list[str], voice: str, outs: list[Path]) -> list[float]:
        """Speaks several lines in one batch (about as fast on the GPU as one line)."""
        instruct, ref_text = DESIGNS.get(voice, DESIGNS["qwen_gamer"])
        return self.speak_ref(texts, outs, voice, instruct, ref_text)

    def speak_ref(self, texts: list[str], outs: list[Path], ref_id: str, instruct: str,
                  ref_text: str) -> list[float]:  # fmt: skip
        """Lines cloned from one reference voice, designed once from `instruct` and kept.
        A reference per personality and mood carries the delivery: the model has no tags."""
        if not self.ready():
            raise SphuraError("The natural voice isn't installed yet")
        ref = self._voices / f"{re.sub(r'[^a-z0-9_]', '', ref_id.lower())}.wav"
        for out in outs:
            out.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._ensure_running()
            if not ref.is_file():  # first use: design this voice once, keep it forever
                self._voices.mkdir(parents=True, exist_ok=True)
                self._post("/design", {"instruct": instruct, "text": ref_text, "out": str(ref)})
            items = [{"text": t, "out": str(o)} for t, o in zip(texts, outs, strict=True)]
            self._post("/speak", {"ref_audio": str(ref), "ref_text": ref_text,
                                  "items": items, "language": "English"})  # fmt: skip
        return [_trim_file(o) for o in outs]

    def design_many(self, texts: list[str], outs: list[Path], instruct: str) -> list[float]:
        """Lines spoken by the VoiceDesign model itself (no cloning): each call designs
        afresh, so the voice can drift between lines. Used to compare, not by default."""
        if not self.ready():
            raise SphuraError("The natural voice isn't installed yet")
        for out in outs:
            out.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._ensure_running()
            items = [{"text": t, "out": str(o)} for t, o in zip(texts, outs, strict=True)]
            self._post("/design", {"instruct": instruct, "items": items, "text": "", "out": ""})
        return [_trim_file(o) for o in outs]


def _trim_file(path: Path) -> float:
    """Trims silence at both ends in place; returns the length in seconds."""
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    mono = a.reshape(-1, ch).mean(axis=1) if ch > 1 else a
    loud = np.flatnonzero(np.abs(mono) > 330)
    if len(loud):
        pad = int(0.05 * sr)
        mono = mono[max(0, loud[0] - pad) : loud[-1] + pad]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(mono.astype(np.int16).tobytes())
    return len(mono) / sr
