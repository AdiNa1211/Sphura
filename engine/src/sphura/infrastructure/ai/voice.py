"""Narrator voice: Kokoro-82M (Apache 2.0) through kokoro-onnx, on this PC. Installed on
first use (pip package + ~350 MB of model files). Game names come as IPA from the
pronunciation dictionary (Voice Intelligence): Kokoro's own guess at "Kratos" is wrong."""

# ruff: noqa: RUF001  (IPA phonemes look like Latin letters on purpose)
from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

from sphura.domain.errors import SphuraError

log = logging.getLogger(__name__)

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
FILES = {"kokoro-v1.0.onnx": 325_532_387, "voices-v1.0.bin": 28_214_398}
PACKAGE = "kokoro-onnx>=0.4,<1"

# id -> (Kokoro voice, phonemizer language, label, narration language, engine)
VOICES: dict[str, tuple[str, str, str, str, str]] = {
    "qwen_gamer": ("", "en", "English · Gamer (natural)", "en", "qwen"),
    "qwen_story": ("", "en", "English · Storyteller (natural)", "en", "qwen"),
    "en_michael": ("am_michael", "en-us", "English · Michael (fast)", "en", "kokoro"),
    "en_george": ("bm_george", "en-gb", "English · George, British (fast)", "en", "kokoro"),
    "hi_psi": ("hm_psi", "hi", "Hinglish · Psi", "hi", "kokoro"),
}
DEFAULT_VOICE = "en_michael"


class KokoroVoice:
    def __init__(self, model_dir: Path) -> None:
        self._dir = model_dir
        self._lock = threading.Lock()
        self._engine: Any = None
        self.setup: dict[str, Any] = {"running": False, "step": "", "error": None}

    # --- install -----------------------------------------------------------------------

    def ready(self) -> bool:
        if not all((self._dir / f).is_file() and (self._dir / f).stat().st_size == n
                   for f, n in FILES.items()):  # fmt: skip
            return False
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            return False
        return True

    def install(self) -> dict[str, Any]:
        with self._lock:
            if self.setup["running"]:
                return self.setup
            self.setup = {"running": True, "step": "starting", "error": None}
        threading.Thread(target=self._install, daemon=True, name="voice-install").start()
        return self.setup

    def _install(self) -> None:
        try:
            try:
                import kokoro_onnx  # noqa: F401
            except ImportError:
                self.setup["step"] = "installing the voice engine (kokoro-onnx)"
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                     PACKAGE],
                    check=True, capture_output=True, text=True,
                )  # fmt: skip
            self._dir.mkdir(parents=True, exist_ok=True)
            for name, size in FILES.items():
                path = self._dir / name
                if path.is_file() and path.stat().st_size == size:
                    continue
                self.setup["step"] = f"downloading {name} ({size // 1_000_000} MB)"
                part = path.with_suffix(path.suffix + ".part")
                with (
                    urllib.request.urlopen(MODEL_URL + name, timeout=60) as r,
                    part.open("wb") as f,
                ):
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                if part.stat().st_size != size:
                    part.unlink(missing_ok=True)
                    raise OSError(f"{name} didn't download completely")
                part.replace(path)
            self.setup["step"] = "done"
        except subprocess.CalledProcessError as e:
            self.setup["error"] = "couldn't install kokoro-onnx: " + (e.stderr or "")[-400:]
        except Exception as e:
            self.setup["error"] = f"couldn't set up the narrator voice: {e}"
        finally:
            self.setup["running"] = False

    # --- speech ------------------------------------------------------------------------

    def _load(self) -> Any:
        if self._engine is None:
            if not self.ready():
                raise SphuraError("The narrator voice isn't installed yet")
            from kokoro_onnx import Kokoro

            self._engine = Kokoro(
                str(self._dir / "kokoro-v1.0.onnx"), str(self._dir / "voices-v1.0.bin")
            )
        return self._engine

    def phonemes(self, text: str, lang: str, lexicon: dict[str, str] | None = None) -> str:
        """Phonemes with dictionary names spliced in as IPA; drops espeak's "(en)" language
        marks, which Kokoro would otherwise read out as sounds."""
        k = self._load()
        lexicon = lexicon or {}
        names = (re.compile(r"(?<![\w'’-])(" + "|".join(re.escape(w) for w in sorted(
            lexicon, key=len, reverse=True)) + r")(?![\w'’-])") if lexicon else None)  # fmt: skip
        out = []
        for part in names.split(text) if names else [text]:
            if part in lexicon:
                out.append(lexicon[part])
            elif part.strip():
                out.append(re.sub(r"\((?:en|hi)\)", "", k.tokenizer.phonemize(part, lang)))
        return re.sub(r"\s+", " ", " ".join(out)).strip()

    def speak(self, text: str, voice: str, out_path: Path, speed: float = 1.0,
              lexicon: dict[str, str] | None = None) -> float:  # fmt: skip
        """Speaks one line into a 24 kHz mono WAV; returns its length in seconds."""
        kv, lang, *_ = VOICES.get(voice, VOICES[DEFAULT_VOICE])
        with self._lock:
            k = self._load()
            samples, sr = k.create(self.phonemes(text, lang, lexicon), voice=kv,
                                   speed=min(max(speed, 0.5), 2.0), is_phonemes=True)  # fmt: skip
        audio = _trim(np.asarray(samples, dtype=np.float32), sr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return float(len(audio) / sr)


def _trim(a: np.ndarray, sr: int) -> np.ndarray:
    """Cut leading/trailing silence so lines can sit tight in gaps."""
    loud = np.flatnonzero(np.abs(a) > 0.01)
    if not len(loud):
        return a
    pad = int(0.05 * sr)
    return a[max(0, loud[0] - pad) : loud[-1] + pad]
