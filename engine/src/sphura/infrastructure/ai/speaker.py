"""Speaker embeddings for voice-consistency checks: WeSpeaker ResNet34 (VoxCeleb, Apache-2.0)
as ONNX, run by onnxruntime (already installed with faster-whisper). 26 MB, downloaded on
first use from the sherpa-onnx model release; without it SPHURA falls back to timbre stats."""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from sphura.pipeline.voice_audio import F32, fbank

log = logging.getLogger(__name__)

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx")  # fmt: skip
MODEL_BYTES = 26_534_365


class SpeakerEmbedder:
    def __init__(self, model_dir: Path) -> None:
        self._path = model_dir / "speaker" / "wespeaker_en_voxceleb_resnet34.onnx"
        self._session: Any = None
        self._failed: str | None = None
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {"ready": self._path.is_file() and self._path.stat().st_size == MODEL_BYTES,
                "error": self._failed, "model": "WeSpeaker ResNet34 (VoxCeleb)"}  # fmt: skip

    def embed(self, audio: F32, sr: int) -> F32 | None:
        """A 256-number voice print, or None when the model can't be used."""
        if len(audio) < int(0.5 * sr):
            return None  # too short for a stable print
        s = self._load()
        if s is None:
            return None
        feats = fbank(audio, sr)[None]
        out = s.run(None, {s.get_inputs()[0].name: feats})[0][0]
        return np.asarray(out, np.float32)

    def _load(self) -> Any:
        with self._lock:
            if self._session is not None or self._failed:
                return self._session
            try:
                if not self.status()["ready"]:
                    self._download()
                import onnxruntime as ort

                self._session = ort.InferenceSession(
                    str(self._path), providers=["CPUExecutionProvider"]
                )
            except Exception as e:  # offline, no onnxruntime, bad file: timbre stats instead
                log.warning("speaker model unavailable: %s", e)
                self._failed = f"{type(e).__name__}: {e}"
            return self._session

    def _download(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        part = self._path.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, part.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        if part.stat().st_size != MODEL_BYTES:
            part.unlink(missing_ok=True)
            raise OSError("speaker model download was incomplete")
        part.replace(self._path)
