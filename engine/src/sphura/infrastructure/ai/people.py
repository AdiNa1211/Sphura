"""On-device person detection (YOLOX-S, Apache-2.0) via onnxruntime, which faster-whisper
already installs. The model (~36 MB) is downloaded on first use; without it, None."""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sphura.ports.media import PersonBox

log = logging.getLogger(__name__)

MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
MODEL_BYTES = 35_858_002
SIZE = 640
MIN_SCORE = 0.35
NMS_IOU = 0.5


class YoloxPeopleDetector:
    def __init__(self, model_dir: Path) -> None:
        self._path = model_dir / "yolox_s.onnx"
        self._lock = threading.Lock()
        self._session: Any = None
        self._failed = False
        self._grid: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None = None

    def detect(self, rgb: bytes, width: int, height: int) -> list[PersonBox] | None:
        sess = self._load()
        if sess is None:
            return None
        img = np.frombuffer(rgb, np.uint8).reshape(height, width, 3)
        r = min(SIZE / width, SIZE / height)
        w, h = round(width * r), round(height * r)
        canvas = np.full((SIZE, SIZE, 3), 114, np.float32)
        canvas[:h, :w] = _resize(img, w, h)[:, :, ::-1]  # the model expects BGR, top-left
        out = sess.run(None, {"images": canvas.transpose(2, 0, 1)[None]})[0][0]
        return _people(self._decode(out), r, width, height)

    def _decode(self, out: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self._grid is None:
            grids, strides = [], []
            for s in (8, 16, 32):
                n = SIZE // s
                yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
                grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
                strides.append(np.full((n * n, 1), s))
            self._grid = (
                np.concatenate(grids).astype(np.float32),
                np.concatenate(strides).astype(np.float32),
            )
        grid, stride = self._grid
        out = out.copy()
        out[:, :2] = (out[:, :2] + grid) * stride
        out[:, 2:4] = np.exp(out[:, 2:4]) * stride
        return out

    def _load(self) -> Any:
        with self._lock:
            if self._session is not None or self._failed:
                return self._session
            try:
                import onnxruntime as ort

                if not self._path.is_file() or self._path.stat().st_size != MODEL_BYTES:
                    self._download()
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                self._session = ort.InferenceSession(
                    str(self._path), opts, providers=["CPUExecutionProvider"]
                )
            except Exception as e:  # no onnxruntime, offline, bad file: fall back to motion
                log.warning("person detection unavailable (%s); framing follows motion", e)
                self._failed = True
            return self._session

    def _download(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        part = self._path.with_suffix(".part")
        log.info("downloading the person detector (36 MB) once")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, part.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        if part.stat().st_size != MODEL_BYTES:
            part.unlink(missing_ok=True)
            raise OSError("incomplete model download")
        part.replace(self._path)


def _resize(img: npt.NDArray[np.uint8], w: int, h: int) -> npt.NDArray[np.float32]:
    """Nearest-neighbour resize: frames arrive near model size, so quality loss is nil."""
    ys = (np.arange(h) * img.shape[0] / h).astype(np.intp)
    xs = (np.arange(w) * img.shape[1] / w).astype(np.intp)
    out: npt.NDArray[np.float32] = img[ys][:, xs].astype(np.float32)
    return out


def _people(out: npt.NDArray[np.float32], r: float, width: int, height: int) -> list[PersonBox]:
    score = out[:, 4] * out[:, 5]  # objectness x class 0 ("person")
    keep = score >= MIN_SCORE
    if not keep.any():
        return []
    b, s = out[keep, :4], score[keep]
    x0, y0 = (b[:, 0] - b[:, 2] / 2) / r, (b[:, 1] - b[:, 3] / 2) / r
    x1, y1 = (b[:, 0] + b[:, 2] / 2) / r, (b[:, 1] + b[:, 3] / 2) / r
    boxes: list[PersonBox] = []
    for i in np.argsort(-s):
        cand = (x0[i], y0[i], x1[i], y1[i])
        if all(
            _iou(cand, (k.x0 * width, k.y0 * height, k.x1 * width, k.y1 * height)) < NMS_IOU
            for k in boxes
        ):
            boxes.append(
                PersonBox(
                    float(np.clip(x0[i] / width, 0, 1)),
                    float(np.clip(y0[i] / height, 0, 1)),
                    float(np.clip(x1[i] / width, 0, 1)),
                    float(np.clip(y1[i] / height, 0, 1)),
                    float(s[i]),
                )
            )
    return boxes


def _iou(a: tuple[Any, ...], b: tuple[Any, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0
