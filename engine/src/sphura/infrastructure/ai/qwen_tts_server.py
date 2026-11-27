"""Qwen3-TTS worker, run with its own Python (it needs torch + transformers==4.57.3, which
SPHURA's environment doesn't have). Local HTTP on 127.0.0.1 only; keeps the model loaded.

POST /design {instruct, text, out}  -> designs a voice once (VoiceDesign model), saves it
POST /speak  {ref_audio, ref_text, items: [{text, out}], language} -> lines in that voice
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LOCK = threading.Lock()
STATE: dict[str, Any] = {"base": None, "prompts": {}}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _load(name: str) -> Qwen3TTSModel:
    cuda = torch.cuda.is_available()
    log(f"loading {name} on {'GPU' if cuda else 'CPU (slow)'}")
    # Local folder, not the repo id: with a repo id transformers asks the Hub for model info
    path = snapshot_download(name, local_files_only=True)
    return Qwen3TTSModel.from_pretrained(
        path,
        device_map="cuda:0" if cuda else "cpu",
        dtype=torch.bfloat16 if cuda else torch.float32,
    )


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def design(body: dict[str, Any]) -> dict[str, Any]:
    # The design model is only needed once per voice: load, use, unload (VRAM).
    # With "items", several lines are spoken by the design model directly (benchmarks)
    model = _load(DESIGN)
    items = body.get("items") or [{"text": body["text"], "out": body["out"]}]
    if len(items) == 1:  # the original single-line call, unchanged
        wavs, sr = model.generate_voice_design(
            text=items[0]["text"], instruct=body["instruct"], language="English"
        )
    else:
        try:
            wavs, sr = model.generate_voice_design(
                text=[i["text"] for i in items],
                instruct=[body["instruct"]] * len(items),
                language=["English"] * len(items),
            )
        except (TypeError, ValueError):  # no batch support in this version: one by one
            wavs = []
            for i in items:
                w, sr = model.generate_voice_design(
                    text=i["text"], instruct=body["instruct"], language="English"
                )
                wavs.append(w[0])
    for i, w in zip(items, wavs, strict=True):
        sf.write(i["out"], w, sr)
    del model
    _free()
    return {"sr": sr, "seconds": [len(w) / sr for w in wavs]}


def speak(body: dict[str, Any]) -> dict[str, Any]:
    if STATE["base"] is None:
        STATE["base"] = _load(BASE)
    model = STATE["base"]
    key = (body["ref_audio"], os.path.getmtime(body["ref_audio"]))
    if key not in STATE["prompts"]:
        STATE["prompts"][key] = model.create_voice_clone_prompt(
            ref_audio=body["ref_audio"], ref_text=body["ref_text"]
        )
    items = body["items"]
    wavs, sr = model.generate_voice_clone(
        text=[i["text"] for i in items],
        language=[body.get("language", "English")] * len(items),
        voice_clone_prompt=STATE["prompts"][key],
    )
    out = []
    for i, w in zip(items, wavs, strict=True):
        sf.write(i["out"], w, sr)
        out.append(len(w) / sr)
    return {"sr": sr, "seconds": out}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, code: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        info = {"ok": True, "cuda": torch.cuda.is_available(), "pid": os.getpid()}
        self._send(200, {**info, "version": 3})  # matches SERVER_VERSION in qwen_voice.py

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        fn = {"/design": design, "/speak": speak}.get(self.path)
        if fn is None:
            self._send(404, {"error": "unknown endpoint"})
            return
        try:
            with LOCK:  # one generation at a time on the GPU
                t = time.monotonic()
                log(f"{self.path} started")
                res = fn(body)
                log(f"{self.path} done in {time.monotonic() - t:.1f}s")
                self._send(200, res)
        except Exception as e:
            log(traceback.format_exc())
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 47823
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    log(f"ready on port {port}, torch {torch.__version__}, GPU: {gpu}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
