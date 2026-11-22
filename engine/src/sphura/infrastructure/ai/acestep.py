"""Local AI music with ACE-Step 1.5 (MIT licence; output usable commercially).

ACE-Step runs as its own local API server in its own Python environment (managed by uv),
so its PyTorch/CUDA stack never touches SPHURA's. Installed once, on request, into
<root>/repo; models download on first use into <root>/hf. The server is started on the
first generation and stopped with SPHURA.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sphura.domain.errors import SphuraError
from sphura.infrastructure.media.ffmpeg.process import POPEN_FLAGS

log = logging.getLogger(__name__)

REPO_ZIP = "https://codeload.github.com/ace-step/ACE-Step-1.5/zip/refs/heads/main"
UV_BASE = "https://github.com/astral-sh/uv/releases/latest/download/"
PORT = 8011  # not ACE-Step's default 8001, to stay clear of anything else using it
FIRST_RUN_TIMEOUT_S = 45 * 60  # first generation downloads the models (several GB)
RUN_TIMEOUT_S = 6 * 60

Progress = Callable[[str, float], None]


class AiMusicError(SphuraError):
    pass


def default_root() -> Path:
    """Next to the SPHURA install (D:\\clipforge\\ai\\ace-step): big files stay off C:."""
    here = Path(__file__).resolve()
    app = here.parents[5] if len(here.parents) > 5 else None  # .../clipforge
    if app is not None and (app / "engine").is_dir():
        return app / "ai" / "ace-step"
    return Path.home() / "SPHURA-AI" / "ace-step"


class AceStep:
    def __init__(self, root: Path, port: int = PORT) -> None:
        self.root = root
        self.port = port
        self.repo = root / "repo"
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self.status: dict[str, Any] = {"running": False, "step": "", "progress": 0.0,
                                       "error": None}  # fmt: skip

    # ---------- install ----------

    def ready(self) -> bool:
        return (self.root / ".installed").is_file() and self._uv().is_file()

    def install(self) -> None:
        """Background install; progress in self.status."""
        with self._lock:
            if self.status["running"] or self.ready():
                return
            self.status.update(running=True, error=None, step="Starting", progress=0.0)
        threading.Thread(target=self._install, daemon=True, name="acestep-install").start()

    def _install(self) -> None:
        def step(text: str, f: float) -> None:
            self.status.update(step=text, progress=f)

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self._uv().is_file():
                step("Downloading uv (Python manager)", 0.02)
                self._get_uv()
            if not (self.repo / "pyproject.toml").is_file():
                step("Downloading ACE-Step", 0.05)
                data = _fetch(REPO_ZIP)
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    tmp = self.root / "repo-unpack"
                    shutil.rmtree(tmp, ignore_errors=True)
                    z.extractall(tmp)
                    inner = next(p for p in tmp.iterdir() if p.is_dir())
                    shutil.rmtree(self.repo, ignore_errors=True)
                    inner.rename(self.repo)
                    shutil.rmtree(tmp, ignore_errors=True)
            step("Installing Python, PyTorch and ACE-Step (a few GB, takes a while)", 0.1)
            self._run_logged([str(self._uv()), "sync"], "install.log", step, 0.1, 0.5)
            step("Downloading and loading the music model (about 5-8 GB, first time only)", 0.5)
            self.ensure_running(FIRST_RUN_TIMEOUT_S)
            warm = self.root / "warmup.wav"
            self.generate("calm ambient pad, instrumental", 10, 90, 1, warm, first_run=True,
                          on_wait=lambda s: step(f"Downloading and loading the music model ({s})",
                                                 0.6))  # fmt: skip
            warm.unlink(missing_ok=True)
            (self.root / ".installed").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            step("AI music is ready", 1.0)
        except Exception as e:
            log.exception("ACE-Step install failed")
            self.status["error"] = f"{e} (details: {self.root / 'install.log'})"
        finally:
            self.status["running"] = False

    def _server_cmd(self) -> list[str]:
        """Run the server with the install's own Python. The acestep-api.exe launcher uv
        generates is unsigned, so Windows Smart App Control blocks it (os error 4551)."""
        venv = self.repo / ".venv"
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if py.is_file():
            return [str(py), "-m", "acestep.api_server"]
        return [str(self._uv()), "run", "acestep-api"]

    def _uv(self) -> Path:
        return self.root / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")

    def _get_uv(self) -> None:
        if sys.platform == "win32":
            data = _fetch(UV_BASE + "uv-x86_64-pc-windows-msvc.zip")
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                (self.root / "uv").mkdir(parents=True, exist_ok=True)
                for name in z.namelist():
                    if name.endswith("uv.exe"):
                        self._uv().write_bytes(z.read(name))
        else:
            import tarfile

            data = _fetch(UV_BASE + "uv-x86_64-unknown-linux-gnu.tar.gz")
            with tarfile.open(fileobj=io.BytesIO(data)) as t:
                member = next(m for m in t.getmembers() if m.name.endswith("/uv"))
                f = t.extractfile(member)
                assert f is not None
                self._uv().parent.mkdir(parents=True, exist_ok=True)
                self._uv().write_bytes(f.read())
            self._uv().chmod(0o755)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            UV_PYTHON_INSTALL_DIR=str(self.root / "python"),
            UV_CACHE_DIR=str(self.root / "uv-cache"),
            HF_HOME=str(self.root / "hf"),
            ACESTEP_API_HOST="127.0.0.1",
            ACESTEP_API_PORT=str(self.port),
            ACESTEP_CONFIG_PATH="acestep-v15-turbo",  # 2B, 8 steps: fast, fits beside Gemma
            ACESTEP_INIT_LLM="false",  # no planning LM: less VRAM, prompts are enough
            PYTHONUTF8="1",
        )
        return env

    def _run_logged(
        self, cmd: list[str], log_name: str, step: Callable[[str, float], None],
        lo: float, hi: float,
    ) -> None:  # fmt: skip
        with (self.root / log_name).open("a", encoding="utf-8") as logf:
            logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd, cwd=self.repo, env=self._env(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                **POPEN_FLAGS,
            )  # fmt: skip
            assert proc.stdout is not None
            for n, line in enumerate(proc.stdout, 1):
                logf.write(line)
                text = line.strip()
                if text and n % 3 == 0:
                    frac = lo + (hi - lo) * (1 - 1 / (1 + n / 60))  # creeps up; size unknown
                    step(f"Installing: {text[:90]}", min(hi, frac))
            if proc.wait() != 0:
                raise AiMusicError(f"'{' '.join(cmd[1:])}' failed")

    # ---------- server ----------

    def ensure_running(self, timeout_s: float = 600) -> None:
        with self._lock:
            if self._healthy():
                return
            if self._proc is None or self._proc.poll() is not None:
                logf = (self.root / "server.log").open("a", encoding="utf-8")
                logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} starting\n")
                logf.flush()
                self._proc = subprocess.Popen(
                    self._server_cmd(), cwd=self.repo, env=self._env(),
                    stdout=logf, stderr=subprocess.STDOUT, **POPEN_FLAGS,
                )  # fmt: skip
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self._healthy():
                    return
                if self._proc.poll() is not None:
                    raise AiMusicError(f"the music server stopped (see {self.root / 'server.log'})")
                time.sleep(2)
            raise AiMusicError("the music server didn't start in time")

    def stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(self._proc.pid)],
                                   capture_output=True, **POPEN_FLAGS)  # fmt: skip
                else:
                    self._proc.kill()
            self._proc = None

    def _healthy(self) -> bool:
        try:
            d = self._call("GET", "/health", timeout=3)
        except (OSError, ValueError):
            return False
        return str((d.get("data") or {}).get("status", d.get("status", ""))).lower() == "ok"

    # ---------- generation ----------

    def generate(
        self, prompt: str, duration_s: float, bpm: int, seed: int, dest: Path,
        first_run: bool = False, on_wait: Callable[[str], None] | None = None,
    ) -> Path:  # fmt: skip
        """Instrumental track for `prompt`, written to dest (wav)."""
        self.ensure_running()
        body = {
            "prompt": prompt, "lyrics": "[Instrumental]", "instrumental": True,
            "audio_duration": round(max(10.0, min(duration_s, 240.0)), 1),
            "bpm": int(max(30, min(bpm, 300))), "time_signature": "4",
            "inference_steps": 8, "thinking": False, "batch_size": 1,
            "use_random_seed": False, "seed": int(seed % 2**31), "audio_format": "wav",
        }  # fmt: skip
        task = self._call("POST", "/release_task", body, timeout=30)
        task_id = (task.get("data") or task).get("task_id")
        if not task_id:
            raise AiMusicError(f"the music server didn't accept the request: {task}")
        deadline = time.monotonic() + (FIRST_RUN_TIMEOUT_S if first_run else RUN_TIMEOUT_S)
        started = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(1.5)
            res = self._call("POST", "/query_result", {"task_id_list": [task_id]}, timeout=30)
            items = res.get("data") or []
            item = items[0] if isinstance(items, list) and items else {}
            status = item.get("status", 0)
            if status == 2:
                raise AiMusicError(f"music generation failed: {item.get('result') or item}")
            if status == 1:
                result = item.get("result")
                files = json.loads(result) if isinstance(result, str) else result or []
                url = files[0]["file"] if isinstance(files, list) and files else None
                if not url:
                    raise AiMusicError("the music server returned no audio")
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(self._get_bytes(url))
                tmp.replace(dest)
                return dest
            if on_wait:
                on_wait(f"{int(time.monotonic() - started)} s")
        raise AiMusicError("music generation timed out")

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 30
    ) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"},
        )  # fmt: skip
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out: dict[str, Any] = json.loads(r.read() or b"{}")
            return out

    def _get_bytes(self, url: str) -> bytes:
        if url.startswith("/"):
            url = f"http://127.0.0.1:{self.port}{url}"
        if urllib.parse.urlparse(url).hostname not in ("127.0.0.1", "localhost"):
            raise AiMusicError("unexpected audio location from the music server")
        with urllib.request.urlopen(url, timeout=120) as r:
            data: bytes = r.read()
            return data


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SPHURA"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data: bytes = r.read()
            return data
    except OSError as e:
        raise AiMusicError(f"couldn't download {url.rsplit('/', 1)[-1]}: {e}") from e
