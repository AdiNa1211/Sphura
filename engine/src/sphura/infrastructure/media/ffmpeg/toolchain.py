from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from sphura.domain.errors import ToolchainError
from sphura.domain.models import HardwareProfile
from sphura.infrastructure.media.ffmpeg.process import run

log = logging.getLogger(__name__)

_EXE = ".exe" if sys.platform == "win32" else ""

# Best first. vaapi is omitted: it needs device setup + hwupload, handled in the renderer later.
_H264_PREFERENCE = (
    "h264_nvenc",
    "h264_qsv",
    "h264_amf",
    "h264_videotoolbox",
    "h264_mf",
    "libx264",
    "libopenh264",
)
_HEVC_PREFERENCE = ("hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_videotoolbox", "libx265")

# GUI-launched processes on macOS don't inherit the shell PATH, so check Homebrew explicitly
_EXTRA_DIRS = ("/opt/homebrew/bin", "/usr/local/bin") if sys.platform == "darwin" else ()

INSTALL_HINT = {
    "darwin": "Install it with `brew install ffmpeg`",
    "win32": "Install it with `winget install Gyan.FFmpeg`",
}.get(sys.platform, "Install it with your package manager, e.g. `sudo apt install ffmpeg`")


def _candidate_dirs(explicit: Path | None) -> list[Path]:
    dirs: list[Path] = []
    if explicit:
        dirs.append(explicit)
    if getattr(sys, "frozen", False):  # PyInstaller sidecar ships ffmpeg/ next to the exe
        dirs.append(Path(sys.executable).parent / "ffmpeg")
    return dirs


def locate(name: str, explicit_dir: Path | None) -> str:
    for d in _candidate_dirs(explicit_dir):
        p = d / f"{name}{_EXE}"
        if p.is_file():
            return str(p)
    found = shutil.which(name)
    if found:
        return found
    for extra in _EXTRA_DIRS:
        candidate = Path(extra) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise ToolchainError(
        f"{name} not found. {INSTALL_HINT}, or set SPHURA_FFMPEG_DIR to its folder."
    )


class FFmpegToolchain:
    """Finds FFmpeg/FFprobe and determines which encoders actually work on this machine."""

    def __init__(self, explicit_dir: Path | None, cache_path: Path | None = None) -> None:
        self._dir = explicit_dir
        self._cache_path = cache_path
        self._profile: HardwareProfile | None = None
        self._filters: set[str] | None = None
        self._lock = threading.Lock()

    @property
    def ffprobe(self) -> str:
        # Prefer the ffprobe next to ffmpeg so both come from the same build
        sibling = Path(self.ffmpeg).with_name(f"ffprobe{_EXE}")
        return str(sibling) if sibling.is_file() else locate("ffprobe", self._dir)

    @property
    def ffmpeg(self) -> str:
        return locate("ffmpeg", self._dir)

    def has_filter(self, name: str) -> bool:
        if self._filters is None:
            out = run([self.ffmpeg, "-hide_banner", "-filters"]).stdout
            self._filters = set(re.findall(r"^\s*\S{2,3}\s+(\w+)\s", out, re.MULTILINE))
        return name in self._filters

    def profile(self, refresh: bool = False) -> HardwareProfile:
        with self._lock:
            if self._profile is None or refresh:
                self._profile = self._load_or_build(refresh)
            return self._profile

    def _load_or_build(self, refresh: bool) -> HardwareProfile:
        ffmpeg, ffprobe = self.ffmpeg, self.ffprobe
        version = self._version(ffmpeg)
        key = f"{ffmpeg}|{version}|{Path(ffmpeg).stat().st_mtime_ns}"

        if not refresh and (cached := self._read_cache(key)):
            return cached

        encoders = self._listed_encoders(ffmpeg)
        profile = HardwareProfile(
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            ffmpeg_version=version,
            hwaccels=self._hwaccels(ffmpeg),
            h264_encoders=self._working(ffmpeg, [e for e in _H264_PREFERENCE if e in encoders]),
            hevc_encoders=self._working(ffmpeg, [e for e in _HEVC_PREFERENCE if e in encoders]),
            gpl_build="--enable-gpl" in self._buildconf(ffmpeg),
        )
        self._write_cache(key, profile)
        return profile

    @staticmethod
    def _version(ffmpeg: str) -> str:
        first = run([ffmpeg, "-hide_banner", "-version"]).stdout.splitlines()[0]
        m = re.match(r"ffmpeg version (\S+)", first)
        return m.group(1) if m else first

    @staticmethod
    def _listed_encoders(ffmpeg: str) -> set[str]:
        out = run([ffmpeg, "-hide_banner", "-encoders"]).stdout
        # Lines look like: " V....D h264_nvenc   NVIDIA NVENC H.264 encoder"
        return {m.group(1) for m in re.finditer(r"^\s*V\S{5}\s+(\S+)", out, re.MULTILINE)}

    @staticmethod
    def _hwaccels(ffmpeg: str) -> tuple[str, ...]:
        out = run([ffmpeg, "-hide_banner", "-hwaccels"]).stdout
        lines = [ln.strip() for ln in out.splitlines()]
        return tuple(ln for ln in lines if ln and not ln.endswith(":"))

    @staticmethod
    def _buildconf(ffmpeg: str) -> str:
        return run([ffmpeg, "-hide_banner", "-buildconf"]).stdout

    @staticmethod
    def _works(ffmpeg: str, encoder: str) -> bool:
        # Listed != usable: hardware encoders fail without a matching GPU/driver
        try:
            run(
                [
                    ffmpeg, "-hide_banner", "-v", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30:d=0.2",
                    "-pix_fmt", "yuv420p", "-c:v", encoder, "-f", "null", "-",
                ],
                timeout_s=20,
            )  # fmt: skip
            return True
        except ToolchainError as e:
            detail = str(e).split("): ", 1)[-1].splitlines()
            log.debug("encoder %s unavailable: %s", encoder, detail[0][:200] if detail else "")
            return False

    def _working(self, ffmpeg: str, candidates: list[str]) -> tuple[str, ...]:
        if not candidates:
            return ()
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            ok = list(pool.map(lambda e: self._works(ffmpeg, e), candidates))
        return tuple(e for e, good in zip(candidates, ok, strict=True) if good)

    def _read_cache(self, key: str) -> HardwareProfile | None:
        if not self._cache_path or not self._cache_path.is_file():
            return None
        try:
            data = json.loads(self._cache_path.read_text())
            if data.get("key") != key:
                return None
            p = data["profile"]
            return HardwareProfile(
                ffmpeg_path=str(p["ffmpeg_path"]),
                ffprobe_path=str(p["ffprobe_path"]),
                ffmpeg_version=str(p["ffmpeg_version"]),
                hwaccels=tuple(p["hwaccels"]),
                h264_encoders=tuple(p["h264_encoders"]),
                hevc_encoders=tuple(p["hevc_encoders"]),
                gpl_build=bool(p["gpl_build"]),
            )
        except (ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, key: str, profile: HardwareProfile) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps({"key": key, "profile": asdict(profile)}, indent=2))
