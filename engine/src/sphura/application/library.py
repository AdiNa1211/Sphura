"""Finds recent recordings to offer in the UI: OBS's recording folder, Videos, Downloads,
and the top two levels of other drives (where long recordings often live)."""

from __future__ import annotations

import os
import re
import string
import sys
from pathlib import Path
from typing import Any

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".flv", ".ts", ".webm", ".m4v"}
SKIP = {"windows", "program files", "program files (x86)", "programdata", "$recycle.bin",
        "system volume information", "appdata", "node_modules"}  # fmt: skip


def obs_recording_folder() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    obs = Path(appdata) / "obs-studio"
    profile = None
    for name in ("user.ini", "global.ini"):  # OBS 31+ keeps the profile in user.ini
        ini = obs / name
        if ini.is_file():
            m = re.search(r"(?m)^ProfileDir=(.+)$", ini.read_text(errors="ignore"))
            if m:
                profile = m.group(1).strip()
                break
    inis = [obs / "basic" / "profiles" / profile / "basic.ini"] if profile else []
    inis += sorted((obs / "basic" / "profiles").glob("*/basic.ini"),
                   key=lambda p: p.stat().st_mtime, reverse=True)  # fmt: skip
    for ini in inis:
        if not ini.is_file():
            continue
        text = ini.read_text(errors="ignore")
        key = "RecFilePath" if re.search(r"(?m)^Mode=Advanced", text) else "FilePath"
        m = re.search(rf"(?m)^{key}=(.+)$", text)
        if m and Path(m.group(1).strip()).is_dir():
            return Path(m.group(1).strip())
    return None


def search_folders() -> list[tuple[Path, int]]:
    """(folder, depth) pairs to scan."""
    home = Path.home()
    out: list[tuple[Path, int]] = []
    if obs := obs_recording_folder():
        out.append((obs, 1))
    out += [(home / "Videos", 2), (home / "Downloads", 1), (home / "Desktop", 1)]
    if sys.platform == "win32":
        for letter in string.ascii_uppercase[3:]:  # D: onwards; C:\ is mostly system files
            root = Path(f"{letter}:\\")
            if root.exists():
                out.append((root, 2))
    return out


def recent_videos(limit: int = 40, min_bytes: int = 50_000_000) -> list[dict[str, Any]]:
    seen: set[str] = set()
    found: list[dict[str, Any]] = []
    obs = obs_recording_folder()
    for folder, depth in search_folders():
        for p in _walk(folder, depth):
            key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size >= min_bytes:
                found.append(
                    {
                        "path": str(p),
                        "name": p.name,
                        "folder": str(p.parent),
                        "size_bytes": st.st_size,
                        "modified": st.st_mtime,
                        "from_obs": bool(obs and p.parent == obs),
                    }
                )
    found.sort(key=lambda f: f["modified"], reverse=True)
    return found[:limit]


def _walk(folder: Path, depth: int) -> list[Path]:
    out: list[Path] = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return out
    for e in entries:
        try:
            if e.is_file() and Path(e.name).suffix.lower() in VIDEO_EXT:
                out.append(Path(e.path))
            elif (
                depth > 1
                and e.is_dir()
                and e.name.lower() not in SKIP
                and not e.name.startswith(".")
            ):
                out += _walk(Path(e.path), depth - 1)
        except OSError:
            continue
    return out
