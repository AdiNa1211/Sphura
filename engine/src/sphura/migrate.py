"""One-time move from the app's old names (ClipForge, then SPHURI) to SPHURA.

Renames the default data, music and output folders when only the old ones exist, then
rewrites file paths saved inside them (database rows, analysis manifests) so every
project, analysis and rendered Short carries over. Safe to run on every start: it does
nothing once the new folders exist. If a folder is in use (an old copy still running),
it keeps using the old one for this session and tries again next time.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from sphura.config import Settings

log = logging.getLogger(__name__)

OLD_NAMES = ("SPHURI", "ClipForge")  # newest first
NEW = "SPHURA"


def legacy_data_dir(old: str) -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / old
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / old.lower()


def migrate_legacy(settings: Settings, default_data_dir: Path) -> Settings:
    moves: list[tuple[str, str]] = []
    data_dir, music_dir = settings.data_dir, settings.music_dir

    for old in OLD_NAMES:  # whichever old folder exists is moved to the new name
        if data_dir == default_data_dir:
            data_dir = _move(legacy_data_dir(old), data_dir, moves)
        if music_dir == Path.home() / "Music" / NEW:
            music_dir = _move(Path.home() / "Music" / old, music_dir, moves)
        _move(Path.home() / "Videos" / old, Path.home() / "Videos" / NEW, moves)

    _rename_db(data_dir)
    new_meta = music_dir / "sphura_tracks.json"
    for old in OLD_NAMES:
        old_meta = music_dir / f"{old.lower()}_tracks.json"
        if old_meta.is_file() and not new_meta.exists():
            old_meta.rename(new_meta)

    _rename_partials(data_dir)
    if moves:
        _rewrite_paths(data_dir, moves)
        log.info("moved data to the new name: %s", ", ".join(n for _, n in moves))
    return replace(settings, data_dir=data_dir, music_dir=music_dir)


def _rename_partials(data_dir: Path) -> None:
    """Half-finished downloads keep resuming: their work folder follows the new name."""
    try:
        cfg = json.loads((data_dir / "downloads.json").read_text(encoding="utf-8"))
        folders = {Path(str(cfg["folder"]))}
    except (OSError, ValueError, KeyError):
        folders = set()
    folders.add(Path.home() / "Videos" / NEW / "Downloads")
    for folder in folders:
        for old in OLD_NAMES:
            src, dst = folder / f".{old.lower()}-partial", folder / f".{NEW.lower()}-partial"
            if src.is_dir() and not dst.exists():
                try:
                    src.rename(dst)
                except OSError as e:
                    log.warning("could not rename %s: %s", src, e)


def _move(old: Path, new: Path, moves: list[tuple[str, str]]) -> Path:
    """Rename old -> new when only old exists. Returns the folder to use now."""
    if new.exists() or not old.is_dir():
        return new
    try:
        old.rename(new)  # same drive: instant, nothing is copied
    except OSError as e:
        log.warning("could not rename %s to %s (%s); using the old folder for now", old, new, e)
        return old
    moves.append((str(old), str(new)))
    return new


def _rename_db(data_dir: Path) -> None:
    new = data_dir / "sphura.db"
    old = next((data_dir / f"{n.lower()}.db" for n in OLD_NAMES
                if (data_dir / f"{n.lower()}.db").is_file()), None)  # fmt: skip
    if old is None or new.exists():
        return
    try:
        for suffix in ("-wal", "-shm"):  # SQLite's side files travel with it
            side = old.with_name(old.name + suffix)
            if side.exists():
                side.rename(new.with_name(new.name + suffix))
        old.rename(new)
    except OSError as e:
        log.warning("could not rename the database yet (%s); using %s for now", e, old.name)


def _rewrite_paths(data_dir: Path, moves: list[tuple[str, str]]) -> None:
    def fix(s: str) -> str:
        for old, new in moves:
            if s.lower().startswith(old.lower()):
                return new + s[len(old) :]
        return s

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return fix(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {fix(k): walk(x) for k, x in v.items()}
        return v

    def fix_json(text: str | None) -> str | None:
        if not text:
            return text
        try:
            return json.dumps(walk(json.loads(text)), ensure_ascii=False)
        except ValueError:
            return fix(text)

    db = data_dir / "sphura.db"
    for old in OLD_NAMES:
        if not db.exists():
            db = data_dir / f"{old.lower()}.db"
    if db.is_file():
        conn = sqlite3.connect(db)
        try:
            for cid, edl, out in conn.execute("SELECT id, edl, output_path FROM clips").fetchall():
                conn.execute(
                    "UPDATE clips SET edl = ?, output_path = ? WHERE id = ?",
                    (fix_json(edl), fix(out) if out else out, cid),
                )
            for jid, payload, result in conn.execute("SELECT id, payload, result FROM jobs"):
                conn.execute(
                    "UPDATE jobs SET payload = ?, result = ? WHERE id = ?",
                    (fix_json(payload), fix_json(result), jid),
                )
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            if "downloads" in tables:
                for did, fp, tp in conn.execute(
                    "SELECT id, file_path, thumbnail_path FROM downloads"
                ).fetchall():
                    conn.execute(
                        "UPDATE downloads SET file_path = ?, thumbnail_path = ? WHERE id = ?",
                        (fix(fp) if fp else fp, fix(tp) if tp else tp, did),
                    )
            conn.commit()
        except sqlite3.Error as e:
            log.warning("could not update saved paths: %s", e)
        finally:
            conn.close()

    for f in [*data_dir.glob("*.json"), *data_dir.glob("workspace/projects/*/analysis/*.json")]:
        if f.name == "music_cache.json":
            f.unlink(missing_ok=True)  # keyed by track paths; rebuilt on demand
            continue
        try:
            f.write_text(fix_json(f.read_text(encoding="utf-8")) or "", encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            log.warning("could not update paths in %s: %s", f.name, e)
