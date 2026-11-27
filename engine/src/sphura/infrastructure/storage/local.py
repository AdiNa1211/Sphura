from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

_ARTIFACT_KINDS = ("proxy", "analysis", "renders", "drafts", "tmp", "thumbs", "music", "voice")


class LocalSourceReader:
    """Reads originals in place. Opens read-only; never writes, moves or copies them."""

    def _path(self, uri: str) -> Path:
        return Path(uri)

    def exists(self, uri: str) -> bool:
        return self._path(uri).is_file()

    def size(self, uri: str) -> int:
        return self._path(uri).stat().st_size

    def open(self, uri: str) -> BinaryIO:
        return self._path(uri).open("rb")

    def read_range(self, uri: str, offset: int, length: int) -> bytes:
        with self.open(uri) as f:
            f.seek(offset)
            return f.read(length)

    def iter_chunks(self, uri: str, chunk_size: int = 8 * 1024 * 1024) -> Iterator[bytes]:
        with self.open(uri) as f:
            while chunk := f.read(chunk_size):
                yield chunk


class LocalWorkspaceStorage:
    """Derived artifacts under <root>/projects/<id>/{proxy,analysis,renders,tmp}."""

    def __init__(self, root: Path) -> None:
        self._root = root
        (root / "projects").mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        d = self._root / "projects" / project_id
        for kind in _ARTIFACT_KINDS:
            (d / kind).mkdir(parents=True, exist_ok=True)
        return d

    def artifact_path(self, project_id: str, kind: str, name: str) -> Path:
        if kind not in _ARTIFACT_KINDS:
            raise ValueError(f"unknown artifact kind: {kind}")
        if Path(name).name != name:
            raise ValueError("artifact name must not contain path separators")
        return self.project_dir(project_id) / kind / name

    def temp_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "tmp"

    def delete_project(self, project_id: str) -> None:
        d = self._root / "projects" / project_id
        if d.exists():
            shutil.rmtree(d)
