"""Per-project record of completed pipeline stages, so re-runs skip finished work."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from sphura.domain.models import utcnow


def stage_key(*parts: Any) -> str:
    """Identity of a stage's inputs; any change (file, options, stage version) invalidates it."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


class StageManifest:
    _lock = threading.Lock()

    def __init__(self, path: Path) -> None:
        self._path = path

    def _load(self) -> dict[str, Any]:
        try:
            data: dict[str, Any] = json.loads(self._path.read_text())
            return data
        except (FileNotFoundError, ValueError):
            return {}

    def completed(self, stage: str, key: str) -> dict[str, Any] | None:
        """Outputs of a finished stage, if its key matches and its files still exist."""
        entry = self._load().get(stage)
        if not entry or entry.get("key") != key:
            return None
        outputs: dict[str, Any] = entry.get("outputs", {})
        files = [Path(p) for p in outputs.get("files", [])]
        return outputs if all(f.is_file() for f in files) else None

    def mark(self, stage: str, key: str, outputs: dict[str, Any]) -> None:
        with self._lock:
            data = self._load()
            data[stage] = {"key": key, "completed_at": utcnow().isoformat(), "outputs": outputs}
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self._path)

    def invalidate(self, stage: str) -> None:
        with self._lock:
            data = self._load()
            if data.pop(stage, None) is not None:
                tmp = self._path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2))
                os.replace(tmp, self._path)
