from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload


@overload
def _env(name: str) -> str | None: ...
@overload
def _env(name: str, default: str) -> str: ...
def _env(name: str, default: str | None = None) -> str | None:
    """SPHURA_<name>, falling back to the older names SPHURI_<name> and CLIPFORGE_<name>."""
    for prefix in ("SPHURA", "SPHURI", "CLIPFORGE"):
        if value := os.environ.get(f"{prefix}_{name}"):
            return value
    return default


def default_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "SPHURA"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "sphura"


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    workers: int = 2
    host: str = "127.0.0.1"
    port: int = 47821
    api_token: str | None = None  # set by the desktop app when it spawns the engine
    ffmpeg_dir: Path | None = None  # folder containing ffmpeg + ffprobe; else bundled/PATH
    whisper_model: str = "auto"  # auto | tiny | base | small | medium | large-v3 | turbo ...
    whisper_device: str = "auto"  # auto | cpu | cuda | mps (experimental Apple GPU)
    # AI judge: auto = Anthropic if a key is set, else a local Ollama model if running, else none
    ai_provider: str = "auto"  # auto | anthropic | ollama | none
    anthropic_api_key: str | None = field(default=None, repr=False)
    anthropic_model: str = "claude-sonnet-5-5"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"
    music_dir: Path = field(default_factory=lambda: Path.home() / "Music" / "SPHURA")

    @property
    def db_path(self) -> Path:
        new = self.data_dir / "sphura.db"
        if not new.exists():  # from before a rename, until it's moved
            for old in ("sphuri.db", "clipforge.db"):
                if (self.data_dir / old).is_file():
                    return self.data_dir / old
        return new

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(_env("DATA_DIR") or default_data_dir()),
            workers=int(_env("WORKERS", "2")),
            host=_env("HOST", "127.0.0.1"),
            port=int(_env("PORT", "47821")),
            api_token=_env("API_TOKEN") or None,
            whisper_model=_env("WHISPER_MODEL", "auto"),
            whisper_device=_env("WHISPER_DEVICE", "auto"),
            ai_provider=_env("AI", "auto"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or None,
            anthropic_model=_env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=_env("OLLAMA_MODEL", "gemma4:12b"),
            ffmpeg_dir=Path(d) if (d := _env("FFMPEG_DIR")) else None,
            music_dir=Path(m) if (m := _env("MUSIC_DIR")) else Path.home() / "Music" / "SPHURA",
        )
