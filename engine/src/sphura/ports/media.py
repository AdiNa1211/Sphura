"""Video/audio processing ports. Adapters (FFmpeg etc.) arrive in Phases 2, 4, 6."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sphura.domain.models import Clip, Container, HardwareProfile, MediaInfo
from sphura.ports.queue import JobContext

SNIFF_HEADER_BYTES = 4096


class ContainerSniffer(Protocol):
    """Identify container from file bytes, not the extension."""

    def sniff(self, header: bytes) -> Container | None: ...


class MediaProbe(Protocol):
    def probe(self, uri: str) -> MediaInfo:
        """Raises InvalidMediaError for unreadable/broken files, ToolchainError if FFprobe fails."""
        ...


class HardwareProbe(Protocol):
    def profile(self, refresh: bool = False) -> HardwareProfile: ...


@dataclass(frozen=True, slots=True)
class FrameBatch:
    timestamps_s: tuple[float, ...]
    width: int
    height: int
    rgb: bytes  # len == n * width * height * 3


class MediaDecoder(Protocol):
    """Low-res streaming decode for analysis; never loads the whole video."""

    def iter_proxy_frames(
        self,
        uri: str,
        start_s: float,
        duration_s: float,
        fps: float,
        width: int,
        height: int,
        batch: int,
    ) -> Iterator[FrameBatch]:
        """Frames letterboxed to width x height. Closing the iterator stops decoding."""
        ...


class AudioExtractor(Protocol):
    def extract(
        self,
        uri: str,
        outputs: dict[int, Path],
        sample_rate: int,
        duration_s: float,
        on_progress: Callable[[float], None],
    ) -> None:
        """Decode several audio streams (index -> output WAV) to mono PCM in one pass."""
        ...


class FrameSampler(Protocol):
    def sample(self, uri: str, times_s: list[float], width: int) -> list[bytes]:
        """JPEG stills at the given source times (same order)."""
        ...


class SceneDetector(Protocol):
    def detect_cuts(self, uri: str, ctx: JobContext | None = None) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class RenderOutcome:
    path: Path
    encoder: str
    duration_s: float


class VideoRenderer(Protocol):
    def render(
        self,
        clip: Clip,
        source_uri: str,
        info: MediaInfo,
        audio_stream: int | None,
        out_path: Path,
        on_progress: Callable[[float], None],
    ) -> RenderOutcome: ...
