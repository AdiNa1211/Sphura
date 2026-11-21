"""Highlight pipeline ports (Phases 3-9)."""

from __future__ import annotations

from typing import Protocol

from sphura.domain.models import (
    CaptionWord,
    Clip,
    CropKeyframe,
    EditingMode,
    HighlightCandidate,
    MediaAsset,
    TimeRange,
)
from sphura.domain.transcript import TranscriptSegment
from sphura.ports.queue import JobContext


class TranscriptService(Protocol):
    def transcript(self, asset: MediaAsset, ctx: JobContext) -> list[TranscriptSegment]: ...


class VideoAnalyzer(Protocol):
    """Runs signal extraction over the whole video and persists a SignalTimeline artifact."""

    def analyze(self, asset: MediaAsset, ctx: JobContext) -> str:
        """Returns the analysis run id."""
        ...


class HighlightDetector(Protocol):
    def detect(self, run_id: str, ctx: JobContext) -> list[HighlightCandidate]: ...


class HighlightRanker(Protocol):
    def rank(
        self, candidates: list[HighlightCandidate], mode: EditingMode, ctx: JobContext
    ) -> list[HighlightCandidate]: ...


class CaptionGenerator(Protocol):
    def captions(
        self, transcript: list[TranscriptSegment], segments: list[TimeRange]
    ) -> list[CaptionWord]: ...


class CropTracker(Protocol):
    def track(
        self, asset: MediaAsset, segments: list[TimeRange], ctx: JobContext
    ) -> list[CropKeyframe]: ...


class ClipGenerator(Protocol):
    def generate(
        self, candidate: HighlightCandidate, mode: EditingMode, ctx: JobContext
    ) -> Clip: ...
