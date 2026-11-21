"""Vendor-neutral AI ports. Local (ONNX, faster-whisper, Ollama) and cloud adapters both fit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from sphura.domain.transcript import TranscriptSegment


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    name: str
    runs_locally: bool  # False => data leaves the machine; UI must disclose it


AudioSamples = npt.NDArray[np.float32]  # mono, range -1..1
ProgressFn = Callable[[float], None]  # 0..1 within the call; may raise to cancel


class TranscriptionProvider(Protocol):
    @property
    def info(self) -> ProviderInfo: ...

    @property
    def model_name(self) -> str: ...

    def detect_language(self, audio: AudioSamples, sample_rate: int) -> tuple[str, float]:
        """Returns (language code, probability)."""
        ...

    def transcribe(
        self,
        audio: AudioSamples,
        sample_rate: int,
        language: str | None,
        offset_s: float,
        on_progress: ProgressFn,
    ) -> list[TranscriptSegment]:
        """language=None: detect per window (mixed-language streams). Timestamps are shifted
        by offset_s (absolute source time)."""
        ...


@dataclass(frozen=True, slots=True)
class ImageInput:
    t_s: float
    jpeg: bytes
    label: str = ""  # shown to the model right before the image, e.g. "Frame 3 at +12.0s"


class VisionProvider(Protocol):
    info: ProviderInfo

    def describe(self, frames: list[ImageInput], prompt: str) -> str: ...

    def embed(self, frames: list[ImageInput]) -> list[list[float]]: ...


@dataclass(slots=True)
class LLMRequest:
    system: str
    prompt: str
    images: list[ImageInput] = field(default_factory=list)
    json_schema: dict[str, Any] | None = None  # request structured output when supported
    max_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str  # JSON text when the request carried a json_schema
    input_tokens: int = 0
    output_tokens: int = 0


class LLMProvider(Protocol):
    @property
    def info(self) -> ProviderInfo: ...

    @property
    def max_concurrency(self) -> int: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class HighlightScoringProvider(Protocol):
    info: ProviderInfo

    def score(self, context: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(slots=True)
class AIProvider:
    """Bundle of capability providers, selectable per capability."""

    transcription: TranscriptionProvider | None = None
    vision: VisionProvider | None = None
    llm: LLMProvider | None = None
    scoring: HighlightScoringProvider | None = None

    def uses_cloud(self) -> bool:
        return any(
            p is not None and not p.info.runs_locally
            for p in (self.transcription, self.vision, self.llm, self.scoring)
        )
