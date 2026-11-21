from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    text: str
    start_s: float  # absolute time in the source video
    end_s: float
    probability: float


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    words: tuple[TranscriptWord, ...] = ()
    confidence: float = 1.0  # mean word probability; low values are often hallucinations
    language: str | None = None


@dataclass(slots=True)
class Transcript:
    language: str
    language_probability: float
    model: str
    audio_stream_index: int
    duration_s: float
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "model": self.model,
            "audio_stream_index": self.audio_stream_index,
            "duration_s": self.duration_s,
            "segments": [segment_to_dict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transcript:
        return cls(
            language=d["language"],
            language_probability=d["language_probability"],
            model=d["model"],
            audio_stream_index=d["audio_stream_index"],
            duration_s=d["duration_s"],
            segments=[segment_from_dict(s) for s in d["segments"]],
        )


def segment_to_dict(s: TranscriptSegment) -> dict[str, Any]:
    return {
        "start": round(s.start_s, 3),
        "end": round(s.end_s, 3),
        "text": s.text,
        "conf": round(s.confidence, 3),
        "lang": s.language,
        "words": [
            [w.text, round(w.start_s, 3), round(w.end_s, 3), round(w.probability, 3)]
            for w in s.words
        ],
    }


def segment_from_dict(d: dict[str, Any]) -> TranscriptSegment:
    return TranscriptSegment(
        start_s=d["start"],
        end_s=d["end"],
        text=d["text"],
        words=tuple(TranscriptWord(w[0], w[1], w[2], w[3]) for w in d["words"]),
        confidence=d.get("conf", 1.0),
        language=d.get("lang"),
    )
