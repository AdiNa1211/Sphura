"""Core domain entities. Pure data — no IO, no FFmpeg, no AI vendors."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


MIN_CLIP_S = 15.0  # shortest clip worth posting


class EditingMode(StrEnum):
    AUTO = "auto"
    HIGHLIGHT = "highlight"
    FUNNY = "funny"
    CINEMATIC = "cinematic"
    VIRAL = "viral"
    MINIMAL = "minimal"


class ProjectStatus(StrEnum):
    CREATED = "created"
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class StorageKind(StrEnum):
    LOCAL_PATH = "local_path"  # referenced in place, never copied (Windows local-first)
    OBJECT = "object"  # object storage key (cloud)


class Container(StrEnum):
    MP4 = "mp4"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    AVI = "avi"


class HighlightCategory(StrEnum):
    FUNNY = "funny"
    CINEMATIC = "cinematic"
    ACTION = "action"
    CLUTCH = "clutch"
    UNEXPECTED = "unexpected"


class HighlightTier(StrEnum):
    TOP = "top"
    STRONG = "strong"
    MAYBE = "maybe"


class AspectRatio(StrEnum):
    VERTICAL = "9:16"
    LANDSCAPE = "16:9"
    SQUARE = "1:1"
    PORTRAIT = "3:4"


@dataclass(slots=True)
class Project:
    name: str
    mode: EditingMode = EditingMode.AUTO
    status: ProjectStatus = ProjectStatus.CREATED
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class VideoStream:
    index: int
    codec: str
    width: int  # coded size; see display_width/height for rotation
    height: int
    fps: float  # average frame rate
    variable_frame_rate: bool
    rotation: int = 0  # degrees clockwise to display correctly
    pix_fmt: str | None = None
    bit_depth: int = 8
    hdr: bool = False
    bit_rate: int | None = None

    @property
    def display_width(self) -> int:
        return self.height if self.rotation % 180 else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation % 180 else self.height


@dataclass(frozen=True, slots=True)
class AudioStream:
    index: int
    codec: str
    channels: int
    sample_rate: int
    title: str | None = None  # e.g. OBS track names: "Mic", "Game"
    language: str | None = None
    is_default: bool = False
    bit_rate: int | None = None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_s: float
    video: tuple[VideoStream, ...]
    audio: tuple[AudioStream, ...]
    format_name: str
    bit_rate: int | None = None
    start_time_s: float = 0.0  # container start offset; timestamps are relative to this

    @property
    def primary_video(self) -> VideoStream:
        return self.video[0]


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """What this machine's FFmpeg can actually do. Drives decoder/encoder choices."""

    ffmpeg_path: str
    ffprobe_path: str
    ffmpeg_version: str
    hwaccels: tuple[str, ...]  # decode accelerations FFmpeg was built with
    h264_encoders: tuple[str, ...]  # verified working, best first
    hevc_encoders: tuple[str, ...]
    gpl_build: bool  # libx264/libx265 present => GPL; matters for redistribution

    @property
    def best_h264_encoder(self) -> str | None:
        return self.h264_encoders[0] if self.h264_encoders else None

    @property
    def has_hw_encoder(self) -> bool:
        return any(not e.startswith("lib") for e in self.h264_encoders + self.hevc_encoders)


@dataclass(slots=True)
class MediaAsset:
    project_id: str
    source_uri: str
    storage_kind: StorageKind
    container: Container
    size_bytes: int
    fingerprint: str  # fast identity: size + head/tail hash
    original_name: str
    sha256: str | None = None  # full checksum, computed by a background job
    media_info: MediaInfo | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class TimeRange:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError(f"invalid range {self.start_s}..{self.end_s}")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class AIJudgement:
    """What the AI judge concluded about one candidate, grounded in frames + transcript."""

    provider: str  # e.g. "anthropic/claude-sonnet-5-5", "ollama/qwen2.5vl:7b"
    is_highlight: bool
    category: HighlightCategory | None
    quality: int  # 1-10
    standalone: bool  # understandable without the preceding gameplay
    what_happens: str
    reason: str
    title: str
    hook: str
    confidence: float
    refined: TimeRange | None = None  # AI-suggested tighter boundaries, if any
    tags: tuple[str, ...] = ()  # factual checklist answers that were true, e.g. "boss"
    model_quality: int | None = None  # the model's own 1-10 impression (poorly calibrated)
    # Moment database v2 (judge prompt v4); defaults keep older verdicts loadable
    dialogue: tuple[tuple[str, str], ...] = ()  # (speaker, line) read from on-screen subtitles
    names: tuple[str, ...] = ()  # boss bars, HUD and location text, read exactly
    characters: tuple[str, ...] = ()  # who is visibly there
    formats: tuple[str, ...] = ()  # content formats this moment could carry
    emotion: int = 0  # 0-10 emotional / story weight
    intensity: int = 0  # 0-10 action intensity
    dialogue_matters: bool = False  # the original lines carry the moment: never talk over
    payoff_seen: bool = False  # the outcome is visible inside the window
    payoff_s: float | None = None  # source time of the payoff
    hook_s: float | None = None  # source time of the most gripping instant
    visual: str = ""  # good | dark | cluttered | blurry


@dataclass(slots=True)
class HighlightCandidate:
    project_id: str
    asset_id: str
    range: TimeRange
    peak_s: float
    signal_score: float  # 0..1 percentile of the peak within this video
    peak_score: float  # raw interest value at the peak
    drivers: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)
    judgement: AIJudgement | None = None
    compare_score: float | None = None  # 0..1 average placing in AI side-by-side comparisons
    id: str = field(default_factory=new_id)

    @property
    def clip_range(self) -> TimeRange:
        j = self.judgement
        if j and j.refined and j.refined.duration_s >= MIN_CLIP_S:
            return j.refined
        return self.range


# --- Edit Decision List: the editable representation of a generated short ---


@dataclass(frozen=True, slots=True)
class CropKeyframe:
    t_s: float  # clip-relative time
    cx: float  # crop centre, normalised 0..1 of source width
    cy: float
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class CaptionWord:
    text: str
    start_s: float
    end_s: float
    emphasis: bool = False


@dataclass(slots=True)
class TextOverlay:
    """Times are on the Short's own (output) timeline."""

    text: str
    start_s: float
    end_s: float
    style: str = "hook"  # hook (pop-in) | caps | plain (subtitle) | bar (meme caption bar)
    y: float | None = None  # vertical centre as a share of the frame height; None = default
    size: float = 1.0  # multiplier on the style's font size
    color: str = ""  # "#RRGGBB"; "" = the style's own colours
    owner: str = "ai"  # ai (planned, may be re-planned) | user (made or edited by hand)


@dataclass(slots=True)
class AudioMix:
    game_gain: float = 1.0
    voice_gain: float = 1.0
    music_gain: float = 0.0
    sfx_gain: float = 1.0


@dataclass(slots=True)
class Clip:
    project_id: str
    asset_id: str
    title: str
    segments: list[TimeRange]  # source ranges, concatenated in order (jump cuts)
    aspect: AspectRatio = AspectRatio.VERTICAL
    layout: str = "auto"  # auto | crop (follow the action) | fit (whole frame on blurred fill)
    style: str = "hype"  # hype | meme (caption bar) | cinematic (mono grade, glitch, music)
    crop_path: list[CropKeyframe] = field(default_factory=list)
    crop_version: int = 0  # planner version that made crop_path; older paths are re-planned
    captions: list[CaptionWord] = field(default_factory=list)
    caption_style: str = "bold_pop"
    overlays: list[TextOverlay] = field(default_factory=list)
    audio: AudioMix = field(default_factory=AudioMix)
    effects: list[dict[str, Any]] = field(default_factory=list)
    publish: dict[str, Any] = field(default_factory=dict)  # title, description, hashtags...
    candidate_id: str | None = None
    rank: int | None = None  # position in the highlight list when generated
    version: int = 1
    id: str = field(default_factory=new_id)
    # Per-part game audio set in the editor, parallel to segments: {"gain_db", "mute"}
    segment_audio: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    def render_segments(self) -> list[TimeRange]:
        """Source ranges as rendered: an optional cold-open teaser, then the clip itself."""
        teaser = [
            TimeRange(e["start_s"], e["end_s"]) for e in self.effects if e["type"] == "cold_open"
        ]
        return teaser[:1] + self.segments

    def speed_pieces(self) -> list[tuple[float, float, float]]:
        """(start, end, factor) speed changes on the pre-speed timeline, in order."""
        return sorted(
            (e["start_s"], e["end_s"], e["factor"])
            for e in self.effects
            if e["type"] in ("speed", "slowmo") and e["end_s"] > e["start_s"]
        )

    def to_output_time(self, t: float) -> float:
        """Pre-speed timeline (teaser included) -> time in the finished video."""
        out = t
        for a, b, f in self.speed_pieces():
            if t > a:
                out += (min(t, b) - a) * (1 / f - 1)
        return out

    @property
    def output_duration_s(self) -> float:
        """Length of the finished video, including the teaser and speed changes."""
        return self.to_output_time(sum(s.duration_s for s in self.render_segments()))

    def to_source_time(self, clip_t: float) -> float:
        """Map a time in the edited clip back to the original recording."""
        for seg in self.segments:
            if clip_t <= seg.duration_s:
                return seg.start_s + clip_t
            clip_t -= seg.duration_s
        return self.segments[-1].end_s


# --- Jobs ---


class JobType(StrEnum):
    CHECKSUM = "checksum"
    ANALYZE = "analyze"
    RENDER = "render"
    LORE_INDEX = "lore_index"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass(slots=True)
class Job:
    type: JobType
    project_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    stage: str | None = None
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
