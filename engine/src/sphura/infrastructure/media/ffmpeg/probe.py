from __future__ import annotations

import itertools
import json
import logging
import re
import statistics
from fractions import Fraction
from typing import Any

from sphura.domain.errors import InvalidMediaError, ToolchainError
from sphura.domain.models import AudioStream, MediaInfo, VideoStream
from sphura.infrastructure.media.ffmpeg.process import run, run_with_progress
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain

log = logging.getLogger(__name__)

_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10/DV) and HLG
_VFR_SAMPLE_S = 10.0
_VFR_DEVIATION = 0.25  # frame interval more than 25% off the median counts as irregular
_VFR_IRREGULAR_SHARE = 0.05


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def _rate(v: Any) -> float:
    try:
        r = Fraction(str(v))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(r) if r > 0 else 0.0


def _bit_depth(s: dict[str, Any]) -> int:
    if d := _int(s.get("bits_per_raw_sample")):
        return d
    pix = str(s.get("pix_fmt") or "")
    for depth in (16, 12, 10):
        if str(depth) in pix:
            return depth
    return 10 if pix.startswith("p010") else 8


def _rotation(s: dict[str, Any]) -> int:
    for sd in s.get("side_data_list") or ():
        if "rotation" in sd:
            # Display matrix rotation is counter-clockwise; store clockwise
            return int(-float(sd["rotation"])) % 360
    if r := _int((s.get("tags") or {}).get("rotate")):
        return r % 360
    return 0


_GENERIC_HANDLER = re.compile(r"handler|media|produced by|isom|gpac|l-smash|mainconcept", re.I)


def _track_title(tags: dict[str, Any]) -> str | None:
    """Real track names only; MP4 handler_name is usually an encoder default, not a label."""
    if title := str(tags.get("title") or "").strip():
        return title
    handler = str(tags.get("handler_name") or "").strip()
    if handler and len(handler) <= 32 and not _GENERIC_HANDLER.search(handler):
        return handler
    return None


def _broken_media_message(stderr: str, name: str) -> str | None:
    low = stderr.lower()
    if "moov atom not found" in low:
        return (
            f"{name}: the MP4/MOV index is missing, so the recording probably didn't finish "
            "(crash or power loss). Record to MKV in OBS to avoid this; the file may be "
            "recoverable with a repair tool such as untrunc."
        )
    if "invalid data found" in low or "end of file" in low:
        return f"{name}: the file is corrupt or truncated and can't be read."
    return None


class FfprobeMediaProbe:
    def __init__(self, toolchain: FFmpegToolchain) -> None:
        self._tc = toolchain

    def probe(self, uri: str) -> MediaInfo:
        name = uri.replace("\\", "/").rsplit("/", 1)[-1]
        try:
            out = run(
                [
                    self._tc.ffprobe, "-v", "error", "-print_format", "json",
                    "-show_format", "-show_streams", uri,
                ],
                timeout_s=120,
            ).stdout  # fmt: skip
        except ToolchainError as e:
            if msg := _broken_media_message(str(e), name):
                raise InvalidMediaError(msg) from e
            raise

        data = json.loads(out or "{}")
        fmt: dict[str, Any] = data.get("format") or {}
        streams: list[dict[str, Any]] = data.get("streams") or []

        raw_video = [
            s
            for s in streams
            if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")  # skip cover art
        ]
        if not raw_video:
            raise InvalidMediaError(f"{name}: no video stream found")

        duration = _float(fmt.get("duration")) or max(
            (_float(s.get("duration")) or 0.0 for s in streams), default=0.0
        )
        if duration <= 0:
            # e.g. browser-recorded WebM without a duration header: scan packets
            duration = self._scan_duration(uri)
        if duration <= 0:
            raise InvalidMediaError(f"{name}: could not determine the video duration")

        start = _float(fmt.get("start_time")) or 0.0
        video = tuple(self._video(uri, s, start, duration, i == 0) for i, s in enumerate(raw_video))
        audio = tuple(self._audio(s) for s in streams if s.get("codec_type") == "audio")

        return MediaInfo(
            duration_s=duration,
            video=video,
            audio=audio,
            format_name=str(fmt.get("format_name") or ""),
            bit_rate=_int(fmt.get("bit_rate")),
            start_time_s=start,
        )

    def _video(
        self, uri: str, s: dict[str, Any], start: float, duration: float, primary: bool
    ) -> VideoStream:
        avg, real = _rate(s.get("avg_frame_rate")), _rate(s.get("r_frame_rate"))
        vfr, sampled_fps = (
            self._vfr(uri, int(s["index"]), start, duration)
            if primary
            else (
                False,
                None,
            )
        )
        fps = avg or sampled_fps or real
        return VideoStream(
            index=int(s["index"]),
            codec=str(s.get("codec_name") or "unknown"),
            width=_int(s.get("width")) or 0,
            height=_int(s.get("height")) or 0,
            fps=round(fps, 3),
            variable_frame_rate=vfr,
            rotation=_rotation(s),
            pix_fmt=s.get("pix_fmt"),
            bit_depth=_bit_depth(s),
            hdr=s.get("color_transfer") in _HDR_TRANSFERS,
            bit_rate=_int(s.get("bit_rate")),
        )

    @staticmethod
    def _audio(s: dict[str, Any]) -> AudioStream:
        tags = s.get("tags") or {}
        return AudioStream(
            index=int(s["index"]),
            codec=str(s.get("codec_name") or "unknown"),
            channels=_int(s.get("channels")) or 0,
            sample_rate=_int(s.get("sample_rate")) or 0,
            title=_track_title(tags),
            language=tags.get("language") if tags.get("language") != "und" else None,
            is_default=bool((s.get("disposition") or {}).get("default")),
            bit_rate=_int(s.get("bit_rate")),
        )

    def _vfr(
        self, uri: str, index: int, start: float, duration: float
    ) -> tuple[bool, float | None]:
        """Sample frame intervals at the start and middle (recordings often stutter early)."""
        mid = start + duration * 0.5
        intervals = f"{start}%+{_VFR_SAMPLE_S},{mid:.3f}%+{_VFR_SAMPLE_S}"
        try:
            out = run(
                [
                    self._tc.ffprobe, "-v", "error", "-select_streams", str(index),
                    "-read_intervals", intervals, "-show_entries", "packet=pts_time",
                    "-of", "csv=p=0", uri,
                ],
                timeout_s=60,
            ).stdout  # fmt: skip
        except ToolchainError:
            log.warning("frame-interval sampling failed for %s", uri, exc_info=True)
            return False, None

        pts = sorted(t for ln in out.splitlines() if (t := _float(ln.strip(" ,"))) is not None)
        deltas = [b - a for a, b in itertools.pairwise(pts) if 0 < b - a < 1.0]
        if len(deltas) < 30:
            return False, None
        med = statistics.median(deltas)
        irregular = sum(abs(d - med) > med * _VFR_DEVIATION for d in deltas) / len(deltas)
        return irregular > _VFR_IRREGULAR_SHARE, 1.0 / med

    def _scan_duration(self, uri: str) -> float:
        try:
            return run_with_progress(
                [self._tc.ffmpeg, "-v", "error", "-i", uri, "-map", "0:v:0", "-c", "copy",
                 "-f", "null", "-"],
                None,
                lambda _f, _t: None,
            )  # fmt: skip
        except ToolchainError:
            return 0.0
