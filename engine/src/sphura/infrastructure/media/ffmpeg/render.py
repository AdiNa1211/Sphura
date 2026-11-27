"""Phase 6b: render a Clip's EDL from the original recording.

Each segment is its own input with a fast input seek, so only the clip's ranges are decoded
(never the whole multi-hour file). Audio is loudness-normalised for social platforms.
Phase 8 effects (punch-ins, flash, slow motion) and text overlays are applied last.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sphura.domain.errors import ToolchainError
from sphura.domain.models import AspectRatio, Clip, CropKeyframe, MediaInfo, TextOverlay
from sphura.infrastructure.media.ffmpeg.ass import overlays_ass
from sphura.infrastructure.media.ffmpeg.process import run_with_progress
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain
from sphura.pipeline import sfx
from sphura.pipeline.reframe import crop_x_expression
from sphura.pipeline.styles import Geometry, geometry
from sphura.ports.media import RenderOutcome

log = logging.getLogger(__name__)

MAX_FPS = 60
MAX_HEIGHT = 1080
MAX_SOURCE_HEIGHT = 2160
TARGET_LUFS = -14  # what TikTok/Shorts/Reels normalise towards

_ENCODER_ARGS: dict[str, list[str]] = {
    "h264_nvenc": ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0",
                   "-maxrate", "20M", "-bufsize", "40M"],
    "h264_qsv": ["-preset", "medium", "-global_quality", "21"],
    "h264_amf": ["-quality", "balanced", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"],
    "h264_videotoolbox": ["-b:v", "12M", "-maxrate", "16M"],
    "h264_mf": ["-b:v", "12M"],
    "libx264": ["-preset", "veryfast", "-crf", "20"],
    "libopenh264": ["-b:v", "12M"],
}  # fmt: skip

# Editor previews: same filter graph at half size, 30 fps, fastest encoder settings
DRAFT_SCALE = 0.5
DRAFT_FPS = 30
_DRAFT_ARGS: dict[str, list[str]] = {
    "h264_nvenc": ["-preset", "p1", "-tune", "ll", "-rc", "vbr", "-cq", "28", "-b:v", "0"],
    "h264_qsv": ["-preset", "veryfast", "-global_quality", "28"],
    "h264_amf": ["-quality", "speed", "-rc", "cqp", "-qp_i", "26", "-qp_p", "28"],
    "h264_videotoolbox": ["-b:v", "4M", "-realtime", "1"],
    "h264_mf": ["-b:v", "4M"],
    "libx264": ["-preset", "ultrafast", "-crf", "26"],
    "libopenh264": ["-b:v", "4M"],
}  # fmt: skip


def _geo(clip: Clip, frame_w: int, frame_h: int, draft: bool) -> Geometry:
    g = geometry(clip, frame_w, frame_h)
    if not draft:
        return g

    def even(v: int) -> int:
        return max(2, round(v * DRAFT_SCALE) // 2 * 2)

    return Geometry(even(g.out_w), even(g.out_h), even(g.video_h), even(g.bar_h) if g.bar_h else 0)


class FFmpegRenderer:
    def __init__(self, toolchain: FFmpegToolchain) -> None:
        self._tc = toolchain

    def render(
        self,
        clip: Clip,
        source_uri: str,
        info: MediaInfo,
        audio_stream: int | None,
        out_path: Path,
        on_progress: Callable[[float], None],
        draft: bool = False,
    ) -> RenderOutcome:
        profile = self._tc.profile()
        encoders = list(profile.h264_encoders) or ["libx264"]
        last: ToolchainError | None = None
        for enc in encoders:  # hardware first; fall back if the GPU path fails mid-render
            try:
                done = self._run(
                    clip, source_uri, info, audio_stream, out_path, enc, on_progress, draft
                )
                return RenderOutcome(done, enc, clip.output_duration_s)
            except ToolchainError as e:
                log.warning("render with %s failed, trying next encoder: %s", enc, e)
                last = e
        raise last or ToolchainError("no working H.264 encoder")

    def build_command(
        self,
        clip: Clip,
        source_uri: str,
        info: MediaInfo,
        audio_stream: int | None,
        out: Path,
        encoder: str,
        text_file: str | None = None,
        sfx_file: str | None = None,
        draft: bool = False,
    ) -> list[str]:
        """text_file / sfx_file: names relative to the directory ffmpeg runs in."""
        v = info.primary_video
        hw = ["-hwaccel", "auto"] if self._tc.profile().hwaccels else []
        cmd = [self._tc.ffmpeg, "-hide_banner", "-v", "error", "-y", "-nostdin"]
        segments = clip.render_segments()
        for s in segments:
            cmd += [*hw, "-ss", f"{s.start_s:.3f}", "-t", f"{s.duration_s:.3f}", "-i", source_uri]

        extra = len(segments)  # index of the next extra input
        sfx_in = music_in = None
        if sfx_file:
            cmd += ["-i", sfx_file]
            sfx_in, extra = extra, extra + 1
        music = next((e for e in clip.effects if e["type"] == "music"), None)
        if music and Path(music["path"]).is_file():
            cmd += ["-ss", f"{music['offset_s']:.3f}", "-i", music["path"]]
            music_in, extra = extra, extra + 1
        voice = next((e for e in clip.effects if e["type"] == "voiceover"), None)
        voice_in = None
        if voice and Path(voice["path"]).is_file():
            cmd += ["-i", voice["path"]]
            voice_in = extra
        landscape = clip.aspect == AspectRatio.LANDSCAPE
        # Vertical/square crop a slice of the frame, so keep source detail until the crop
        height = min(MAX_HEIGHT if landscape else MAX_SOURCE_HEIGHT, v.display_height) // 2 * 2
        scale = (
            f",scale=-2:{height}:flags=lanczos" if landscape or height < v.display_height else ""
        )
        cap = DRAFT_FPS if draft else MAX_FPS
        fps = f",fps={cap}" if v.fps > cap + 0.5 else ""
        parts: list[str] = []
        lead = len(segments) - len(clip.segments)  # the cold-open teaser comes first
        for i, s in enumerate(segments):
            parts.append(f"[{i}:{v.index}]setpts=PTS-STARTPTS{scale}{fps},setsar=1[v{i}]")
            fade_out = max(0.0, s.duration_s - 0.05)
            src = (
                f"[{i}:{audio_stream}]asetpts=PTS-STARTPTS,"
                if audio_stream is not None
                else f"anullsrc=r=48000:cl=stereo,atrim=duration={s.duration_s:.3f},"
            )
            # Short fades at every join stop clicks at the jump cuts
            parts.append(
                f"{src}aresample=48000,aformat=channel_layouts=stereo,"
                f"afade=t=in:d=0.03,afade=t=out:st={fade_out:.3f}:d=0.05"
                f"{_part_volume(clip, i - lead)}[a{i}]"
            )
        n = len(segments)
        joined = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{joined}concat=n={n}:v=1:a=1[vc][ac]")
        frame_h = height
        frame_w = round(frame_h * v.display_width / v.display_height) // 2 * 2
        parts += self._frame_chain(clip, frame_w, frame_h, draft)
        g = _geo(clip, frame_w, frame_h, draft)
        parts += _effects_chain(clip, g.out_w, g.video_h, text_file, g.bar_h)
        parts += _audio_mix(clip, sfx_in, music_in, music, voice_in)
        cmd += [
            "-filter_complex", ";".join(parts),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", encoder, *(_DRAFT_ARGS if draft else _ENCODER_ARGS).get(encoder, []),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            "-f", "mp4", str(out),
        ]  # fmt: skip
        return cmd

    @staticmethod
    def _frame_chain(clip: Clip, frame_w: int, frame_h: int, draft: bool = False) -> list[str]:
        """[vc] (concatenated, source framing) -> [vout] in the clip's aspect and layout."""
        if clip.aspect == AspectRatio.LANDSCAPE:
            if draft:  # half size, like the other drafts
                w, h = frame_w // 4 * 2, frame_h // 4 * 2
                return [f"[vc]scale={w}:{h},format=yuv420p[vf]"]
            return ["[vc]format=yuv420p[vf]"]
        g = _geo(clip, frame_w, frame_h, draft)
        out_w, out_h = g.out_w, g.video_h  # footage area; a caption bar is added later
        if clip.layout == "fit":
            # Whole frame centred over a blurred, zoomed copy: nothing important cut off
            return [
                "[vc]split=2[bgi][fgi]",
                f"[bgi]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},gblur=sigma=24,eq=brightness=-0.08[bg]",
                f"[fgi]scale={out_w}:-2:flags=lanczos[fg]",
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[vf]",
            ]
        crop_w = min(frame_w, round(frame_h * out_w / out_h) // 2 * 2)
        keys = clip.crop_path or [CropKeyframe(0.0, 0.5, 0.5)]
        x = crop_x_expression(keys, frame_w, crop_w)
        upscale = out_h / frame_h
        sharpen = ",unsharp=5:5:0.5" if upscale > 1.3 else ""  # soften the cost of upscaling
        return [
            f"[vc]crop=w={crop_w}:h={frame_h}:x='{x}':y=0,"
            f"scale={out_w}:{out_h}:flags=lanczos{sharpen},setsar=1,format=yuv420p[vf]"
        ]

    def _run(
        self,
        clip: Clip,
        source_uri: str,
        info: MediaInfo,
        audio_stream: int | None,
        out_path: Path,
        encoder: str,
        on_progress: Callable[[float], None],
        draft: bool = False,
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp names: two renders of one clip must never write the same file
        tag = uuid.uuid4().hex[:8]
        part = out_path.with_name(f"{out_path.stem}.{tag}.part.mp4")
        text = out_path.with_name(f"{out_path.stem}.{tag}.ass")
        text_name: str | None = None
        overlays = clip.overlays + _narration_captions(clip) + _dialogue_captions(clip)
        if overlays:
            if self._tc.has_filter("ass"):
                g = geometry(clip, 1920, 1080)
                text.write_text(overlays_ass(overlays, g.out_w, g.out_h, g.bar_h), encoding="utf-8")
                text_name = text.name  # relative: avoids drive-letter escaping in filters
            else:
                log.warning("this FFmpeg build has no libass; on-screen text is skipped")
        sounds = out_path.with_name(f"{out_path.stem}.{tag}.sfx.wav")
        events = [
            {**e, "out_s": clip.to_output_time(e["t_s"])}
            for e in clip.effects
            if e["type"] == "sfx"
        ]
        sfx_name: str | None = None
        if events:
            sfx.write_wav(sounds, sfx.mix_track(events, clip.output_duration_s))
            sfx_name = sounds.name
        cmd = self.build_command(
            clip, source_uri, info, audio_stream, part, encoder, text_name, sfx_name, draft
        )
        try:
            run_with_progress(
                cmd, clip.output_duration_s, lambda frac, _t: on_progress(frac), out_path.parent
            )
        except BaseException:
            with contextlib.suppress(OSError):
                part.unlink()
            raise
        finally:
            for tmp in (text, sounds):
                with contextlib.suppress(OSError):
                    tmp.unlink()
        return _place(part, out_path)


def _place(part: Path, out_path: Path) -> Path:
    """Move the finished render to its name. Windows refuses while the old file is open
    (playing in SPHURA or another app): wait a little, then save it under a new name."""
    for _ in range(10):
        try:
            part.replace(out_path)  # never leave a half-written file under the final name
            return out_path
        except PermissionError:
            time.sleep(0.5)
    fallback = out_path.with_name(f"{out_path.stem}.{part.stem.split('.')[-2]}.mp4")
    part.replace(fallback)
    log.warning("%s is in use; saved the new render as %s", out_path.name, fallback.name)
    return fallback


def _effects_chain(
    clip: Clip, out_w: int, out_h: int, text_file: str | None, bar_h: int = 0
) -> list[str]:
    """[vf][ac] -> [vout][as]. Grade, zooms/shake, flashes and RGB splits run on the
    pre-speed timeline; speed ramps come next; text goes on the final timeline."""
    fx = clip.effects
    chain = ""
    for e in fx:
        if e["type"] == "grade":
            chain += (
                f",eq=contrast={e['contrast']}:saturation={e['saturation']}:gamma={e['gamma']}"
                ",unsharp=5:5:0.35"
            )
            if e.get("tint") == "cool":
                chain += ",colorbalance=rs=-0.05:bs=0.07:rm=-0.03:bm=0.05"
    punches = [e for e in fx if e["type"] == "punch"]
    pushes = [e for e in fx if e["type"] == "push"]
    if punches or pushes:
        z = "+".join([_punch_term(e) for e in punches] + [_push_term(e) for e in pushes])
        shake = "+".join(_punch_env(e) for e in punches if e.get("shake")) or "0"
        amp = out_w * 0.022
        chain += (
            f",scale=w='trunc({out_w}*(1+{z})/2)*2':h='trunc({out_h}*(1+{z})/2)*2'"
            f":eval=frame:flags=bicubic,crop={out_w}:{out_h}"
            # Shake rides on the punch zoom, so it never reveals the frame edge
            f":x='clip((iw-ow)/2+{amp:.1f}*({shake})*sin(2*PI*13*t),0,iw-ow)'"
            f":y='clip((ih-oh)/2+{amp * 0.7:.1f}*({shake})*cos(2*PI*11*t),0,ih-oh)'"
        )
    for e in fx:
        a, b = e.get("t_s", 0.0) - e.get("dur_s", 0.0) / 2, e.get("t_s", 0.0) + e.get("dur_s", 0.0)
        if e["type"] == "flash":
            chain += f",drawbox=c=white@0.6:t=fill:enable='between(t,{a:.3f},{b:.3f})'"
        elif e["type"] == "rgb":
            chain += f",rgbashift=rh=-7:bh=7:enable='between(t,{e['t_s']:.3f},{b:.3f})'"
        elif e["type"] == "glitch":
            chain += (
                f",rgbashift=rh=-16:bh=12:gv=4:enable='between(t,{a:.3f},{b:.3f})'"
                f",noise=alls=28:allf=t:enable='between(t,{a:.3f},{b:.3f})'"
            )
        elif e["type"] == "pixel":
            # Datamosh-style smear: chunky blocks plus a colour split, for a few frames
            w = f"between(t,{a:.3f},{b:.3f})"
            chain += (
                f",pixelize=width=36:height=36:enable='{w}',rgbashift=rh=-12:bh=12:enable='{w}'"
            )
        elif e["type"] == "ghost":
            w = f"between(t,{e['start_s']:.3f},{e['end_s']:.3f})"
            chain += f",tmix=frames=4:weights='1 1 2 4':enable='{w}'"
        elif e["type"] == "vignette":
            chain += f",vignette=angle=0.75:enable='between(t,{e['start_s']:.3f},{e['end_s']:.3f})'"
        elif e["type"] == "letterbox":
            bh = round(out_h * 0.1)
            w = f"between(t,{e['start_s']:.3f},{e['end_s']:.3f})"
            chain += (
                f",drawbox=x=0:y=0:w=iw:h={bh}:c=black:t=fill:enable='{w}'"
                f",drawbox=x=0:y=ih-{bh}:w=iw:h={bh}:c=black:t=fill:enable='{w}'"
            )
    parts = [f"[vf]null{chain}[vp]"]

    pieces = _speed_plan(clip)
    if pieces:
        n = len(pieces)
        labels = [f"x{i}" for i in range(n)]
        parts.append(f"[vp]split={n}{''.join(f'[{x}]' for x in labels)}")
        parts.append(f"[ac]asplit={n}{''.join(f'[a{x}]' for x in labels)}")
        vo, ao = "", ""
        for x, (a, b, k) in zip(labels, pieces, strict=True):
            rng = f"start={a:.3f}" + (f":end={b:.3f}" if b is not None else "")
            parts.append(f"[{x}]trim={rng},setpts=(PTS-STARTPTS)/{k}[{x}v]")
            # Slowed game audio drops in pitch and gets muffled: the "time slows" feel
            slow = (
                f",asetrate={round(48000 * k)},aresample=48000,lowpass=f=2500,volume=0.7"
                if k < 1
                else ""
            )
            parts.append(f"[a{x}]atrim={rng},asetpts=PTS-STARTPTS{slow}[{x}a]")
            vo += f"[{x}v]"
            ao += f"[{x}a]"
        parts.append(f"{vo}concat=n={n}:v=1:a=0[vt]")
        parts.append(f"{ao}concat=n={n}:v=0:a=1[as]")
    else:
        parts += ["[vp]null[vt]", "[ac]anull[as]"]

    bar = f",pad={out_w}:{out_h + bar_h}:0:{bar_h}:color=white" if bar_h else ""
    text = f",ass={text_file}" if text_file else ""
    parts.append(f"[vt]setsar=1{bar}{text},format=yuv420p[vout]")
    return parts


def _part_volume(clip: Clip, i: int) -> str:
    """Volume set for one part in the editor; nothing when it's untouched."""
    if i < 0 or i >= len(clip.segment_audio):
        return ""
    a = clip.segment_audio[i]
    if a.get("mute"):
        return ",volume=0"
    db = float(a.get("gain_db") or 0.0)
    return f",volume={10 ** (db / 20):.4f}" if abs(db) >= 0.05 else ""


def _speed_plan(clip: Clip) -> list[tuple[float, float | None, float]]:
    """Contiguous (start, end, factor) pieces covering the whole clip; [] when no speed
    changes. The last piece is open-ended."""
    changes = [(a, b, f) for a, b, f in clip.speed_pieces() if b - a > 0.04]
    if not changes:
        return []
    out: list[tuple[float, float | None, float]] = []
    cursor = 0.0
    for a, b, f in changes:
        a = max(a, cursor)
        if a - cursor > 0.04:
            out.append((cursor, a, 1.0))
        if b > a:
            out.append((a, b, f))
            cursor = b
    out.append((cursor, None, 1.0))
    return out


def _audio_mix(
    clip: Clip,
    sfx_in: int | None,
    music_in: int | None,
    music: dict[str, Any] | None,
    voice_in: int | None = None,
) -> list[str]:
    """[as] game audio (+ generated SFX, + music ducked under the game, + narration with
    everything else ducked under it) -> [aout]."""
    parts = [f"[as]volume={clip.audio.game_gain:.3f}[g0]"]
    mix = ["[g0]"]
    if music_in is not None and music is not None:
        parts[0] = f"[as]volume={clip.audio.game_gain:.3f},asplit=2[g0][gsc]"
        dur = clip.output_duration_s
        gain = 10 ** (music["gain_db"] / 20)
        env = _music_envelope(music, dur)
        parts.append(
            f"[{music_in}:a]aresample=48000,aformat=channel_layouts=stereo,apad,"
            f"atrim=duration={dur:.3f},volume='{gain:.4f}*{env}':eval=frame[mu]"
        )
        if music.get("duck", True):
            # Duck under the game: the music backs off whenever the game gets loud
            parts.append(
                "[mu][gsc]sidechaincompress=threshold=0.04:ratio=6:attack=15:release=350[md]"
            )
        else:  # music-led edit: the game sits underneath instead
            parts.append("[gsc]anullsink")
            parts.append("[mu]anull[md]")
        mix.append("[md]")
    if sfx_in is not None:
        parts.append(f"[{sfx_in}:a]aresample=48000,aformat=channel_layouts=stereo[fx]")
        mix.append("[fx]")
    tail = f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11,aresample=48000[aout]"
    if voice_in is not None:
        dur = clip.output_duration_s
        bed = (
            "[g0]anull[bed]"
            if len(mix) == 1
            else (f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0[bed]")
        )
        parts.append(bed)
        parts.append(
            f"[{voice_in}:a]aresample=48000,aformat=channel_layouts=stereo,apad,"
            f"atrim=duration={dur:.3f},volume=1.6,asplit=2[vo][vsc]"
        )
        # The game and music back off while the narrator speaks
        parts.append("[bed][vsc]sidechaincompress=threshold=0.015:ratio=10:attack=20:"
                     "release=400:makeup=1[bd]")  # fmt: skip
        parts.append(f"[bd][vo]amix=inputs=2:duration=first:normalize=0,{tail}")
    elif len(mix) == 1:
        parts.append(f"[g0]{tail}")
    else:
        parts.append(f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0,{tail}")
    return parts


def _dialogue_captions(clip: Clip) -> list[TextOverlay]:
    """The characters' lines, big enough for a phone, with the speaker's name."""
    subs = next((e for e in clip.effects if e["type"] == "subs"), None)
    if not subs:
        return []
    return [
        TextOverlay(
            (f"{ln['speaker']}|" if ln.get("speaker") else "") + str(ln["text"]),
            ln["start"], ln["end"], "sub", y=0.755,
        )
        for ln in subs.get("lines", [])
        if ln["end"] > ln["start"]
    ]  # fmt: skip


def _narration_captions(clip: Clip) -> list[TextOverlay]:
    """Each spoken narration line as a subtitle, higher than the characters' captions."""
    voice = next((e for e in clip.effects if e["type"] == "voiceover"), None)
    if not voice:
        return []
    return [
        TextOverlay(
            str(ln.get("caption") or ln.get("text")), ln["start"], ln["end"], "plain", y=0.62
        )
        for ln in voice.get("lines", [])
        if not ln.get("dropped") and "start" in ln
    ]


def _music_envelope(music: dict[str, Any], dur: float) -> str:
    """Silent through the teaser, fade in, dip out before the key hit, fade out at the end."""
    t0 = music.get("in_s", 0.0)
    env = f"clip((t-{t0:.3f})/0.25,0,1)*clip(({dur:.3f}-t)/1.2,0,1)"
    if music.get("dip"):
        a, b = music["dip"]
        env += f"*(1-clip((t-{a - 0.15:.3f})/0.15,0,1)*clip(({b:.3f}-t)/0.05,0,1))"
    for a, b in music.get("dialogue") or []:
        # Down to ~35% under dialogue, easing in before the line and out after it
        env += f"*(1-0.65*clip((t-{a - 0.25:.3f})/0.25,0,1)*clip(({b + 0.3:.3f}-t)/0.3,0,1))"
    return env


def _punch_env(e: dict[str, Any]) -> str:
    """0..1 bump: 60 ms ramp in, eased decay out."""
    t, d, up = float(e["t_s"]), float(e["dur_s"]), 0.06
    return (
        f"if(between(t,{t - up:.3f},{t:.3f}),(t-{t - up:.3f})/{up},"
        f"if(between(t,{t:.3f},{t + d:.3f}),pow(1-(t-{t:.3f})/{d:.3f},2),0))"
    )


def _punch_term(e: dict[str, Any]) -> str:
    return f"{float(e['zoom']) - 1:.3f}*{_punch_env(e)}"


def _push_term(e: dict[str, Any]) -> str:
    """Slow push-in over [start, end], released quickly afterwards."""
    a, b, z = float(e["start_s"]), float(e["end_s"]), float(e["zoom"])
    return (
        f"{z:.3f}*if(between(t,{a:.3f},{b:.3f}),(t-{a:.3f})/{max(b - a, 0.01):.3f},"
        f"if(between(t,{b:.3f},{b + 0.6:.3f}),1-(t-{b:.3f})/0.6,0))"
    )
