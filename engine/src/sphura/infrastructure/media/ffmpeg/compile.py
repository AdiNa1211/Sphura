"""Joins rendered moments into one highlight reel: one pass, one encode.

Each part is fitted to the output size and frame rate, parts are joined with crossfades
(a few frames long = a clean cut), the music comes in under the game and backs off when
the game gets loud, and the whole mix is normalised for social platforms. Branding (a text
watermark and/or a logo) is burned in last.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from sphura.domain.errors import ToolchainError
from sphura.infrastructure.media.ffmpeg.process import run_with_progress
from sphura.infrastructure.media.ffmpeg.render import _ENCODER_ARGS, TARGET_LUFS, _place
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain
from sphura.ports.media import ReelMusic, ReelPart, RenderOutcome

log = logging.getLogger(__name__)

_AFMT = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"


class FFmpegReelCompiler:
    def __init__(self, toolchain: FFmpegToolchain) -> None:
        self._tc = toolchain

    def compile(
        self,
        parts: list[ReelPart],
        joins: list[float],
        size: tuple[int, int],
        fps: int,
        music: ReelMusic | None,
        watermark_text: str,
        watermark_logo: Path | None,
        out_path: Path,
        on_progress: Callable[[float], None],
    ) -> RenderOutcome:
        if not parts:
            raise ToolchainError("nothing to compile")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        part_file = out_path.with_name(f"{out_path.stem}.{tag}.part.mp4")
        wm = out_path.with_name(f"{out_path.stem}.{tag}.ass")
        total = sum(p.duration_s for p in parts) - sum(joins[: len(parts) - 1])
        wm_name = None
        if watermark_text.strip() and self._tc.has_filter("ass"):
            wm.write_text(_watermark_ass(watermark_text.strip(), size, total, bool(watermark_logo)),
                          encoding="utf-8")  # fmt: skip
            wm_name = wm.name  # relative to the working dir: no drive-letter escaping
        encoders = list(self._tc.profile().h264_encoders) or ["libx264"]
        last: ToolchainError | None = None
        try:
            for enc in encoders:  # hardware first, like single Shorts
                cmd = self.build_command(parts, joins, size, fps, music, wm_name, watermark_logo,
                                         part_file, enc, total)  # fmt: skip
                try:
                    run_with_progress(cmd, total, lambda f, _t: on_progress(f), out_path.parent)
                    return RenderOutcome(_place(part_file, out_path), enc, total)
                except ToolchainError as e:
                    log.warning("reel encode with %s failed, trying next encoder: %s", enc, e)
                    last = e
        finally:
            for tmp in (wm, part_file):
                with contextlib.suppress(OSError):
                    tmp.unlink()
        raise last or ToolchainError("no working H.264 encoder")

    def build_command(
        self,
        parts: list[ReelPart],
        joins: list[float],
        size: tuple[int, int],
        fps: int,
        music: ReelMusic | None,
        watermark_ass: str | None,
        logo: Path | None,
        out: Path,
        encoder: str,
        total: float,
    ) -> list[str]:
        w, h = size
        cmd = [self._tc.ffmpeg, "-hide_banner", "-v", "error", "-y", "-nostdin"]
        for p in parts:
            cmd += ["-i", str(p.path)]
        n = len(parts)
        music_in = logo_in = None
        if music is not None:
            cmd += ["-ss", f"{max(0.0, music.offset_s):.3f}", "-i", str(music.path)]
            music_in = n
        if logo is not None:
            cmd += ["-loop", "1", "-t", f"{total:.3f}", "-i", str(logo)]
            logo_in = n + (music_in is not None)

        f: list[str] = []
        for i, p in enumerate(parts):
            d = f"{p.duration_s:.3f}"
            f.append(
                f"[{i}:v:0]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},"
                # Hold the last frame so every part is exactly its length: crossfade
                # offsets are counted from these lengths, and a few short frames add up
                f"format=yuv420p,tpad=stop_mode=clone:stop_duration=2,trim=duration={d},"
                f"setpts=PTS-STARTPTS,settb=AVTB[v{i}]"
            )
            src = f"[{i}:a:0]" if p.has_audio else f"anullsrc=r=48000:cl=stereo,atrim=duration={d},"
            f.append(f"{src}{_AFMT},apad,atrim=duration={d},asetpts=PTS-STARTPTS[a{i}]")
        v, a, t = "[v0]", "[a0]", parts[0].duration_s
        for i in range(1, n):
            j = max(0.04, min(joins[i - 1], parts[i].duration_s / 2, t / 2))
            f.append(f"{v}[v{i}]xfade=transition=fade:duration={j:.3f}:offset={t - j:.3f}[vx{i}]")
            f.append(f"{a}[a{i}]acrossfade=d={j:.3f}:c1=tri:c2=tri[ax{i}]")
            v, a, t = f"[vx{i}]", f"[ax{i}]", t + parts[i].duration_s - j

        if watermark_ass:
            f.append(f"{v}ass={watermark_ass}[vw]")
            v = "[vw]"
        if logo_in is not None:
            m = round(min(w, h) * 0.03)
            f.append(f"[{logo_in}:v]scale={round(w * 0.13)}:-1,format=rgba,"
                     "colorchannelmixer=aa=0.85[lg]")  # fmt: skip
            f.append(f"{v}[lg]overlay=W-w-{m}:{m}:shortest=1[vl]")
            v = "[vl]"
        f.append(f"{v}null[vout]")

        loud = f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11,aresample=48000"
        if music is not None and music_in is not None:
            span = max(1.0, music.end_s - music.start_s)
            delay = round(music.start_s * 1000)
            f.append(
                f"[{music_in}:a]{_AFMT},atrim=duration={span:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d=0.6,afade=t=out:st={max(0.0, span - 1.6):.3f}:d=1.5,"
                f"volume={music.gain_db:.1f}dB,adelay={delay}|{delay},apad[mu]"
            )
            f.append(f"{a}asplit=2[g0][gsc]")
            # Duck under the game: the music backs off whenever the game gets loud
            f.append("[mu][gsc]sidechaincompress=threshold=0.04:ratio=5:attack=15:release=350[md]")
            f.append(f"[g0][md]amix=inputs=2:duration=first:normalize=0,{loud}[aout]")
        else:
            f.append(f"{a}{loud}[aout]")

        cmd += [
            "-filter_complex", ";".join(f),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", encoder, *_ENCODER_ARGS.get(encoder, []),
            "-r", str(fps), "-t", f"{total:.3f}",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            "-f", "mp4", str(out),
        ]  # fmt: skip
        return cmd


def _watermark_ass(text: str, size: tuple[int, int], total: float, logo: bool) -> str:
    """Channel name in a corner: top right, or top left when a logo takes the right."""
    w, h = size
    fs = round(min(w, h) * 0.034)
    margin = round(min(w, h) * 0.03)
    align = 7 if logo else 9
    end = f"{int(total // 3600)}:{int(total % 3600 // 60):02d}:{total % 60:05.2f}"
    safe = text.replace("{", "(").replace("}", ")").replace("\n", " ")
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: wm,Arial,{fs},&H30FFFFFF,&H80000000,&H00000000,-1,2,0,{align},"
        f"{margin},{margin},{margin}\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, "
        "Text\n"
        f"Dialogue: 0,0:00:00.00,{end},wm,,0,0,0,,{safe}\n"
    )
