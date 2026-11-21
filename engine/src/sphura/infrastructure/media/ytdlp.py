"""Import videos from a YouTube link with yt-dlp.

SPHURA keeps its own copies of the official yt-dlp executable and of Deno (the JavaScript
runtime yt-dlp needs for YouTube) in <data>/tools, installed once when the user enables
YouTube import. Nothing is installed system-wide. Finished videos only: live streams that
are still running are refused.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import stat
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from sphura.domain.errors import SphuraError
from sphura.infrastructure.media.ffmpeg.process import POPEN_FLAGS

log = logging.getLogger(__name__)

YTDLP_BASE = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/"
DENO_BASE = "https://github.com/denoland/deno/releases/latest/download/"
BEST = "best"
UPDATE_EVERY_S = 24 * 3600  # YouTube changes often; keep yt-dlp fresh


@dataclass(frozen=True, slots=True)
class Step:
    """One progress report from a download. Fractions are of the whole download (0..1)."""

    text: str
    fraction: float
    speed_bps: float | None = None
    eta_s: float | None = None
    total_bytes: float | None = None  # size of the stream currently downloading
    part: str = ""  # "video" or "audio"
    throttled: bool = False


Progress = Callable[[Step], None]


class ImportError_(SphuraError):
    pass


class Stopped(Exception):
    """The download was paused or cancelled."""


@dataclass(frozen=True, slots=True)
class Quality:
    height: int
    fps: int
    size_bytes: int  # estimate incl. best audio; 0 when YouTube doesn't say

    @property
    def label(self) -> str:
        return f"{self.height}p{self.fps if self.fps > 30 else ''}"


@dataclass(frozen=True, slots=True)
class VideoInfo:
    id: str
    title: str
    duration_s: float
    uploader: str
    is_live: bool
    processing: bool  # stream ended recently and YouTube hasn't finished processing it
    qualities: tuple[Quality, ...]  # highest first
    thumbnail: str
    url: str


@dataclass(frozen=True, slots=True)
class PlaylistEntry:
    id: str
    title: str
    duration_s: float
    channel: str
    thumbnail: str
    url: str
    unavailable: str | None  # reason it can't be downloaded, or None


@dataclass(frozen=True, slots=True)
class PlaylistInfo:
    id: str
    title: str
    channel: str
    thumbnail: str
    entries: tuple[PlaylistEntry, ...]


def _qualities(formats: list[dict[str, Any]]) -> tuple[Quality, ...]:
    def size(f: dict[str, Any]) -> int:
        return int(f.get("filesize") or f.get("filesize_approx") or 0)

    audio = max((size(f) for f in formats if f.get("vcodec") == "none"), default=0)
    best: dict[int, Quality] = {}
    ranks: dict[int, tuple[bool, int, int]] = {}
    for f in formats:
        h, vcodec = f.get("height"), f.get("vcodec")
        if not isinstance(h, int) or h < 144 or vcodec in (None, "none"):
            continue
        fps = int(f.get("fps") or 30)
        q = Quality(h, fps, size(f) + (audio if f.get("acodec") == "none" else 0))
        rank = (f.get("dynamic_range") in (None, "SDR"), q.fps, q.size_bytes)
        if h not in best or rank > ranks[h]:
            best[h], ranks[h] = q, rank
    return tuple(sorted(best.values(), key=lambda q: -q.height))


def format_for(quality: str, hls: bool = False) -> str:
    """yt-dlp format selector. 'best' = highest resolution and frame rate available."""
    cap = "" if quality == BEST else f"[height<={int(quality)}]"
    if hls:  # YouTube's HLS streams come with audio already in them
        return f"b{cap}[protocol^=m3u8]/bv*{cap}[protocol^=m3u8]+ba/b[protocol^=m3u8]"
    return f"bv*{cap}+ba/b{cap}/bv*+ba/b"


# Tried in order when YouTube hands back an empty or blocked file
ATTEMPTS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("default", (), False),
    (
        "other YouTube clients",
        ("--extractor-args", "youtube:player_client=tv,web_safari,mweb"),
        False,
    ),
    ("HLS streams", ("--extractor-args", "youtube:player_client=web_safari,ios"), True),
)
RETRYABLE = re.compile(
    r"file is empty|HTTP Error 403|Forbidden|Requested format is not available|"
    r"fragment|did not get any data|PO Token",
    re.IGNORECASE,
)


def _platform_assets() -> tuple[str, str, str]:
    """(yt-dlp asset, deno zip asset, deno binary name) for this OS."""
    if sys.platform == "win32":
        return "yt-dlp.exe", "deno-x86_64-pc-windows-msvc.zip", "deno.exe"
    if sys.platform == "darwin":
        arch = "aarch64" if os.uname().machine == "arm64" else "x86_64"
        return "yt-dlp_macos", f"deno-{arch}-apple-darwin.zip", "deno"
    return "yt-dlp_linux", "deno-x86_64-unknown-linux-gnu.zip", "deno"


class YouTubeImporter:
    def __init__(self, tools_dir: Path, ffmpeg: str) -> None:
        self.tools_dir = tools_dir
        self._ffmpeg = ffmpeg
        ytdlp, _, deno = _platform_assets()
        self.ytdlp = tools_dir / ytdlp
        self.deno = tools_dir / deno

    def ready(self) -> bool:
        return self.ytdlp.is_file() and self.deno.is_file()

    def install(self, progress: Callable[[str, float], None]) -> None:
        """Download the official yt-dlp and Deno builds into the tools folder."""
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        ytdlp_asset, deno_zip, deno_name = _platform_assets()
        if not self.ytdlp.is_file():
            progress("Downloading yt-dlp (about 18 MB)", 0.0)
            _download(YTDLP_BASE + ytdlp_asset, self.ytdlp, lambda f: progress(
                "Downloading yt-dlp (about 18 MB)", 0.4 * f))  # fmt: skip
        if not self.deno.is_file():
            progress("Downloading Deno (about 45 MB)", 0.4)
            buf = io.BytesIO()
            _download(DENO_BASE + deno_zip, buf, lambda f: progress(
                "Downloading Deno (about 45 MB)", 0.4 + 0.55 * f))  # fmt: skip
            with zipfile.ZipFile(buf) as z:
                self.deno.write_bytes(z.read(deno_name))
        for exe in (self.ytdlp, self.deno):
            exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        (self.tools_dir / ".updated").write_text(str(time.time()))
        progress("YouTube import is ready", 1.0)

    def update_if_stale(self) -> None:
        stamp = self.tools_dir / ".updated"
        try:
            last = float(stamp.read_text())
        except (OSError, ValueError):
            last = 0.0
        if time.time() - last < UPDATE_EVERY_S:
            return
        try:
            subprocess.run([str(self.ytdlp), "-U"], capture_output=True, timeout=120,
                           **POPEN_FLAGS)  # fmt: skip
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("yt-dlp self-update failed: %s", e)
        stamp.write_text(str(time.time()))

    def probe(self, url: str, playlist: bool) -> VideoInfo | PlaylistInfo:
        """Details for a video, or the entry list of a playlist (fast, no per-video fetch)."""
        mode = ["--yes-playlist", "--flat-playlist"] if playlist else ["--no-playlist"]
        out = self._run([*self._common(), *mode, "-J", url], timeout=180)
        try:
            d = json.loads(out)
        except ValueError as e:
            raise ImportError_("YouTube didn't return details for that link") from e
        if d.get("_type") == "playlist":
            entries = tuple(_entry(e) for e in d.get("entries") or [] if e)
            return PlaylistInfo(
                id=str(d.get("id", "")), title=str(d.get("title") or "Playlist"),
                channel=str(d.get("channel") or d.get("uploader") or ""),
                thumbnail=_thumb(d) or next((e.thumbnail for e in entries if e.thumbnail), ""),
                entries=entries,
            )  # fmt: skip
        if playlist:
            raise ImportError_("That's a single video link. Paste it in the video box.")
        return _video(d)

    def download(
        self, url: str, quality: str, work_dir: Path, progress: Progress, stop: threading.Event
    ) -> Path:
        """Download into work_dir (kept between runs so a paused download resumes)."""
        work_dir.mkdir(parents=True, exist_ok=True)
        log_path = work_dir / "sphura-download.log"
        tail: list[str] = []
        with log_path.open("a", encoding="utf-8") as logf:
            for i, (name, extra, hls) in enumerate(ATTEMPTS):
                if i:
                    progress(Step(f"Retrying with {name}", 0.0))
                logf.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {i + 1}: {name}\n")
                final, tail = self._attempt(
                    url, quality, work_dir, extra, hls, progress, stop, logf
                )
                if stop.is_set():
                    raise Stopped
                if final is not None:
                    return final
                if not RETRYABLE.search("\n".join(tail)):
                    break
        raise ImportError_(_explain(tail))

    def _attempt(
        self, url: str, quality: str, work_dir: Path, extra: tuple[str, ...], hls: bool,
        progress: Progress, stop: threading.Event, logf: TextIO,
    ) -> tuple[Path | None, list[str]]:  # fmt: skip
        cmd = [
            *self._common(), "--no-playlist", *extra,
            "-f", format_for(quality, hls),
            "-S", "res,fps,hdr:sdr",  # SDR: HDR footage would wash out the colour grade
            "--merge-output-format", "mp4",
            "--concurrent-fragments", "8",
            "--newline", "--progress", "--color", "no_color",
            "--progress-template",
            "download:SPHURA-PROGRESS|%(progress._percent_str)s|%(progress.speed)s|"
            "%(progress.eta)s|%(progress.total_bytes)s|%(progress.total_bytes_estimate)s|"
            "%(info.vcodec)s",
            "--print", "after_move:SPHURA-FILE %(filepath)s",
            "-o", str(work_dir / "%(id)s.%(ext)s"),
            url,
        ]  # fmt: skip
        final, throttled = None, False
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", **POPEN_FLAGS,
        )  # fmt: skip
        watcher = threading.Thread(target=_kill_on, args=(stop, proc), daemon=True)
        watcher.start()
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("SPHURA-PROGRESS"):
                progress(_parse_progress(line, throttled))
                continue
            logf.write(line + "\n")
            tail = [*tail, line][-15:]
            if "WARNING" in line and re.search(r"challenge|n function|JS runtime|throttl",
                                               line, re.IGNORECASE):  # fmt: skip
                log.warning("yt-dlp: %s", line)
                throttled = True
            elif line.startswith("SPHURA-FILE"):
                final = Path(line.split(" ", 1)[1].strip())
            elif "[Merger]" in line:
                progress(Step("Joining video and audio", 0.97))
        proc.wait()
        ok = (
            proc.returncode == 0 and final is not None and final.is_file()
            and final.stat().st_size > 0
        )  # fmt: skip
        return (final if ok else None), tail

    def _common(self) -> list[str]:
        # Warnings stay on: a failed YouTube challenge shows up there and means throttling
        return [
            str(self.ytdlp),
            "--js-runtimes", f"deno:{self.deno}",
            "--ffmpeg-location", self._ffmpeg,
        ]  # fmt: skip

    def _run(self, cmd: list[str], timeout: int) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout, **POPEN_FLAGS)  # fmt: skip
        except subprocess.TimeoutExpired as e:
            raise ImportError_("YouTube took too long to respond. Try again.") from e
        if r.returncode != 0 and not r.stdout.strip().startswith("{"):
            raise ImportError_(_explain((r.stderr or r.stdout).splitlines()[-15:]))
        return str(r.stdout)


def _parse_progress(line: str, throttled: bool) -> Step:
    _, pct_s, speed, eta, total, estimate, vcodec = (f.strip() for f in line.split("|"))
    m = re.search(r"([\d.]+)%", pct_s)
    pct = min(float(m.group(1)) / 100, 1.0) if m else 0.0
    audio = vcodec == "none"
    return Step(
        "Downloading audio" if audio else "Downloading video",
        0.85 + 0.1 * pct if audio else 0.85 * pct,
        speed_bps=_num(speed), eta_s=_num(eta),
        total_bytes=_num(total) or _num(estimate), part="audio" if audio else "video",
        throttled=throttled,
    )  # fmt: skip


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None  # "NA" while yt-dlp doesn't know yet


def _kill_on(stop: threading.Event, proc: subprocess.Popen[str]) -> None:
    while proc.poll() is None:
        if stop.wait(0.5):
            kill_tree(proc)
            return


def kill_tree(proc: subprocess.Popen[str]) -> None:
    """Stop yt-dlp and the ffmpeg it may be running (it would otherwise keep the file open)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                       **POPEN_FLAGS)  # fmt: skip
    else:
        proc.kill()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp (pid %s) did not exit", proc.pid)


def _thumb(d: dict[str, Any]) -> str:
    if t := d.get("thumbnail"):
        return str(t)
    thumbs = [t for t in d.get("thumbnails") or [] if t.get("url")]
    best = max(thumbs, key=lambda t: t.get("width") or 0, default=None)
    return str(best["url"]) if best else ""


def _video(d: dict[str, Any]) -> VideoInfo:
    return VideoInfo(
        id=str(d.get("id", "")),
        title=str(d.get("title") or "YouTube video"),
        duration_s=float(d.get("duration") or 0),
        uploader=str(d.get("channel") or d.get("uploader") or ""),
        is_live=bool(d.get("is_live")) or d.get("live_status") in ("is_live", "is_upcoming"),
        processing=d.get("live_status") == "post_live",
        qualities=_qualities(list(d.get("formats") or [])),
        thumbnail=_thumb(d),
        url=str(d.get("webpage_url") or f"https://www.youtube.com/watch?v={d.get('id', '')}"),
    )


_UNAVAILABLE = {
    "private": "Private video",
    "needs_auth": "Needs sign-in (age-restricted or private)",
    "subscriber_only": "Channel members only",
    "premium_only": "YouTube Premium only",
}


def _entry(e: dict[str, Any]) -> PlaylistEntry:
    vid = str(e.get("id") or "")
    title = str(e.get("title") or "")
    reason = _UNAVAILABLE.get(str(e.get("availability") or ""))
    if title in ("[Private video]", "[Deleted video]"):
        reason = title.strip("[]")
    elif e.get("live_status") in ("is_live", "is_upcoming"):
        reason = "Live or upcoming stream"
    elif e.get("live_status") == "post_live":
        reason = "YouTube is still processing this stream"
    return PlaylistEntry(
        id=vid, title=title or vid, duration_s=float(e.get("duration") or 0),
        channel=str(e.get("channel") or e.get("uploader") or ""),
        thumbnail=_thumb(e) or (f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else ""),
        url=f"https://www.youtube.com/watch?v={vid}", unavailable=reason,
    )  # fmt: skip


def _explain(lines: list[str]) -> str:
    text = "\n".join(lines)
    for pattern, message in (
        (r"Private video", "That video is private."),
        (r"Sign in to confirm your age|age-restricted", "That video is age-restricted."),
        (r"members-only|Join this channel", "That video is for channel members only."),
        (r"Video unavailable|not available", "That video isn't available."),
        (r"Sign in to confirm you.re not a bot", "YouTube asked for a bot check. Try again later."),
        (
            r"Unable to connect|getaddrinfo|timed out|Network is unreachable|"
            r"Unable to download API",
            "Couldn't reach YouTube. Check your internet connection and try again.",
        ),
        (r"Unsupported URL|is not a valid URL", "That doesn't look like a YouTube video link."),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            return message
    errors = [ln for ln in lines if "ERROR" in ln]
    return (errors[-1] if errors else (lines[-1] if lines else "download failed"))[:300]


def _download(url: str, dest: Path | io.BytesIO, on_fraction: Callable[[float], None]) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "SPHURA"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            data = bytearray()
            while chunk := r.read(1 << 16):
                data += chunk
                if total:
                    on_fraction(len(data) / total)
    except OSError as e:
        raise ImportError_(f"Couldn't download {url.rsplit('/', 1)[-1]}: {e}") from e
    if isinstance(dest, io.BytesIO):
        dest.write(bytes(data))
        dest.seek(0)
    else:
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(bytes(data))
        tmp.replace(dest)


def fetch_file(url: str, dest: Path) -> None:
    """Small HTTP download (thumbnails); written atomically."""
    _download(url, dest, lambda _f: None)
