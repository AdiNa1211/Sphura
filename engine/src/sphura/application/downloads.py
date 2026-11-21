"""Download manager: YouTube videos and playlists into a local library.

Kept apart from the editing engine. A finished download is a plain local file; "Edit Video"
hands it to the normal ingest + analyse flow, which references the file and never changes it.
Each download works in <folder>/.sphura-partial/<video id> so pause, crashes and restarts
resume from the partial files, and cancel removes them.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sphura.application.analyze import AnalysisService
from sphura.application.ingest import IngestService
from sphura.application.queries import QueryService
from sphura.domain.errors import NotFoundError, SphuraError
from sphura.domain.models import JobStatus, JobType, utcnow
from sphura.infrastructure.media.ytdlp import (
    BEST,
    PlaylistInfo,
    Step,
    Stopped,
    YouTubeImporter,
    fetch_file,
)
from sphura.infrastructure.persistence.sqlite.downloads import (
    ACTIVE,
    Download,
    SqliteDownloadRepository,
)
from sphura.ports.media import MediaProbe

log = logging.getLogger(__name__)

PARTIAL = ".sphura-partial"
MAX_HOURS = 12
MIN_FREE_BYTES = 500 * 1024**2
FILENAME_FORMATS = {
    "title_id": "{title} [{id}]",
    "title": "{title}",
    "channel_title": "{channel} - {title}",
    "date_title": "{date} {title}",
}
QUALITIES = (BEST, "2160", "1440", "1080", "720", "480")


@dataclass(slots=True)
class DownloadSettings:
    folder: str
    quality: str = BEST
    filename: str = "title_id"
    save_thumbnails: bool = True
    parallel: int = 2


def default_folder() -> Path:
    return Path.home() / "Videos" / "SPHURA" / "Downloads"


class DownloadManager:
    def __init__(
        self,
        repo: SqliteDownloadRepository,
        yt: YouTubeImporter,
        probe: MediaProbe,
        ingest: IngestService,
        analysis: AnalysisService,
        queries: QueryService,
        data_dir: Path,
        sample: Callable[[str, list[float], int], list[bytes]],
    ) -> None:
        self._repo = repo
        self._yt = yt
        self._probe = probe
        self._ingest = ingest
        self._analysis = analysis
        self._queries = queries
        self._sample = sample
        self._settings_path = data_dir / "downloads.json"
        self._thumbs = data_dir / "thumbnails"
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._live: dict[str, dict[str, Any]] = {}  # running downloads: stop event, speed, …
        self._threads: dict[str, threading.Thread] = {}
        self._scheduler: threading.Thread | None = None
        self.setup: dict[str, Any] = {"running": False, "step": "", "progress": 0.0, "error": None}

    # ---------- lifecycle ----------

    def start(self) -> None:
        n = self._repo.interrupted_to_queued()
        if n:
            log.info("resuming %d interrupted download(s)", n)
        self._clean_orphans()
        self._closing.clear()
        self._scheduler = threading.Thread(target=self._schedule, daemon=True, name="downloads")
        self._scheduler.start()

    def stop(self) -> None:
        """Shutting down: running downloads stop and go back to the queue for next time."""
        self._closing.set()
        with self._lock:
            for live in self._live.values():
                live["mode"] = "shutdown"
                live["stop"].set()
        self._wake.set()
        for t in list(self._threads.values()):
            t.join(15)
        if self._scheduler:
            self._scheduler.join(5)

    # ---------- tools ----------

    def tools_ready(self) -> bool:
        return self._yt.ready()

    def install_tools(self) -> None:
        with self._lock:
            if self.setup["running"] or self._yt.ready():
                return
            self.setup.update(running=True, error=None, step="Starting", progress=0.0)

        def work() -> None:
            try:
                self._yt.install(lambda step, f: self.setup.update(step=step, progress=f))
            except Exception as e:
                self.setup["error"] = str(e)
            finally:
                self.setup["running"] = False
                self._wake.set()

        threading.Thread(target=work, daemon=True, name="yt-setup").start()

    # ---------- settings & storage ----------

    def settings(self) -> DownloadSettings:
        try:
            d = json.loads(self._settings_path.read_text(encoding="utf-8"))
            s = DownloadSettings(**{k: v for k, v in d.items() if k in DownloadSettings.__slots__})
        except (OSError, ValueError, TypeError):
            s = DownloadSettings(folder=str(default_folder()))
        return s

    def save_settings(self, changes: dict[str, Any]) -> DownloadSettings:
        s = self.settings()
        if "folder" in changes:
            folder = Path(str(changes["folder"]).strip().strip('"')).expanduser()
            if not folder.is_absolute():
                raise SphuraError("Enter a full folder path, like D:\\Videos\\Downloads")
            try:
                folder.mkdir(parents=True, exist_ok=True)
                probe = folder / ".sphura-write-test"
                probe.write_text("ok")
                probe.unlink()
            except OSError as e:
                raise SphuraError(f"Can't save downloads in {folder}: {e.strerror or e}") from e
            s.folder = str(folder)
        if "quality" in changes:
            if str(changes["quality"]) not in QUALITIES:
                raise SphuraError(f"unknown quality: {changes['quality']}")
            s.quality = str(changes["quality"])
        if "filename" in changes:
            if changes["filename"] not in FILENAME_FORMATS:
                raise SphuraError(f"unknown filename format: {changes['filename']}")
            s.filename = str(changes["filename"])
        if "save_thumbnails" in changes:
            s.save_thumbnails = bool(changes["save_thumbnails"])
        if "parallel" in changes:
            s.parallel = max(1, min(4, int(changes["parallel"])))
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
        self._wake.set()
        return s

    def storage(self) -> dict[str, Any]:
        folder = Path(self.settings().folder)
        used = sum(d.size_bytes or 0 for d in self._repo.all() if d.status == "completed")
        info: dict[str, Any] = {"folder": str(folder), "used_bytes": used}
        try:
            folder.mkdir(parents=True, exist_ok=True)
            du = shutil.disk_usage(folder)
            info.update(free_bytes=du.free, total_bytes=du.total)
        except OSError:
            info.update(free_bytes=None, total_bytes=None)
        return info

    # ---------- import ----------

    def preview(self, url: str, playlist: bool) -> dict[str, Any]:
        self._need_tools()
        self._yt.update_if_stale()
        info = self._yt.probe(url, playlist)
        if isinstance(info, PlaylistInfo):
            entries = [asdict(e) | {"duplicate": self._duplicate(e.id)} for e in info.entries]
            return {
                "kind": "playlist", "title": info.title, "channel": info.channel,
                "thumbnail": info.thumbnail, "count": len(entries), "entries": entries,
            }  # fmt: skip
        problem = None
        if info.is_live:
            problem = "This stream is still live. Download it after it ends."
        elif info.processing:
            problem = ("YouTube is still processing this stream, so it can't be downloaded yet. "
                       "This can take a few hours for long streams.")  # fmt: skip
        elif info.duration_s > MAX_HOURS * 3600:
            problem = f"Videos longer than {MAX_HOURS} hours aren't supported."
        elif not info.qualities:
            problem = "YouTube didn't offer any video quality for this link."
        return {
            "kind": "video", "id": info.id, "title": info.title, "channel": info.uploader,
            "duration_s": info.duration_s, "thumbnail": info.thumbnail, "url": info.url,
            "problem": problem, "duplicate": self._duplicate(info.id),
            "qualities": [{"value": str(q.height), "label": q.label, "size_bytes": q.size_bytes}
                          for q in info.qualities],
        }  # fmt: skip

    def add(
        self, items: list[dict[str, Any]], quality: str | None, playlist_title: str | None
    ) -> dict[str, Any]:
        self._need_tools()
        quality = quality or self.settings().quality
        if quality not in QUALITIES and not quality.isdigit():
            raise SphuraError(f"unknown quality: {quality}")
        added, skipped = [], []
        with self._lock:
            for it in items:
                vid = str(it.get("id") or "")
                title = str(it.get("title") or vid)
                if not re.fullmatch(r"[\w-]{6,20}", vid):
                    skipped.append({"title": title, "reason": "Not a YouTube video"})
                    continue
                if dup := self._duplicate(vid):
                    skipped.append({"title": title, "reason": dup})
                    continue
                stale = [d for d in self._repo.by_video(vid) if d.status in ("failed", "cancelled")]
                if stale:
                    self._repo.update(stale[0], status="queued", error=None, quality=quality)
                    for extra in stale[1:]:
                        self._repo.delete(extra.id)
                    added.append(stale[0].id)
                    continue
                d = Download(
                    video_id=vid, url=f"https://www.youtube.com/watch?v={vid}", title=title,
                    quality=quality, channel=str(it.get("channel") or ""),
                    duration_s=float(it.get("duration_s") or 0),
                    thumbnail_url=str(it.get("thumbnail") or ""), playlist_title=playlist_title,
                )  # fmt: skip
                self._repo.add(d)
                added.append(d.id)
        self._wake.set()
        return {"added": len(added), "skipped": skipped}

    def _duplicate(self, video_id: str) -> str | None:
        for d in self._repo.by_video(video_id):
            if d.status in ACTIVE:
                return "Already in the download queue"
            if d.status == "completed":
                if d.file_path and Path(d.file_path).is_file():
                    return "Already downloaded"
                self._repo.delete(d.id)  # file was removed outside SPHURA; allow again
        folder = Path(self.settings().folder)
        if folder.is_dir() and any(folder.glob(f"*[[]{video_id}[]]*")):
            return "Already in your downloads folder"
        return None

    # ---------- queue controls ----------

    def pause(self, download_id: str) -> None:
        d = self._get(download_id)
        if d.status == "queued":
            self._repo.update(d, status="paused")
        elif d.status == "downloading":
            self._signal(d.id, "pause")

    def resume(self, download_id: str) -> None:
        d = self._get(download_id)
        if d.status == "paused":
            self._repo.update(d, status="queued")
            self._wake.set()

    def cancel(self, download_id: str) -> None:
        d = self._get(download_id)
        if d.status == "downloading":
            self._signal(d.id, "cancel")
        elif d.status in ("queued", "paused", "failed"):
            self._discard_partial(d)
            self._repo.update(d, status="cancelled", progress=0.0)

    def retry(self, download_id: str) -> None:
        d = self._get(download_id)
        if d.status in ("failed", "cancelled"):
            self._repo.update(d, status="queued", error=None)
            self._wake.set()

    def remove(self, download_id: str) -> None:
        """Remove from the queue (not for finished downloads: those are deleted as files)."""
        d = self._get(download_id)
        if d.status == "completed":
            raise SphuraError("Finished downloads are removed with Delete in Downloaded Videos")
        if d.status == "downloading":
            self._signal(d.id, "remove")
            return
        self._discard_partial(d)
        self._repo.delete(d.id)

    def delete_file(self, download_id: str) -> None:
        """Delete a finished download. Shorts made from it live elsewhere and are kept."""
        d = self._get(download_id)
        if d.status != "completed":
            raise SphuraError("Only finished downloads can be deleted here")
        for p in (d.file_path, d.thumbnail_path):
            if p:
                Path(p).unlink(missing_ok=True)
        self._repo.delete(d.id)

    def _signal(self, download_id: str, mode: str) -> None:
        with self._lock:
            live = self._live.get(download_id)
            if live:
                live["mode"] = mode
                live["stop"].set()

    def _get(self, download_id: str) -> Download:
        d = self._repo.get(download_id)
        if d is None:
            raise NotFoundError("download not found")
        return d

    # ---------- editing hand-off ----------

    def edit(self, download_id: str) -> str:
        """Open the download as a SPHURA project (analysis starts if it hasn't run)."""
        d = self._get(download_id)
        if d.status != "completed" or not d.file_path or not Path(d.file_path).is_file():
            raise SphuraError("The downloaded file is missing. Download it again.")
        r = self._ingest.ingest_local_file(d.file_path, project_name=d.title)
        jobs = [j for j in self._queries.jobs(r.project.id) if j.type == JobType.ANALYZE]
        if not any(j.status in (JobStatus.SUCCEEDED, JobStatus.QUEUED, JobStatus.RUNNING)
                   for j in jobs):  # fmt: skip
            self._analysis.start(r.project.id)
        if d.project_id != r.project.id:
            self._repo.update(d, project_id=r.project.id)
        return r.project.id

    def editing_status(self, d: Download) -> str:
        if not d.project_id:
            return "not_edited"
        try:
            jobs = [j for j in self._queries.jobs(d.project_id) if j.type == JobType.ANALYZE]
            clips = self._queries.clips(d.project_id)
        except NotFoundError:
            return "not_edited"
        if any(out for _, out in clips):
            return "shorts"
        latest = max(jobs, key=lambda j: j.created_at, default=None)
        if latest is None:
            return "not_edited"
        if latest.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            return "analysing"
        return "analysed" if latest.status == JobStatus.SUCCEEDED else "analysis_failed"

    # ---------- listing ----------

    def items(self) -> list[dict[str, Any]]:
        out = []
        for d in self._repo.all():
            item = d.to_dict()
            with self._lock:
                live = self._live.get(d.id)
                if live:
                    item.update({k: v for k, v in live.items() if k not in ("stop", "mode")})
            if d.status == "completed":
                item["file_exists"] = bool(d.file_path and Path(d.file_path).is_file())
                item["editing"] = self.editing_status(d)
            out.append(item)
        return out

    def thumbnail(self, download_id: str) -> tuple[Path | None, str]:
        d = self._get(download_id)
        p = Path(d.thumbnail_path) if d.thumbnail_path else None
        return (p if p and p.is_file() else None), d.thumbnail_url

    def file_of(self, download_id: str) -> Path:
        d = self._get(download_id)
        if not d.file_path or not Path(d.file_path).is_file():
            raise NotFoundError("the downloaded file is missing")
        return Path(d.file_path)

    # ---------- workers ----------

    def _schedule(self) -> None:
        while not self._closing.is_set():
            self._wake.wait(2)
            self._wake.clear()
            if self._closing.is_set() or not self._yt.ready():
                continue
            with self._lock:
                slots = self.settings().parallel - len(self._live)
                for d in [d for d in self._repo.all() if d.status == "queued"][: max(slots, 0)]:
                    self._live[d.id] = {"stop": threading.Event(), "mode": None,
                                        "step": "Starting", "speed_bps": None, "eta_s": None,
                                        "total_bytes": None, "throttled": False}  # fmt: skip
                    t = threading.Thread(target=self._run, args=(d,), daemon=True,
                                         name=f"download-{d.video_id}")  # fmt: skip
                    self._threads[d.id] = t
                    self._repo.update(d, status="downloading", error=None)
                    t.start()

    def _run(self, d: Download) -> None:
        live = self._live[d.id]
        settings = self.settings()
        folder = Path(settings.folder)
        work = folder / PARTIAL / d.video_id
        last_save = [0.0]

        def on_progress(step: Step) -> None:
            live.update(step=step.text, speed_bps=step.speed_bps, eta_s=step.eta_s,
                        total_bytes=step.total_bytes, throttled=step.throttled)  # fmt: skip
            live["progress"] = step.fraction
            if step.total_bytes and step.fraction < 0.5 and not live.get("space_checked"):
                live["space_checked"] = True
                free = shutil.disk_usage(folder).free
                if step.total_bytes > free - MIN_FREE_BYTES:
                    live["mode"] = "nospace"
                    live["needed"] = step.total_bytes
                    live["free"] = free
                    live["stop"].set()
            if time.monotonic() - last_save[0] > 3:
                last_save[0] = time.monotonic()
                self._repo.update(d, progress=step.fraction)

        try:
            folder.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(folder).free < MIN_FREE_BYTES:
                raise SphuraError(f"Not enough disk space in {folder}. Free up space and retry.")
            if settings.save_thumbnails and d.thumbnail_url and not d.thumbnail_path:
                self._save_thumbnail(d)
            path = self._yt.download(d.url, d.quality, work, on_progress, live["stop"])
            live["step"] = "Finishing"
            final = self._place(path, d, folder, settings.filename)
            self._finish(d, final)
            shutil.rmtree(work, ignore_errors=True)
        except Stopped:
            self._stopped(d, live, work)
        except SphuraError as e:
            self._repo.update(d, status="failed", error=str(e))
        except Exception as e:
            log.exception("download %s failed", d.video_id)
            self._repo.update(d, status="failed", error=f"Unexpected error: {e}")
        finally:
            with self._lock:
                self._live.pop(d.id, None)
                self._threads.pop(d.id, None)
            self._wake.set()

    def _stopped(self, d: Download, live: dict[str, Any], work: Path) -> None:
        mode = live.get("mode")
        if mode == "pause":
            self._repo.update(d, status="paused", progress=live.get("progress", d.progress))
        elif mode == "shutdown":
            self._repo.update(d, status="queued", progress=live.get("progress", d.progress))
        elif mode == "nospace":
            self._repo.update(d, status="failed", error=(
                f"Not enough disk space: this video needs about {_gb(live['needed'])}, "
                f"but only {_gb(live['free'])} is free in {work.parent.parent}. "
                "Free up space or change the download folder, then retry."))  # fmt: skip
        else:
            shutil.rmtree(work, ignore_errors=True)
            if mode == "remove":
                self._repo.delete(d.id)
            else:
                self._repo.update(d, status="cancelled", progress=0.0)

    def _place(self, path: Path, d: Download, folder: Path, fmt: str) -> Path:
        name = FILENAME_FORMATS.get(fmt, FILENAME_FORMATS["title_id"]).format(
            title=d.title,
            id=d.video_id,
            channel=d.channel or "YouTube",
            date=f"{datetime.now():%Y-%m-%d}",
        )
        stem = _safe_name(name)
        dest = folder / f"{stem}{path.suffix}"
        n = 2
        while dest.exists():
            dest = folder / f"{stem} ({n}){path.suffix}"
            n += 1
        shutil.move(str(path), dest)
        return dest

    def _finish(self, d: Download, final: Path) -> None:
        width = height = None
        fps = None
        duration = d.duration_s
        try:
            info = self._probe.probe(str(final))
            duration = duration or info.duration_s
            if info.video:
                v = info.video[0]
                width, height, fps = v.width, v.height, round(v.fps, 2)
        except SphuraError as e:
            log.warning("couldn't read %s: %s", final, e)
        thumb = d.thumbnail_path
        if not thumb:  # no YouTube thumbnail saved: use a frame from the video
            try:
                dest = self._thumbs / f"{d.video_id}.jpg"
                self._thumbs.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(self._sample(str(final), [duration * 0.1], 480)[0])
                thumb = str(dest)
            except (SphuraError, OSError, IndexError) as e:
                log.info("no thumbnail for %s: %s", d.video_id, e)
        self._repo.update(
            d, status="completed", progress=1.0, error=None, file_path=str(final),
            size_bytes=final.stat().st_size, width=width, height=height, fps=fps,
            duration_s=duration, thumbnail_path=thumb, completed_at=utcnow().isoformat(),
        )  # fmt: skip

    def _save_thumbnail(self, d: Download) -> None:
        dest = self._thumbs / f"{d.video_id}.jpg"
        try:
            self._thumbs.mkdir(parents=True, exist_ok=True)
            if not dest.is_file():
                fetch_file(d.thumbnail_url, dest)
            self._repo.update(d, thumbnail_path=str(dest))
        except SphuraError as e:
            log.info("thumbnail for %s not saved: %s", d.video_id, e)

    def _discard_partial(self, d: Download) -> None:
        shutil.rmtree(Path(self.settings().folder) / PARTIAL / d.video_id, ignore_errors=True)

    def _clean_orphans(self) -> None:
        """Remove partial files whose download no longer exists (temporary-file cleanup)."""
        root = Path(self.settings().folder) / PARTIAL
        if not root.is_dir():
            return
        keep = {d.video_id for d in self._repo.all() if d.status in (*ACTIVE, "failed")}
        for p in root.iterdir():
            if p.name not in keep:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    def _need_tools(self) -> None:
        if not self._yt.ready():
            raise SphuraError("Turn on YouTube downloads first")


def _safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(".")
    return (name or "video")[:150]


def _gb(n: float) -> str:
    return f"{n / 1024**3:.1f} GB"
