from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sphura.domain.errors import (
    InvalidMediaError,
    NotFoundError,
    SphuraError,
    UnsupportedContainerError,
)
from sphura.domain.models import (
    EditingMode,
    Job,
    JobType,
    MediaAsset,
    Project,
    StorageKind,
)
from sphura.ports.media import SNIFF_HEADER_BYTES, ContainerSniffer, MediaProbe
from sphura.ports.queue import JobContext, JobQueue
from sphura.ports.repositories import MediaAssetRepository, ProjectRepository
from sphura.ports.storage import SourceReader, StorageProvider

log = logging.getLogger(__name__)

_FINGERPRINT_SAMPLE = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IngestResult:
    project: Project
    asset: MediaAsset
    checksum_job: Job | None
    already_registered: bool


def fast_fingerprint(reader: SourceReader, uri: str, size: int) -> str:
    """Constant-time identity for multi-GB files: size + head + tail samples."""
    h = hashlib.sha256(str(size).encode())
    h.update(reader.read_range(uri, 0, _FINGERPRINT_SAMPLE))
    if size > _FINGERPRINT_SAMPLE:
        h.update(reader.read_range(uri, max(size - _FINGERPRINT_SAMPLE, 0), _FINGERPRINT_SAMPLE))
    return "v1:" + h.hexdigest()[:40]


class IngestService:
    """Registers a source video. The original is referenced, never copied or modified."""

    def __init__(
        self,
        projects: ProjectRepository,
        assets: MediaAssetRepository,
        reader: SourceReader,
        storage: StorageProvider,
        sniffer: ContainerSniffer,
        probe: MediaProbe,
        queue: JobQueue,
    ) -> None:
        self._projects = projects
        self._assets = assets
        self._reader = reader
        self._storage = storage
        self._sniffer = sniffer
        self._probe = probe
        self._queue = queue

    def move_source(self, project_id: str, old: str | None, new: str) -> int:
        """The user moved a source file: point the project's media at its new path.
        Fingerprints are size + head/tail, so a moved (not edited) file still matches."""
        n = 0
        for a in self._assets.list_for_project(project_id):
            if a.source_uri == old or not Path(a.source_uri).is_file():
                a.source_uri = new
                self._assets.update(a)
                n += 1
        return n

    def relink(self, asset_id: str, path: str) -> MediaAsset:
        """A moved recording: point its media at the new path, only if it's the same file
        (size + head/tail fingerprint), so a different video can never be linked by mistake."""
        asset = self._assets.get(asset_id)
        if asset is None:
            raise NotFoundError("this recording isn't in SPHURA any more")
        p = Path(path.strip().strip('"')).expanduser()
        uri = str(p.resolve()) if p.exists() else str(p)
        if not self._reader.exists(uri):
            raise InvalidMediaError(f"file not found: {uri}")
        if fast_fingerprint(self._reader, uri, self._reader.size(uri)) != asset.fingerprint:
            raise SphuraError(
                f"{p.name} isn't the same recording (its size or content differs). "
                "Pick the file this Short was made from."
            )
        asset.source_uri = uri
        self._assets.update(asset)
        return asset

    def ingest_local_file(
        self,
        path: str | Path,
        project_id: str | None = None,
        project_name: str | None = None,
        mode: EditingMode = EditingMode.AUTO,
        reprobe: bool = False,
    ) -> IngestResult:
        p = Path(path).expanduser().resolve()
        uri = str(p)
        if not self._reader.exists(uri):
            raise InvalidMediaError(f"file not found: {p}")
        size = self._reader.size(uri)
        if size == 0:
            raise InvalidMediaError(f"file is empty: {p}")

        container = self._sniffer.sniff(self._reader.read_range(uri, 0, SNIFF_HEADER_BYTES))
        if container is None:
            raise UnsupportedContainerError(f"{p.name}: not a recognised MP4/MOV/MKV/WebM/AVI file")
        if p.suffix.lower().lstrip(".") not in (container.value, "m4v"):
            log.info("%s: extension does not match detected container %s", p.name, container)

        fingerprint = fast_fingerprint(self._reader, uri, size)

        # Same file again: reuse its project unless a specific project was requested
        for existing in self._assets.find_by_fingerprint(fingerprint):
            if project_id is None or existing.project_id == project_id:
                changed = existing.source_uri != uri  # moved on disk
                existing.source_uri = uri
                if existing.media_info is None or reprobe:
                    existing.media_info = self._probe.probe(uri)
                    changed = True
                if changed:
                    self._assets.update(existing)
                owner = self._resolve_project(existing.project_id, p.stem, mode)
                return IngestResult(owner, existing, None, already_registered=True)

        # Probe before creating anything so broken files are rejected up front
        media_info = self._probe.probe(uri)
        project = self._resolve_project(project_id, project_name or p.stem, mode)

        asset = MediaAsset(
            project_id=project.id,
            source_uri=uri,
            storage_kind=StorageKind.LOCAL_PATH,
            container=container,
            size_bytes=size,
            fingerprint=fingerprint,
            original_name=p.name,
            media_info=media_info,
        )
        self._assets.add(asset)
        self._storage.project_dir(project.id)

        job = self._queue.enqueue(
            Job(type=JobType.CHECKSUM, project_id=project.id, payload={"asset_id": asset.id})
        )
        return IngestResult(project, asset, job, already_registered=False)

    def _resolve_project(self, project_id: str | None, name: str, mode: EditingMode) -> Project:
        if project_id:
            project = self._projects.get(project_id)
            if project is None:
                raise NotFoundError(f"project {project_id} not found")
            return project
        project = Project(name=name, mode=mode)
        self._projects.add(project)
        return project


class ChecksumHandler:
    """Background full SHA-256 (streamed, constant memory). Used for dedup and upload integrity."""

    def __init__(self, assets: MediaAssetRepository, reader: SourceReader) -> None:
        self._assets = assets
        self._reader = reader

    def __call__(self, ctx: JobContext) -> dict[str, Any]:
        asset_id = ctx.job.payload["asset_id"]
        asset = self._assets.get(asset_id)
        if asset is None:
            raise NotFoundError(f"asset {asset_id} not found")
        uri = asset.source_uri
        if not self._reader.exists(uri):
            raise InvalidMediaError(f"source file missing: {uri}")
        if fast_fingerprint(self._reader, uri, self._reader.size(uri)) != asset.fingerprint:
            raise InvalidMediaError("source file changed since it was added")

        h = hashlib.sha256()
        done = 0
        ctx.report("checksum", 0.0)
        for chunk in self._reader.iter_chunks(uri):
            h.update(chunk)
            done += len(chunk)
            ctx.report("checksum", done / asset.size_bytes)

        asset.sha256 = h.hexdigest()
        self._assets.update(asset)
        return {"asset_id": asset.id, "sha256": asset.sha256, "bytes": done}
