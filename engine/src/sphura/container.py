"""Composition root: the only place that knows which adapters implement which ports."""

from __future__ import annotations

from dataclasses import dataclass

from sphura.application.analyze import AnalysisService, AnalyzeHandler
from sphura.application.clips import ClipService, RenderHandler
from sphura.application.downloads import DownloadManager
from sphura.application.feedback import FeedbackService
from sphura.application.ingest import ChecksumHandler, IngestService
from sphura.application.queries import QueryService
from sphura.application.reels import ReelService
from sphura.application.research import ResearchService
from sphura.config import Settings, default_data_dir
from sphura.domain.models import JobType
from sphura.infrastructure.ai.accel import WhisperAccelChecker
from sphura.infrastructure.ai.acestep import AceStep
from sphura.infrastructure.ai.acestep import default_root as default_acestep_root
from sphura.infrastructure.ai.anthropic import AnthropicProvider
from sphura.infrastructure.ai.ollama import OllamaProvider
from sphura.infrastructure.ai.people import YoloxPeopleDetector
from sphura.infrastructure.ai.whisper import FasterWhisperProvider
from sphura.infrastructure.events.bus import InProcessEventBus
from sphura.infrastructure.media.ffmpeg.audio import FFmpegAudioExtractor
from sphura.infrastructure.media.ffmpeg.compile import FFmpegReelCompiler
from sphura.infrastructure.media.ffmpeg.frames import FFmpegFrameDecoder
from sphura.infrastructure.media.ffmpeg.music import MusicLibrary
from sphura.infrastructure.media.ffmpeg.probe import FfprobeMediaProbe
from sphura.infrastructure.media.ffmpeg.render import FFmpegRenderer
from sphura.infrastructure.media.ffmpeg.stills import FFmpegFrameSampler
from sphura.infrastructure.media.ffmpeg.toolchain import FFmpegToolchain
from sphura.infrastructure.media.sniff import MagicBytesSniffer
from sphura.infrastructure.media.ytdlp import YouTubeImporter
from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase
from sphura.infrastructure.persistence.sqlite.downloads import SqliteDownloadRepository
from sphura.infrastructure.persistence.sqlite.reels import SqliteReelRepository
from sphura.infrastructure.persistence.sqlite.repositories import (
    SqliteCandidateRepository,
    SqliteClipRepository,
    SqliteFeedbackRepository,
    SqliteJobRepository,
    SqliteMediaAssetRepository,
    SqliteProjectRepository,
)
from sphura.infrastructure.persistence.sqlite.research import SqliteResearchRepository
from sphura.infrastructure.queue.local_queue import LocalJobQueue
from sphura.infrastructure.storage.local import LocalSourceReader, LocalWorkspaceStorage
from sphura.migrate import migrate_legacy
from sphura.pipeline.compare import CompareStage
from sphura.pipeline.judge import JudgeStage
from sphura.pipeline.signals import SignalsStage
from sphura.pipeline.stages import ExtractAudioStage, TranscribeStage
from sphura.ports.ai import LLMProvider


@dataclass(slots=True)
class LocalEngine:
    settings: Settings
    events: InProcessEventBus
    queue: LocalJobQueue
    toolchain: FFmpegToolchain
    probe: FfprobeMediaProbe
    whisper: FasterWhisperProvider
    accel: WhisperAccelChecker
    llm: LLMProvider | None
    ingest: IngestService
    analysis: AnalysisService
    clips: ClipService
    feedback: FeedbackService
    queries: QueryService
    sampler: FFmpegFrameSampler
    storage: LocalWorkspaceStorage
    music: MusicLibrary
    downloads: DownloadManager
    ai_music: AceStep
    reels: ReelService
    research: ResearchService

    def start(self) -> None:
        self.queue.start()
        self.downloads.start()
        self.reels.start()

    def stop(self) -> None:
        self.reels.stop()
        self.downloads.stop()
        self.ai_music.stop()
        self.queue.stop()


def build_local_engine(settings: Settings) -> LocalEngine:
    settings = migrate_legacy(settings, default_data_dir())
    db = SqliteDatabase(settings.db_path)
    projects = SqliteProjectRepository(db)
    assets = SqliteMediaAssetRepository(db)
    jobs = SqliteJobRepository(db)
    reader = LocalSourceReader()
    storage = LocalWorkspaceStorage(settings.workspace_dir)
    events = InProcessEventBus()
    toolchain = FFmpegToolchain(settings.ffmpeg_dir, cache_path=settings.data_dir / "hardware.json")
    probe = FfprobeMediaProbe(toolchain)

    queue = LocalJobQueue(jobs, events, workers=settings.workers)
    queue.register(JobType.CHECKSUM, ChecksumHandler(assets, reader))

    candidates = SqliteCandidateRepository(db)
    llm = build_llm(settings)
    sampler = FFmpegFrameSampler(toolchain)
    judge = JudgeStage(llm, sampler, storage)
    accel = WhisperAccelChecker(
        settings.models_dir / "whisper", settings.data_dir / "whisper_accel.json"
    )
    whisper = FasterWhisperProvider(
        settings.whisper_model, settings.models_dir / "whisper", settings.whisper_device, accel
    )
    queue.register(
        JobType.ANALYZE,
        AnalyzeHandler(
            assets,
            storage,
            ExtractAudioStage(FFmpegAudioExtractor(toolchain), storage),
            TranscribeStage(whisper, storage),
            SignalsStage(FFmpegFrameDecoder(toolchain), storage),
            judge,
            CompareStage(llm, sampler, storage),
            candidates,
        ),
    )

    clip_repo = SqliteClipRepository(db)
    people = YoloxPeopleDetector(settings.models_dir / "people")
    queue.register(
        JobType.RENDER,
        RenderHandler(
            clip_repo,
            assets,
            storage,
            FFmpegRenderer(toolchain),
            FFmpegFrameDecoder(toolchain),
            people,
        ),
    )
    feedback_repo = SqliteFeedbackRepository(db)
    queries = QueryService(projects, assets, jobs, storage, candidates, clip_repo, feedback_repo)
    music = MusicLibrary(settings.music_dir, toolchain, settings.data_dir / "music_cache.json")
    ai_music = AceStep(default_acestep_root())
    ingest = IngestService(projects, assets, reader, storage, MagicBytesSniffer(), probe, queue)
    analysis = AnalysisService(projects, assets, jobs, queue)
    clips = ClipService(
        queries,
        clip_repo,
        queue,
        assets,
        storage,
        music.tracks,
        llm,
        lambda uri, times: sampler.sample(uri, times, 640),
        ai_music,
        music.analyze,
        music.envelope,
    )

    return LocalEngine(
        settings=settings,
        events=events,
        queue=queue,
        toolchain=toolchain,
        probe=probe,
        whisper=whisper,
        accel=accel,
        llm=llm,
        ingest=ingest,
        analysis=analysis,
        clips=clips,
        feedback=FeedbackService(clip_repo, candidates, feedback_repo),
        queries=queries,
        sampler=sampler,
        storage=storage,
        music=music,
        ai_music=ai_music,
        downloads=DownloadManager(
            SqliteDownloadRepository(db),
            YouTubeImporter(settings.data_dir / "tools", toolchain.ffmpeg),
            probe,
            ingest,
            analysis,
            queries,
            settings.data_dir,
            lambda uri, times, width: sampler.sample(uri, times, width),
        ),
        reels=ReelService(
            SqliteReelRepository(db),
            ingest,
            analysis,
            queries,
            queue,
            assets,
            clips,
            FFmpegRenderer(toolchain),
            FFmpegReelCompiler(toolchain),
            FFmpegFrameDecoder(toolchain),
            probe,
            lambda uri, times, width: sampler.sample(uri, times, width),
            llm,
            settings.data_dir,
            people,
        ),
        research=ResearchService(SqliteResearchRepository(db), settings.data_dir),
    )


def build_llm(settings: Settings) -> LLMProvider | None:
    """Pick the AI judge. Cloud is only used when a key is configured, never implicitly."""
    choice = settings.ai_provider
    if choice in ("auto", "anthropic") and settings.anthropic_api_key:
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    if choice in ("auto", "ollama"):
        ollama = OllamaProvider(settings.ollama_host, settings.ollama_model)
        if choice == "ollama" or ollama.available():
            return ollama
    return None
