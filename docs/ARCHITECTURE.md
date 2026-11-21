# SPHURA — Architecture & Technical Plan

AI gaming highlight clipper for Windows (local-first) and Web (cloud).
Long gameplay → signal analysis → candidate moments → AI ranking → editable clips (EDL) → 9:16 render.

---

## 1. Recommended overall architecture

**One processing engine, two hosts.**

```text
                ┌──────────────────────┐        ┌────────────────────────┐
                │ Flutter Desktop (Win)│        │ Web client (browser)   │
                └──────────┬───────────┘        └───────────┬────────────┘
                 localhost HTTP + WS                   HTTPS + WS
                ┌──────────▼───────────┐        ┌───────────▼────────────┐
                │ Engine (local mode)  │        │ Cloud API (same engine │
                │ SQLite · local FS    │        │ package) Postgres · S3 │
                │ in-process workers   │        │ queue → worker pools   │
                └──────────┬───────────┘        └───────────┬────────────┘
                           └──────────────┬─────────────────┘
                              sphura core (Python package)
             domain · application services · ports (interfaces) · pipeline
                                          │
          adapters: FFmpeg · faster-whisper · ONNX vision/audio · LLM/VLM · storage
```

Key design decisions:

- **Hexagonal core.** `domain` + `application` depend only on `ports` (Python `Protocol`s). FFmpeg, AI vendors, SQLite/Postgres and S3/local disk are adapters chosen at composition time (`container.py`).
- **Same engine, different adapters.** Local mode = SQLite + local filesystem + in-process worker pool. Cloud mode = Postgres + object storage + external queue + separate worker processes. Pipeline code is identical.
- **The Edit Decision List (EDL) is the product.** A `Clip` is data: source ranges, crop keyframes, caption track, overlays, audio mix, effects. Rendering is a pure function of the EDL plus the original file. That makes every AI result editable, regenerable and re-renderable at any aspect ratio without re-analysing.
- **Two-pass analysis.** Cheap low-res signals over the whole video → candidate windows → expensive AI only on candidates (typically <5% of runtime).
- **The UI is a thin client.** It never touches FFmpeg or models; it sends commands and subscribes to job progress.

## 2. Recommended technology stack

| Concern | Choice | Why |
|---|---|---|
| Engine language | **Python 3.12** | Best AI/media ecosystem; heavy work runs in native code (FFmpeg, ONNX Runtime, CTranslate2), so Python overhead is small. Rust/C# considered: faster glue, far weaker ML tooling. |
| Engine API | FastAPI + uvicorn, WebSocket for progress | Same API locally and in cloud; OpenAPI → generated Dart/TS clients. |
| Video | FFmpeg/FFprobe binaries (LGPL build for distribution) | Subprocess, never linked; hardware encoders NVENC/QSV/AMF with libx264 fallback. |
| Transcription | faster-whisper (CTranslate2), Silero VAD | Word timestamps, runs on CPU or CUDA, local. |
| Audio events | YAMNet/PANNs (ONNX) + numpy DSP | Laughter, screaming, gunshots, explosions, cheering. |
| Vision | ONNX Runtime (DirectML on Windows, CUDA in cloud); SigLIP/CLIP embeddings, RapidOCR for HUD text, saliency model for reframing | DirectML works on NVIDIA/AMD/Intel GPUs on Windows. |
| Scene cuts | FFmpeg `scdet` / PySceneDetect on proxy | Cheap and robust. |
| Reasoning | Pluggable `LLMProvider`/`VisionLLMProvider`: local (Ollama/llama.cpp, e.g. Qwen-VL class) or cloud (Claude, Gemini, OpenAI) | Swappable; local-only mode possible. |
| Windows UI | **Flutter Desktop**, BLoC, `media_kit` (libmpv) for preview | Frame-accurate playback, hardware decode. |
| Web UI | **Recommendation: web-native (React + TypeScript)** for the editor; Flutter Web acceptable for upload/review only | Browser video editing needs `<video>`, WebCodecs, precise scrubbing and canvas overlays — Flutter Web's video story is weak. Decision deferred to Phase 12; API-first design keeps both options open. |
| Local DB | SQLite (WAL) | Zero-config, single file per install. |
| Cloud DB | PostgreSQL | Same repository interfaces. |
| Cloud queue | Postgres-backed queue (`SELECT … FOR UPDATE SKIP LOCKED`) at first; Redis Streams / SQS if scale demands | Fewer moving parts; per-queue worker pools (cpu / gpu / render). |
| Object storage | S3-compatible (S3, R2, MinIO) + CDN | Multipart presigned uploads = resumable. |
| Packaging | Engine → PyInstaller onedir sidecar; Flutter MSIX/Inno Setup installer; models downloaded on demand | Keeps installer small; models cached in `%LOCALAPPDATA%`. |

## 3. Windows architecture

```text
Flutter UI (presentation, BLoC)
   │  generated Dart client (OpenAPI) + WebSocket events
   ▼
sphura-engine.exe  (sidecar, spawned & supervised by the app, bound to 127.0.0.1, random port + token)
   ├─ API layer (FastAPI)
   ├─ Application services (ingest, analyze, rank, generate, render)
   ├─ Local job queue (SQLite-persisted, worker threads/processes, resumable after crash)
   ├─ Pipeline stages → FFmpeg subprocesses, ONNX Runtime (DirectML), faster-whisper
   └─ Workspace: %LOCALAPPDATA%\SPHURA\projects\<id>\{proxy,analysis,renders,tmp}
```

- The original file is **referenced by path, never copied or modified**; a fast fingerprint (size + head/tail hash) detects if it moved or changed.
- The UI never blocks: every long operation is a job with stage-level progress pushed over WebSocket.
- GPU: capability probe at startup (`ffmpeg -encoders`, ONNX providers, CUDA availability) → `HardwareProfile` drives decoder/encoder/model choices.
- Crash safety: jobs and stage checkpoints persisted; analysis artifacts are per-stage files, so a restarted job resumes from the last completed stage.

## 4. Web architecture

```text
Browser ──multipart presigned PUTs (resumable)──► Object storage (originals)
   │                                                   │
   ▼                                                   ▼
API (FastAPI, auth, projects, clips, signed URLs) ─► Postgres ◄─ workers update state
   │ enqueue                                           ▲
   ▼                                                   │
Queue ──► cpu-workers   (probe, proxy, audio DSP, scene detection, render light)
      ──► gpu-workers   (whisper, vision models, VLM, NVENC render)
      ──► llm-workers   (ranking / hooks via external LLM API, rate-limited)
Renders ──► Object storage ──► CDN ──► preview/download
```

- Workers stream the original from object storage with HTTP range reads (FFmpeg reads presigned URLs directly), so no full download is needed for proxy generation; final render seeks only the needed ranges.
- Horizontal scaling per pool; fairness via per-user concurrency limits in the queue.
- Lifecycle rules delete originals/proxies after N days; renders kept per plan.

## 5. AI / video analysis pipeline

```text
Stage 0  Probe            ffprobe → streams, duration, fps (VFR?), audio tracks, rotation, HDR
Stage 1  Proxy decode     ONE ffmpeg pass over the original:
                            → 16 kHz mono WAV per audio track (mic/game tracks kept separate if present)
                            → 2–4 fps 360p frames streamed to the analyzer (no giant intermediate file)
                            → scene-cut scores (scdet)
Stage 2  Signal extraction (whole video, fixed 0.5 s hop → SignalTimeline)
            audio:  loudness (LUFS/RMS), spectral flux, onset density, event classes (laugh, scream, gunshot, explosion, music)
            speech: VAD + faster-whisper words with timestamps, speech rate, pitch/energy excitement
            visual: motion energy, frame-embedding novelty, brightness/colour change, HUD OCR (kill feed, "VICTORY", damage numbers) at ~1 fps on ROIs
            structure: scene cuts, static/menu/loading detection (low motion + UI-like text + no game audio)
Stage 3  Candidate detection
            fused interest curve → peak picking with non-max suppression
            + rule triggers (OCR "eliminated", laughter burst, scream, sudden audio spike after silence)
            → candidate windows (anchor time + rough bounds), typically 30–80 per 2 h
Stage 4  Deep analysis (candidates only)
            higher-res frames (720p, 4–8 fps) around each anchor
            VLM: what happens, category, does it stand alone, setup/payoff times
            LLM: transcript excerpt + signals + VLM description → category, reason, hook text, boundary refinement
Stage 5  Ranking & dedup  merge overlaps, diversity across categories and time, mode-specific weights
Stage 6  Clip generation  EDL: boundaries snapped to scene cuts/sentence ends, dead-time jump cuts, hook, captions, crop path, effects plan
Stage 7  Render           FFmpeg filter graph from EDL, original-quality source, hardware encoder
```

Every stage writes a versioned artifact (`analysis/<stage>.v1.parquet|json`), so re-ranking with a different mode reuses stages 0–4.

## 6. Highlight detection strategy

**Hybrid: interpretable signals find candidates; AI judges them.**

1. *Signal score (cheap, whole video)* — weighted sum of normalised (per-video z-score, so a loud streamer isn't always "exciting") features: action intensity, audio excitement, reaction (laugh/scream/speech-energy), event triggers, visual novelty. Penalties: menu/loading/static, low context.
2. *AI judgement (expensive, candidates only)* — VLM + LLM produce `category`, `reason`, `standalone` (makes sense without prior context), `hookability`, setup/payoff timestamps.
3. *Final rank* = mode-weighted blend (Funny weights reactions/laughter; Cinematic weights visual novelty/low HUD/scene quality; Highlight weights kill events/action; Viral weights hook + surprise + pace).
4. *Diversity* — maximal marginal relevance across category and timeline so the top 10 aren't 10 similar fights.

**Score honesty:** store component scores internally; show users a tier (🔥 Top / Strong / Maybe) plus the AI's reason. A 0–100 number is only shown once it's calibrated against real outcomes (user keeps/exports/deletes become training labels; later, real post performance if users connect accounts).

**Boundaries:** anchor event → walk backwards to the start of the build-up (activity rise, scene cut, sentence start, max pre-roll by category) → walk forward through payoff until reaction energy decays (speech/laughter end) → snap to cuts/sentence boundaries → remove internal dead spans (>1.5 s low activity, menus) as jump cuts if the result still flows.

## 7. Database / storage architecture

Relational metadata, file/object artifacts.

```text
projects(id, name, created_at, mode, status)
media_assets(id, project_id, source_uri, storage_kind, container, size_bytes, fingerprint, sha256, probe_json, created_at)
jobs(id, project_id, type, status, stage, progress, payload, result, error, attempts, created_at, started_at, finished_at, heartbeat_at)
analysis_runs(id, asset_id, pipeline_version, config_json, status, artifacts_json)
candidates(id, run_id, start, end, category, signal_score, ai_score, tier, reason, features_json)
clips(id, project_id, candidate_id, title, aspect, edl_json, version, created_at, updated_at)
renders(id, clip_id, preset, status, output_uri, size_bytes, duration)
```

- **SQLite** locally, **Postgres** in cloud, behind repository ports. Schema migrations are plain SQL files, versioned.
- **Large data never in the DB:** signal timelines (Parquet/npz), transcripts (JSON), frame thumbnails, proxies, renders live in the workspace (local FS) or object storage under `projects/<id>/…`.
- EDL stored as versioned JSON → undo/history and "regenerate" keep prior versions.

## 8. Processing queue architecture

- `Job` with type, payload, status (`queued → running → succeeded|failed|cancelled`), stage + progress, attempts, heartbeat.
- **Local:** SQLite-persisted queue; claim is an atomic `UPDATE … RETURNING`; worker threads for IO/subprocess-bound work, process pool for CPU-bound Python; GPU jobs serialised by a semaphore (one GPU job at a time on consumer PCs). Stale `running` jobs (no heartbeat) are re-queued at startup.
- **Cloud:** same `JobQueue` port backed by Postgres `SKIP LOCKED`; queues per resource class; retries with backoff; idempotent stages (artifact exists → skip).
- **Progress:** handlers report `(stage, fraction)`; events flow through an `EventBus` port → WebSocket. UI shows the stage list from the spec (Uploading / Analyzing / Finding moments / Generating / Rendering).
- **Batch render:** one job per clip, so 20 renders progress and fail independently.

## 9. FFmpeg architecture

- `FFmpegToolchain` adapter: locates binaries (bundled → PATH), probes version, encoders, hwaccels once and caches a `HardwareProfile`.
- Commands are built by small typed builders (no string concatenation), executed via `subprocess` with `-progress pipe:1` parsed into progress events, cancellable (process kill), stderr captured to the job log.
- **Never full-res decode for analysis.** Proxy pass uses hardware decode when available (`-hwaccel d3d11va/cuda`), scales to 360p and drops to 2–4 fps inside the filter graph; frames streamed as raw RGB over a pipe in fixed-size chunks.
- **Seek-accurate extraction for render:** input-seek (`-ss` before `-i`) to nearest keyframe + accurate trim in filter graph; only clip ranges are decoded.
- **Render graph from EDL:** `trim/atrim` per segment → `concat` → crop (keyframed via `sendcmd` or per-segment crops) → scale 1080×1920 → `ass` captions → overlays → audio mix (`amix`/volume per track) → encoder (h264_nvenc / h264_qsv / h264_amf / libx264), AAC 48 kHz, `+faststart`.
- **Avoid re-encoding:** proxies are not saved as video; stream copy is used for raw "export original segment" options.
- VFR safety: timestamps always in seconds from the container's PTS, never frame indices.

## 10. Local vs cloud processing strategy

| | Local (Windows) | Cloud (Web, optional on Windows) |
|---|---|---|
| Video bytes | Never leave the machine | Uploaded to object storage |
| Transcription / vision / audio models | Local (ONNX/faster-whisper) | GPU workers |
| LLM/VLM reasoning | Local model (if hardware allows) **or** cloud API with explicit opt-in | Cloud API |
| What a cloud LLM sees in local mode | Transcript excerpts + a few downscaled candidate frames only, shown in a consent dialog | — |

The UI displays a persistent badge: **Local processing** / **Cloud AI enabled** / **Cloud processing**. Provider selection is per-capability (e.g. local whisper + cloud VLM).

Low-end PCs: a "light" profile (smaller whisper model, lower frame rate, skip embeddings) and offer "process in cloud".

## 11. MVP feature boundaries

**In:** single video per project; MP4/MOV/MKV/WebM/AVI; probe; proxy + audio extraction; whisper transcript; audio/motion/scene signals; candidate detection; LLM/VLM ranking with reasons; 5–10 clips; automatic boundaries with basic dead-time trimming; 9:16 reframing with a smoothed single-crop path + blurred-fill fallback; word-timed burned captions (2–3 styles); one hook text overlay; render MP4 1080×1920; preview, trim, rename, delete, export; Auto/Highlight/Funny modes.

**Out (post-MVP):** effects (slow-mo, zooms, SFX, music), split facecam layout, game-specific detectors, Cinematic/Viral/Minimal modes, batch "export all" polish, web app, accounts, publishing to platforms, multi-video projects, score calibration.

## 12. Project folder structure

```text
sphura/
├── docs/ARCHITECTURE.md
├── engine/                         # Python processing engine (shared by desktop + cloud)
│   ├── pyproject.toml
│   └── src/sphura/
│       ├── domain/                 # entities, value objects, errors — no IO
│       ├── ports/                  # Protocol interfaces (media, ai, storage, repos, queue, events)
│       ├── application/            # use-case services (ingest, analyze, rank, generate, render)
│       ├── pipeline/               # analysis stages (Phase 2+)
│       ├── infrastructure/
│       │   ├── persistence/sqlite/ # repositories + migrations
│       │   ├── storage/            # local workspace, S3 (Phase 12)
│       │   ├── media/              # container sniffing, ffmpeg/ffprobe adapters
│       │   ├── ai/                 # whisper, onnx, llm providers
│       │   ├── queue/              # local SQLite queue, postgres queue
│       │   └── events/             # in-process event bus
│       ├── api/                    # FastAPI app, schemas, websocket
│       ├── config.py
│       ├── container.py            # composition root (adapter wiring)
│       └── cli.py
├── apps/
│   ├── desktop/                    # Flutter Windows app (presentation/domain/data, BLoC)
│   └── web/                        # Web client (Phase 12)
├── packages/
│   ├── api_contract/               # OpenAPI spec exported from engine
│   └── sphura_client_dart/      # generated Dart client used by desktop (and Flutter Web if chosen)
└── deploy/                         # Dockerfiles, worker configs (Phase 12)
```

## 13. Key domain models

- **Project** — id, name, mode (`AUTO|HIGHLIGHT|FUNNY|CINEMATIC|VIRAL|MINIMAL`), status.
- **MediaAsset** — source URI (local path or object key), container, size, fingerprint, sha256, `MediaInfo` (duration, video/audio streams, fps, VFR flag).
- **Job** — type, status, stage, progress, payload/result, attempts, timestamps.
- **AnalysisRun** — asset, pipeline version, config, artifact locations.
- **SignalTimeline** — hop size + named feature arrays (artifact, not a DB row).
- **TranscriptWord / TranscriptSegment** — text, start, end, confidence, speaker/track.
- **HighlightCandidate** — start, end, duration, category, confidence, signal score, AI score, tier, reason, feature breakdown.
- **Clip (EDL)** — segments `[SourceRange]`, aspect ratio, `CropPath` (keyframes: t, cx, cy, zoom), `CaptionTrack` (words + style), `TextOverlay` (hook), `AudioMix` (per-track gains), effects list, version.
- **RenderJob / RenderOutput** — preset, encoder, output URI, status.

## 14. Main interfaces / services

Ports (engine `ports/`): `MediaProbe`, `MediaDecoder` (proxy frames/audio), `SceneDetector`, `AudioAnalyzer`, `TranscriptService`, `VisionAnalyzer`, `HighlightDetector`, `HighlightRanker`, `CaptionGenerator`, `CropTracker`, `ClipGenerator`, `VideoRenderer`, `AIProvider` (`TranscriptionProvider`, `VisionProvider`, `LLMProvider`, `HighlightScoringProvider`), `StorageProvider`, repositories (`ProjectRepository`, `MediaAssetRepository`, `JobRepository`, …), `JobQueue`, `EventBus`.

Application services: `IngestService`, `AnalysisService` (orchestrates stages), `HighlightService`, `ClipService` (EDL CRUD, regenerate), `RenderService`, `JobService`.

`VideoAnalyzer` from the spec is the `AnalysisService` façade over the stage ports.

## 15. Development phases

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 1 | Monorepo, domain, ports, SQLite persistence, workspace storage, job queue, ingestion, local API + CLI | Register a multi-GB file instantly; background checksum job reports progress |
| 2 | FFmpeg toolchain + probe + hardware profile | Correct `MediaInfo` for all 5 containers incl. VFR & multi-track |
| 3 | Audio extraction + whisper transcript (word timestamps) | 2 h video transcribed locally, resumable |
| 4 | Proxy decode + audio/visual/scene signals → SignalTimeline | Interest curve visibly peaks on real action |
| 5 | Candidate detection + AI ranking (provider abstraction) | Top 10 judged good on a test set of real recordings |
| 6 | EDL clip generation + render of trimmed 16:9 clips | Real MP4s out |
| 7 | 9:16 reframing (crop path + smoothing) | Action stays in frame, no jitter |
| 8 | Captions (ASS, word highlight, safe zones) | Readable on phone, avoid HUD |
| 9 | Hooks + editing effects | Selective, non-misleading |
| 10 | Batch rendering, performance, GPU paths | 2 h → shorts in target time on mid-range GPU |
| 11 | Flutter Windows app + packaging (sidecar, installer, model downloads) | Clean install on fresh Windows |
| 12 | Cloud: uploads, Postgres/S3 adapters, workers, web client | Same results as local |

The Flutter UI is started alongside Phase 6 (once real clips exist) so it is never built on fake data.

## 16. Major technical risks

1. **Highlight quality across genres** — the core product risk. Mitigate with a labelled evaluation set (10–20 real recordings across FPS/BR/horror/story games) from Phase 4, and measure precision@10 every change.
2. **Consumer hardware** — whisper + vision on CPU-only laptops may be slow for 3–5 h videos. Light profile, GPU detection, cloud fallback.
3. **VLM/LLM cost & latency** — bounded by candidate cap per hour of video and frame budget per candidate.
4. **Packaging size** — CUDA/cuDNN runtimes are GBs; prefer DirectML on Windows and download models on demand.
5. **Licensing** — FFmpeg must be an LGPL build for closed distribution (no libx264 → use hardware encoders or OpenH264/ship GPL separately); several detectors (e.g. Ultralytics YOLO) are AGPL — avoid or license; check model licenses.
6. **VFR / A/V sync** in screen recordings — always use PTS seconds; test with OBS and ShadowPlay files.
7. **Multi-track audio** (mic vs game on separate tracks) — big advantage for reaction detection when present; must also work with a single mixed track.
8. **Smart reframing** on fast games — jitter vs. missed action trade-off; needs smoothing + HUD awareness.
9. **Misleading hooks/hallucinated reasons** — ground LLM output in detected signals; reject hooks referencing events not observed.
10. **Game audio copyright** on exported shorts (licensed music) — warn/mute music track option.

## 17. Estimated complexity

| Component | Complexity | Notes |
|---|---|---|
| Ingestion, persistence, queue, API | Medium | Mostly engineering discipline |
| FFmpeg toolchain, probe, hw profile | Medium | Many edge cases across containers/GPUs |
| Proxy decode + audio extraction | Medium | Streaming without big temp files |
| Transcription | Low–Medium | Library does the hard part; long-file chunking |
| Audio/visual signal extraction | High | Feature engineering + per-video normalisation |
| Candidate detection + boundaries | High | Core IP; needs evaluation set |
| AI ranking (VLM/LLM) | High | Prompting, grounding, cost control |
| EDL + renderer | High | Filter graphs, keyframed crop, A/V sync |
| Smart reframing | Very high | Tracking + smoothing + layouts |
| Captions | Medium | ASS styling, safe zones |
| Effects/hooks | Medium–High | Taste matters more than code |
| Flutter desktop app + editor | High | Timeline, preview, crop editor |
| Windows packaging | Medium–High | Sidecar lifecycle, GPU runtimes, installer |
| Cloud pipeline + web app | High | Uploads, scaling, auth, cost |

## 18. Downloads (YouTube videos and playlists)

A separate module from the editing engine. A finished download is a plain local file; **Edit Video** hands it to the normal `IngestService` + `AnalysisService` flow, which references the file and never modifies it.

```text
Browser UI (Downloads: queue · downloaded videos · settings)
   │  /v1/downloads/*
   ▼
DownloadManager (application/downloads.py)
   ├─ scheduler thread → N worker threads (setting: 1–4 at once)
   ├─ downloads table (SQLite, migration 0007) = queue + history
   ├─ settings: <data>/downloads.json (folder, quality, file name, thumbnails, parallel)
   └─ YouTubeImporter (infrastructure/media/ytdlp.py) → official yt-dlp + Deno in <data>/tools
```

- **States:** queued → downloading → completed, or paused / failed / cancelled. Retry and resume put an entry back to queued.
- **Partial files:** each download works in `<folder>/.sphura-partial/<video id>`. Pause and shutdown stop yt-dlp (whole process tree) and keep the partial files, so the download resumes from where it was. Cancel and remove delete them. At startup, `downloading` rows become `queued` (crash recovery), and partial folders with no matching download are removed.
- **Duplicates:** a video already queued, already downloaded (file still present) or already in the folder (`[video id]` in the name) is skipped. Re-adding a failed or cancelled video reuses its entry.
- **Retry ladder:** empty or 403 responses are retried with other YouTube clients, then HLS.
- **Disk space:** a download refuses to start below 500 MB free, and stops with a clear message once yt-dlp reports a size larger than the free space.
- **Thumbnails:** YouTube's thumbnail is saved to `<data>/thumbnails`. If none was saved, a frame is taken from the video with the existing frame sampler.
- **Editing status** is derived, not stored: no project → Not edited; analysis queued or running → Analysing; analysed → Ready for Shorts; any rendered clip → Shorts made.
- **Delete** removes only the downloaded file and its thumbnail. Shorts live in `Videos/SPHURA/<date> <name>` and are kept.
- **Rights:** adding to the queue requires the user to confirm that the videos are theirs or that they have permission. Private, members-only, Premium, live and still-processing videos are shown as unavailable and are never downloaded.
