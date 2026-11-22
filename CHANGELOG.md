# Changelog

The full development history of SPHURA, phase by phase. This was the project README until the docs were reorganised; the new [README](README.md) covers setup and usage.

---

# SPHURA

AI Shorts editor for gaming streams. Point it at a long recording (or a YouTube video), and it finds the best moments and cuts them into ready-to-post vertical Shorts with effects, captions, music, titles and hashtags. Everything runs locally on your PC.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical plan.

## Quick start (Windows)

1. Install [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) (`winget install Gyan.FFmpeg`) and [Ollama](https://ollama.com) with a vision model (`ollama pull gemma4:12b`).
2. Set up the engine once:
   ```bash
   cd engine
   python -m venv .venv
   .venv\Scripts\pip install -e ".[dev]"
   ```
3. Double-click **`SPHURA.bat`**. It starts Ollama if needed and opens SPHURA at `http://127.0.0.1:47821/`. Keep the black window open while you use it, and restart it after updating the code.

## What it does

- **New stream:** paste a file path or **Browse** for a recording, use a YouTube video from Downloads, or pick one of your recent recordings (OBS folder, Videos, Downloads, other drives). SPHURA analyses it with live progress per stage.
- **Best moments:** ranked moments with thumbnails, tiers and the AI's notes. 👍 / 👎 teach SPHURA your taste across videos. Fight finales (the end of a long boss fight) are always found and kept.
- **Make Shorts:** pick styles, the game name, the number of moments and a **max length per style**. Each Short keeps the length its moment needs, up to that limit.
- **Shorts:** preview every Short, copy the title and description (with hashtags), change music, edit the on-screen text, or open the **timeline editor**.
- **Downloads:** a YouTube queue and library for videos and playlists you own or have permission to use, with quality choice, pause/resume and **Edit video** straight into the Shorts pipeline.
- **Music:** original music made for every Short (local AI music, or SPHURA's own composer), or tracks from your music folder picked by the AI. The edit is synced to the beat.

### Styles

| Style | Format | Look |
|---|---|---|
| **Hype** | 9:16 | Cold open, zoom punches, shake, speed ramps, slow-mo, sound effects. Optional meme caption bar |
| **Meme** | 3:4, ~20 s | White caption bar: the setup until the key hit, then the punchline. Game audio |
| **Cinematic** | 1:1, ~20 s | Cool grade, glitch cuts, motion trails, all-caps opener, music on the beat. Optional caption bar |

Effects follow what happens in the clip: a finisher gets a freeze-frame and letterbox bars, an explosion a bigger flash and shake, a death or a funny moment a punch-in and freeze, and so on. Each effect can be switched off per Short in the editor.

### Timeline editor

Open any Short to trim or split its cuts on a timeline, add text at any time with its own style, size, colour and position, turn effects on or off, and toggle the caption bar. **Update preview** renders a quick draft; **Save & export** makes the final Short with the same music re-synced to the new cuts.

### Where things are saved

| What | Where |
|---|---|
| Shorts, with a `.txt` of title, description and hashtags | `Videos\SPHURA\<date> <stream name>\<style>` |
| Downloads | `Videos\SPHURA\Downloads` (changeable in Downloads settings) |
| Your music | `Music\SPHURA` |
| Database, analysis, tools | `%LOCALAPPDATA%\SPHURA` |
| AI music model (ACE-Step 1.5, installed from the Music page) | `<install folder>\ai\ace-step` |

Data from the earlier names (SPHURI, ClipForge) moves over automatically on first start, and their `SPHURI_*` / `CLIPFORGE_*` environment variables still work.

## Layout

| Path | What | Status |
|---|---|---|
| `engine/` | Python engine (FastAPI, SQLite, FFmpeg) and the browser UI in `src/sphura/web` | In use |
| `apps/desktop/` | Flutter Windows app | Planned |
| `apps/web/` | Cloud web client | Planned |
| `packages/api_contract/` | OpenAPI spec exported from the engine | Generated |
| `deploy/` | Cloud workers / Docker | Planned |

## How it was built

The sections below record each build phase and how the engine works inside.

## Phase 1 — what works

- Clean architecture: `domain` → `ports` (Protocols) → `application` → `infrastructure` adapters, wired only in `container.py`.
- Ingestion of MP4 / MOV / MKV / WebM / AVI, detected from **file bytes** (not extension). The original is referenced in place and never copied or modified.
- Constant-time fingerprint (size + 4 MB head/tail) → a 6 GB file registers in ~20 ms; re-adding the same file returns its existing project (also follows the file if it moved).
- SQLite persistence (WAL, versioned SQL migrations).
- Persistent job queue with worker threads, throttled progress, cooperative cancel, crash recovery (interrupted jobs requeue on restart).
- Background streamed SHA-256 job (constant memory) — exercises the queue the analysis stages will use.
- Local API (FastAPI) with token auth + WebSocket progress events; CLI.

## Phase 2 — what works

- **FFmpeg toolchain:** finds `ffmpeg`/`ffprobe` (in `SPHURA_FFMPEG_DIR`, then bundled next to the app, then PATH, then Homebrew paths on macOS).
- **Encoder testing:** checks which encoders actually work on this machine by test-encoding a tiny clip. Being listed by FFmpeg isn't enough, because hardware encoders need a matching GPU and driver.
  - H.264 preference: NVENC → QSV → AMF → VideoToolbox → Media Foundation → libx264.
  - The results are cached in `hardware.json` and retested when FFmpeg changes or when you run `doctor --refresh`.
- **Media probe:** reads duration, codec, resolution after rotation, frame rate, HDR, bit depth, and every audio track with its name (e.g. OBS "Game"/"Mic"). Cover-art images are skipped.
- **Real VFR detection:** samples frame timestamps at the start and middle of the video instead of trusting the header.
- **Broken files are rejected before anything is registered:**
  - an MP4 whose recording crashed (missing index), with a recovery hint
  - corrupt or truncated files
  - audio-only files
- **WebM without a duration header** (e.g. browser recordings): the duration is measured by scanning the file.
- **Ingest now probes the file:** files registered in Phase 1 get their details filled in when added again.
- **FFmpeg runner:** reports live progress and can be cancelled; later phases use it.

## Phase 3 — what works

- **`analyze` job:** runs the stages in order, with stage-level progress over WebSocket and the CLI. Finished stages are recorded in `analysis/manifest.json` and skipped on re-runs. Any change (a different file, language or model, or a new stage version) invalidates them.
- **Audio extraction:** one FFmpeg read of the source decodes every audio track to 16 kHz mono WAV, written atomically to `proxy/audio_s<stream>.wav`.
  - That's about 115 MB per hour per track, read through memory-mapping so the audio is never loaded into RAM.
- **Track choice:** a track named Mic/Voice/Commentary is transcribed (OBS multi-track); otherwise the default track, otherwise the first.
- **Local transcription with faster-whisper:** word timestamps, the Silero speech filter (bundled, no download), and repetition loops and hallucinations removed.
  - Model `auto` picks `large-v3-turbo` on NVIDIA GPUs and `small` (int8) on CPU. Override it with `--model` or `SPHURA_WHISPER_MODEL`.
  - The model is downloaded on first use into `<data>/models/whisper`.
- **GPU safety check:** before using an NVIDIA (or Apple) GPU, a tiny model is run once in a separate process, because missing cuDNN/cuBLAS libraries can crash CTranslate2 outright. If the check fails, transcription falls back to CPU automatically. The result is cached in `whisper_accel.json`; `sphura doctor` shows it and `--refresh` re-tests.
  - On Windows, the DLLs from `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` are found automatically.
- **Language:** auto-detected from 8 samples spread across the whole recording (intros are often music); force it with `--language`.
- **Resumable:** audio is split into roughly 10-minute chunks cut at the quietest moment, and each finished chunk is saved to `transcript.chunks.jsonl`. Ctrl+C, a crash or power loss loses at most the chunk in progress, and a torn last line is repaired.
- Output: `analysis/transcript.json`; `sphura transcript <project> [--srt file.srt]`.

## Phase 4 — what works

- **`signals` stage** (runs at the end of `analyze`): a 0.5 s `SignalTimeline` saved as `analysis/signals.npz`.
- **Audio** (from the default/mix track): loudness, loudness relative to the surrounding minute, spectral flux (impacts, gunshots, hits) and the 1–4 kHz energy share (shouting).
  - About 3 s for a 108-minute stream.
- **Visual:** one low-res decode (256×144 at 4 fps, hardware-decoded when available) gives motion, novelty against a ~10 s memory, brightness, saturation and scene cuts.
  - Decoded in 5-minute segments that are checkpointed, so an interrupted run resumes.
- **Speech:** words per second (weighted by transcript confidence), exclamations, and laughter written as text.
- **Normalisation:** every feature is robust-z-normalised per video, so a naturally loud streamer isn't "exciting" all the time.
- **Preliminary interest curve:** a weighted fusion with a penalty for still, quiet stretches (menus, loading, AFK).
  - `sphura signals <project> [--top 15] [--html chart.html]` lists the strongest moments with the reasons and draws the curve.
  - Phase 5 replaces this ranking with candidate windows plus AI judgement.
- **Synthetic check:** on a 7.5-minute test video with three action bursts and two menu stretches, the top 3 peaks were exactly the three bursts (scores 6.2–6.4, next 2.2).

### Transcription fixes from the first real stream (Phase 3.1)

- **Mixed language:** auto language mode detects per 30 s window (`multilingual`), so English game dialogue and Hindi commentary each come out in their own language instead of being force-translated.
- **Hallucination filtering,** measured on a real stream (real dialogue ≥ 0.75 mean word probability, music hallucinations 0.05–0.55):
  - segments under 0.25 confidence are dropped;
  - segments whose words are more than 8 s apart (Whisper stretching over music) are dropped unless confidence ≥ 0.7;
  - kept segments are split at gaps over 3 s so timings are accurate.

## Phase 5 — what works

- **Candidate windows** (`pipeline/candidates.py`): peaks of the interest curve become 15–90 s windows.
  - Each window covers build-up → peak → payoff, plus about 4 s of lead-in and 3 s of tail.
  - Starts snap to nearby scene cuts, and windows never start or end mid-sentence.
  - Overlapping windows are merged, and windows that are mostly menu or idle are dropped.
  - Capped at about 15 per hour (minimum 12) to bound AI cost. A 108-minute stream gives 27 windows.
- **AI judge** (`pipeline/judge.py`): for each window, 8 frames (512 px, one at the peak) plus plain-language signal evidence plus transcript lines go to a vision LLM.
  - It returns a structured verdict: footage type, what happens, highlight yes/no, category, quality 1–10, standalone, title, hook, tighter boundaries and confidence.
  - The rules forbid inventing events and require hooks to be true to the clip.
  - Verdicts are cached per window in `analysis/judgements.jsonl`, so re-runs never pay twice.
  - One bad response doesn't sink the batch, but if every call fails (bad key, server down) the job fails with the real error.
- **Providers** (stdlib HTTP, no SDK dependency), chosen with `--ai` or `SPHURA_AI`:
  - `anthropic`: Claude via the Messages API with forced tool-use JSON; 4 in parallel; retries on 429/5xx. **Cloud:** sends frames and transcript lines, never the video. `ANTHROPIC_API_KEY`; `SPHURA_ANTHROPIC_MODEL` (default `claude-sonnet-5-5`).
  - `ollama`: a local vision model through Ollama's JSON-schema output. Everything stays on the PC. `SPHURA_OLLAMA_MODEL` (default `gemma4:12b`), `SPHURA_OLLAMA_HOST`.
  - `auto` (default): Anthropic if a key is set, else Ollama if it's running with the model, else none (signals-only ranking, clearly labelled). Cloud is never used implicitly; `analyze` states which judge it uses and what leaves the machine.
- **Ranking** (`pipeline/ranking.py`):
  - The score blends AI quality (scaled by confidence, 75%) with signal strength (25%), plus a mode bonus (Highlight → action/clutch, Funny → funny/unexpected, Cinematic, Viral).
  - Non-highlights are halved, and moments that don't stand alone get a penalty.
  - Greedy diversity keeps the top of the list from being five slices of one fight.
  - The ranking is computed when you ask, so switching mode is instant and makes no new AI calls.
- **Side-by-side comparison round** (`pipeline/compare.py`): small vision models rate nearly everything 8–9 when judging one moment at a time, but compare well.
  - AI-accepted moments are shown in groups of 4 (3 frames + description each) and the model orders each group.
  - 4 Swiss-style rounds: the first is random, the later ones group moments with similar standing.
  - Every group ranking becomes head-to-head results, and Bradley-Terry strengths turn them into a 0–1 `compare_score`, which leads the AI part of the ranking (60%).
  - Simulated with a noisy judge on 27 moments: it recovers on average 4.3 of the true top 5 from comparisons alone, where flat absolute scores carry no information.
  - Cached in `analysis/comparisons.jsonl`; deterministic grouping; local model at temperature 0.
- **Tag rarity:** checklist tags most moments share in a video (e.g. "boss/giant" on 70%) are discounted by how common they are, so only distinctive features lift a moment.
- **Honest tiers:** TOP means quality ≥ 8, a highlight, and standalone; STRONG means quality ≥ 6. Moments the AI hasn't reviewed get no tier, and no 0–100 scores are shown.
- **Storage:** candidates and verdicts live in the `candidates` table (migration 0002); `sphura highlights <project> [--mode funny] [--details]`.

## Phase 6 — what works

- **Clips as edit lists** (`pipeline/clips.py`, the `clips` table): the top-ranked moments become editable `Clip` EDLs.
  - Each clip holds source segments, the AI title, and the hook as a text overlay.
  - Dead stretches (still picture, or quiet with no motion, and nobody talking) of 2 s or more become jump cuts. Dead lead-ins and tails of 1.5 s or more are trimmed.
  - Cuts are skipped if they would leave the clip under 15 s.
  - Re-running keeps an existing clip for the same moment, so edits survive.
- **Renderer** (`infrastructure/media/ffmpeg/render.py`):
  - Every segment is a separate input with a fast input seek, so only the clip's ranges are decoded, never the whole recording. Frame accuracy is verified at every jump cut.
  - 16:9 up to 1080p, capped at 60 fps, with short audio fades at each join so cuts don't click.
  - Loudness is normalised to -14 LUFS, the TikTok/Shorts/Reels target.
  - Silent sources get a silent track.
  - Encoders: NVENC → QSV → AMF → VideoToolbox → Media Foundation → libx264, falling back automatically if a GPU encoder fails mid-render.
  - Files are written as `.part.mp4` and renamed on success.
- **Render jobs:** one per clip, so a batch can be cancelled or resumed per clip. Output goes to `renders/NN_title.mp4` in the workspace, with `--out` to also copy finished clips to a folder.
- `sphura render <project> [--top 5] [--mode funny] [--out D:\Shorts]`, `sphura clips <project>`.

## Taste learning

- **Rate what you watched:** `sphura feedback <project> --good 1 2 --bad 3 4 5`. The numbers are the clip numbers from the file names; only clips of the current analysis count.
- **Direct ratings are final for that moment.** A rejected moment drops out of the list and is never rendered. A kept moment gets a strong boost and may be TOP/STRONG even if the AI called it routine.
- **Taste across videos:** every rating stores a snapshot of the moment's traits (AI tags, category, signal drivers, length bucket).
  - Each trait's weight is its log-odds lift among good vs bad ratings, shrunk toward 0 until about 3 ratings back it.
  - Taste moves a ranking score by at most ±0.2, so a handful of ratings nudges rather than overrides.
- `sphura taste` shows what it learned, e.g. "you like: boss/giant, cutscene · you dislike: routine, finisher".

## Run

```bash
cd engine
pip install -e ".[dev]"

sphura ingest "D:\Recordings\session.mkv" --wait   # register + checksum with progress
sphura projects                                    # list projects/jobs
sphura analyze "<video or project id>" [--language en] [--model small] [--device mps]
sphura transcript <project id> [--limit 20] [--srt out.srt]
sphura signals <project id> [--top 15] [--html chart.html]
sphura highlights <project id> [--mode funny] [--limit 15] [--details]
sphura render <project id> [--top 5] [--out <folder>]
sphura clips <project id>
sphura info "<video>"                              # show details without registering
sphura doctor [--refresh]                          # FFmpeg + working GPU encoders
sphura serve --port 47821                          # local API (docs at /docs)
```

Requires FFmpeg (`brew install ffmpeg` / `winget install Gyan.FFmpeg`).
Env: `ANTHROPIC_API_KEY`, `SPHURA_AI`, `SPHURA_WHISPER_MODEL`, `SPHURA_WHISPER_DEVICE` (auto|cpu|cuda|mps), `SPHURA_FFMPEG_DIR`, `SPHURA_DATA_DIR`, `SPHURA_WORKERS`, `SPHURA_PORT`, `SPHURA_API_TOKEN`
(default data dir on Windows: `%LOCALAPPDATA%\SPHURA`).

### API (v1)

| Method | Path | |
|---|---|---|
| GET | `/health` | no auth |
| POST | `/v1/ingest` | `{path, project_id?, project_name?, mode?}` |
| GET | `/v1/projects`, `/v1/projects/{id}`, `/v1/projects/{id}/assets`, `/v1/projects/{id}/jobs` | |
| GET / POST | `/v1/jobs/{id}`, `/v1/jobs/{id}/cancel` | |
| GET | `/v1/system/hardware?refresh=` | hardware profile |
| POST | `/v1/projects/{id}/analyze` | `{language?}` → analyze job |
| GET | `/v1/projects/{id}/transcript` | transcript with word timings |
| GET | `/v1/projects/{id}/signals?points=&top=` | downsampled curves + top peaks |
| GET | `/v1/projects/{id}/highlights?mode=&limit=` | ranked moments with AI verdicts |
| POST / GET | `/v1/projects/{id}/clips?top=&mode=` | generate / list clip EDLs |
| POST | `/v1/projects/{id}/clips/render?top=` | render jobs for the top clips |
| WS | `/v1/events?token=…&project_id=…` | `job.progress` events |

Auth header: `X-Sphura-Token` (only enforced when `SPHURA_API_TOKEN` is set; the desktop app will set a random one per launch).

## Phase 7 — vertical 9:16 reframing

Renders are now **1080×1920 by default** (also `--aspect 1:1` → 1080×1080, `--aspect 16:9` → original framing).

- **Action tracking.** Each clip is scanned at 8 fps (320×180). The crop follows where the motion is. The frame's median motion is subtracted, so camera pans don't drag the crop around. A soft centre prior keeps it calm when nothing happens.
- **Smooth virtual camera.** The whole clip is known in advance, so the camera path is smoothed symmetrically (no lag behind the action). Pan speed is capped, and tiny wobbles are held still. The path goes into FFmpeg as one piecewise-linear crop expression, with no side files.
- **Auto layout.**
  - **crop**: when the action fits one window.
  - **fit**: when the action is scattered across the frame (the best window keeps jumping and covers little of it). Fit shows the full 16:9 frame over a blurred, zoomed copy of itself.
  - Override with `--layout crop|fit`.
- Jump-cut clips are planned per segment, and the crop path is saved on the clip, so it stays editable.
- **Tested on synthetic videos:** a fast-moving subject stays fully inside the crop at every sample (0/160 cut). Scattered noise picks "fit".

```
sphura render <project_id> --out $HOME\Videos\SPHURA                  # 9:16, auto layout
sphura render <project_id> --aspect 1:1 --layout fit --out ...
```

## Phase 8 — engagement edits

Every clip now gets these edits automatically, planned from signals the analysis already has:

- **Hook text:** the AI's hook line pops in at the top for the first 3 seconds, placed clear of the TikTok/Shorts buttons. It's drawn with libass in Arial Black with a thick outline.
- **Cold open:** the climax plays for about 1 second, then a white flash cuts to the start of the clip. This is skipped when the clip already opens on the action.
- **Punch-ins:** a quick 12% zoom on the strongest audio impacts. Each one is snapped to the exact hit (10 ms precision, from the audio proxy), so it lands on the beat.
- **Slow motion:** the key hit plays at 0.5× for about a second. The audio drops in pitch too, for the classic "time slows" feel.

These edits are stored as plain entries in the clip's `effects` list, so they stay editable. Use `sphura render ... --plain` to turn them off; the hook text stays.

## Phase 9 — pro short-form edit

Every vertical clip is now edited like a creator's Short. Each effect is anchored to a real hit or cut in the footage; nothing is invented.

- **Hook:** the climax plays first, then a glitch flash and whoosh cut back to the start. The hook text is in capitals, with the key word in yellow.
- **Build-up:** a slow push-in and a riser lead into the key hit.
- **Key hit:**
  - a speed ramp of 0.7× → 0.4× → 0.7× around it;
  - a bass drop;
  - a zoom with camera shake;
  - an RGB split;
  - a vignette.
- **Other impacts:** a punch-in with shake, a sub-bass hit and a short RGB split.
- **Pacing:** each jump cut gets a whoosh and a small zoom, and stretches longer than 5 s without an accent get a gentle zoom.
- **Look:** a light contrast and saturation grade, plus sharpening.
- **Sound effects** are generated by SPHURA itself, so there are no sample packs and nothing to license.
- **Music** comes from your own folder: `%USERPROFILE%\Music\SPHURA` (or set `SPHURA_MUSIC_DIR`). Use royalty-free tracks such as the YouTube Audio Library.
  - SPHURA measures where each track drops and lines that drop up with the moment slow-mo ends.
  - The music sits out the teaser and cuts out just before the key hit, while the riser builds.
  - It is ducked under the game audio.
- `--plain` turns all of this off (the hook text stays). Changing the style later re-plans existing clips automatically.

## Music: AI pick, beat sync, licenses

- **Track profiles:** each track in your music folder is measured for tempo (spectral-flux beat tracking, with half-tempo correction and drones rejected), dark vs bright tone, build-ups and drops, and the drop's position.
- **AI music supervisor:** the local AI reads the moment, the edit style and every track's profile plus your tags, picks a track, gives a short reason and two alternatives, and suggests what to search for when nothing fits. Without AI it falls back to a lively track.
- **Beat sync for every style:**
  - the key hit lands on a beat;
  - jump cuts are trimmed (≤ ¼ beat) so they land on the beat grid;
  - accents snap to nearby beats (≤ 0.12 s);
  - the drop lines up with the end of the slow-mo.
- **Dialogue:** the music dips to about 35% under spoken lines.
- **Licenses:** each track's source, license, credit and tags live in `sphura_tracks.json` in the music folder (editable on the Music page). A required credit is added to the Short's description `.txt`.
- **Finding music:** the Music page links to the YouTube Audio Library, Pixabay Music and Mixkit. Downloading stays manual, because these sites don't offer permitted automated access.

## Phase 9b — styles modelled on top-performing Shorts

These were researched from the most-viewed `#phantombladezero` Shorts (Nov 2026).

| `--style` | Format | Modelled on |
|---|---|---|
| `hype` (default) | 9:16. Cold open, punch-ins, shake, speed ramps, SFX | — |
| `meme` | 3:4. A white caption bar, game audio only, payoff near the end (now ~20 s: setup, then punchline) | a 2.5M-view caption-bar Short |
| `cinematic` | 1:1, about 18–20 s. Near-mono cool grade, pixel-glitch cuts, ghost trails in slow-mo, all-caps opener, music-led with game audio underneath | 2M- and 657K-view edit Shorts |

- **Beat sync:** the cinematic style detects the music's tempo (±10 ms). It slides the track so the key hit lands on a beat, and snaps nearby cuts, punches and glitches onto beats. It never moves them more than 120 ms away from the real hit.
- **Copy:** the local AI writes a reaction line, an all-caps hype line, a title (`<line> <emoji> - <Game>`), a description ending in a question, and hashtags. Everything is checked against made-up claims ("one hit", "flawless", names…). A `.txt` with the title, description and hashtags is exported next to each video.
- **Commands:**
  `sphura render <id> --style meme --game "Phantom Blade Zero" --out ...`
  `sphura render <id> --style cinematic --game "Phantom Blade Zero" --out ...`

## Later additions

- **Original music:** SPHURA composes a score per Short in one of 10 moods (phonk, drift, drill, trap, rage, dark, epic, tension, bounce, funk), varying moods between clips. With **AI music** installed (ACE-Step 1.5, MIT licence, runs on the GPU), it writes a prompt from the moment and generates a fitting instrumental instead, falling back to the composer if that fails. Nothing to license or credit.
- **Fight finales:** long stretches of sustained action are detected as fights, and the last 20 s plus aftermath becomes its own moment that is always included.
- **Context-matched effects:** each moment is classified (finisher, clutch, explosion, death, funny, story, action) and the effect plan is reshaped to fit, including freeze-frames and letterbox bars.
- **Max length per style:** a limit, not a target: shorter moments keep their own length; longer ones are cut around the key moment.
- **Caption bar on Hype/Cinematic:** the meme-style white bar with setup and punchline, per Short or for a whole batch.
- **Timeline editor** with draft previews, and music that stays the same track after edits.
- **YouTube downloads:** official yt-dlp + Deno, retries across YouTube clients, crash-safe queue, duplicate detection, disk-space checks. See section 18 of the architecture doc.
- **Robust re-renders:** music changes run in the background, and a render whose video is open elsewhere is saved under a new name instead of failing.

## Next

Ideas: a cover frame, a loop ending, captions for dialogue scenes, and a Flutter desktop app.
