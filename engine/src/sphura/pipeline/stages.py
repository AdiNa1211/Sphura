"""Analysis stages. Idempotent: finished work is recorded in the StageManifest and skipped."""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from sphura.domain.errors import InvalidMediaError
from sphura.domain.models import MediaAsset
from sphura.domain.transcript import (
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    segment_from_dict,
    segment_to_dict,
)
from sphura.pipeline.hinglish import romanize
from sphura.pipeline.manifest import StageManifest, stage_key
from sphura.pipeline.wav import WavFile
from sphura.ports.ai import TranscriptionProvider
from sphura.ports.media import AudioExtractor
from sphura.ports.storage import StorageProvider

log = logging.getLogger(__name__)

StageProgress = Callable[[float, str | None], None]

ANALYSIS_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class ExtractedAudio:
    files: dict[int, Path]  # audio stream index -> 16 kHz mono WAV
    skipped: bool


class ExtractAudioStage:
    name = "extract_audio"
    version = 1

    def __init__(self, extractor: AudioExtractor, storage: StorageProvider) -> None:
        self._extractor = extractor
        self._storage = storage

    def run(
        self, asset: MediaAsset, manifest: StageManifest, progress: StageProgress
    ) -> ExtractedAudio:
        info = asset.media_info
        if info is None:
            raise InvalidMediaError("media details missing; re-run ingest")
        indices = [a.index for a in info.audio]
        if not indices:
            return ExtractedAudio({}, skipped=True)

        key = stage_key(self.name, self.version, asset.fingerprint, indices, ANALYSIS_SAMPLE_RATE)
        if done := manifest.completed(self.name, key):
            return ExtractedAudio({int(k): Path(v) for k, v in done["tracks"].items()}, True)

        files = {
            i: self._storage.artifact_path(asset.project_id, "proxy", f"audio_s{i}.wav")
            for i in indices
        }
        progress(0.0, f"decoding {len(files)} audio track(s)")
        self._extractor.extract(
            asset.source_uri,
            files,
            ANALYSIS_SAMPLE_RATE,
            info.duration_s,
            lambda f: progress(f, None),
        )
        manifest.mark(
            self.name,
            key,
            {
                "tracks": {str(i): str(p) for i, p in files.items()},
                "files": [str(p) for p in files.values()],
            },
        )
        return ExtractedAudio(files, skipped=False)


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    start_s: float
    end_s: float


def plan_chunks(wav: WavFile, target_s: float = 600.0, search_s: float = 60.0) -> list[Chunk]:
    """Split long audio near target_s, cutting at the quietest moment so words aren't split.
    Deterministic for a given file, which keeps resumed runs aligned with saved chunks."""
    chunks: list[Chunk] = []
    start, total = 0.0, wav.duration_s
    while start < total:
        if total - start <= target_s + search_s:
            end = total
        else:
            win_start = start + target_s - search_s
            rms = wav.rms_frames(win_start, start + target_s, frame_s=0.1)
            if len(rms):
                # Latest of the quietest frames, so chunks stay close to target length
                quiet = np.flatnonzero(rms <= rms.min() * 1.1 + 1e-4)
                end = win_start + float(quiet[-1]) * 0.1 + 0.05
            else:
                end = start + target_s
        chunks.append(Chunk(len(chunks), start, end))
        start = end
    return chunks


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


_MAX_WORD_GAP_S = 3.0
_STRETCH_GAP_S = 8.0  # words this far apart inside one segment = Whisper stretched over music
# Measured on a real stream: dialogue >= 0.75 mean word probability, hallucinations 0.05-0.55
_MIN_CONFIDENCE = 0.25
_TRUSTED_CONFIDENCE = 0.7


def split_on_gaps(seg: TranscriptSegment) -> list[TranscriptSegment]:
    """Whisper sometimes stretches a few words over minutes of music; word timings are
    reliable, so split where speech actually stops and recompute spans and confidence."""
    if not seg.words:
        return [seg]
    groups: list[list[TranscriptWord]] = [[seg.words[0]]]
    for prev, w in itertools.pairwise(seg.words):
        if w.start_s - prev.end_s > _MAX_WORD_GAP_S:
            groups.append([])
        groups[-1].append(w)
    if len(groups) == 1:
        g = groups[0]
        return [replace(seg, start_s=g[0].start_s, end_s=g[-1].end_s)]
    return [
        replace(
            seg,
            start_s=g[0].start_s,
            end_s=g[-1].end_s,
            text=" ".join(w.text for w in g),
            words=tuple(g),
            confidence=sum(w.probability for w in g) / len(g),
        )
        for g in groups
    ]


def _max_gap(seg: TranscriptSegment) -> float:
    return max((b.start_s - a.end_s for a, b in itertools.pairwise(seg.words)), default=0.0)


def clean_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Judge each segment whole first, then fix timings, so hallucinations aren't kept as
    split-off fragments."""
    kept: list[TranscriptSegment] = []
    for s in segments:
        if s.words and s.confidence < _MIN_CONFIDENCE:
            continue
        if _max_gap(s) > _STRETCH_GAP_S and s.confidence < _TRUSTED_CONFIDENCE:
            continue
        kept.extend(p for p in split_on_gaps(s) if not p.words or p.confidence >= _MIN_CONFIDENCE)
    return drop_repetition_loops(kept)


def drop_repetition_loops(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Whisper can loop on noisy audio ("thank you. thank you. ..."); keep at most 2 in a row."""
    out: list[TranscriptSegment] = []
    for s in segments:
        n = _norm(s.text)
        if not n:
            continue
        if len(out) >= 2 and _norm(out[-1].text) == n == _norm(out[-2].text):
            continue
        out.append(s)
    return out


class TranscribeStage:
    name = "transcribe"
    version = 3  # v3: Hinglish (Hindi windows transcribed as Hindi, then romanized)
    _LANG_WINDOWS = 8
    _LANG_WINDOW_S = 30.0

    def __init__(self, provider: TranscriptionProvider, storage: StorageProvider) -> None:
        self._provider = provider
        self._storage = storage

    def transcript_path(self, project_id: str) -> Path:
        return self._storage.artifact_path(project_id, "analysis", "transcript.json")

    def run(
        self,
        asset: MediaAsset,
        wav_path: Path,
        stream_index: int,
        language: str | None,
        manifest: StageManifest,
        progress: StageProgress,
    ) -> Transcript:
        key = stage_key(
            self.name,
            self.version,
            asset.fingerprint,
            stream_index,
            self._provider.model_name,
            language,
        )
        out_path = self.transcript_path(asset.project_id)
        if manifest.completed(self.name, key) and out_path.is_file():
            return Transcript.from_dict(json.loads(out_path.read_text()))

        wav = WavFile(wav_path)
        chunks = plan_chunks(wav)
        ckpt = self._storage.artifact_path(asset.project_id, "analysis", "transcript.chunks.jsonl")
        header, done = self._load_checkpoint(ckpt, key)

        if header is None:
            progress(0.0, "loading speech model")
            lang, prob = (language, 1.0) if language else self._detect_language(wav)
            header = {"type": "header", "key": key, "language": lang, "probability": prob}
            ckpt.write_text(json.dumps(header) + "\n")
            done = {}
            log.info("transcript language: %s (p=%.2f)", lang, prob)

        lang = str(header["language"])  # dominant language; per-window unless forced
        total = sum(c.end_s - c.start_s for c in chunks) or 1.0
        finished = sum(c.end_s - c.start_s for c in chunks if c.index in done)
        resumed = len(done)

        with ckpt.open("a", encoding="utf-8") as f:
            for c in chunks:
                if c.index in done:
                    continue
                length = c.end_s - c.start_s
                base = finished
                msg = f"chunk {c.index + 1}/{len(chunks)}" + (
                    f" (resumed after {resumed})" if resumed else ""
                )

                def on_chunk(
                    frac: float, b: float = base, ln: float = length, m: str = msg
                ) -> None:
                    progress((b + frac * ln) / total, m)

                segs = self._provider.transcribe(
                    wav.read(c.start_s, c.end_s), wav.sample_rate, language, c.start_s, on_chunk
                )
                segs = [t for s in segs if (t := _tidy(s)).text]
                done[c.index] = segs
                f.write(
                    json.dumps(
                        {
                            "type": "chunk",
                            "index": c.index,
                            "segments": [segment_to_dict(s) for s in segs],
                        }
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())  # survive power loss mid-run
                finished += length
                progress(finished / total, msg)

        segments = clean_segments([s for i in sorted(done) for s in done[i]])
        transcript = Transcript(
            language=lang,
            language_probability=float(header["probability"]),
            model=self._provider.model_name,
            audio_stream_index=stream_index,
            duration_s=wav.duration_s,
            segments=segments,
        )
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False))
        os.replace(tmp, out_path)
        manifest.mark(
            self.name,
            key,
            {"files": [str(out_path)], "segments": len(segments), "version": self.version},
        )
        return transcript

    def _detect_language(self, wav: WavFile) -> tuple[str, float]:
        # Sample across the whole recording: the first minutes are often intro music/silence
        dur = wav.duration_s
        n = self._LANG_WINDOWS if dur > self._LANG_WINDOWS * self._LANG_WINDOW_S * 2 else 1
        starts = [dur * (0.05 + 0.9 * i / max(n - 1, 1)) for i in range(n)] if n > 1 else [0.0]
        audio = np.concatenate([wav.read(s, s + self._LANG_WINDOW_S) for s in starts])
        return self._provider.detect_language(audio, wav.sample_rate)

    @staticmethod
    def _load_checkpoint(
        path: Path, key: str
    ) -> tuple[dict[str, Any] | None, dict[int, list[TranscriptSegment]]]:
        if not path.is_file():
            return None, {}
        header: dict[str, Any] | None = None
        done: dict[int, list[TranscriptSegment]] = {}
        raw = path.read_bytes()
        valid_end = 0
        for line in raw.splitlines(keepends=True):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("unterminated")
                rec = json.loads(line)
            except ValueError:
                # Torn write from a crash: cut it off so new records start on a clean line
                with path.open("r+b") as f:
                    f.truncate(valid_end)
                break
            valid_end += len(line)
            if rec.get("type") == "header":
                if rec.get("key") != key:
                    return None, {}  # inputs changed: start over
                header = rec
            elif rec.get("type") == "chunk" and header is not None:
                done[int(rec["index"])] = [segment_from_dict(s) for s in rec["segments"]]
        return header, done


def _tidy(seg: TranscriptSegment) -> TranscriptSegment:
    """Captions stay in Roman script (Hindi as Hinglish: "bhai", never "brother"), without
    stutter loops ("aan aan aan aan") or characters Whisper cut in half."""
    words = [replace(w, text=romanize(w.text).replace("\ufffd", "")) for w in seg.words]
    keep = [w for w in _drop_word_loops(words) if w.text]
    text = romanize(seg.text).replace("\ufffd", "").strip()
    if len(keep) != len(seg.words):
        text = " ".join(w.text for w in keep)
    return replace(seg, text=text, words=tuple(keep))


def _drop_word_loops(words: list[TranscriptWord], limit: int = 4) -> list[TranscriptWord]:
    out: list[TranscriptWord] = []
    i = 0
    while i < len(words):
        j = i
        key = _norm(words[i].text)
        while j < len(words) and _norm(words[j].text) == key:
            j += 1
        if j - i < limit or not key:
            out += words[i:j]
        i = j
    return out
