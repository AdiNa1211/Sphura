"""faster-whisper (CTranslate2) speech-to-text. Runs fully on-device; CUDA when available."""

from __future__ import annotations

import itertools
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sphura.domain.errors import ToolchainError
from sphura.domain.transcript import TranscriptSegment, TranscriptWord
from sphura.infrastructure.ai.accel import WhisperAccelChecker, cuda_device_count
from sphura.ports.ai import AudioSamples, ProgressFn, ProviderInfo

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions

log = logging.getLogger(__name__)

# Segments Whisper itself marks as probably-not-speech with low confidence are hallucinations
_NO_SPEECH_PROB = 0.6
_LOW_LOGPROB = -1.0
_LANG_WINDOW_S = 30.0
_HINDI_MIN = 0.03  # hi+ur share: Hinglish scores 0.03-0.25, English dialogue ~0.002
_HINGLISH_PROMPT = "भाई, ये तो pro move था! चल, next level पे चलते हैं।"
_VAD_PARAMS = {"min_silence_duration_ms": 500, "speech_pad_ms": 300}


def resolve_model(requested: str, device: str) -> tuple[str, str]:
    """(model, compute_type) for a concrete device. 'auto' model picks by device."""
    compute = {"cuda": "float16", "cpu": "int8"}.get(device, "auto")
    if requested != "auto":
        return requested, compute
    # turbo is accurate and fast on GPUs; on CPU, small is the best speed/quality trade-off
    return ("small" if device == "cpu" else "large-v3-turbo"), compute


class FasterWhisperProvider:
    _run_lock = threading.Lock()  # one model run at a time: it saturates CPU/GPU on its own

    def __init__(
        self,
        model: str,
        models_dir: Path,
        device: str = "auto",
        checker: WhisperAccelChecker | None = None,
    ) -> None:
        self._requested_model = model
        self._requested_device = device
        self._models_dir = models_dir
        self._checker = checker
        self._resolved: tuple[str, str, str] | None = None
        self._model: WhisperModel | None = None
        self._load_lock = threading.Lock()
        self._resolve_lock = threading.Lock()
        self.fallback_reason: str | None = None

    def resolve(self) -> tuple[str, str, str]:
        """(model, device, compute). Lazy: the GPU check only runs when transcription needs it."""
        with self._resolve_lock:
            if self._resolved is None:
                device = self._requested_device
                if device == "auto":
                    device = "cuda" if cuda_device_count() > 0 else "cpu"
                if device != "cpu" and self._checker is not None:
                    check = self._checker.check(device)
                    if not check.ok:
                        log.warning(
                            "Whisper on %s failed its test (%s); using CPU", device, check.detail
                        )
                        self.fallback_reason = f"{device} unavailable: {check.detail}"
                        device = "cpu"
                model, compute = resolve_model(self._requested_model, device)
                self._resolved = (model, device, compute)
            return self._resolved

    @property
    def model_name(self) -> str:
        return self.resolve()[0]

    @property
    def device(self) -> str:
        return self.resolve()[1]

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name=f"faster-whisper/{self.model_name}", runs_locally=True)

    def _get(self) -> WhisperModel:
        with self._load_lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as e:
                    raise ToolchainError(
                        "faster-whisper is not installed; run `pip install -e .` in engine/"
                    ) from e
                self._models_dir.mkdir(parents=True, exist_ok=True)
                _, device, compute = self.resolve()
                try:
                    self._model = self._load(WhisperModel, device, compute)
                except Exception as e:
                    if device != "mps":
                        raise ToolchainError(
                            f"could not load Whisper model '{self.model_name}': {e}. The first "
                            "run downloads it from Hugging Face, so check your internet connection."
                        ) from e
                    # Apple GPU backend is experimental: fall back to CPU rather than fail
                    log.warning("Whisper on MPS unavailable (%s); using CPU", e)
                    self.fallback_reason = f"mps unavailable: {e}"
                    self._resolved = (resolve_model(self._requested_model, "cpu")[0], "cpu", "int8")
                    self._model = self._load(WhisperModel, "cpu", "int8")
            return self._model

    def _load(self, cls: type[WhisperModel], device: str, compute: str) -> WhisperModel:
        log.info("loading whisper %s on %s (%s)", self.model_name, device, compute)
        return cls(
            self.model_name,
            device=device,
            compute_type=compute,
            cpu_threads=os.cpu_count() or 4,
            download_root=str(self._models_dir),
        )

    @staticmethod
    def _vad() -> VadOptions:
        # detect_language() needs a VadOptions object; transcribe() accepts either
        from faster_whisper.vad import VadOptions

        return VadOptions(**_VAD_PARAMS)

    def detect_language(self, audio: AudioSamples, sample_rate: int) -> tuple[str, float]:
        model = self._get()
        with self._run_lock:
            try:
                lang, prob, _ = model.detect_language(
                    audio,
                    vad_filter=True,
                    vad_parameters=self._vad(),
                    language_detection_segments=4,
                )
            except ValueError:  # VAD found no speech at all
                return "en", 0.0
        return lang, float(prob)

    def transcribe(
        self,
        audio: AudioSamples,
        sample_rate: int,
        language: str | None,
        offset_s: float,
        on_progress: ProgressFn,
    ) -> list[TranscriptSegment]:
        if language is not None:
            return self._transcribe(audio, sample_rate, language, offset_s, on_progress)
        # Auto: Hindi/Hinglish windows are transcribed as Hindi. Whisper often scores
        # Hinglish as English and then translates it ("bhai" -> "brother")
        out: list[TranscriptSegment] = []
        length = len(audio) or 1
        for a, b, lang in self._language_runs(audio, sample_rate):

            def part(f: float, a: int = a, b: int = b) -> None:
                on_progress((a + f * (b - a)) / length)

            out += self._transcribe(audio[a:b], sample_rate, lang, offset_s + a / sample_rate, part)
        on_progress(1.0)
        return out

    def _language_runs(self, audio: AudioSamples, sr: int) -> list[tuple[int, int, str | None]]:
        """[(start, end, "hi" or None = auto)] over the audio, cut near quiet points."""
        step = int(_LANG_WINDOW_S * sr)
        cuts = [0]
        while cuts[-1] + step < len(audio) - sr * 5:
            cuts.append(_quiet_point(audio, cuts[-1] + step, sr))
        cuts.append(len(audio))
        model = self._get()
        marks: list[str | None] = []
        for a, b in itertools.pairwise(cuts):
            with self._run_lock:
                try:
                    _, _, probs = model.detect_language(
                        audio[a:b],
                        vad_filter=True,
                        vad_parameters=self._vad(),
                        language_detection_segments=1,
                    )
                except ValueError:  # no speech in this window: follow the previous one
                    marks.append(marks[-1] if marks else None)
                    continue
            p = dict(probs)
            marks.append("hi" if p.get("hi", 0.0) + p.get("ur", 0.0) >= _HINDI_MIN else None)
        runs: list[tuple[int, int, str | None]] = []
        for (a, b), m in zip(itertools.pairwise(cuts), marks, strict=True):
            if runs and runs[-1][2] == m:
                runs[-1] = (runs[-1][0], b, m)
            else:
                runs.append((a, b, m))
        return runs

    def _transcribe(
        self,
        audio: AudioSamples,
        sample_rate: int,
        language: str | None,
        offset_s: float,
        on_progress: ProgressFn,
    ) -> list[TranscriptSegment]:
        model = self._get()
        length = len(audio) / sample_rate or 1.0
        out: list[TranscriptSegment] = []
        with self._run_lock:
            try:
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    # Mixed streams (English game dialogue + Hindi commentary): re-detect per
                    # window instead of forcing one language, which translates the rest
                    multilingual=language is None,
                    # Hinglish style: keeps English words in Latin script inside Hindi
                    initial_prompt=_HINGLISH_PROMPT if language == "hi" else None,
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=self._vad(),
                    # Off: on long noisy gameplay audio, conditioning causes repetition loops
                    condition_on_previous_text=False,
                    hallucination_silence_threshold=2.0,
                )
                for s in segments:  # lazy generator: decoding happens while iterating
                    on_progress(min(s.end / length, 1.0))
                    if s.no_speech_prob > _NO_SPEECH_PROB and s.avg_logprob < _LOW_LOGPROB:
                        continue
                    words = tuple(
                        TranscriptWord(
                            w.word.strip(), w.start + offset_s, w.end + offset_s, w.probability
                        )
                        for w in (s.words or ())
                        if w.word.strip()
                    )
                    conf = sum(w.probability for w in words) / len(words) if words else 0.0
                    out.append(
                        TranscriptSegment(
                            s.start + offset_s,
                            s.end + offset_s,
                            s.text.strip(),
                            words,
                            confidence=conf,
                            language=info.language if language is None else language,
                        )
                    )
            except ValueError as e:
                if "need at least one array" in str(e):  # VAD removed everything: no speech
                    return []
                raise
        on_progress(1.0)
        return out


def _quiet_point(audio: AudioSamples, at: int, sr: int) -> int:
    """The quietest 100 ms within 3 s of `at`, so a cut doesn't split a word."""
    hop, span = sr // 10, sr * 3
    lo, hi = max(0, at - span), min(len(audio) - hop, at + span)
    if hi <= lo:
        return at
    return min(range(lo, hi, hop), key=lambda i: float(np.abs(audio[i : i + hop]).mean()))
