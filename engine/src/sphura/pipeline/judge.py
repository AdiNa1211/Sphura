"""Phase 5b: an AI judge (vision LLM) reviews each candidate window from sampled frames,
transcript lines and signal evidence, and returns a grounded, structured verdict.

Verdicts are cached per candidate in analysis/judgements.jsonl, so re-runs and resumed
runs never pay for the same window twice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sphura.domain.models import (
    MIN_CLIP_S,
    AIJudgement,
    HighlightCandidate,
    HighlightCategory,
    MediaAsset,
    TimeRange,
)
from sphura.infrastructure.ai.http import AIProviderError
from sphura.pipeline.candidates import CandidateWindow
from sphura.pipeline.manifest import stage_key
from sphura.ports.ai import ImageInput, LLMProvider, LLMRequest
from sphura.ports.media import FrameSampler
from sphura.ports.storage import StorageProvider

log = logging.getLogger(__name__)

PROMPT_VERSION = 3  # v3: factual checklist drives the score; risky-claim guard
FRAMES_PER_CANDIDATE = 8
FRAME_WIDTH = 512
MIN_REFINED_S = MIN_CLIP_S
_HOPELESS_AFTER = 3

StageProgress = Callable[[float, str | None], None]

CHECKLIST = (
    "kill_or_finisher",
    "player_dies_or_fails",
    "boss_or_giant_enemy",
    "multiple_enemies",
    "cutscene_or_story",
    "rare_or_surprising",
    "funny",
    "routine_combat",
    "outcome_clear",
    "spectacle",
    "menu_or_loading",
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "footage": {
            "type": "string",
            "description": "What the frames show: gameplay, cutscene, menu, loading, other",
        },
        "what_happens": {"type": "string"},
        "is_highlight": {"type": "boolean"},
        "category": {
            "type": "string",
            "enum": ["action", "clutch", "funny", "cinematic", "unexpected", "none"],
        },
        "kill_or_finisher": {"type": "boolean"},
        "player_dies_or_fails": {"type": "boolean"},
        "boss_or_giant_enemy": {"type": "boolean"},
        "multiple_enemies": {"type": "boolean"},
        "cutscene_or_story": {"type": "boolean"},
        "rare_or_surprising": {"type": "boolean"},
        "funny": {"type": "boolean"},
        "routine_combat": {"type": "boolean"},
        "outcome_clear": {"type": "boolean"},
        "spectacle": {"type": "string", "enum": ["low", "medium", "high"]},
        "menu_or_loading": {"type": "string", "enum": ["none", "some", "mostly"]},
        "quality": {"type": "integer", "minimum": 1, "maximum": 10},
        "standalone": {"type": "boolean"},
        "reason": {"type": "string"},
        "title": {"type": "string"},
        "hook": {"type": "string"},
        "start_offset_s": {"type": "number"},
        "end_offset_s": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "footage",
        "what_happens",
        "is_highlight",
        "category",
        *CHECKLIST,
        "quality",
        "standalone",
        "reason",
        "title",
        "hook",
        "start_offset_s",
        "end_offset_s",
        "confidence",
    ],
}

SYSTEM = """You are an experienced short-form video editor for gaming content (TikTok, \
YouTube Shorts, Instagram Reels). You review ONE candidate moment cut from a long recording \
and decide whether it would make an engaging standalone short.

Rules:
- Judge only from the frames, transcript lines and signal evidence provided. Never invent \
events, outcomes, kills, deaths or reactions that are not visible or audible in the evidence.
- First say what the footage is. If it is not gameplay or a game cutscene (menus, loading, \
inventory, settings, desktop) or nothing notable happens, set is_highlight=false, \
category="none".
- Answer the checklist strictly from the evidence; it matters more than your overall score:
  kill_or_finisher: an enemy is visibly killed or a finishing move lands.
  player_dies_or_fails: the player visibly dies, is downed, or fails badly.
  boss_or_giant_enemy: a boss or an enemy much larger than the player is fought.
  multiple_enemies: three or more enemies fight the player at once.
  cutscene_or_story: a non-interactive cutscene or story dialogue scene.
  rare_or_surprising: something genuinely unusual for this game (glitch, odd physics, \
unexpected twist). Ordinary kills, deaths and boss hits are NOT rare.
  funny: something a viewer would laugh at.
  routine_combat: an ordinary fight like the many others in a game of this kind. Most \
combat candidates are routine; answer true unless the fight clearly stands out.
  outcome_clear: a viewer can tell what happened and who won.
  spectacle: visual scale and effects; "high" only for exceptional effects or scale, \
"medium" for normal action-game effects.
  menu_or_loading: how much of the window is menus, inventory or loading screens.
- quality 1-10: your overall impression (secondary). Routine fights are 5-6.
- standalone: true only if a viewer with no context would understand what is happening \
and why it matters.
- Never name a character, game, place or item unless the name appears in on-screen text \
or in the transcript. Do not guess which game this is. Say "the player" otherwise.
- category: action (intense combat, boss fight), clutch (survival or win against the odds, \
skillful play), funny (fails, glitches, humour, reactions), cinematic (scenery, cutscene, \
dramatic reveal), unexpected (rare or surprising events), none.
- title: at most 60 characters, specific to what happens.
- hook: at most 45 characters of on-screen opening text, different from the title. It \
must be true to the clip: no counts, "one hit", "1vX", HP or win/loss claims unless the \
frames or transcript clearly show them; no fake stakes. If unsure, describe the moment.
- what_happens: 1-2 factual sentences. reason: 1 sentence on why it would or would not \
work as a short.
- start_offset_s / end_offset_s: the tightest boundaries inside the window (seconds from \
window start) that keep the setup and the payoff or reaction. Use 0 and the full length if \
the window is already right.
- confidence 0-1: how sure you are, given that you only see a few frames.
- The transcript is machine-generated and may be mistranscribed or mistranslated; treat \
low-confidence lines with caution. The recording may contain non-English speech."""


def _level(z: float) -> str:
    return "very strong" if z >= 4 else "strong" if z >= 2.5 else "moderate" if z >= 1.5 else "weak"


def frame_times(w: CandidateWindow, n: int = FRAMES_PER_CANDIDATE) -> list[float]:
    """Evenly spread across the window, with one frame exactly at the signal peak."""
    span = w.end_s - w.start_s
    even = [w.start_s + span * (i + 0.5) / (n - 1) for i in range(n - 1)]
    return sorted([*even, w.peak_s])


def build_request(w: CandidateWindow, frames: list[ImageInput]) -> LLMRequest:
    f = w.features
    evidence = [
        f"Window length: {w.duration_s:.1f}s. Strongest signal at +{w.peak_s - w.start_s:.1f}s.",
        "Signal evidence (relative to the rest of this recording):",
        f"- sudden loudness: {_level(f.get('loud_spike', 0))}",
        f"- impacts/hits/gunfire-like sounds: {_level(f.get('impacts', 0))}",
        f"- shouting-range energy: {_level(f.get('shouting', 0))}",
        f"- on-screen motion: {_level(f.get('motion', 0))}",
        f"- visual change: {_level(f.get('visual_change', 0))}",
        f"- scene cuts in window: {int(f.get('scene_cuts', 0))}",
    ]
    if f.get("finale_of_s"):
        evidence.append(
            f"Context: this window is the END of a sustained fight/action stretch that lasted "
            f"{f['finale_of_s'] / 60:.0f} min. It should contain the final blows and the payoff "
            "right after (finishing move, defeat, cutscene, reactions). Judge it as that fight's "
            "climax: dialogue here is usually the payoff, not filler."
        )
    if w.transcript:
        evidence.append("Transcript lines (time from window start, confidence):")
        for s in w.transcript:
            evidence.append(
                f"- +{max(0.0, s.start_s - w.start_s):.1f}s (conf {s.confidence:.2f}): {s.text}"
            )
    else:
        evidence.append("Transcript: no speech detected in this window.")
    evidence.append(
        f"\nFrames above are in time order. Review the moment and submit your verdict. "
        f"Offsets must be between 0 and {w.duration_s:.1f}."
    )
    return LLMRequest(
        system=SYSTEM,
        prompt="\n".join(evidence),
        images=frames,
        json_schema=SCHEMA,
        max_tokens=800,
    )


_TAG_LABELS = {
    "kill_or_finisher": "finisher",
    "player_dies_or_fails": "player dies",
    "boss_or_giant_enemy": "boss/giant",
    "multiple_enemies": "many enemies",
    "cutscene_or_story": "cutscene",
    "rare_or_surprising": "rare",
    "funny": "funny",
    "routine_combat": "routine",
}

# Claims a few frames can't verify; small models add them anyway
_RISKY_CLAIM = re.compile(
    r"\bone[- ]?(hit|shot|tap|strike|blow|swing|move|attack|combo|slash)\b|\bin one\b"
    r"|\b\d+\s*v\s*\d+\b|\bno[- ]damage\b|\b\d+\s*hp\b"
    r"|\bfirst try\b|\bflawless\b|\bwithout (getting )?hit\b",
    re.IGNORECASE,
)


def rubric_quality(d: dict[str, Any]) -> float:
    """1-10 from factual answers. Small vision models describe well but rate nearly
    everything 8-9; a fixed rubric over what they saw spreads moments apart."""
    q = 4.0
    q += 1.0 if d.get("kill_or_finisher") else 0.0
    q += 1.0 if d.get("player_dies_or_fails") else 0.0
    q += 1.0 if d.get("boss_or_giant_enemy") else 0.0
    q += 0.5 if d.get("multiple_enemies") else 0.0
    q += 1.0 if d.get("cutscene_or_story") else 0.0
    q += 2.0 if d.get("rare_or_surprising") else 0.0
    q += 2.0 if d.get("funny") else 0.0
    q -= 1.5 if d.get("routine_combat") else 0.0
    q += 0.5 if d.get("outcome_clear") else -1.0
    q += {"high": 1.5, "medium": 0.5}.get(str(d.get("spectacle")), 0.0)
    q -= {"some": 1.0, "mostly": 4.0}.get(str(d.get("menu_or_loading")), 0.0)
    return min(max(q, 1.0), 10.0)


def _safe_text(text: str) -> str:
    return "" if _RISKY_CLAIM.search(text) else text


def guard_claims(text: str) -> bool:
    """True when text makes no claim a few frames can't back up."""
    return not _RISKY_CLAIM.search(text)


def parse_judgement(text: str, w: CandidateWindow, provider: str) -> AIJudgement:
    d: dict[str, Any] = json.loads(text)
    cat = str(d.get("category", "none")).lower()
    category = None if cat == "none" else HighlightCategory(cat)
    a = min(max(float(d.get("start_offset_s", 0)), 0.0), w.duration_s)
    b = min(max(float(d.get("end_offset_s", w.duration_s)), 0.0), w.duration_s)
    model_q = int(min(max(int(d.get("quality", 5)), 1), 10))
    tags = tuple(label for key, label in _TAG_LABELS.items() if d.get(key))
    if d.get("spectacle") == "high":
        tags += ("high spectacle",)
    refined = None
    if b - a >= MIN_REFINED_S and (a > 0.5 or b < w.duration_s - 0.5):
        refined = TimeRange(w.start_s + a, w.start_s + b)
    return AIJudgement(
        provider=provider,
        is_highlight=bool(d.get("is_highlight")) and category is not None,
        category=category,
        quality=round(0.8 * rubric_quality(d) + 0.2 * model_q),
        standalone=bool(d.get("standalone")),
        what_happens=str(d.get("what_happens", "")).strip(),
        reason=str(d.get("reason", "")).strip(),
        title=_safe_text(str(d.get("title", "")).strip()[:80])
        or _fallback_title(str(d.get("what_happens", "")), tags, category),
        hook=_safe_text(str(d.get("hook", "")).strip()[:60]),
        confidence=float(min(max(float(d.get("confidence", 0.5)), 0.0), 1.0)),
        refined=refined,
        tags=tags,
        model_quality=model_q,
    )


def _fallback_title(what: str, tags: tuple[str, ...], category: HighlightCategory | None) -> str:
    """Title was dropped for an unverifiable claim: reuse the plain description instead."""
    first = re.split(r"(?<=[.!?])\s|, then | and then |, ", what.strip(), maxsplit=1)[0]
    first = first.rstrip(".")
    if len(first) > 60:
        first = first[:60].rsplit(" ", 1)[0]
    if first and _safe_text(first):
        return re.sub(r"^The player\b", "Player", first)
    parts = [t for t in tags if t != "routine"][:2]
    base = " + ".join(parts) if parts else (category.value if category else "moment")
    return base[:1].upper() + base[1:]


def judgement_to_dict(j: AIJudgement) -> dict[str, Any]:
    return {
        "provider": j.provider,
        "is_highlight": j.is_highlight,
        "category": j.category.value if j.category else None,
        "quality": j.quality,
        "standalone": j.standalone,
        "what_happens": j.what_happens,
        "reason": j.reason,
        "title": j.title,
        "hook": j.hook,
        "confidence": j.confidence,
        "refined": [j.refined.start_s, j.refined.end_s] if j.refined else None,
        "tags": list(j.tags),
        "model_quality": j.model_quality,
    }


def judgement_from_dict(d: dict[str, Any]) -> AIJudgement:
    r = d.get("refined")
    return AIJudgement(
        provider=d["provider"],
        is_highlight=d["is_highlight"],
        category=HighlightCategory(d["category"]) if d.get("category") else None,
        quality=d["quality"],
        standalone=d["standalone"],
        what_happens=d["what_happens"],
        reason=d["reason"],
        title=_safe_text(d["title"])
        or _fallback_title(d["what_happens"], tuple(d.get("tags", ())), None),
        hook=_safe_text(d["hook"]),
        confidence=d["confidence"],
        refined=TimeRange(r[0], r[1]) if r else None,
        tags=tuple(d.get("tags", ())),
        model_quality=d.get("model_quality"),
    )


@dataclass(slots=True)
class JudgeReport:
    judged: int = 0
    cached: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str | None = None
    sends_data_off_device: bool = False


class JudgeStage:
    name = "judge"

    def __init__(
        self, llm: LLMProvider | None, sampler: FrameSampler, storage: StorageProvider
    ) -> None:
        self._llm = llm
        self._sampler = sampler
        self._storage = storage
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str | None:
        return self._llm.info.name if self._llm else None

    def run(
        self,
        asset: MediaAsset,
        windows: list[CandidateWindow],
        progress: StageProgress,
    ) -> tuple[list[HighlightCandidate], JudgeReport]:
        report = JudgeReport()
        candidates = [
            HighlightCandidate(
                project_id=asset.project_id,
                asset_id=asset.id,
                range=TimeRange(w.start_s, w.end_s),
                peak_s=w.peak_s,
                signal_score=w.signal_score,
                peak_score=w.peak_score,
                drivers=w.drivers,
                features=w.features,
            )
            for w in windows
        ]
        if self._llm is None or not windows:
            progress(1.0, "no AI judge configured: ranking by signals only")
            return candidates, report

        llm = self._llm
        report.provider = llm.info.name
        report.sends_data_off_device = not llm.info.runs_locally
        cache_path = self._storage.artifact_path(asset.project_id, "analysis", "judgements.jsonl")
        cache = self._load_cache(cache_path)

        todo: list[tuple[int, str]] = []
        for i, w in enumerate(windows):
            key = self._key(asset, w, llm.info.name)
            if key in cache:
                candidates[i].judgement = cache[key]
                report.cached += 1
            else:
                todo.append((i, key))

        done = report.cached
        progress(done / len(windows), f"judging {len(todo)} moment(s) with {llm.info.name}")
        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=max(1, llm.max_concurrency)) as pool:
            futures = {
                pool.submit(self._judge_one, asset, windows[i], llm): (i, key) for i, key in todo
            }
            for fut in as_completed(futures):
                i, key = futures[fut]
                done += 1
                try:
                    judgement, tin, tout = fut.result()
                except AIProviderError as e:
                    if e.fatal:  # bad key / unknown model: every other call would fail too
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    log.warning("judging candidate at %.0fs failed: %s", windows[i].peak_s, e)
                    errors.append(e)
                    report.failed += 1
                    self._stop_if_hopeless(report, pool, e)
                except Exception as e:  # one bad response shouldn't sink the batch
                    log.warning("judging candidate at %.0fs failed: %s", windows[i].peak_s, e)
                    errors.append(e)
                    report.failed += 1
                    self._stop_if_hopeless(report, pool, e)
                else:
                    candidates[i].judgement = judgement
                    report.judged += 1
                    report.input_tokens += tin
                    report.output_tokens += tout
                    self._append_cache(cache_path, key, judgement)
                progress(done / len(windows), f"judged {done}/{len(windows)}")

        if todo and report.judged == 0 and errors:
            raise errors[0]  # e.g. bad API key or unreachable server: surface it
        return candidates, report

    @staticmethod
    def _stop_if_hopeless(report: JudgeReport, pool: ThreadPoolExecutor, e: Exception) -> None:
        """Same failure on the first few calls and no success: it won't fix itself."""
        if report.judged == 0 and report.failed >= _HOPELESS_AFTER:
            pool.shutdown(wait=False, cancel_futures=True)
            raise AIProviderError(f"first {report.failed} moments all failed; last error: {e}")

    def _judge_one(
        self, asset: MediaAsset, w: CandidateWindow, llm: LLMProvider
    ) -> tuple[AIJudgement, int, int]:
        times = frame_times(w)
        jpegs = self._sampler.sample(asset.source_uri, times, FRAME_WIDTH)
        frames = [
            ImageInput(t, jpg, f"Frame {k + 1} at +{t - w.start_s:.1f}s")
            for k, (t, jpg) in enumerate(zip(times, jpegs, strict=True))
        ]
        resp = llm.complete(build_request(w, frames))
        return parse_judgement(resp.text, w, llm.info.name), resp.input_tokens, resp.output_tokens

    @staticmethod
    def _key(asset: MediaAsset, w: CandidateWindow, provider: str) -> str:
        return stage_key(
            "judge",
            PROMPT_VERSION,
            provider,
            asset.fingerprint,
            round(w.start_s, 1),
            round(w.end_s, 1),
        )

    @staticmethod
    def _load_cache(path: Path) -> dict[str, AIJudgement]:
        out: dict[str, AIJudgement] = {}
        if not path.is_file():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                out[rec["key"]] = judgement_from_dict(rec["judgement"])
            except (ValueError, KeyError, TypeError):
                continue  # torn or stale line: that window is simply judged again
        return out

    def _append_cache(self, path: Path, key: str, j: AIJudgement) -> None:
        line = json.dumps({"key": key, "judgement": judgement_to_dict(j)}, ensure_ascii=False)
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
