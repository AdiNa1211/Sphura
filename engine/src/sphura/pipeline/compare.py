"""Phase 5d: comparative ranking. Small vision models rate almost every moment 8-9 when
judged alone, but are far better at "which of these is best?". Accepted moments are shown
in groups of 4 (3 frames + description each) and ordered by the model, over several
Swiss-style rounds; each moment's average placing becomes its comparison score.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sphura.domain.models import HighlightCandidate, MediaAsset
from sphura.pipeline.manifest import stage_key
from sphura.ports.ai import ImageInput, LLMProvider, LLMRequest
from sphura.ports.media import FrameSampler
from sphura.ports.storage import StorageProvider

log = logging.getLogger(__name__)

COMPARE_VERSION = 1
GROUP_SIZE = 4
ROUNDS = 4
FRAMES_PER_MOMENT = 3
FRAME_WIDTH = 384
MIN_MOMENTS = 3

StageProgress = Callable[[float, str | None], None]

_LETTERS = "ABCDEFGH"

SYSTEM = """You are an experienced short-form video editor for gaming content (TikTok, \
YouTube Shorts, Instagram Reels). You are shown several candidate moments from the SAME \
recording. Rank them from best to worst as standalone shorts.

What makes one better than another:
- A stranger scrolling past would stop and watch: a clear, surprising or dramatic payoff.
- It stands out from the rest of this recording. Ordinary fights that look like the others \
rank below rare events, deaths, comebacks, story reveals or funny moments.
- The action is easy to follow in the frames.
Judge only from the frames and descriptions given. Every moment must appear exactly once."""


def schema(n: int) -> dict[str, Any]:
    letters = list(_LETTERS[:n])
    return {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": letters},
                "minItems": n,
                "maxItems": n,
            },
            "reason": {"type": "string"},
        },
        "required": ["ranking", "reason"],
    }


@dataclass(slots=True)
class CompareReport:
    calls: int = 0
    cached: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class CompareStage:
    name = "compare"

    def __init__(
        self, llm: LLMProvider | None, sampler: FrameSampler, storage: StorageProvider
    ) -> None:
        self._llm = llm
        self._sampler = sampler
        self._storage = storage
        self._lock = threading.Lock()
        self._frames: dict[str, list[bytes]] = {}

    def run(
        self, asset: MediaAsset, candidates: list[HighlightCandidate], progress: StageProgress
    ) -> CompareReport:
        """Sets compare_score (0..1, higher = better) on accepted candidates."""
        report = CompareReport()
        pool = [c for c in candidates if c.judgement and c.judgement.is_highlight]
        if self._llm is None or len(pool) < MIN_MOMENTS:
            progress(1.0, "not enough AI-accepted moments to compare")
            return report
        llm = self._llm
        cache_path = self._storage.artifact_path(asset.project_id, "analysis", "comparisons.jsonl")
        cache = _load_cache(cache_path)
        self._frames.clear()

        points: dict[str, list[float]] = {c.id: [] for c in pool}
        wins: dict[tuple[str, str], int] = {}  # (winner, loser) -> count
        rng = random.Random(COMPARE_VERSION)  # deterministic grouping => cache hits on re-runs
        groups_per_round = max(1, round(len(pool) / GROUP_SIZE))
        total_calls = groups_per_round * ROUNDS
        done = 0
        for rnd in range(ROUNDS):
            if rnd == 0:
                order = sorted(pool, key=lambda c: c.range.start_s)
                rng.shuffle(order)
            else:
                # Swiss: group moments with similar standing so the top gets contested
                order = sorted(pool, key=lambda c: -_mean(points[c.id]))
            for group in _chunks(order, GROUP_SIZE):
                done += 1
                progress(done / total_calls, f"comparing round {rnd + 1}/{ROUNDS}")
                key = self._key(asset, llm.info.name, group)
                ranking = cache.get(key)
                if ranking is not None:
                    report.cached += 1
                else:
                    try:
                        ranking, tin, tout = self._compare(asset, group, llm)
                    except Exception as e:  # a failed group just contributes no points
                        log.warning("comparison failed: %s", e)
                        report.failed += 1
                        continue
                    report.calls += 1
                    report.input_tokens += tin
                    report.output_tokens += tout
                    self._append_cache(cache_path, key, ranking)
                n = len(group)
                for pos, idx in enumerate(ranking):
                    points[group[idx].id].append((n - 1 - pos) / (n - 1))
                    for lower in ranking[pos + 1 :]:
                        pair = (group[idx].id, group[lower].id)
                        wins[pair] = wins.get(pair, 0) + 1

        strength = bradley_terry([c.id for c in pool], wins)
        for c in pool:
            if points[c.id]:
                c.compare_score = round(strength[c.id], 4)
        return report

    def _compare(
        self, asset: MediaAsset, group: list[HighlightCandidate], llm: LLMProvider
    ) -> tuple[list[int], int, int]:
        images: list[ImageInput] = []
        lines = []
        for i, c in enumerate(group):
            letter = _LETTERS[i]
            for k, jpg in enumerate(self._moment_frames(asset, c)):
                images.append(
                    ImageInput(0.0, jpg, f"Moment {letter}, frame {k + 1}/{FRAMES_PER_MOMENT}")
                )
            j = c.judgement
            assert j is not None
            tags = f" [{', '.join(j.tags)}]" if j.tags else ""
            lines.append(
                f"Moment {letter} ({c.clip_range.duration_s:.0f}s): {j.what_happens}{tags}"
            )
        letters = ", ".join(_LETTERS[: len(group)])
        prompt = (
            "\n".join(lines)
            + f"\n\nRank moments {letters} from best to worst as standalone shorts."
        )
        resp = llm.complete(
            LLMRequest(
                system=SYSTEM,
                prompt=prompt,
                images=images,
                json_schema=schema(len(group)),
                max_tokens=300,
            )
        )
        d = json.loads(resp.text)
        order = [str(x).strip().upper()[:1] for x in d.get("ranking", [])]
        expected = set(_LETTERS[: len(group)])
        if set(order) != expected or len(order) != len(group):
            raise ValueError(f"invalid ranking {order}")
        return [_LETTERS.index(x) for x in order], resp.input_tokens, resp.output_tokens

    def _moment_frames(self, asset: MediaAsset, c: HighlightCandidate) -> list[bytes]:
        # Build-up, peak, payoff: enough to tell moments apart at a glance
        if c.id not in self._frames:
            r = c.clip_range
            peak = min(max(c.peak_s, r.start_s), r.end_s)
            times = [r.start_s + (peak - r.start_s) * 0.5, peak, peak + (r.end_s - peak) * 0.6]
            self._frames[c.id] = self._sampler.sample(asset.source_uri, times, FRAME_WIDTH)
        return self._frames[c.id]

    @staticmethod
    def _key(asset: MediaAsset, provider: str, group: list[HighlightCandidate]) -> str:
        spans = [(round(c.range.start_s, 1), round(c.range.end_s, 1)) for c in group]
        return stage_key("compare", COMPARE_VERSION, provider, asset.fingerprint, spans)

    def _append_cache(self, path: Path, key: str, ranking: list[int]) -> None:
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "ranking": ranking}) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _load_cache(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                out[rec["key"]] = [int(x) for x in rec["ranking"]]
            except (ValueError, KeyError, TypeError):
                continue
    return out


def _chunks(items: list[HighlightCandidate], size: int) -> list[list[HighlightCandidate]]:
    """Groups of `size`; a short tail is merged into the previous group (max size+1)."""
    groups = [items[i : i + size] for i in range(0, len(items), size)]
    if len(groups) > 1 and len(groups[-1]) < 2:
        groups[-2].extend(groups.pop())
    return groups


def bradley_terry(
    ids: list[str], wins: dict[tuple[str, str], int], iters: int = 200
) -> dict[str, float]:
    """Strength from head-to-head results, crediting wins over strong moments more than
    wins over weak ones (plain average placing punishes moments grouped with the best).
    Returned as 0..1 percentile of strength within this video."""
    p = dict.fromkeys(ids, 1.0)
    total_wins = {i: 0.5 for i in ids}  # small prior keeps winless moments finite
    for (w, _), k in wins.items():
        total_wins[w] += k
    games: dict[tuple[str, str], int] = {}
    for (w, lo), k in wins.items():
        key = (w, lo) if w < lo else (lo, w)
        games[key] = games.get(key, 0) + k
    for _ in range(iters):
        new = {}
        for i in ids:
            denom = 0.0
            for (a, b), n in games.items():
                if i in (a, b):
                    other = b if a == i else a
                    denom += n / (p[i] + p[other])
            new[i] = total_wins[i] / denom if denom else p[i]
        norm = sum(new.values()) / len(new)
        p = {i: v / norm for i, v in new.items()}
    order = sorted(ids, key=lambda i: p[i])
    return {i: r / max(len(ids) - 1, 1) for r, i in enumerate(order)}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.5
