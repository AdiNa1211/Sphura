"""Formats that get views: suggest Shorts from the user's own moments in the formats the
research shows working for the game, then make the approved ones with the Shorts editor."""

from __future__ import annotations

import logging
import math
from typing import Any

from sphura.application.clips import ClipService
from sphura.application.queries import QueryService
from sphura.application.research import ResearchService
from sphura.domain.errors import SphuraError
from sphura.domain.models import Clip, HighlightTier
from sphura.pipeline import research as R
from sphura.pipeline import trending as T
from sphura.ports.ai import LLMProvider

log = logging.getLogger(__name__)

MIN_FIT = 0.34  # below this a moment doesn't really fit any format
PER_FORMAT = 3  # suggestions ticked by default per format


class TrendingService:
    def __init__(
        self,
        queries: QueryService,
        clips: ClipService,
        research: ResearchService,
        llm: LLMProvider | None,
    ) -> None:
        self._queries = queries
        self._clips = clips
        self._research = research
        self._llm = llm

    def plan(self, project_id: str, game: str | None = None) -> dict[str, Any]:
        project = self._queries.project(project_id)
        game = game or T.detect_game(project.name)
        formats = T.formats_for(game or "")
        if not formats or game is None:
            raise SphuraError(
                "Formats are researched for God of War Ragnarök and Phantom Blade Zero so "
                "far; choose the game for this stream"
            )
        research = self._research.videos(game)
        ev = {f.id: T.evidence(f, research) for f in formats}
        strength = _strength(ev)
        formats.sort(key=lambda f: (f.makeable, ev[f.id].get("median_views") or 0), reverse=True)
        makeable = [f for f in formats if f.makeable]
        transcript = self._optional(self._queries.transcript, project_id)
        moments: list[dict[str, Any]] = []
        for r in self._queries.highlights(project_id, None):
            c, j = r.candidate, r.candidate.judgement
            if j is None or (r.tier == HighlightTier.MAYBE and j.quality < 5):
                continue
            lines = _lines(transcript, c.range.start_s, c.range.end_s)
            moments.append({"r": r, "id": c.id, "lines": lines,
                            "text": T.moment_text(j, lines)})  # fmt: skip
        ai = T.ai_fit(self._llm, makeable, moments) if self._llm and moments else {}
        out = []
        for m in moments:
            c, j = m["r"].candidate, m["r"].candidate.judgement
            fits = {f.id: T.keyword_fit(f, j, m["lines"]) for f in makeable}
            # ties go to the format with more evidence behind it
            best = max(makeable, key=lambda f: (fits[f.id], strength[f.id]))
            fit, why, source = fits[best.id], _keyword_why(best, j), "keywords"
            a = ai.get(c.id)
            if a and (a["fit"] >= MIN_FIT or fit < MIN_FIT):
                best = next(f for f in makeable if f.id == a["format"])
                fit = max(a["fit"], fits[best.id] * 0.8)
                why, source = a["why"] or why, "ai"
            if best.needs_dialogue and fits[best.id] == 0 and source == "keywords":
                continue
            if fit < MIN_FIT:
                continue
            score = 0.45 * fit + 0.3 * (j.quality / 10 if j else 0.5) + 0.25 * strength[best.id]
            out.append({
                "candidate_id": c.id, "format": best.id, "fit": round(fit, 2),
                "score": round(score * 100), "why": why, "matched_by": source,
                "title": j.title if j else "", "what": j.what_happens if j else "",
                "quality": j.quality if j else None, "start_s": c.range.start_s,
                "end_s": c.range.end_s, "dialogue": m["lines"][:3],
                "thumb_url": f"/v1/projects/{project_id}/moments/{c.id}/thumb.jpg",
            })  # fmt: skip
        out.sort(key=lambda x: x["score"], reverse=True)
        taken: dict[str, int] = {}
        for x in out:  # more suggestions from formats with stronger evidence
            taken[x["format"]] = taken.get(x["format"], 0) + 1
            cap = PER_FORMAT if strength[x["format"]] >= 0.6 else 2
            great = (x["quality"] or 0) >= 8  # the judge's best moments always make the list
            x["suggested"] = x["fit"] >= 0.5 and (taken[x["format"]] <= cap or great)
        return {
            "game": game,
            "game_name": R.GAMES[game]["name"],
            "research_count": len(research),
            "research_updated": self._research.status()["last_updated"],
            "ai": self._llm.info.name if self._llm else None,
            "formats": [
                {
                    "id": f.id,
                    "label": f.label,
                    "style": f.style,
                    "length_s": f.length_s,
                    "about": f.about,
                    "edit": f.edit,
                    "title_guide": f.title_guide,
                    "evidence": ev[f.id],
                    "strength": round(strength[f.id], 2),
                    "makeable": f.makeable,
                    "caption_bar": f.caption_bar,
                }
                for f in formats
            ],
            "moments": out,
        }

    def make(
        self, project_id: str, game: str, picks: list[dict[str, str]]
    ) -> list[tuple[str, list[Clip]]]:
        """Clips per format for the approved picks: [(format id, clips)]; not rendered."""
        formats = {f.id: f for f in T.formats_for(game) if f.makeable}
        if not formats:
            raise SphuraError("unknown game")
        research = self._research.videos(game)
        by_format: dict[str, list[str]] = {}
        for p in picks:
            if p.get("format") in formats:
                by_format.setdefault(p["format"], []).append(p["candidate_id"])
        made = []
        for fid, ids in by_format.items():
            f = formats[fid]
            ev = T.evidence(f, research)
            keep = ("n", "of", "median_views", "median_ops")
            trend = {"format": f.id, "label": f.label, "game": game,
                     "evidence": {k: ev.get(k) for k in keep}}  # fmt: skip
            hints = {cid: {"copy": T.title_hint(f, ev), "trend": trend,
                           "hashtags": ev.get("hashtags", [])} for cid in ids}  # fmt: skip
            clips = self._clips.generate(
                project_id, len(ids), None, True, style=f.style, game=R.GAMES[game]["name"],
                length_s=f.length_s, caption_bar=True if f.caption_bar else None,
                only=ids, hints=hints,
            )  # fmt: skip
            made.append((fid, clips))
        return made

    @staticmethod
    def _optional(fn: Any, project_id: str) -> Any:
        try:
            return fn(project_id)
        except Exception:
            return None


def _strength(ev: dict[str, dict[str, Any]]) -> dict[str, float]:
    """0-1 per format: rank of its median views among formats with 3+ researched Shorts.
    Formats with fewer examples get a neutral 0.4: too little evidence either way."""
    solid = {k: math.log10(max(e["median_views"], 1)) for k, e in ev.items() if e.get("n", 0) >= 3}
    out = {k: 0.4 for k in ev}
    if len(solid) > 1:
        lo, hi = min(solid.values()), max(solid.values())
        for k, v in solid.items():
            out[k] = 0.15 + 0.85 * (v - lo) / ((hi - lo) or 1)
    return out


def _lines(transcript: Any, start: float, end: float) -> list[str]:
    if transcript is None:
        return []
    return [
        s.text.strip()
        for s in transcript.segments
        if s.end_s > start and s.start_s < end and s.text.strip()
    ]


def _keyword_why(f: T.Format, j: Any) -> str:
    what = (j.what_happens if j else "").split(". ")[0].rstrip(".")
    return f"{f.label}: {what}" if what else f.label
