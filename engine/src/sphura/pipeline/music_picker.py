"""The AI music supervisor: matches a Short's moment to a track from the user's folder.

It sees what happens (from the judge), the edit style and each track's measured profile
(tempo, dark/bright, build-ups, drop, the user's tags), returns a pick with a short reason
and two alternatives, and says what to search for when nothing in the folder fits well.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sphura.domain.models import AIJudgement
from sphura.pipeline.effects import MusicTrack
from sphura.ports.ai import LLMProvider, LLMRequest

log = logging.getLogger(__name__)

MAX_TRACKS = 25  # keeps the prompt small; the rest are rarely better fits

SYSTEM = """You are the music supervisor for gaming YouTube Shorts.
Pick the track from the list whose feel fits the moment and the edit style best.
Match energy and mood: tense build-ups and boss fights suit dark, heavy, bassy tracks with
big drops; fast chaotic fights suit high-tempo tracks; funny or calm moments suit lighter,
steadier tracks. Judge only from the descriptions given. Keep the reason short and concrete
(tempo, tone, how it fits the moment). If no track fits well, say so and suggest what to
search for in a royalty-free library (genre, mood, tempo range)."""

STYLES = {
    "hype": "punchy edit with zooms, shake, slow motion on the key hit; music ducks under the game",
    "cinematic": "music-led slowed, moody edit (phonk / dark trap style); game audio under it",
    "meme": "short funny moment with a caption; music is light background under the game audio",
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pick": {"type": "integer", "description": "number of the best track"},
        "reason": {"type": "string", "description": "<=20 words, why it fits"},
        "alternatives": {"type": "array", "items": {"type": "integer"},
                         "description": "up to 2 other good track numbers"},
        "fits_well": {"type": "boolean"},
        "search": {"type": "string", "description": "<=8 words to search a royalty-free "
                   "library for an ideal track, e.g. 'dark cinematic phonk 120-130 bpm'"},
    },
    "required": ["pick", "reason", "alternatives", "fits_well", "search"],
}  # fmt: skip


def library_key(tracks: list[MusicTrack]) -> str:
    """Changes whenever tracks or their tags change, so a pick is redone then."""
    return str(hash(tuple(sorted((t.path, t.tags) for t in tracks))))


def pick_music(
    llm: LLMProvider | None,
    judgement: AIJudgement | None,
    style: str,
    duration_s: float,
    tracks: list[MusicTrack],
) -> dict[str, Any] | None:
    if llm is None or judgement is None or not tracks:
        return None
    shortlist = sorted(tracks, key=lambda t: -t.energy)[:MAX_TRACKS]
    lines = [
        f"Moment: {judgement.what_happens}",
        f"Why it stands out: {judgement.reason}",
        f"Kind: {judgement.category.value if judgement.category else 'unknown'}"
        + (f"; details: {', '.join(judgement.tags)}" if judgement.tags else ""),
        f"Edit style: {STYLES.get(style, style)}",
        f"Short length: {duration_s:.0f}s",
        "Tracks:",
        *[f"{i + 1}. {t.profile()}" for i, t in enumerate(shortlist)],
    ]
    try:
        resp = llm.complete(
            LLMRequest(system=SYSTEM, prompt="\n".join(lines), json_schema=SCHEMA, max_tokens=250)
        )
        d = json.loads(resp.text)
        pick = int(d["pick"]) - 1
        if not 0 <= pick < len(shortlist):
            return None
        alts = [int(a) - 1 for a in d.get("alternatives", [])]
        alts = [a for a in dict.fromkeys(alts) if 0 <= a < len(shortlist) and a != pick][:2]
        return {
            "path": shortlist[pick].path,
            "reason": str(d.get("reason", "")).strip()[:160],
            "alternatives": [shortlist[a].path for a in alts],
            "fits_well": bool(d.get("fits_well", True)),
            "search": str(d.get("search", "")).strip()[:80],
            "library": library_key(tracks),
        }
    except Exception as e:
        log.warning("AI music pick failed with %s: %s", llm.info.name, e)
        return None
