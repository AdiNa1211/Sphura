"""Captions, titles, descriptions and hashtags for a Short.

The model sees frames from the moment, any in-game dialogue and the game name, and has to
write about what is actually on screen (the weapon, the enemy, the place, the move). Stock
hype ("a massive threat emerges", "the ultimate strike") is rejected and retried once; if it
still comes back generic, the on-screen text is left out rather than shipped. The user can
edit or remove any of it later (publish["edited"] protects their text from being rewritten).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sphura.domain.models import AIJudgement
from sphura.pipeline.judge import guard_claims
from sphura.ports.ai import ImageInput, LLMProvider, LLMRequest

log = logging.getLogger(__name__)

COPY_VERSION = 3  # bump to rewrite copy the user hasn't edited (v3: meme setup/punchline)

SYSTEM = """You write the on-screen text and post copy for a gaming YouTube Short.
Look at the frames. Write about what is really there: the weapon, the enemy's look, the
place, the move that happens, what the character says. Be specific to THIS moment.
Rules:
- Never invent events, numbers, or names. Use a character, boss or place name only if it
  appears in the dialogue or on-screen text you were given.
- No claims like "one hit", "no damage", "first try", "flawless".
- Banned: generic trailer hype that could fit any game, e.g. "a massive threat emerges",
  "the ultimate strike", "an epic battle", "unleash", "unstoppable", "insane", "you won't
  believe", "legendary", "showdown", "chaos".
Good examples of the style (do not copy them): "He brought chains to a sword fight",
"Parried the axe at the last second", "Why is the puppet still smiling", "The snow pier
was a trap". Casual and punchy, like a real gamer posting."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reaction": {"type": "string", "description": "<=8 words: a relatable reaction or "
                     "question about this exact moment"},
        "hype": {"type": "string", "description": "<=7 words: a bold, specific statement "
                 "about what happens (for an all-caps opener)"},
        "title": {"type": "string", "description": "<=60 chars, specific, no game name, "
                  "no hashtags"},
        "setup": {"type": "string", "description": "<=9 words: meme setup shown BEFORE the "
                  "moment, e.g. 'When you think you can tank one more hit' (specific)"},
        "punchline": {"type": "string", "description": "<=6 words: meme punchline shown AS "
                      "it happens, the reaction to the setup"},
        "emoji": {"type": "string", "description": "one or two emoji fitting the mood"},
        "description": {"type": "string", "description": "2-3 short sentences describing "
                        "what happens, ending with a question to the viewer"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "4-6 topic "
                 "hashtags without '#', lowercase, no spaces"},
    },
    "required": ["reaction", "hype", "setup", "punchline", "title", "emoji", "description",
                 "tags"],
}  # fmt: skip

# Stock phrases that fit any game; the reason the user asked for specific text
_GENERIC = re.compile(
    r"\b(massive|ultimate|epic|legendary|unleash\w*|unstoppable|insane|showdown|chaos|"
    r"intense (battle|fight|combat)|emerges?|you won'?t believe|mind[- ]?blowing|"
    r"jaw[- ]?dropping|next level|godlike|beast mode|absolute (cinema|madness))\b",
    re.IGNORECASE,
)
_TEXT_FIELDS = ("reaction", "hype", "setup", "punchline", "title")


def is_generic(text: str) -> bool:
    return bool(_GENERIC.search(text))


def write_copy(
    llm: LLMProvider | None,
    judgement: AIJudgement | None,
    game: str | None,
    frames: list[ImageInput] | None = None,
    dialogue: list[str] | None = None,
) -> dict[str, Any]:
    copy = _fallback(judgement)
    if llm is not None and judgement is not None:
        prompt = _prompt(judgement, game, dialogue or [])
        for attempt in range(2):
            try:
                resp = llm.complete(
                    LLMRequest(
                        system=SYSTEM,
                        prompt=prompt,
                        images=frames or [],
                        json_schema=SCHEMA,
                        max_tokens=400,
                    )
                )
                raw = json.loads(resp.text)
            except Exception as e:
                log.warning("could not write Short copy with %s: %s", llm.info.name, e)
                break
            cleaned = _clean(raw)
            bad = [str(raw.get(k, "")) for k in _TEXT_FIELDS if k not in cleaned]
            copy.update(cleaned)
            if not bad or attempt == 1:
                break
            prompt += (
                "\n\nYour last answer was too generic or made a claim the footage can't "
                f"back up: {'; '.join(bad)}. Rewrite those lines about the specific things "
                "visible in the frames."
            )
    return _finish(copy, game)


def _prompt(j: AIJudgement, game: str | None, dialogue: list[str]) -> str:
    lines = [
        f"Game: {game or 'unknown (do not name it)'}",
        f"What happens (from the judge): {j.what_happens}",
        f"Why it stands out: {j.reason}",
        f"Kind of moment: {j.category.value if j.category else 'unknown'}",
    ]
    if j.tags:
        lines.append(f"Details: {', '.join(j.tags)}")
    if dialogue:
        lines.append("Dialogue heard in the moment (exact, may name characters):")
        lines += [f'  "{d}"' for d in dialogue[:6]]
    return "\n".join(lines)


def _clean(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("reaction", "hype", "setup", "punchline", "title", "description"):
        text = re.sub(r"#\w+", "", str(raw.get(key, ""))).strip().strip('"')
        if not text or not guard_claims(text):
            continue
        if key in _TEXT_FIELDS and is_generic(text):
            continue
        out[key] = text
    emoji = str(raw.get("emoji", "")).strip()
    if emoji and not re.search(r"[A-Za-z0-9]", emoji):
        out["emoji"] = emoji[:4]
    tags = [re.sub(r"[^a-z0-9]", "", str(t).lower()) for t in raw.get("tags", [])]
    out["tags"] = [t for t in tags if 2 < len(t) <= 30][:6]
    return out


def _fallback(j: AIJudgement | None) -> dict[str, Any]:
    """Without good AI copy, prefer plain facts over hype: the on-screen lines stay empty
    (no text is better than generic text) and the title comes from what happens."""
    what = (j.what_happens if j else "").strip().rstrip(".")
    title = j.title if j and j.title and not is_generic(j.title) else ""
    if not title and what:
        title = " ".join(what.split()[:8])
    return {
        "reaction": "",
        "hype": "",
        "setup": "",
        "punchline": "",
        "title": title or "Gameplay moment",
        "emoji": "",
        "description": what + ("." if what else ""),
        "tags": ["gaming", "gameplay"],
    }


def _finish(copy: dict[str, Any], game: str | None) -> dict[str, Any]:
    game_tag = re.sub(r"[^a-z0-9]", "", game.lower()) if game else ""
    tags = [game_tag] if game_tag else []
    tags += [t for t in copy.get("tags", []) if t not in tags]
    tags += [t for t in ("shorts", "gaming") if t not in tags]
    title = copy["title"].rstrip(" .")
    emoji = f" {copy['emoji']}" if copy.get("emoji") else ""
    copy["hype"] = copy["hype"].upper().rstrip(".")
    copy["youtube_title"] = (f"{title}{emoji}" + (f" - {game}" if game else ""))[:100]
    copy["hashtags"] = " ".join(f"#{t}" for t in tags[:8])
    copy["game"] = game or ""
    copy["version"] = COPY_VERSION
    return copy


def on_screen_text(publish: dict[str, Any], style: str) -> str:
    """The line a style shows on screen ("" = none). An edited "text" always wins."""
    if "text" in publish:
        return str(publish["text"])
    return str(publish.get("hype" if style == "cinematic" else "reaction", ""))


def render_txt(copy: dict[str, Any], credit: str = "") -> str:
    music = f"\n\nMusic: {credit}" if credit else ""
    return (
        f"TITLE\n{copy.get('youtube_title', '')}\n\n"
        f"DESCRIPTION\n{copy.get('description', '')}{music}\n\n{copy.get('hashtags', '')}\n"
    )
