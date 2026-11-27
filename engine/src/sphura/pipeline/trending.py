"""Formats that get views: the kinds of Shorts the research shows working for each game,
how to match the user's own moments to them, and how each format is edited and titled.

A format is a kind of Short (an unexpected moment, a quotable line, a boss fight...). Its
evidence is the researched Shorts whose titles fit it, so every suggestion can point at
real examples. Nothing is copied: titles only borrow structure, never words or edits.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from typing import Any

from sphura.domain.models import AIJudgement
from sphura.ports.ai import LLMProvider, LLMRequest

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Format:
    id: str
    label: str
    style: str  # hype / meme / cinematic: drives cut, effects and music
    length_s: float
    about: str  # what such a moment is (for the AI and the user)
    edit: str  # how it gets edited (shown to the user)
    title_guide: str
    match: str  # regex over the judge's words and the dialogue
    evidence: str  # regex over researched titles
    needs_dialogue: bool = False
    caption_bar: bool = False
    makeable: bool = True  # False: shown as an idea, needs footage a stream doesn't have


# One catalogue for every game; the research decides which types work for which game
CATALOGUE: list[Format] = [
    Format(
        "lore",
        "Lore & story",
        "cinematic",
        45,
        "a moment that reveals story or mythology: who someone is, a prophecy, a god, a "
        "creature, a name, a secret or a truth about the world",
        "Caption bar with the context (who or what this is), then the moment, music-led",
        "A question or fact about the lore, naming the character ('Why...', 'The only...'); "
        "only facts the moment shows",
        r"prophecy|ragnar|loki|odin|\bgods?\b|myth|secret|reveal|truth|identity|ancient|realm"
        r"|giant|jotun|fate|norns|calling him",
        r"\bwhy\b|only being|strongest|explained|lore|secret|existed|every land|special"
        r"|truth|origin|did you know|misconception",
        caption_bar=True,
    ),
    Format(
        "weapon",
        "Weapon & power showcase",
        "hype",
        35,
        "a weapon, gear or power shown off: a special attack, a new weapon, a runic or "
        "elemental ability, an upgrade",
        "Straight to the move, slow motion on the impact, cuts on the beat",
        "Names the weapon or power and what makes it special; one emoji",
        r"weapon|blades?\b|axe|spear|runic|summon|lightning|ability|gear|handles|breastplate"
        r"|fury|elemental|fire|frost|upgrade|wrath",
        r"weapon|blades of chaos|axe|spear|armou?r|build|broken|top \d|moveset",
    ),
    Format(
        "discovery",
        "Hidden detail",
        "cinematic",
        30,
        "something easy to miss: a hidden reaction, a small detail, a discovery, a new area, "
        "an unlock",
        "Caption bar pointing at the detail, then the moment; short so it replays",
        "'Did you catch...' or 'You missed...' about the specific detail",
        r"discover|unlock|hidden|notice|detail|secret|new region|found|mask|easter",
        r"walked past|missed|hidden|easter egg|detail|you didn'?t|\btip\b|trick|catch",
        caption_bar=True,
    ),
    Format(
        "quote",
        "Powerful line",
        "cinematic",
        50,
        "a character delivers a strong, emotional or quotable line: vengeance, consequences, "
        "fate, loss, family, a promise",
        "Open on the line, music-led, captions for every word, hold on the reaction",
        "The strongest line actually spoken, in quotes, nothing invented",
        r"dialogue|monologue|conversation|speaks|speaking|says|line|justice|vengeance"
        r"|consequence|blood debt|promise|somber|advice|confront|prophecy|tells|crying|father"
        r"|son\b",
        r"\"|consequence|remember|must be|tells|confronts|💔|younger self|saves|cries|father"
        r"|\bson\b|emotional|goodbye",
        needs_dialogue=True,
    ),
    Format(
        "unexpected",
        "Funny / unexpected",
        "meme",
        20,
        "something sudden, funny or awkward happens to a character: a fall, a surprise hit, "
        "an interruption, an awkward or funny line",
        "Short setup then the payoff, meme caption, cut right after the reaction",
        "A short reaction about the character, one word in CAPS, ending with 💀 or 😭",
        r"unexpected|sudden|fall|falls|humou?r|funny|awkward|interrupt|surpris|passport"
        r"|calling him|exit",
        r"💀|😭|😂|ruined|thought he was|laugh|funny|british|when (your|you)|pov|meme"
        r"|speechless|ymca|drunk",
    ),
    Format(
        "fail",
        "Brutal fail",
        "meme",
        14,
        "the player gets destroyed: a death screen, a brutal hit, getting cut down",
        "Quick setup, the hit, freeze on the death; very short so it replays",
        "First person and relatable ('How do you even...', 'I got...'), ending 💀 or 😭",
        r"\bdead\b|defeated by|killed by|death|knocked down|heavy hit|sudden defeat|died"
        r"|\bfail",
        r"recover|cut in half|humiliat|itachi|how do you|died|death|\bhard\b|rage",
    ),
    Format(
        "boss",
        "Boss fight",
        "hype",
        45,
        "a boss or giant enemy fight that ends in a big finishing blow",
        "Start mid-fight, fast cuts on the beat, slow motion on the finisher",
        "Who vs who, or a short specific claim about the fight, one emoji",
        r"boss|thor|odin|heimdall|final blow|finish|giant|bear|ancient|zealo|takes? down"
        r"|shatter|mechanical|guardian|robot|red mask",
        r"\bvs\b|fight|boss|kill|beat|🥶|battle|takes? down",
    ),
    Format(
        "style",
        "Stylish combat",
        "hype",
        30,
        "stylish, flowing combat: fast combos, parries and clean kills",
        "Straight into the combo, cuts on the beat, slow motion on the kill, loops back",
        "2-5 words a player would type about the style of the fight, plus 🔥",
        r"swordsmanship|swift|masterful|combo|parry|fast-paced|sword|strike|duel|blade",
        r"aura|edit|goty|unmatched|master|combat style|sword fight|cinematic|🔥",
    ),
    Format(
        "comparison",
        "Comparison with another game",
        "hype",
        40,
        "the same kind of moment shown side by side with another game",
        "Needs footage from the other game too, so SPHURA can't make it from this stream",
        "",
        r"(?!)",
        r"\bvs\.? (sekiro|black myth|ghost|onimusha|wukong|mcu)|compar|which (combat|is better)"
        r"|better than",
        makeable=False,
    ),
    Format(
        "challenge",
        "Challenge / what-if",
        "hype",
        40,
        "gameplay played under a self-imposed rule: no damage, 1v3, a what-if scenario",
        "Needs gameplay recorded for the challenge, so it can't come from a normal stream",
        "",
        r"(?!)",
        r"what if|3v3|1v3|no damage|without|only using",
        makeable=False,
    ),
]


def formats_for(game: str) -> list[Format]:
    return list(CATALOGUE) if game in ("gow", "pbz") else []


def detect_game(name: str) -> str | None:
    n = name.lower()
    if "ragnar" in n or "god of war" in n or "kratos" in n:
        return "gow"
    if "phantom blade" in n or "phantomblade" in n:
        return "pbz"
    return None


_GAME_NAMES = re.compile(r"phantom\s*blade(\s*zero)?|god\s*of\s*war(\s*ragnar\S*)?", re.I)


def evidence(fmt: Format, research: list[dict[str, Any]]) -> dict[str, Any]:
    """Researched Shorts whose titles fit the format: how many, how they did, examples."""
    rx = re.compile(fmt.evidence, re.I)
    hits = [v for v in research if rx.search(_GAME_NAMES.sub(" ", v.get("title", "")))]
    if not hits:
        return {"n": 0, "of": len(research)}
    hits.sort(key=lambda v: v.get("ops") or 0, reverse=True)
    ops = [v["ops"] for v in hits if v.get("ops") is not None]
    views = [v["metrics"]["views"] for v in hits]
    lens = [v["duration_s"] for v in hits[: max(3, len(hits) // 2)]]
    return {
        "n": len(hits),
        "of": len(research),
        "median_ops": round(statistics.median(ops)) if ops else None,
        "median_views": round(statistics.median(views)),
        "median_length_s": round(statistics.median(lens)),
        "examples": [
            {
                "title": v["title"],
                "url": v["url"],
                "views": v["metrics"]["views"],
                "ops": v.get("ops"),
            }
            for v in hits[:3]
        ],
        "hashtags": _hashtags(hits[:20]),
    }


def _hashtags(vids: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for v in vids:
        for h in v.get("hashtags", []):
            if h not in ("#shorts", "#short", "#viral", "#fyp", "#trending"):
                counts[h] = counts.get(h, 0) + 1
    return [h for h, _ in sorted(counts.items(), key=lambda x: -x[1])[:5]]


def moment_text(j: AIJudgement | None, dialogue: list[str]) -> str:
    if j is None:
        return " ".join(dialogue)
    return " ".join(
        [j.what_happens, j.reason, j.title, j.category.value if j.category else "",
         *j.tags, *dialogue]
    )  # fmt: skip


def keyword_fit(fmt: Format, j: AIJudgement | None, dialogue: list[str]) -> float:
    """0-1: how well a moment fits a format, from the judge's words (no AI needed). The
    judge's own title counts extra: it names what the moment is about."""
    if not fmt.makeable or (fmt.needs_dialogue and len(" ".join(dialogue).split()) < 6):
        return 0.0
    text = moment_text(j, dialogue)
    hits = len({m.group(0).lower() for m in re.finditer(fmt.match, text, re.I)})
    if j and j.title and re.search(fmt.match, j.title, re.I):
        hits += 2
    if j and j.category and j.category.value in ("funny", "unexpected") and fmt.id == "unexpected":
        hits += 1
    return min(1.0, hits / 5)


SYSTEM = """You plan YouTube Shorts for a gaming creator. Each moment from their own
gameplay has a description from an AI judge and the dialogue heard. Decide which Short
format fits each moment best, or "none" when nothing fits well. Judge from what really
happens; never assume events that aren't described."""


def ai_fit(
    llm: LLMProvider, formats: list[Format], moments: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """{moment id: {"format", "fit", "why"}} from the language model; {} on any failure."""
    ids = {f.id for f in formats}
    lines = ["Formats:"] + [f'- "{f.id}": {f.about}' for f in formats] + ["", "Moments:"]
    for m in moments:
        lines.append(f'- id "{m["id"]}": {m["text"][:500]}')
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "format": {"type": "string", "enum": [*sorted(ids), "none"]},
                "fit": {"type": "number", "description": "0 to 1"},
                "why": {"type": "string", "description": "<=20 words, specific to the moment"},
            },
            "required": ["id", "format", "fit", "why"],
        }}},
        "required": ["items"],
    }  # fmt: skip
    try:
        resp = llm.complete(
            LLMRequest(
                system=SYSTEM,
                prompt="\n".join(lines),
                json_schema=schema,
                max_tokens=200 + 60 * len(moments),
            )
        )
        items = json.loads(resp.text).get("items", [])
    except Exception as e:
        log.warning("format matching with %s failed: %s", llm.info.name, e)
        return {}
    known = {m["id"] for m in moments}
    out = {}
    for it in items:
        if it.get("id") in known and it.get("format") in ids:
            fit = max(0.0, min(1.0, float(it.get("fit") or 0)))
            out[it["id"]] = {
                "format": it["format"],
                "fit": fit,
                "why": str(it.get("why", ""))[:200],
            }
    return out


def title_hint(fmt: Format, ev: dict[str, Any]) -> str:
    examples = "; ".join(f'"{e["title"][:70]}"' for e in ev.get("examples", []))
    hint = (
        f"This Short uses the '{fmt.label}' format, which is doing well for this game on "
        f"YouTube right now. Title style: {fmt.title_guide}."
    )
    if examples:
        hint += (
            f" Titles of that kind that performed well (for STYLE only; never reuse their "
            f"words, names or claims): {examples}."
        )
    return hint
