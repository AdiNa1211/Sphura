"""Text prompts for the AI music model: one per Short.

A style-specific base (genre, instruments, energy) plus a few seeded extras so no two
Shorts ask for the same thing; the local LLM then tailors it to the game and the moment.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from sphura.ports.ai import LLMProvider, LLMRequest

log = logging.getLogger(__name__)

BASE = {
    "phonk": "dark memphis phonk, distorted cowbell melody, heavy 808 bass, punchy trap drums",
    "drift": "drift phonk, fast distorted cowbell riff, booming sliding 808s, hard-hitting drums",
    "drill": "dark drill beat, sliding 808 bass, ominous piano and bells, skittering hi-hats",
    "trap": "dark cinematic trap, deep 808, haunting synth lead, crisp hi-hat rolls",
    "rage": "rage beat, bright detuned supersaw lead, distorted 808, energetic drums",
    "dark": "dark cinematic score, deep brass swells, low strings, slow heavy drums",
    "epic": "epic orchestral battle music, thundering taiko drums, driving staccato strings, choir",
    "tension": "tense dark score, pulsing synth bass, ticking percussion, eerie textures",
    "bounce": "playful upbeat beat, plucky synths, bouncy bass, light claps",
    "funk": "funky groove, slap bass, wah guitar, tight drums",
}
EXTRAS = {
    "phonk": ["vinyl crackle", "chopped vocal-free samples", "aggressive", "gritty", "night drive"],
    "drift": ["adrenaline", "high energy", "relentless", "street racing", "overdriven"],
    "drill": ["menacing", "cold", "minor key", "sparse and heavy", "dark atmosphere"],
    "trap": ["intense", "epic build", "brooding", "cinematic", "hard"],
    "rage": ["hype", "euphoric", "chaotic", "video game energy", "loud"],
    "dark": ["tense and epic", "ominous", "slow build", "brooding", "mythic"],
    "epic": ["heroic", "powerful", "war drums", "norse", "climactic"],
    "tension": ["suspense building", "uneasy", "cold", "creeping", "minimal"],
    "bounce": ["cheeky", "fun", "lighthearted", "quirky", "cartoonish"],
    "funk": ["groovy", "sunny", "playful", "retro", "smooth"],
}
SYSTEM = (
    "You write prompts for an AI music generator that makes INSTRUMENTAL background music "
    "for gaming YouTube Shorts. Given a style, the game and the moment, write one prompt of "
    "8-25 words: genre, 2-4 instruments, mood and energy. Keep the given style's genre. "
    "Fit the game's world when it helps (e.g. norse, samurai, sci-fi). No artist, song or "
    "game names, no lyrics, no vocals. Reply as JSON."
)
SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
}


def base_prompt(mood: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    extras = EXTRAS.get(mood, [])
    picks = [extras[int(i)] for i in rng.choice(len(extras), size=min(2, len(extras)),
                                                replace=False)] if extras else []  # fmt: skip
    return ", ".join([BASE.get(mood, BASE["dark"]), *picks, "instrumental, no vocals"])


def write_music_prompt(
    llm: LLMProvider | None, judgement: Any, game: str | None, mood: str, label: str,
    seed: int,
) -> str:  # fmt: skip
    base = base_prompt(mood, seed)
    if llm is None or judgement is None:
        return base
    lines = [
        f"Style: {label} ({base})",
        f"Game: {game or 'unknown'}",
        f"Moment: {judgement.what_happens}",
        f"Tags: {', '.join(judgement.tags) if judgement.tags else 'none'}",
    ]
    try:
        resp = llm.complete(
            LLMRequest(system=SYSTEM, prompt="\n".join(lines), json_schema=SCHEMA, max_tokens=120)
        )
        text = str(json.loads(resp.text).get("prompt", "")).strip().strip(".")
    except Exception as e:
        log.info("music prompt fell back to the template: %s", e)
        return base
    if not 4 <= len(text.split()) <= 40:
        return base
    return f"{text}, instrumental, no vocals"
