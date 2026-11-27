"""A fixed set of narration scenes for comparing voice engines: every scene type, names
from different games, delivery tags. Original lines, written for this benchmark."""

from __future__ import annotations

from typing import Any

SCENES: list[dict[str, Any]] = [
    {"id": "emotional", "kind": "Emotional dialogue", "game": "gow",
     "text": "[sad, slow] He didn't say goodbye. He just stood there... and for once, Kratos "
             "let Atreus see how much it hurt."},
    {"id": "boss", "kind": "Boss fight", "game": "gow",
     "text": "[epic] Thor doesn't hold back this time. Every swing of Mjölnir shakes the ice, "
             "and Kratos has maybe one opening left."},
    {"id": "intro", "kind": "Character introduction", "game": "gow",
     "text": "[warm] Meet Mímir. Smartest man in the nine realms, according to Mímir. He's also "
             "just a head, which somehow doesn't slow him down."},
    {"id": "lore", "kind": "Lore narration", "game": "gow",
     "text": "[mysterious] Long before Ragnarök, Jörmungandr was flung back in time. That's why "
             "the World Serpent already knows how this story ends."},
    {"id": "suspense", "kind": "Suspense", "game": "pbz",
     "text": "[tense, slow] The hall goes quiet. Soul doesn't draw his blade yet... he's "
             "counting the footsteps behind the door."},
    {"id": "funny", "kind": "Humorous scene", "game": "gow",
     "text": "[playful] Brok just insulted Sindri, Freya and a dragon in one sentence. Honestly? "
             "Impressive."},
    {"id": "action", "kind": "Fast-paced action", "game": "pbz",
     "text": "[urgent, fast] Parry. Dodge. Counter. Soul cuts through three of them before the "
             "first one even hits the ground."},
]  # fmt: skip

# What a "pass" means for each benchmark number (shown next to results, never hidden)
NOTES = {
    "latency_s": "Wall time to generate the line on this PC (includes model loading if cold).",
    "rtf": "Generation time divided by audio length: below 1 is faster than real time.",
    "names_ok": "Share of game names speech recognition heard as expected. Recognition also "
                "mishears correct fictional names, so treat this as a lower bound.",
    "consistency": "Average speaker-model similarity between this engine's scenes (same "
                   "narrator expected). Measured, not a judgement of quality.",
    "loudness_spread_lu": "Difference between the quietest and loudest scene.",
    "rating": "Your own 1-5 rating while listening: the only naturalness score here.",
}  # fmt: skip
