"""Effects that match what happens in the moment (boss finisher, death, funny, story...).

The base plan (effects.plan_effects) anchors every effect to real impacts and cuts; this
module reads the AI's verdict for the moment and reshapes that plan: harder hits for a
finisher, a slow desaturated fall for a death, a freeze-frame punchline for something
funny, letterbox and no shake for a cutscene. Also: the user's effect on/off switches.
"""

from __future__ import annotations

import re
from typing import Any

FX = list[dict[str, Any]]

# Checked in order: the first match wins
KINDS: tuple[tuple[str, str], ...] = (
    ("finisher", r"finish|final blow|execution|boss (kill|down|defeat)|kills? the boss|"
                 r"fatality|takedown|killing blow"),
    ("clutch", r"clutch|close call|last second|parry|perfect dodge|barely|low health|1 ?hp"),
    ("explosion", r"explo|blast|detonat|grenade|nuke"),
    ("death", r"player dies|you died|death|dies\b|killed by|game over|wiped"),
    ("funny", r"funny|laugh|fail|glitch|bug|ragdoll|troll|meme|awkward|unexpected"),
    ("story", r"cutscene|dialogue|story|conversation|reveal|lore|cinematic"),
)  # fmt: skip
LABELS = {
    "finisher": "Finisher: harder hit, white flash, motion trail in the slow-mo",
    "clutch": "Clutch: deeper slow-mo with a motion trail",
    "explosion": "Explosion: big flash, glitch and heavy shake",
    "death": "Death: drained colour, long slow fall, vignette",
    "funny": "Funny: freeze-frame punchline with a zoom",
    "story": "Story: letterbox bars, smooth push-ins, no shake",
    "action": "Action: punch-ins, shake and slow-mo on the hit",
}
# User switches (publish["fx_off"]) -> effect types they remove
TOGGLES = {
    "cold_open": ("cold_open",),
    "zoom": ("punch", "push"),
    "shake": (),  # handled below: keeps the zoom, drops the shake
    "slowmo": ("speed",),
    "flash": ("flash", "glitch", "pixel"),
    "rgb": ("rgb",),
    "trail": ("ghost",),
    "vignette": ("vignette",),
    "letterbox": ("letterbox",),
    "sfx": ("sfx",),
}


def classify(judgement: Any) -> str:
    if judgement is None:
        return "action"
    cat = judgement.category.value if getattr(judgement, "category", None) else ""
    text = " ".join([cat, *getattr(judgement, "tags", ()), judgement.what_happens or ""])
    for kind, pattern in KINDS:
        if re.search(pattern, text, re.IGNORECASE):
            return kind
    return "funny" if cat == "funny" else "story" if cat == "cinematic" else "action"


def key_time(fx: FX) -> float | None:
    """The key hit: the strongest punch-in on the clip timeline."""
    punches = [e for e in fx if e["type"] == "punch"]
    return max(punches, key=lambda e: e["zoom"])["t_s"] if punches else None


def apply_context(fx: FX, kind: str, style: str, total_s: float) -> FX:
    k = key_time(fx)
    out = [e for e in fx if e["type"] != "context"]
    out.append({"type": "context", "kind": kind})
    if k is None:
        return out
    if kind == "finisher":
        for e in out:
            if e["type"] == "punch" and e["t_s"] == k:
                e["zoom"] = round(e["zoom"] + 0.06, 3)
        out += [{"type": "flash", "t_s": k, "dur_s": 0.12},
                {"type": "ghost", "start_s": round(k - 0.25, 3), "end_s": round(k + 0.55, 3)},
                {"type": "sfx", "kind": "boom", "t_s": k, "gain_db": -3}]  # fmt: skip
    elif kind == "clutch":
        for e in out:
            if e["type"] == "speed" and e["factor"] < 0.5:
                e["factor"] = 0.3
        out.append({"type": "ghost", "start_s": round(k - 0.3, 3), "end_s": round(k + 0.6, 3)})
    elif kind == "explosion":
        out += [{"type": "flash", "t_s": k, "dur_s": 0.2},
                {"type": "glitch", "t_s": round(k + 0.05, 3), "dur_s": 0.18}]  # fmt: skip
    elif kind == "death":
        for e in out:
            if e["type"] == "grade":
                e.update(saturation=0.35, contrast=1.15)
            if e["type"] == "punch":
                e["shake"] = False
            if e["type"] == "speed" and e["factor"] < 0.5:
                e["end_s"] = round(e["end_s"] + 0.7, 3)
        out = [e for e in out if e["type"] != "rgb"]
        out.append({"type": "vignette", "start_s": k, "end_s": round(total_s, 3)})
    elif kind == "funny":
        out = [e for e in out if e["type"] not in ("speed", "vignette")]
        for e in out:
            if e["type"] == "punch" and e["t_s"] == k:
                e.update(zoom=1.24, shake=False)
        out += freeze(k)
    elif kind == "story":
        for e in out:
            if e["type"] == "punch":
                e["shake"] = False
        out = [e for e in out if e["type"] not in ("rgb", "glitch", "pixel")]
        out.append({"type": "letterbox", "start_s": 0.0, "end_s": round(total_s, 3)})
    if style == "cinematic":  # its own look stays: no extra shake or colour swaps there
        for e in out:
            if e["type"] == "punch":
                e["shake"] = False
    return out


def freeze(t: float, hold_s: float = 0.6) -> FX:
    """Hold the image for hold_s (a 0.05 s slice slowed right down), with a flash and a hit."""
    src = 0.05  # the renderer ignores shorter speed pieces
    return [{"type": "speed", "start_s": round(t, 3), "end_s": round(t + src, 3),
             "factor": round(src / hold_s, 4), "freeze": True},
            {"type": "flash", "t_s": t, "dur_s": 0.08},
            {"type": "sfx", "kind": "hit", "t_s": t, "gain_db": -4}]  # fmt: skip


def filter_off(fx: FX, off: list[str]) -> FX:
    """Apply the user's effect switches."""
    drop = {t for name in off for t in TOGGLES.get(name, ())}
    out = []
    for e in fx:
        if e["type"] in drop:
            continue
        if "shake" in off and e["type"] == "punch":
            e = {**e, "shake": False}
        out.append(e)
    return out
