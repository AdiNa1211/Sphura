"""On-screen text as an ASS subtitle file, rendered by libass (proper fonts on every OS,
outlines and pop-in animation without font-file paths in the filter graph)."""

from __future__ import annotations

import re

from sphura.domain.models import TextOverlay

FONT = "Arial Black"  # ships with Windows and macOS; libass falls back elsewhere
BAR_FONT = "Arial"


def overlays_ass(overlays: list[TextOverlay], width: int, height: int, bar_h: int = 0) -> str:
    """Styles: hook (pop-in, key word in yellow), caps (plain all-caps opener),
    bar (black reaction line centred in the white caption bar)."""
    size = round(min(width, height) * 0.085)
    margin_v = round(height * (0.2 if height > width * 1.5 else 0.1))  # clear of platform UI
    bar_size = round(width * 0.064)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: hook,{FONT},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(3, size // 12)},2,8,{width // 12},{width // 12},"
        f"{margin_v},1",
        f"Style: caps,{FONT},{round(size * 0.9)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,1,0,1,{max(3, size // 12)},0,8,{width // 14},{width // 14},"
        f"{margin_v},1",
        f"Style: bar,{BAR_FONT},{bar_size},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,"
        f"-1,0,0,0,100,100,0,0,1,0,0,5,{width // 16},{width // 16},0,1",
        f"Style: plain,{BAR_FONT},{round(size * 0.7)},&H00FFFFFF,&H00FFFFFF,&H00000000,"
        f"&H64000000,-1,0,0,0,100,100,0,0,1,{max(3, size // 14)},1,2,{width // 14},"
        f"{width // 14},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for o in overlays:
        if o.style == "bar":
            body = rf"{{\pos({width // 2},{bar_h // 2})}}" + _esc(o.text)
        elif o.style == "caps":
            body = r"{\fad(120,250)}" + _esc(o.text).upper()
        elif o.style == "plain":
            body = r"{\fad(80,120)}" + _esc(o.text)
        else:
            # Pop in: start small, overshoot, settle; fade out at the end
            anim = (
                r"{\fad(60,220)\fscx60\fscy60"
                r"\t(0,140,\fscx108\fscy108)\t(140,240,\fscx100\fscy100)}"
            )
            # A chosen colour applies to the whole line instead of the yellow key word
            body = anim + (_esc(o.text).upper() if o.color else _styled(o.text))
        lines.append(f"Dialogue: 0,{_ts(o.start_s)},{_ts(o.end_s)},{o.style},,0,0,0,,"
                     f"{_place(o, width, height, size, bar_size)}{body}")  # fmt: skip
    return "\n".join(lines) + "\n"


STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "is",
        "was", "he", "she", "it", "this", "that", "his", "her", "you", "your", "i", "my",
        "be", "did", "do", "not", "just", "how", "what", "who", "all", "out", "up"}  # fmt: skip


def _styled(text: str) -> str:
    """Uppercase, with the words the AI stressed (written in caps) or else the longest
    meaningful word in yellow, the way short-form creators emphasise a hook."""
    words = _esc(text).split()
    caps = {i for i, w in enumerate(words) if len(w) > 1 and w.isupper()}
    if not caps:
        content = [(len(w.strip(".,!?")), i) for i, w in enumerate(words)
                   if w.lower().strip(".,!?") not in STOP]  # fmt: skip
        caps = {max(content)[1]} if content else set()
    return " ".join(
        (r"{\c&H00D7FF&}" + w.upper() + r"{\c&HFFFFFF&}") if i in caps else w.upper()
        for i, w in enumerate(words)
    )


def base_size(style: str, width: int, height: int) -> float:
    """Font size a style uses before the overlay's own multiplier (also used by the editor)."""
    size = min(width, height) * 0.085
    return {"caps": size * 0.9, "plain": size * 0.7, "bar": width * 0.064}.get(style, size)


def _place(o: TextOverlay, width: int, height: int, size: int, bar_size: int) -> str:
    """Overrides for text moved, resized or recoloured in the timeline editor."""
    tags = ""
    if o.y is not None and o.style != "bar":  # the meme bar keeps its place
        tags += rf"\an5\pos({width // 2},{round(min(max(o.y, 0.02), 0.98) * height)})"
    if abs(o.size - 1.0) > 0.01:
        tags += rf"\fs{round(base_size(o.style, width, height) * o.size)}"
    if o.color and re.fullmatch(r"#[0-9a-fA-F]{6}", o.color):
        r, g, b = o.color[1:3], o.color[3:5], o.color[5:7]
        tags += rf"\1c&H{b}{g}{r}&"
    return "{" + tags + "}" if tags else ""


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", r"\N")


def _ts(t: float) -> str:
    cs = max(0, round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"
