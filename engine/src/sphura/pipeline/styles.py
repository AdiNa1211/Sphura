"""Short-form styles, modelled on what performs for this kind of footage:

- hype: punchy creator edit (cold open, zooms, shake, speed ramps, SFX), 9:16
- meme: one moment, ~10 s, a white caption bar with a reaction line on top, game audio only,
  3:4 (the format of the most-viewed Phantom Blade Zero Short at the time of writing)
- cinematic: ~20 s, square, desaturated high-contrast grade, pixel-glitch transitions and
  accents on the music's beat, a short all-caps line at the start
"""

from __future__ import annotations

from dataclasses import dataclass

from sphura.domain.models import AspectRatio, Clip, TimeRange

STYLES = ("hype", "meme", "cinematic")
MEME_BAR_H = 300


@dataclass(frozen=True, slots=True)
class Geometry:
    out_w: int
    out_h: int
    video_h: int  # height of the footage area (below any caption bar)
    bar_h: int = 0

    @property
    def video_aspect(self) -> float:
        return self.out_w / self.video_h


def style_aspect(style: str, requested: AspectRatio) -> AspectRatio:
    return {"meme": AspectRatio.PORTRAIT, "cinematic": AspectRatio.SQUARE}.get(style, requested)


def geometry(clip: Clip, frame_w: int, frame_h: int) -> Geometry:
    if clip.style == "meme":
        return Geometry(1080, 1440, 1440 - MEME_BAR_H, MEME_BAR_H)
    size = {
        AspectRatio.VERTICAL: (1080, 1920),
        AspectRatio.SQUARE: (1080, 1080),
        AspectRatio.PORTRAIT: (1080, 1440),
    }.get(clip.aspect, (frame_w, frame_h))
    if clip.publish.get("caption_bar"):  # meme-style white caption bar on top
        bar = round(size[1] * 0.18) // 2 * 2
        return Geometry(size[0], size[1], size[1] - bar, bar)
    return Geometry(size[0], size[1], size[1])


# (seconds before the payoff, seconds after it): meme lands the payoff late, like a punchline;
# cinematic builds for longer and lets it breathe
_WINDOW = {"meme": (14.0, 6.0), "cinematic": (11.0, 8.0)}  # meme: setup, then payoff


def window_after(style: str) -> float:
    """Seconds a style keeps after its key moment."""
    return _WINDOW.get(style, (0.0, 0.0))[1]


CUT_VERSION = 3  # bump when cut rules change, so older cuts are re-made (v3: length = limit)

# Seconds a style adds on top of the source footage (teaser, slow-mo, freeze-frame)
_EXTRA_S = {"hype": 2.0, "cinematic": 1.0, "meme": 0.6}
_SHARE_BEFORE = {"hype": 0.65, "meme": 0.7, "cinematic": 0.58}


def fit_length(
    segments: list[TimeRange], key: float, style: str, length_s: float, source_s: float
) -> list[TimeRange]:
    """length_s is a limit: a moment that already fits keeps its own length; a longer one
    is cut to fit, around the key moment like the style's own window."""
    if not segments:
        return segments
    target = max(3.0, length_s - _EXTRA_S.get(style, 1.0))
    share = _SHARE_BEFORE.get(style, 0.65)
    if sum(s.duration_s for s in segments) <= target:
        return segments
    lo, hi = segments[0].start_s, segments[-1].end_s
    a = min(max(key - target * share, lo), max(lo, hi - target))
    out, left = [], target
    for seg in segments:  # walk the parts from a, keeping jump cuts, until target is used
        s0 = max(seg.start_s, a)
        if s0 >= seg.end_s or left <= 0:
            continue
        e0 = min(seg.end_s, s0 + left)
        if e0 - s0 >= 0.5:
            out.append(TimeRange(s0, e0))
            left -= e0 - s0
    return out or segments


def trim_for_style(segments: list[TimeRange], key: float, style: str) -> list[TimeRange]:
    if style not in _WINDOW:
        return segments
    before, after = _WINDOW[style]
    a, b = key - before, key + after
    out = [TimeRange(max(s.start_s, a), min(s.end_s, b)) for s in segments]
    out = [s for s in out if s.duration_s >= 0.8]
    return out or segments
