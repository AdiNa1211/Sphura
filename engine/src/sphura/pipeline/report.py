"""Self-contained HTML chart of a SignalTimeline, for eyeballing where the action is."""

from __future__ import annotations

import html

import numpy as np

from sphura.pipeline.signals import Peak
from sphura.pipeline.timeline import Curve, SignalTimeline

_W, _H, _PAD = 1400, 260, 40
_CSS = (
    "body{font:14px system-ui,sans-serif;margin:24px;color:#222}"
    "svg text{font-size:11px;fill:#555}"
    "table{border-collapse:collapse;margin-top:16px}"
    "td,th{padding:4px 12px;border-bottom:1px solid #eee;text-align:left}"
    ".k span{display:inline-block;width:14px;height:3px;margin:0 6px 3px 14px;"
    "vertical-align:middle}"
)
_LEGEND = (
    "Preliminary interest curve (Phase 4) &mdash;"
    '<span style="background:#2b59c3"></span>interest'
    '<span style="background:#f2a541"></span>loudness spikes'
    '<span style="background:#5fad56"></span>motion &nbsp;| '
    "grey ticks = scene cuts, red dots = top peaks"
)


def _hms(t: float) -> str:
    s = int(t)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _downsample(x: Curve, n: int) -> Curve:
    if x.size <= n:
        return x
    k = int(np.ceil(x.size / n))
    pad = (-x.size) % k
    return np.nanmax(np.pad(x, (0, pad), mode="edge").reshape(-1, k), axis=1).astype(np.float32)


def _path(y: Curve, lo: float, hi: float) -> str:
    xs = np.linspace(_PAD, _W - _PAD, y.size)
    ys = _H - _PAD - (np.clip(y, lo, hi) - lo) / (hi - lo or 1) * (_H - 2 * _PAD)
    return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys, strict=True))


def render_html(title: str, tl: SignalTimeline, peaks: list[Peak]) -> str:
    interest = _downsample(np.nan_to_num(tl.features["interest"]), 1400)
    loud = _downsample(np.nan_to_num(tl.features.get("z_loud_rel", np.zeros(1, np.float32))), 1400)
    motion = _downsample(np.nan_to_num(tl.features.get("z_motion", np.zeros(1, np.float32))), 1400)
    lo, hi = float(np.percentile(interest, 1)), float(interest.max() or 1)
    dur = tl.duration_s or 1.0

    def x_of(t: float) -> float:
        return _PAD + t / dur * (_W - 2 * _PAD)

    cuts = np.flatnonzero(tl.features.get("scene_cut", np.zeros(0)) > 0) * tl.hop_s
    marks = "".join(
        f'<line x1="{x_of(t):.1f}" x2="{x_of(t):.1f}" y1="{_H - _PAD}" y2="{_H - _PAD + 6}" '
        f'stroke="#999"/>'
        for t in cuts
    )
    ticks = "".join(
        f'<text x="{x_of(t):.1f}" y="{_H - 12}" text-anchor="middle">{_hms(t)}</text>'
        for t in np.linspace(0, dur, 9)
    )
    pk = ""
    rows = ""
    for i, p in enumerate(peaks, 1):
        y = _H - _PAD - (min(p.score, hi) - lo) / (hi - lo or 1) * (_H - 2 * _PAD)
        pk += (
            f'<circle cx="{x_of(p.t_s):.1f}" cy="{y:.1f}" r="5" fill="#e4572e"/>'
            f'<text x="{x_of(p.t_s):.1f}" y="{y - 9:.1f}" text-anchor="middle" '
            f'font-weight="600">{i}</text>'
        )
        rows += (
            f"<tr><td>{i}</td><td>{_hms(p.t_s)}</td><td>{p.score:.2f}</td>"
            f"<td>{html.escape(', '.join(p.drivers) or '-')}</td></tr>"
        )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head>
<body><h2>{html.escape(title)}</h2>
<p class="k">{_LEGEND}</p>
<svg viewBox="0 0 {_W} {_H}" width="100%">
<path d="{_path(loud, -2, 6)}" fill="none" stroke="#f2a541" stroke-width="1" opacity=".45"/>
<path d="{_path(motion, -2, 6)}" fill="none" stroke="#5fad56" stroke-width="1" opacity=".45"/>
<path d="{_path(interest, lo, hi)}" fill="none" stroke="#2b59c3" stroke-width="1.6"/>
{marks}{pk}{ticks}</svg>
<table><tr><th>#</th><th>time</th><th>score</th><th>why</th></tr>{rows}</table></body></html>"""
