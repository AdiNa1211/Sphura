"""JSON form of a Clip's edit decision list (stored in the DB, served by the API)."""

from __future__ import annotations

from typing import Any

from sphura.domain.models import (
    AspectRatio,
    AudioMix,
    CaptionWord,
    Clip,
    CropKeyframe,
    TextOverlay,
    TimeRange,
)


def clip_to_dict(c: Clip) -> dict[str, Any]:
    d = {
        "id": c.id,
        "project_id": c.project_id,
        "asset_id": c.asset_id,
        "title": c.title,
        "segments": [[s.start_s, s.end_s] for s in c.segments],
        "aspect": c.aspect.value,
        "layout": c.layout,
        "style": c.style,
        "publish": c.publish,
        "crop_path": [[k.t_s, k.cx, k.cy, k.zoom] for k in c.crop_path],
        "crop_version": c.crop_version,
        "captions": [[w.text, w.start_s, w.end_s, w.emphasis] for w in c.captions],
        "caption_style": c.caption_style,
        "overlays": [
            [o.text, o.start_s, o.end_s, o.style, o.y, o.size, o.color, o.owner] for o in c.overlays
        ],
        "audio": {
            "game_gain": c.audio.game_gain,
            "voice_gain": c.audio.voice_gain,
            "music_gain": c.audio.music_gain,
            "sfx_gain": c.audio.sfx_gain,
        },
        "effects": c.effects,
        "candidate_id": c.candidate_id,
        "rank": c.rank,
        "version": c.version,
        "duration_s": round(c.duration_s, 3),
    }
    if c.segment_audio:  # only when set, so untouched Shorts store exactly what they did
        d["segment_audio"] = c.segment_audio
    return d


def clip_from_dict(d: dict[str, Any]) -> Clip:
    return Clip(
        id=d["id"],
        project_id=d["project_id"],
        asset_id=d["asset_id"],
        title=d["title"],
        segments=[TimeRange(a, b) for a, b in d["segments"]],
        aspect=AspectRatio(d["aspect"]),
        layout=d.get("layout", "auto"),
        style=d.get("style", "hype"),
        publish=d.get("publish", {}),
        crop_path=[CropKeyframe(*k) for k in d.get("crop_path", [])],
        crop_version=d.get("crop_version", 0),
        captions=[CaptionWord(*w) for w in d.get("captions", [])],
        caption_style=d.get("caption_style", "bold_pop"),
        overlays=[TextOverlay(*o) for o in d.get("overlays", [])],
        audio=AudioMix(**d.get("audio", {})),
        effects=d.get("effects", []),
        candidate_id=d.get("candidate_id"),
        rank=d.get("rank"),
        version=d.get("version", 1),
        segment_audio=d.get("segment_audio", []),
    )
