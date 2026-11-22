"""Shorts research from public data only: which researched Shorts did unusually well for
their age and channel size, and which metadata patterns the strong ones share.

Observed Performance Score (OPS, 0-100) is relative to the other Shorts of the same game:
  40% reach vs channel size   log(views / max(subscribers, 1,000)) percentile
  40% age-adjusted views      views / sqrt(days since upload) percentile
  20% engagement              (likes + 2 x comments) / views percentile
Hidden likes or subscriber counts drop that part and re-weight the rest. It says nothing
about retention, swipe-away or watch time: YouTube doesn't publish those.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from datetime import datetime
from typing import Any

GAMES: dict[str, dict[str, Any]] = {
    "gow": {
        "name": "God of War Ragnarök",
        "queries": [
            "God of War Ragnarok #shorts",
            "God of War Ragnarok boss fight #shorts",
            "Kratos Atreus Ragnarok #shorts",
            "Kratos vs Thor #shorts",
        ],
        "keywords": ["ragnarok", "ragnarök", "ragnarøk", "gowr"],  # not 2018 / older GoW games
    },
    "pbz": {
        "name": "Phantom Blade Zero",
        "queries": [
            "Phantom Blade Zero #shorts",
            "Phantom Blade Zero boss #shorts",
            "Phantom Blade Zero combat #shorts",
        ],
        "keywords": ["phantom blade", "phantomblade", "phantom blade zero", "pbz"],
    },
    "similar": {
        "name": "Similar action games",
        "queries": [
            "Black Myth Wukong boss #shorts",
            "Sekiro boss fight #shorts",
            "Stellar Blade combat #shorts",
        ],
        "keywords": ["wukong", "sekiro", "stellar blade", "elden ring", "nioh"],
    },
}

MAX_SHORT_S = 180  # YouTube Shorts can be up to 3 minutes
RECENT_DAYS = 30
SMALL_CHANNEL = 50_000
SMALL_REACH = 10  # views per subscriber that counts as unusual reach
SMALL_SAMPLE = 20  # fewer Shorts than this: patterns are hints, not findings


def iso_duration_s(text: str) -> float:
    """ISO 8601 duration ("PT1M5S") to seconds."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", text or "")
    if not m:
        return 0.0
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return float(d * 86400 + h * 3600 + mi * 60 + s)


def is_relevant(game: str, item: dict[str, Any]) -> bool:
    sn = item.get("snippet", {})
    return mentions(game, sn.get("title", ""), sn.get("description", ""), sn.get("tags", []))


def mentions(game: str, title: str, description: str, tags: list[str]) -> bool:
    text = " ".join([title, description, *tags]).lower()
    return any(k in text for k in GAMES[game]["keywords"])


def record(item: dict[str, Any], query: str) -> dict[str, Any]:
    sn, cd = item.get("snippet", {}), item.get("contentDetails", {})
    desc = sn.get("description", "")
    thumbs = sn.get("thumbnails", {})
    thumb = (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url")
    return {
        "url": f"https://www.youtube.com/shorts/{item['id']}",
        "title": sn.get("title", ""),
        "description": desc[:2000],
        "channel_id": sn.get("channelId", ""),
        "channel": sn.get("channelTitle", ""),
        "published_at": sn.get("publishedAt", ""),
        "duration_s": iso_duration_s(cd.get("duration", "")),
        "tags": sn.get("tags", [])[:30],
        "hashtags": sorted(
            {h.lower() for h in re.findall(r"#\w+", sn.get("title", "") + " " + desc)}
        ),
        "language": sn.get("defaultAudioLanguage") or sn.get("defaultLanguage"),
        "thumbnail": thumb,
        "queries": [query],
    }


def stats(item: dict[str, Any], channel: dict[str, Any] | None) -> dict[str, int | None]:
    st = item.get("statistics", {})
    cst = (channel or {}).get("statistics", {})

    def num(d: dict[str, Any], k: str) -> int | None:
        return int(d[k]) if k in d else None

    subs = None if cst.get("hiddenSubscriberCount") else num(cst, "subscriberCount")
    return {
        "views": num(st, "viewCount"),
        "likes": num(st, "likeCount"),
        "comments": num(st, "commentCount"),
        "channel_subs": subs,
    }


def _age_days(published_at: str, now: datetime) -> float:
    try:
        t = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 365.0
    return max((now - t).total_seconds() / 86400, 1.0)


def _pct(values: dict[str, float]) -> dict[str, float]:
    """Percentile rank (0-100) of each value among all of them."""
    if not values:
        return {}
    ordered = sorted(values.values())
    n = len(ordered)
    out = {}
    for k, v in values.items():
        below = sum(1 for x in ordered if x < v)
        equal = sum(1 for x in ordered if x == v)
        out[k] = 100.0 * (below + 0.5 * equal) / n
    return out


def score(videos: list[dict[str, Any]], now: datetime) -> None:
    """Adds metrics, OPS and categories to each video (all of one game) in place."""
    reach: dict[str, float] = {}
    pace: dict[str, float] = {}
    engage: dict[str, float] = {}
    for v in videos:
        s = v.get("latest") or {}
        views = s.get("views") or 0
        age = _age_days(v.get("published_at", ""), now)
        m: dict[str, Any] = {"views": views, "age_days": round(age, 1)}
        m["views_per_day"] = round(views / age)
        if s.get("likes") is not None and views:
            m["like_rate"] = round(s["likes"] / views, 4)
        if s.get("comments") is not None and views:
            m["comment_rate"] = round(s["comments"] / views, 5)
        if s.get("channel_subs") is not None:
            m["views_per_sub"] = round(views / max(s["channel_subs"], 1), 2)
            reach[v["video_id"]] = math.log10(max(views, 1) / max(s["channel_subs"], 1000))
        pace[v["video_id"]] = views / math.sqrt(age)
        if s.get("likes") is not None and views >= 1000:
            engage[v["video_id"]] = (s["likes"] + 2 * (s.get("comments") or 0)) / views
        prev = v.get("previous")
        if prev and prev.get("views") is not None:
            m["views_change"] = views - prev["views"]
        v["metrics"] = m
    pr, pp, pe = _pct(reach), _pct(pace), _pct(engage)
    for v in videos:
        vid = v["video_id"]
        parts = [(pr.get(vid), 0.4), (pp.get(vid), 0.4), (pe.get(vid), 0.2)]
        known = [(x, w) for x, w in parts if x is not None]
        total = sum(w for _, w in known)
        v["ops"] = round(sum(x * w for x, w in known) / total) if total else None
        v["ops_parts"] = {"reach": _r(pr.get(vid)), "pace": _r(pp.get(vid)),
                          "engagement": _r(pe.get(vid))}  # fmt: skip
    _categorise(videos, pp)


def _r(x: float | None) -> int | None:
    return None if x is None else round(x)


def _categorise(videos: list[dict[str, Any]], pace_pct: dict[str, float]) -> None:
    views = sorted((v["metrics"]["views"] for v in videos), reverse=True)
    top10 = views[max(0, len(views) // 10 - 1)] if views else 0
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for v in videos:
        by_channel.setdefault(v.get("channel_id", ""), []).append(v)
    for v in videos:
        m, cats, why = v["metrics"], [], []
        if m["views"] >= 100_000 and m["views"] >= top10:
            cats.append("viral")
            why.append(f"{m['views']:,} views, top 10% of this game's Shorts")
        if m["age_days"] <= RECENT_DAYS and pace_pct.get(v["video_id"], 0) >= 75:
            cats.append("trending")
            why.append(f"uploaded {m['age_days']:.0f} days ago, top 25% for age-adjusted views")
        subs = (v.get("latest") or {}).get("channel_subs")
        if subs is not None and subs < SMALL_CHANNEL and m.get("views_per_sub", 0) >= SMALL_REACH:
            cats.append("small_creator")
            why.append(f"{m['views_per_sub']}x views per subscriber on a {subs:,}-sub channel")
        mine = by_channel.get(v.get("channel_id", ""), [])
        if len(mine) >= 3 and statistics.median(x["ops"] or 0 for x in mine) >= 60:
            cats.append("consistent")
            why.append(f"channel has {len(mine)} researched Shorts with a median OPS of 60+")
        v["categories"], v["category_reasons"] = cats, why


_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")


def title_features(title: str) -> dict[str, bool]:
    words = re.findall(r"[A-Za-z']+", title)
    return {
        "question": "?" in title,
        "number": bool(re.search(r"\d", title)),
        "caps_word": any(len(w) >= 3 and w.isupper() for w in words),
        "emoji": bool(_EMOJI.search(title)),
        "pov_when_this": bool(re.match(r"\s*(pov|when|this|that moment|me when)\b", title, re.I)),
        "exclamation": "!" in title,
        "names_character": bool(
            re.search(r"kratos|atreus|thor|freya|odin|baldur|soul|lei dang|wukong", title, re.I)
        ),
    }


FEATURE_LABELS = {
    "question": "Title asks a question",
    "number": "Title has a number",
    "caps_word": "Title has an ALL-CAPS word",
    "emoji": "Title uses an emoji",
    "pov_when_this": 'Title opens with "POV / When / This"',
    "exclamation": "Title has an exclamation mark",
    "names_character": "Title names a character",
}


def patterns(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Metadata patterns of the top quarter by OPS versus the rest (observed, not causal)."""
    scored = [v for v in videos if v.get("ops") is not None]
    out: dict[str, Any] = {"n": len(scored), "small_sample": len(scored) < SMALL_SAMPLE}
    if len(scored) < 4:
        return out
    scored.sort(key=lambda v: v["ops"], reverse=True)
    k = max(1, len(scored) // 4)
    top, rest = scored[:k], scored[k:]

    def share(group: list[dict[str, Any]], feat: str) -> float:
        return sum(title_features(v["title"])[feat] for v in group) / len(group)

    out["features"] = sorted(
        (
            {
                "feature": f,
                "label": FEATURE_LABELS[f],
                "top": round(share(top, f), 2),
                "rest": round(share(rest, f), 2),
            }
            for f in FEATURE_LABELS
        ),
        key=lambda x: x["top"] - x["rest"],
        reverse=True,
    )
    out["duration"] = {
        "top_median_s": round(statistics.median(v["duration_s"] for v in top)),
        "rest_median_s": round(statistics.median(v["duration_s"] for v in rest)),
    }
    out["title_words"] = {
        "top_median": statistics.median(len(v["title"].split()) for v in top),
        "rest_median": statistics.median(len(v["title"].split()) for v in rest),
    }
    tags = Counter(h for v in top for h in v.get("hashtags", []))
    out["top_hashtags"] = [{"tag": t, "count": c} for t, c in tags.most_common(12)]
    out["top_n"], out["rest_n"] = len(top), len(rest)
    return out
