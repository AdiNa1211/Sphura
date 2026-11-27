"""Lore Studio logic with no I/O: frame tagging prompts, scenes from frame tags + transcript,
lore passages from wiki pages, and story ideas that must cite both footage and sources."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Any

from sphura.pipeline.judge import CAST

INDEX_VERSION = 1
FRAME_STEP_S = 10.0  # one look every 10 s: scenes in games rarely last less
FRAMES_PER_CALL = 6
FRAME_WIDTH = 512
SCENE_MAX_S = 120.0

KINDS = ("cinematic", "dialogue", "combat", "exploration", "puzzle", "menu", "other")
SKIP_KINDS = {"menu"}  # menus, maps and loading screens are never story footage

# Where lore comes from per game. Fandom and Wikipedia both serve the MediaWiki API.
GAMES: dict[str, dict[str, Any]] = {
    "gow": {
        "name": "God of War Ragnarök",
        "wiki": ("https://godofwar.fandom.com/api.php", "God of War Wiki", "CC BY-SA 3.0"),
        "myth": "Norse mythology",
        "cast": sorted(CAST - {"soul", "leidang", "grandmaster"}),
    },
    "pbz": {
        "name": "Phantom Blade Zero",
        "wiki": None,  # no established community wiki yet: Wikipedia only
        "myth": "",
        "cast": ["soul", "leidang", "grandmaster"],
    },
}
WIKIPEDIA = ("https://en.wikipedia.org/w/api.php", "Wikipedia", "CC BY-SA 4.0")

CATEGORIES = {
    "hidden": "Hidden details & easter eggs",
    "lore": "Lore explained",
    "moment": "Story moments explained",
    "facts": "Interesting facts",
    "foreshadow": "Foreshadowing & connections",
    "character": "Character deep dive",
    "myth": "Myth vs game",
}


def game_of(name: str) -> str | None:
    n = name.lower()
    if "ragnar" in n or "god of war" in n or "kratos" in n:
        return "gow"
    if "phantom blade" in n or "phantomblade" in n:
        return "pbz"
    return None


def plain(s: str) -> str:
    """'Frækni' -> 'fraekni', 'Mímir' -> 'mimir': names compare across spellings."""
    s = s.lower().replace("æ", "ae").replace("ø", "o").replace("ð", "d").replace("þ", "th")
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9 ]", "", "".join(c for c in s if not unicodedata.combining(c))).strip()


def nice(name: str) -> str:
    return " ".join(w if w.isupper() and len(w) <= 3 else w.capitalize() for w in name.split())


# ---------- frame tagging ----------

FRAME_SYSTEM = (
    "You catalogue gameplay footage for a lore video editor. Describe only what is visible "
    "or written on screen in each frame. Name a character only if you recognise them for "
    "sure or the subtitles/HUD name them; otherwise describe them (e.g. 'hooded woman')."
)

FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "characters": {"type": "array", "items": {"type": "string"}},
                    "location": {"type": "string"},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "what": {"type": "string"},
                },
                "required": ["n", "kind", "characters", "location", "objects", "what"],
            },
        }
    },
    "required": ["frames"],
}


def frame_times(duration_s: float) -> list[float]:
    n = int(max(0.0, duration_s - 3.0) // FRAME_STEP_S) + 1
    return [round(2.0 + i * FRAME_STEP_S, 2) for i in range(n)]


def frame_prompt(game: str | None, times: list[float], lines: list[str]) -> str:
    g = GAMES.get(game or "")
    cast = f"Recurring characters: {', '.join(nice(c) for c in g['cast'])}.\n" if g else ""
    heard = "\n".join(lines) or "(nothing said)"
    return (
        f"Game: {g['name'] if g else 'unknown'}.\n{cast}"
        f"{len(times)} frames, 10 s apart. Speech heard around them:\n{heard}\n\n"
        "For each frame give: kind (cinematic = cutscene, letterboxed or no HUD; dialogue = "
        "characters talking during play; combat; exploration; puzzle; menu = menus, maps, "
        "loading or pause screens; other), the characters on screen, the location if it can "
        "be told (e.g. 'Sindri's house', 'snowy forest'), notable objects (weapons, symbols, "
        "murals, runes, statues) and one short sentence of what happens. Number frames from 1."
    )


def parse_frames(data: dict[str, Any], times: list[float]) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    for f in data.get("frames") or []:
        try:
            i = int(f.get("n", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= i < len(times):
            continue
        kind = str(f.get("kind", "other")).lower()
        out[times[i]] = {
            "kind": kind if kind in KINDS else "other",
            "characters": _names(f.get("characters")),
            "location": _short(f.get("location"), 60),
            "objects": [_short(o, 40) for o in (f.get("objects") or [])[:6] if _short(o, 40)],
            "what": _short(f.get("what"), 160),
        }
    for t in times:  # frames the model skipped still count as looked at
        out.setdefault(
            t,
            {
                "kind": "other",
                "characters": [],
                "location": "",
                "objects": [],
                "what": "",
                "missing": True,
            },
        )
    return out


def _short(v: Any, n: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:n]


_VAGUE = re.compile(
    r"\b(man|woman|boy|girl|warrior|enemy|enemies|player|character|figure|person|people|"
    r"creature|soldier|guard|someone|unknown|npc|monster|child|hooded|bearded)\b",
    re.I,
)


def _names(v: Any) -> list[str]:
    """Named characters only; 'bearded man' is a description, not a name to research."""
    out: list[str] = []
    for n in v or []:
        n = _short(n, 40)
        if n and not _VAGUE.search(n) and len(n) > 2 and plain(n) not in {plain(x) for x in out}:
            out.append(n)
    return out[:6]


# ---------- scenes ----------


def canonical(name: str, cast: list[str]) -> str:
    p = plain(name)
    for c in cast:
        if p == c or p.split(" ")[0] == c:
            return nice(c)
    return nice(name)


def build_scenes(
    frames: dict[float, dict[str, Any]],
    transcript: list[dict[str, Any]],
    moments: list[dict[str, Any]],
    game: str | None,
    asset_id: str,
    project_id: str,
) -> list[dict[str, Any]]:
    """Neighbouring frames that show the same kind of footage in the same place with the
    same people make one scene; speech and found highlights inside it are attached."""
    cast = GAMES.get(game or "", {}).get("cast", [])
    groups: list[list[tuple[float, dict[str, Any]]]] = []
    for t in sorted(frames):
        f = frames[t]
        if f.get("missing") or f["kind"] in SKIP_KINDS:
            groups.append([])  # a gap ends the scene
            continue
        cur = groups[-1] if groups and groups[-1] else None
        if cur and _same_scene(cur, f) and t - cur[0][0] < SCENE_MAX_S:
            cur.append((t, f))
        else:
            groups.append([(t, f)])
    scenes = []
    for g in (g for g in groups if g):
        start = max(0.0, g[0][0] - FRAME_STEP_S / 2)
        end = g[-1][0] + FRAME_STEP_S / 2
        chars = Counter(canonical(n, cast) for _, f in g for n in f["characters"])
        heard = [s for s in transcript if s["end"] > start and s["start"] < end]
        for c in cast:  # a name said aloud counts as being in the scene
            if any(re.search(rf"\b{c}\b", plain(s["text"])) for s in heard):
                chars[nice(c)] += 0
        kinds = Counter(f["kind"] for _, f in g)
        locs = Counter(f["location"] for _, f in g if f["location"])
        objs = Counter(o for _, f in g for o in f["objects"])
        scenes.append(
            {
                "id": hashlib.sha1(f"{asset_id}:{start:.1f}".encode()).hexdigest()[:16],
                "asset_id": asset_id,
                "project_id": project_id,
                "game": game,
                "start_s": round(start, 2),
                "end_s": round(end, 2),
                "kind": kinds.most_common(1)[0][0],
                "characters": [c for c, _ in chars.most_common(6)],
                "seen": [c for c, n in chars.most_common(6) if n > 0],
                "location": locs.most_common(1)[0][0] if locs else "",
                "objects": [o for o, _ in objs.most_common(6)],
                "what": list(dict.fromkeys(f["what"] for _, f in g if f["what"]))[:4],
                "dialogue": [{"t": round(s["start"], 1), "text": s["text"].strip()} for s in heard][
                    :24
                ],
                "moments": [m for m in moments if m["start_s"] < end and m["end_s"] > start][:3],
            }
        )
    return scenes


def _same_scene(cur: list[tuple[float, dict[str, Any]]], f: dict[str, Any]) -> bool:
    last = cur[-1][1]
    cine = {"cinematic", "dialogue"}
    if (last["kind"] in cine) != (f["kind"] in cine):
        return False
    a, b = {plain(n) for n in last["characters"]}, {plain(n) for n in f["characters"]}
    same_people = not a or not b or bool(a & b)
    la, lb = plain(last["location"]), plain(f["location"])
    same_place = not la or not lb or la == lb or la in lb or lb in la
    return same_people and same_place


def entity_stats(scenes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """How much footage each character / place has: what discovery can show on screen."""
    stats: dict[str, dict[str, Any]] = {}
    for s in scenes:
        dur = s["end_s"] - s["start_s"]
        for kind, names in (("character", s["characters"]), ("place", [s["location"]])):
            for n in names:
                if not n or (kind == "place" and len(n) < 4):
                    continue
                e = stats.setdefault(
                    nice(n),
                    {"kind": kind, "scenes": 0, "seconds": 0.0, "cinematic": 0, "videos": set()},
                )
                e["scenes"] += 1
                e["seconds"] += dur
                e["cinematic"] += s["kind"] in ("cinematic", "dialogue")
                e["videos"].add(s["asset_id"])
    for e in stats.values():
        e["videos"] = len(e["videos"])
        e["seconds"] = round(e["seconds"])
    return stats


# ---------- lore passages ----------

_SKIP_SECTION = re.compile(
    r"gallery|navigation|appearances|see also|^references|external links|notes$|"
    r"bibliography|further reading|sources|videos?$|concept art|merchandise|achievements|trophies",
    re.I,
)


def passage_kind(source: str, section: str) -> str | None:
    s = section.lower()
    if _SKIP_SECTION.search(s):
        return None
    if re.search(r"theor|speculat", s):
        return "theory"
    if re.search(r"trivia|behind the scenes|easter|reference|development", s):
        return "trivia"
    if re.search(r"quote", s):
        return "quote"
    if source == "myth" or re.search(r"mytholog|norse|etymolog|legend", s):
        return "myth"
    return "canon"


def passages_from(
    sections: list[tuple[str, list[str]]],
    game: str,
    entity: str,
    source: str,
    page: dict[str, str],
) -> list[dict[str, Any]]:
    """Paragraphs as citable passages; long ones split at sentence ends."""
    out = []
    for section, paras in sections:
        kind = passage_kind(source, section)
        if kind is None:
            continue
        for para in paras:
            for chunk in _chunks(para):
                if len(chunk) < 60:
                    continue
                pid = hashlib.sha1(f"{game}|{source}|{page['url']}|{chunk}".encode()).hexdigest()[
                    :14
                ]
                out.append({"id": pid, "kind": kind, "text": chunk, "section": section, **page})
    return out


def _chunks(text: str, n: int = 520) -> list[str]:
    text = re.sub(r"\s+", " ", re.sub(r"\[\d+\]|\[citation needed\]", "", text)).strip()
    if len(text) <= n:
        return [text]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(sent) > n:
            out.append(cur.strip())
            cur = ""
        cur += sent + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def pick_passages(
    passages: list[dict[str, Any]], footage_words: set[str], n: int = 14
) -> list[dict[str, Any]]:
    """The passages worth showing the writer: trivia and myth first (they make ideas),
    then canon that mentions what the footage shows, spread across sections."""
    order = {"trivia": 0, "myth": 1, "quote": 2, "canon": 3, "theory": 4}

    def score(p: dict[str, Any]) -> tuple[int, int]:
        words = set(plain(p["text"]).split())
        return (order.get(p["kind"], 5), -len(words & footage_words))

    chosen: list[dict[str, Any]] = []
    per_section: Counter[str] = Counter()
    for p in sorted(passages, key=score):
        key = p["source"] + p.get("section", "")
        if per_section[key] >= 3:
            continue
        per_section[key] += 1
        chosen.append(p)
        if len(chosen) >= n:
            break
    return chosen


# ---------- discovery ----------

IDEA_SYSTEM = (
    "You are a gaming lore YouTuber planning videos from the creator's own footage. "
    "You never state anything that is not in the numbered SOURCES or said in the FOOTAGE "
    "dialogue. Fan theories stay labelled as theories. Every idea must be showable with the "
    "listed footage."
)

IDEA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "pitch": {"type": "string"},
                    "why": {"type": "string"},
                    "format": {"type": "string", "enum": ["short", "long", "both"]},
                    "seconds": {"type": "integer"},
                    "beats": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "says": {"type": "string"},
                                "sources": {"type": "array", "items": {"type": "string"}},
                                "scenes": {"type": "array", "items": {"type": "string"}},
                                "shows": {"type": "string"},
                            },
                            "required": ["says", "sources", "scenes", "shows"],
                        },
                    },
                },
                "required": ["title", "category", "pitch", "why", "format", "seconds", "beats"],
            },
        }
    },
    "required": ["ideas"],
}


def idea_prompt(
    game: str,
    subject: list[str],
    scenes: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    names: dict[str, str],
) -> str:
    g = GAMES[game]
    foot = "\n".join(
        f"{names[s['id']]} [{s['kind']}] {_mmss(s['start_s'])}-{_mmss(s['end_s'])} · "
        f"{', '.join(s['characters']) or 'no named characters'}"
        f"{' · ' + s['location'] if s['location'] else ''} · {' / '.join(s['what'][:2])}"
        + "".join(f'\n    "{d["text"]}"' for d in s["dialogue"][:3])
        for s in scenes
    )
    src = "\n".join(
        f"{names[p['id']]} ({p['kind']}, {p['site']}: {p['page']} > {p['section']}) {p['text']}"
        for p in passages
    )
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    return (
        f"Game: {g['name']}. Subject: {' & '.join(subject)}.\n\n"
        "FOOTAGE the creator has (scene ids, time in the video, who is on screen, speech):\n"
        f"{foot}\n\n"
        f"SOURCES (facts you may use; cite their ids):\n{src}\n\n"
        f"Propose up to 3 video ideas about {' & '.join(subject)}, each in a different "
        f"category:\n{cats}\n\n"
        "Rules:\n"
        "- Title: a curiosity hook a gamer would click, not a wiki heading. No clickbait lies.\n"
        "- Beats: 3 to 6 story steps. 'says' is the point made in plain words; 'sources' lists the "
        "S ids that back it (or [] if a scene's own dialogue proves it); 'scenes' lists the F "
        "ids that show it; 'shows' says what is on screen.\n"
        "- Only use F and S ids from the lists. Don't invent events, quotes or details.\n"
        "- Theories (sources marked theory) must be called theories in 'says'.\n"
        "- hidden ideas need a scene where the detail can actually be seen.\n"
        "- format: short (30-90 s) when one point carries it, long (3-10 min) when it needs "
        "context, both when it works either way. seconds = a sensible length."
    )


def _mmss(t: float) -> str:
    t = int(t)
    return (
        f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"
    )


def validate_idea(
    raw: dict[str, Any],
    ids: dict[str, str],
    scenes: dict[str, dict[str, Any]],
    passages: dict[str, dict[str, Any]],
    subject: list[str],
) -> dict[str, Any] | None:
    """Keeps only beats backed by real sources or the footage's own speech; an idea needs
    at least two such beats. Scene picks the writer didn't make are matched by name."""
    cat = str(raw.get("category", ""))
    if cat not in CATEGORIES:
        return None
    beats = []
    for b in raw.get("beats") or []:
        srcs = [ids[x] for x in b.get("sources") or [] if ids.get(x) in passages]
        scs = [ids[x] for x in b.get("scenes") or [] if ids.get(x) in scenes]
        text = _short(b.get("says"), 300)
        if not text:
            continue
        spoken = any(_quote_in(text, scenes[s]) for s in scs)
        if not srcs and not spoken:
            continue  # an unsupported claim: dropped, never shown as fact
        shows = _short(b.get("shows"), 160)
        match = _match(scs, f"{text} {shows}", scenes, subject)
        beats.append(
            {
                "says": text,
                "shows": shows,
                "sources": srcs,
                "scenes": match["scenes"],
                "fit": match["fit"],
            }
        )
    if len(beats) < 2:
        return None
    used_p = list(dict.fromkeys(p for b in beats for p in b["sources"]))
    used_s = list(dict.fromkeys(s for b in beats for s in b["scenes"]))
    fmt = str(raw.get("format", "short"))
    return {
        "title": _short(raw.get("title"), 110),
        "category": cat,
        "pitch": _short(raw.get("pitch"), 400),
        "why": _short(raw.get("why"), 300),
        "format": fmt if fmt in ("short", "long", "both") else "short",
        "seconds": int(min(max(int(raw.get("seconds") or 60), 20), 1200)),
        "subject": subject,
        "beats": beats,
        "passages": used_p,
        "scenes": used_s,
        "theory": any(passages[p]["kind"] == "theory" for p in used_p),
        **confidence(beats, scenes, passages),
    }


def _quote_in(text: str, scene: dict[str, Any]) -> bool:
    """The beat repeats something actually said in the scene (5+ words in a row)."""
    said = " ".join(plain(d["text"]) for d in scene.get("dialogue", []))
    words = plain(text).split()
    return any(" ".join(words[i : i + 5]) in said for i in range(max(0, len(words) - 4)))


def _match(
    picked: list[str], text: str, scenes: dict[str, dict[str, Any]], subject: list[str]
) -> dict[str, Any]:
    """How well the footage shows a beat: 1 = a picked scene has the named people/place,
    0.5 = only another scene does (swapped in), 0.25 = scenes picked but none fits, 0 = none."""
    want = {plain(n) for n in subject} | {
        c for c in {plain(w) for w in re.findall(r"[A-ZÀ-Þ][\w'\u2019-]+", text)} if len(c) > 2
    }

    def fits(s: dict[str, Any]) -> bool:
        have = {plain(n) for n in s["characters"]} | set(plain(s["location"]).split())
        return bool(want & have)

    good = [s for s in picked if fits(scenes[s])]
    if good:
        return {"scenes": good[:3], "fit": 1.0}
    alt = sorted(
        (s for s in scenes.values() if fits(s)),
        key=lambda s: (s["kind"] not in ("cinematic", "dialogue"), -len(s["dialogue"])),
    )
    if alt:
        return {"scenes": [s["id"] for s in alt[:2]], "fit": 0.5}
    return {"scenes": picked[:2], "fit": 0.25 if picked else 0.0}


def confidence(
    beats: list[dict[str, Any]],
    scenes: dict[str, dict[str, Any]],
    passages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """How well the creator's footage supports the story, plus how solid the sources are.
    Footage counts most: an idea that can't be shown isn't a video yet."""
    footage = sum(b["fit"] for b in beats) / len(beats)
    sourced = sum(
        1.0 if any(passages[p]["kind"] != "theory" for p in b["sources"]) else 0.6 for b in beats
    ) / len(beats)
    value = round(0.7 * footage + 0.3 * sourced, 2)
    level = "high" if value >= 0.75 else "medium" if value >= 0.5 else "low"
    missing = [b["shows"] or b["says"][:60] for b in beats if b["fit"] < 0.5]
    return {
        "confidence": value,
        "level": level,
        "footage_fit": round(footage, 2),
        "sourced": round(sourced, 2),
        "missing": missing[:3],
    }


def idea_id(game: str, title: str) -> str:
    return hashlib.sha1(f"{game}|{plain(title)}".encode()).hexdigest()[:14]
