"""Voice-over for a Short: a short script (hook, fact, reaction) grounded in the fact bank,
then placed on the Short's timeline where nobody in the footage is talking."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from sphura.domain.models import AIJudgement, Clip
from sphura.pipeline import facts as F
from sphura.pipeline import voice_intel as VI
from sphura.pipeline.copywriter import is_generic
from sphura.pipeline.hinglish import has_devanagari, romanize
from sphura.pipeline.judge import guard_claims
from sphura.ports.ai import ImageInput, LLMProvider, LLMRequest

log = logging.getLogger(__name__)

MAX_LINES = 3
WORDS_PER_S = 2.6  # Kokoro at speed 1.0, roughly
SPEECH_SHARE = 0.4  # narration fills at most this much of the Short

SYSTEM = """You write the voice-over for a gaming YouTube Short, spoken by a narrator over
the creator's own gameplay. It must feel like a funny, sharp gamer friend reacting to THIS
exact moment, never a summary or a trailer. Up to 3 short lines:
- "hook": the first words viewers hear; makes them stay. A tease, a question or a joke
  about what is about to happen. Max 12 words.
- "fact": ONE fact from the list you are given that is about who or what is in this
  moment, in your own words, keeping every name and number exactly. Give its fact_id.
  Never state any other fact, number, date or lore detail. If no listed fact is about
  this moment, leave the fact line out: a fact that doesn't fit ruins the Short.
- "reaction": an opinion, roast or fan-style aside about the moment. No facts, no numbers.
  Max 14 words.
Every line must point at something specific you were given: a quoted line of dialogue,
something visible in the frames, or where we are in the story. A line that would fit any
other clip is a failure. Riffing on what a character just said works best.
Good (specific, funny):
- dialogue "Odin won't stop until we stop him" -> "Odin won't stop? Bro hasn't stopped
  scheming since the prologue."
- a giant boss swings at Kratos -> "Imagine picking a fight with the one guy who killed a
  whole pantheon."
- Atreus says "I can handle it" -> "Famous last words, Atreus."
Bad (generic): "Kratos is unstoppable", "Wait for it", "This fight is so good".
Use a character's name only if it appears in the dialogue, story, video title or is
unmistakable in the frames; otherwise say "this guy". Never invent what happens on screen.
Never use: insane, epic, legendary, ultimate, unleash, you won't believe, showdown, chaos.
Write 3 different options, each a complete script, each with a different angle."""

HINGLISH = """Write in Hinglish the way an Indian gamer talks to friends: Hindi sentences
with English gaming words, banter and roasts welcome. Write Hindi words in Devanagari
script (it is read by a Hindi voice) and keep names and English words in English letters,
e.g. "भाई, Kratos ने इसको ज़िंदा नहीं छोड़ना था" or "Odin की planning देखो, मरता हमेशा कोई और है"."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "angle": {"type": "string", "description": "the specific thing this riffs on"},
                    "lines": {
                        "type": "array",
                        "maxItems": MAX_LINES,
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["hook", "fact", "reaction"]},
                                "text": {"type": "string"},
                                "fact_id": {
                                    "type": "string",
                                    "description": "for a fact line; else empty",
                                },
                                "mood": {"type": "string", "enum": list(VI.MOODS)},
                                "pace": {"type": "string", "enum": list(VI.PACES)},
                            },
                            "required": ["kind", "text", "fact_id"],
                        },
                    },
                },
                "required": ["angle", "lines"],
            },
        }
    },
    "required": ["options"],
}

PICK = """You judge voice-over scripts for a gaming Short. Rank the options from best to
worst. Best = most specific to this exact moment (it quotes or plays off the dialogue or
what is visible) and most fun to hear. Penalise anything generic, anything that could fit
another clip, and any fact that is not about who or what is in the moment."""

PICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
}  # fmt: skip


def moment_brief(j: AIJudgement | None, dialogue: list[str]) -> str:
    parts = []
    if j:
        parts += [f"What happens: {j.what_happens}", f"Why it stands out: {j.reason}"]
        if j.tags:
            parts.append(f"Details: {', '.join(j.tags)}")
        who = list(dict.fromkeys([*j.names, *j.characters]))
        if who:
            parts.append(f"On screen (names read from the game): {', '.join(who)}")
        if j.dialogue:
            parts.append(
                "Subtitles read on screen: "
                + " / ".join(f'{sp}: "{t}"' if sp else f'"{t}"' for sp, t in j.dialogue[:6])
            )
    if dialogue:
        parts.append("Dialogue heard (exact): " + " / ".join(f'"{d}"' for d in dialogue[:6]))
    return "\n".join(parts)


def write_script(
    llm: LLMProvider | None,
    game: str,
    judgement: AIJudgement | None,
    dialogue: list[str],
    lang: str,
    duration_s: float,
    format_label: str = "",
    spoilers: bool = False,
    story: list[str] | None = None,
    video: str = "",
    frames: list[ImageInput] | None = None,
    voice_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """{"lines": [{kind, text, caption, fact_id, source}], "facts": [...], "note": str}.
    story: dialogue from just before the moment, so the AI knows who is there and why."""
    brief = moment_brief(judgement, dialogue)
    lore = "lore" in format_label.lower() or "fact" in format_label.lower()
    facts = F.relevant(game, brief, spoilers, context=" ".join(story or []), general=lore)
    out: dict[str, Any] = {"facts": [_fact_dict(f) for f in facts], "note": "", "lines": []}
    budget = max(6, int(duration_s * SPEECH_SHARE * WORDS_PER_S))
    options: list[list[dict[str, Any]]] = []
    if llm is not None:
        prompt = "\n".join(x for x in [
            f"Game: {F_GAME_NAMES.get(game, game)}",
            f"Video: {video}" if video else "",
            f"Kind of Short: {format_label}" if format_label else "",
            f"Short length: about {round(duration_s)} seconds; at most {budget} words in total.",
            "Story so far (dialogue just before this moment): "
            + " / ".join(f'"{d}"' for d in (story or [])[-8:]) if story else "",
            brief,
            "The frames show the moment (before, the key moment, after)." if frames else "",
            "Facts you may use (fact_id: fact):" if facts else "No facts fit: no fact line.",
            *[f"- {f.id}: {f.text}" for f in facts],
            HINGLISH if lang == "hi" else "Write in English.",
            voice_ctx["brief"] if voice_ctx else "",
        ] if x)  # fmt: skip
        try:
            resp = llm.complete(LLMRequest(system=SYSTEM, prompt=prompt, images=frames or [],
                                           json_schema=SCHEMA, max_tokens=900))  # fmt: skip
            raw = [o for o in json.loads(resp.text).get("options", []) if isinstance(o, dict)]
            options = [o.get("lines", []) for o in raw]
            options = _ranked(llm, prompt, raw, options)
        except Exception as e:
            log.warning("voice-over script with %s failed: %s", llm.info.name, e)
            out["note"] = "The AI couldn't write a script; a simple one was used instead."
    lines: list[dict[str, Any]] = []
    for opt in options:  # best first; the first that follows the rules wins
        lines = check(opt, game, lang, budget)
        if len(lines) >= 2 or (lines and opt is options[-1]):
            break
    if not lines:
        lines = _fallback(judgement, facts, lang)
        if not lines:
            out["note"] = out["note"] or (
                "No script: Hinglish needs the AI judge (Ollama)."
                if lang == "hi"
                else "No fact in the fact bank matches this moment. Add a line yourself, "
                "or leave it without a voice-over."
            )
    out["lines"] = [_voiced(ln, raw, voice_ctx, lang) for ln, raw in _with_raw(lines, options)]
    if voice_ctx:
        out["scene"] = voice_ctx["scene"]
        out["personality"] = voice_ctx["personality"]
    return out


def _with_raw(
    lines: list[dict[str, Any]], options: list[list[dict[str, Any]]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Each kept line with the AI's raw line (for its mood and pace)."""
    raw = {
        str(r.get("text", "")).strip().strip('"'): r
        for opt in options
        for r in opt
        if isinstance(r, dict)
    }
    return [(ln, raw.get(ln["text"], {})) for ln in lines]


def _voiced(
    ln: dict[str, Any], raw: dict[str, Any], ctx: dict[str, Any] | None, lang: str
) -> dict[str, Any]:
    """Speech-friendly text and a delivery for the line: the AI's mood when emotion is on,
    else the scene's. The direction travels next to the text, never inside it."""
    if not ctx:
        return ln
    if ctx["optimize"]:
        text, _ = VI.optimize_script(ln["text"], lang)
        if text:
            ln = {**ln, "text": text, "caption": romanize(text)}
    mood, pace = ctx["default"]["mood"], ctx["default"]["pace"]
    if ctx["emotion"] and raw.get("mood") in VI.MOODS:
        mood = raw["mood"]
        pace = raw.get("pace") if raw.get("pace") in VI.PACES else pace
    return {**ln, "direction": {"mood": mood, "pace": pace}}


def _ranked(
    llm: LLMProvider, context: str, raw: list[Any], options: list[list[dict[str, Any]]]
) -> list[list[dict[str, Any]]]:
    """Asks the AI which option is best for this moment; keeps the order on failure."""
    if len(options) < 2:
        return options
    listing = "\n\n".join(
        f"Option {i}: (angle: {o.get('angle', '')})\n"
        + "\n".join(f"  {ln.get('kind')}: {ln.get('text')}" for ln in o.get("lines", []))
        for i, o in enumerate(raw)
    )  # fmt: skip
    try:
        resp = llm.complete(LLMRequest(system=PICK, prompt=f"{context}\n\n{listing}",
                                       json_schema=PICK_SCHEMA, max_tokens=100))  # fmt: skip
        order = [i for i in json.loads(resp.text).get("ranking", []) if 0 <= i < len(options)]
    except Exception as e:
        log.warning("ranking voice-over options failed: %s", e)
        return options
    order = list(dict.fromkeys(order)) + [i for i in range(len(options)) if i not in order]
    return [options[i] for i in order]


F_GAME_NAMES = {"gow": "God of War Ragnarök", "pbz": "Phantom Blade Zero"}


def check(raw: list[dict[str, Any]], game: str, lang: str, budget: int) -> list[dict[str, Any]]:
    """Keeps lines that follow the rules: facts must come from the bank (names and
    numbers intact), reactions make no claims, the whole script fits the word budget."""
    known = F.by_id(game)
    lines: list[dict[str, Any]] = []
    words = 0
    seen_kinds: set[str] = set()
    for r in raw:
        if len(lines) == MAX_LINES:
            break
        kind = str(r.get("kind", ""))
        text = re.sub(r"\s+", " ", str(r.get("text", ""))).strip().strip('"')
        if kind not in ("hook", "fact", "reaction") or not text or kind in seen_kinds:
            continue
        if lang == "hi" and not has_devanagari(text):
            continue  # the Hindi voice can't read romanized Hindi
        fact = known.get(str(r.get("fact_id", ""))) if kind == "fact" else None
        if kind == "fact":
            if fact is None or not _numbers_ok(text, fact.text):
                continue
        elif re.search(r"\d", text) or not guard_claims(text) or is_generic(text):
            continue
        n = len(romanize(text).split())
        if words + n > budget:
            break
        words += n
        seen_kinds.add(kind)
        lines.append(line(kind, text, fact.id if fact else "", fact.source if fact else ""))
    return lines


def line(kind: str, text: str, fact_id: str = "", source: str = "",
         direction: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    out: dict[str, Any] = {"kind": kind, "text": text, "caption": romanize(text),
                           "fact_id": fact_id, "source": source}  # fmt: skip
    if direction:
        out["direction"] = VI.Direction.from_dict(direction).to_dict()
    return out


def _numbers_ok(text: str, fact: str) -> bool:
    nums = set(re.findall(r"\d[\d.,]*", text))
    return nums <= set(re.findall(r"\d[\d.,]*", fact))


def _fallback(j: AIJudgement | None, facts: list[F.Fact], lang: str) -> list[dict[str, Any]]:
    """Without the AI: the judge's hook (if it isn't generic) and the best-matching fact,
    word for word. Hinglish needs the AI to write it, so nothing is said."""
    if lang == "hi":
        return []
    out = []
    hook = (j.hook if j else "").strip()
    if hook and not is_generic(hook) and guard_claims(hook) and len(hook.split()) <= 12:
        out.append(line("hook", hook))
    specific = [f for f in facts if f.keys]
    if specific:
        out.append(line("fact", specific[0].text, specific[0].id, specific[0].source))
    return out


def _fact_dict(f: F.Fact) -> dict[str, Any]:
    return {"id": f.id, "text": f.text, "source": f.source, "spoiler": f.spoiler}


# --- placement ---------------------------------------------------------------------------


def speech_spans(clip: Clip, transcript: Any) -> list[tuple[float, float]]:
    """When people talk in the footage, on the Short's own timeline."""
    if transcript is None:
        return []
    spans: list[tuple[float, float]] = []
    offset = 0.0
    for seg in clip.render_segments():
        for s in transcript.segments:
            a, b = max(s.start_s, seg.start_s), min(s.end_s, seg.end_s)
            if b > a and s.text.strip():
                spans.append((clip.to_output_time(offset + a - seg.start_s),
                              clip.to_output_time(offset + b - seg.start_s)))  # fmt: skip
        offset += seg.duration_s
    return sorted(spans)


def place(
    durations: list[float], spans: list[tuple[float, float]], total: float
) -> list[float | None]:
    """Start time per line (None: it doesn't fit). The hook goes right at the start; the
    others take the earliest spot after the previous line with the least talking under
    it, so the narrator doesn't speak over the characters when it can avoid it."""
    starts: list[float | None] = []
    t_min = 0.15
    for i, d in enumerate(durations):
        latest = total - 0.4 - d
        if i == 0:
            latest = min(latest, 1.2)  # the hook opens the Short
        if latest < t_min:
            starts.append(None)
            continue
        best, best_cost = None, 1e9
        t = t_min
        while t <= latest + 1e-6:
            cost = _overlap(t, t + d, spans) + 0.02 * (t - t_min)  # prefer early
            if cost < best_cost - 1e-6:
                best, best_cost = t, cost
            t += 0.1
        starts.append(round(best, 2) if best is not None else None)
        if best is not None:
            t_min = best + d + 0.35
    return starts


def _overlap(a: float, b: float, spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


def mix_track(
    parts: list[tuple[float, np.ndarray, int]], total: float, sr: int = 48000
) -> np.ndarray:
    """One stereo track the length of the Short with each spoken line at its start time."""
    out = np.zeros(int(total * sr) + 1, np.float32)
    for start, audio, rate in parts:
        if rate != sr:  # linear resample: speech is band-limited, this is plenty
            x = np.arange(int(len(audio) * sr / rate)) * rate / sr
            audio = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
        i = int(start * sr)
        j = min(len(out), i + len(audio))
        out[i:j] += audio[: j - i]
    peak = float(np.abs(out).max()) or 1.0
    out *= min(1.0, 0.9 / peak)
    return np.stack([out, out], axis=1)
