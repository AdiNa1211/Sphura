from __future__ import annotations

import contextlib
import copy
import logging
import math
import re
import shutil
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sphura.application.queries import QueryService
from sphura.domain.errors import InvalidMediaError, NotFoundError, SphuraError
from sphura.domain.models import (
    AspectRatio,
    Clip,
    EditingMode,
    HighlightTier,
    Job,
    JobType,
    TextOverlay,
    TimeRange,
)
from sphura.pipeline import reframe
from sphura.pipeline.candidates import FINALE
from sphura.pipeline.clips import build_clip
from sphura.pipeline.copywriter import COPY_VERSION, on_screen_text, render_txt, write_copy
from sphura.pipeline.effects import (
    STYLE_VERSION,
    MusicTrack,
    align_to_beats,
    beat_sync_cuts,
    dialogue_ranges,
    key_moment,
    plan_effects,
    plan_music,
)
from sphura.pipeline.effects import _anchor as effects_anchor
from sphura.pipeline.fx_context import LABELS, TOGGLES, classify, filter_off
from sphura.pipeline.music_picker import library_key, pick_music
from sphura.pipeline.music_prompt import write_music_prompt
from sphura.pipeline.planner import FORMATS as PLAN_FORMATS
from sphura.pipeline.planner import (
    PLAN_VERSION,
    VARIANTS,
    Plan,
    beat_plans,
    budget,
    subtitle_lines,
    tune_music,
)
from sphura.pipeline.planner import cut as cut_plan
from sphura.pipeline.reframe import plan_reframe
from sphura.pipeline.score import MOODS, ScorePlan, base_bpm, compose, fit_bpm, mood_for
from sphura.pipeline.score import write_wav as write_score
from sphura.pipeline.styles import (
    CUT_VERSION,
    fit_length,
    geometry,
    style_aspect,
    trim_for_style,
    window_after,
)
from sphura.pipeline.tracks import choose_mix_track
from sphura.pipeline.wav import WavFile
from sphura.ports.ai import ImageInput, LLMProvider
from sphura.ports.media import MediaDecoder, PeopleDetector, VideoRenderer
from sphura.ports.queue import JobContext, JobQueue
from sphura.ports.repositories import ClipRepository, MediaAssetRepository
from sphura.ports.storage import StorageProvider


def shorts_folder(project_name: str) -> Path:
    """Where a stream's Shorts are exported: Videos/SPHURA/<date> <stream name>."""
    return (
        Path.home() / "Videos" / "SPHURA" / f"{datetime.now():%Y-%m-%d} {slugify(project_name, 60)}"
    )


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "clip"


OVERLAY_STYLES = ("hook", "caps", "plain", "bar")

if TYPE_CHECKING:
    from sphura.infrastructure.ai.acestep import AceStep

log = logging.getLogger(__name__)


class ClipService:
    def __init__(
        self,
        queries: QueryService,
        clips: ClipRepository,
        queue: JobQueue,
        assets: MediaAssetRepository,
        storage: StorageProvider,
        music: Callable[[], list[MusicTrack]] = list,
        llm: LLMProvider | None = None,
        sample: Callable[[str, list[float]], list[bytes]] | None = None,
        ai_music: AceStep | None = None,
        analyze: Callable[..., MusicTrack] | None = None,
        envelope: Callable[[Path, float, float, float], list[float]] | None = None,
    ) -> None:
        self._envelope = envelope
        self._queries = queries
        self._clips = clips
        self._queue = queue
        self._assets = assets
        self._storage = storage
        self._music = music
        self._llm = llm
        self._sample = sample
        self._ai = ai_music
        self._analyze = analyze
        self._recent_moods: dict[str, list[str]] = {}  # per project: vary music between clips
        self._ctx: tuple[Any, Any] = (None, None)  # (judgement, transcript) of the clip in hand

    def generate(
        self,
        project_id: str,
        top: int,
        mode: EditingMode | None = None,
        include_maybe: bool = False,
        aspect: AspectRatio = AspectRatio.VERTICAL,
        layout: str | None = None,
        effects: bool = True,
        style: str = "hype",
        game: str | None = None,
        length_s: float | None = None,
        caption_bar: bool | None = None,
        only: list[str] | None = None,
        hints: dict[str, dict[str, Any]] | None = None,
    ) -> list[Clip]:
        """Edit lists for up to `top` moments worth posting (TOP/STRONG tiers; or any, with
        include_maybe). Existing clips for the same moment are kept, so edits survive.
        `only`: exactly these moments, in this order. `hints`: per moment copy guidance."""
        everything = self._queries.highlights(project_id, mode)
        self._clips.delete_except(project_id, {r.candidate.id for r in everything})
        hints = hints or {}
        reviewed = any(r.tier is not None for r in everything)
        taste = self._queries.taste()
        worthy = [
            r
            for r in everything
            if not (taste and taste.rejected(r.candidate))
            and (
                include_maybe or not reviewed or r.tier in (HighlightTier.TOP, HighlightTier.STRONG)
            )
        ]
        # A fight's finale always makes it in when the AI judged it a highlight
        ranked = worthy[:top] + [
            r for r in worthy[top:]
            if FINALE in r.candidate.drivers
            and r.candidate.judgement is not None and r.candidate.judgement.is_highlight
        ]  # fmt: skip
        if only is not None:
            by_id = {r.candidate.id: r for r in everything}
            ranked = [by_id[c] for c in only if c in by_id]
        tl = self._optional(self._queries.signals, project_id)
        transcript = self._optional(self._queries.transcript, project_id)
        tracks = self._music() if effects else []
        out: list[Clip] = []
        aspect = style_aspect(style, aspect)
        for r in ranked:
            # One clip per (moment, style): switching styles never overwrites another's edit
            fmt = ((hints.get(r.candidate.id) or {}).get("trend") or {}).get("format")
            clip = self._clips.find_by_candidate(r.candidate.id, style, fmt) or build_clip(
                r, tl, transcript
            )
            clip.rank = r.rank
            wav = self._wav(clip)
            new_style = clip.style != style
            # A different length re-cuts the moment, unless it was cut by hand in the editor
            relength = (
                clip.publish.get("length_s") != length_s
                or clip.publish.get("cut_version") != CUT_VERSION  # cut by older rules
            ) and not clip.publish.get("cuts_edited")
            if new_style or relength:
                # Each style/length cuts the moment differently: start from the full moment
                base = build_clip(r, tl, transcript)
                clip.style = style
                key = key_moment(base, r.candidate.peak_s, tl, wav)
                finale = FINALE in r.candidate.drivers and bool(base.segments)
                if length_s:
                    if finale:  # finales pay off at the end: keep the ending in
                        key = max(s.end_s for s in base.segments) - length_s * 0.3
                    asset = self._assets.get(clip.asset_id)
                    src = asset.media_info.duration_s if asset and asset.media_info else 1e9
                    clip.segments = fit_length(base.segments, key, style, length_s, src)
                else:
                    if finale:
                        key = max(key, max(s.end_s for s in base.segments) - window_after(style))
                    clip.segments = trim_for_style(base.segments, key, style)
                clip.overlays, clip.effects, clip.crop_path = base.overlays, [], []
                if new_style:
                    clip.publish = {}
            hint = hints.get(r.candidate.id) or {}
            trend = hint.get("trend")
            if trend and not clip.publish.get("trend"):
                clip.publish = {}  # new format Short: its own copy, not the plain one's
            stale = (
                clip.publish.get("version") != COPY_VERSION
                or (clip.publish.get("game") or None) != game
                or (trend or {}).get("format") != (clip.publish.get("trend") or {}).get("format")
            )
            if not clip.publish.get("edited") and (stale or not clip.publish):
                clip.publish = write_copy(
                    self._llm,
                    r.candidate.judgement,
                    game,
                    self._frames(clip, r.candidate.peak_s),
                    _dialogue(transcript, clip),
                    hint.get("copy"),
                )
                if trend:
                    clip.publish["trend"] = trend
                    extra = [
                        h for h in hint.get("hashtags", []) if h not in clip.publish["hashtags"]
                    ]
                    clip.publish["hashtags"] = " ".join(
                        clip.publish["hashtags"].split()[:5] + extra[:3]
                    )
                self._caption(clip)
            clip.publish = {**clip.publish, "length_s": length_s, "cut_version": CUT_VERSION}
            if clip.aspect != aspect:
                clip.aspect = aspect
                clip.crop_path = []  # a different window shape needs a new camera path
            if layout is not None and layout != clip.layout:
                clip.layout = layout
                clip.crop_path = []
            if caption_bar is not None and style != "meme":
                self._set_bar(clip, caption_bar)
            if effects:
                self._ctx = (r.candidate.judgement, transcript)
                self._style(clip, r.candidate.peak_s, tl, tracks, wav)
                self._caption(clip)  # timings follow the planned key hit
            elif clip.effects:
                clip.effects = []
                clip.crop_path = []
            self._clips.save(clip)
            out.append(clip)
        return out

    def plans(self, project_id: str, game: str) -> list[Plan]:
        """Proposed Shorts: the best stories and standout moments, each with a format."""
        moments = [r.candidate for r in self._queries.highlights(project_id)]
        tl = self._optional(self._queries.signals, project_id)
        transcript = self._optional(self._queries.transcript, project_id)
        return beat_plans(moments, transcript, tl, game)

    def make_plan(
        self, project_id: str, plan: Plan, variant: str, game: str | None, titles: str = ""
    ) -> Clip:
        """One variant of a planned Short: cut from the plan's moments, then effects,
        music and captions decided by the variant (see pipeline.planner)."""
        v = VARIANTS[variant]
        ranked = {r.candidate.id: r for r in self._queries.highlights(project_id)}
        moments = {m: ranked[m].candidate for m in plan.moments if m in ranked}
        if not moments:
            raise NotFoundError("the moments of this Short are gone; re-run the analysis")
        first = ranked[next(m for m in plan.moments if m in moments)]
        tl = self._optional(self._queries.signals, project_id)
        transcript = self._optional(self._queries.transcript, project_id)
        c = cut_plan(plan, variant, moments, tl, transcript)
        key = f"{plan.key}:{variant}"
        old = self._clips.find_by_candidate(first.candidate.id, None, None, key)
        clip = Clip(
            project_id=project_id,
            asset_id=first.candidate.asset_id,
            title=plan.title,
            segments=c.segments,
            aspect=AspectRatio.VERTICAL,
            layout="crop",
            style=v.style,
            candidate_id=first.candidate.id,
            rank=first.rank,
        )
        if old is not None:
            clip.id = old.id
        # Copy follows the payoff moment: it's what the Short builds to
        inside = [m for m in moments.values() if m.range.start_s <= c.key_s <= m.range.end_s]
        judgement = (inside or list(moments.values()))[-1].judgement
        fmt = PLAN_FORMATS[plan.format]
        hint = (
            f"This Short is one {plan.kind or fmt.label} beat. The title is a take on the "
            "moment, the way the most-viewed Shorts of this game title theirs, not a "
            "description of it. Name who is on screen when known. Never say 'the player'. "
            "Don't give away the payoff in the hook." + (f"\n{titles}" if titles else "")
        )
        if old is not None and old.publish.get("edited"):
            clip.publish = dict(old.publish)
        else:
            clip.publish = write_copy(self._llm, judgement, game, self._frames(clip, c.key_s),
                                      _dialogue(transcript, clip), hint)  # fmt: skip
        read = [x for m in moments.values() if m.judgement for x in m.judgement.dialogue]
        clip.publish = {
            **clip.publish,
            "plan": {"key": key, "story": plan.key, "variant": variant, "label": v.label,
                     "format": plan.format, "format_label": fmt.label, "reason": plan.reason,
                     "about": v.about, "structure": c.structure, "version": PLAN_VERSION},
            "music": {"choice": "none"} if v.music == "none" else
            (old.publish.get("music") if old else None) or {"choice": "auto"},
        }  # fmt: skip
        self._ctx = (judgement, transcript)
        wav = self._wav(clip)
        if variant == "C":
            clip.effects = [{"type": "style", "version": STYLE_VERSION}]
        else:
            self._plan(clip, c.key_s, tl, wav)
        clip.effects = budget(clip.effects, variant, judgement)
        if old is not None:  # an approved narration survives a re-cut
            clip.effects += [e for e in old.effects if e["type"] == "voiceover"]
        tracks = self._music() if v.music != "none" else []
        self._apply_music(clip, tracks)
        clip.audio.game_gain = 1.0  # the game (and its dialogue) always comes through
        # After music: syncing cuts to the beat can move them
        subs = subtitle_lines(clip.render_segments(), transcript, read)
        for i, e in enumerate(clip.effects):
            if e["type"] == "music":
                clip.effects[i] = tune_music(e, variant, bool(subs))
        clip.effects.append({"type": "subs", "lines": [
            {**ln, "start": round(clip.to_output_time(ln["start"]), 2),
             "end": round(clip.to_output_time(ln["end"]), 2)} for ln in subs]})  # fmt: skip
        clip.overlays = self._plan_text(clip, variant)
        self._clips.save(clip)
        return clip

    def choose(self, clip_id: str) -> Clip:
        """Marks this version of a planned Short as the one to post; its siblings stay."""
        clip = self._get(clip_id)
        story = (clip.publish.get("plan") or {}).get("story")
        if not story:
            raise SphuraError("only planned Shorts have versions to choose from")
        for c in self._clips.list_for_project(clip.project_id):
            if (c.publish.get("plan") or {}).get("story") == story:
                c.publish = {**c.publish, "chosen": c.id == clip_id}
                self._clips.save(c)
        return clip

    @staticmethod
    def _plan_text(clip: Clip, variant: str) -> list[TextOverlay]:
        """Only the edit (B) gets a hook line; the clean and cinematic cuts let the scene
        speak, like the top Shorts of these games. Never spoils the payoff."""
        total = clip.output_duration_s
        if variant == "B":
            text = str(clip.publish.get("reaction") or clip.publish.get("hype") or "").strip()
            return [TextOverlay(text, 0.0, min(2.5, total), "hook")] if text else []
        return []

    def render(self, clips: list[Clip], export_dir: Path | None = None) -> list[Job]:
        if export_dir is not None:
            for c in clips:  # remembered so a single re-render lands in the same folder
                c.publish = {**c.publish, "export_dir": str(export_dir)}
                self._clips.save(c)
        return [
            self._queue.enqueue(
                Job(
                    type=JobType.RENDER,
                    project_id=c.project_id,
                    payload={
                        "clip_id": c.id,
                        "export_dir": str(export_dir) if export_dir else None,
                    },
                )
            )
            for c in clips
        ]

    def _style(
        self,
        clip: Clip,
        peak_s: float,
        tl: Any,
        tracks: list[MusicTrack],
        wav: WavFile | None,
    ) -> None:
        """(Re)plan the edit when missing or made by an older style; keep music in step
        with the music folder."""
        version = next((e.get("version") for e in clip.effects if e["type"] == "style"), 0)
        if version != STYLE_VERSION:
            self._plan(clip, peak_s, tl, wav)
        self._apply_music(clip, tracks)

    def _plan(self, clip: Clip, peak_s: float, tl: Any, wav: WavFile | None) -> None:
        """Effects for the moment's kind, minus the ones switched off in the editor."""
        kind = classify(self._ctx[0])
        fx = plan_effects(clip, peak_s, tl, wav, kind)
        voice = [e for e in clip.effects if e["type"] == "voiceover"]  # narration stays
        clip.effects = filter_off(fx, list(clip.publish.get("fx_off") or [])) + voice
        clip.crop_path = []  # the teaser adds footage the camera must cover

    def _apply_music(self, clip: Clip, tracks: list[MusicTrack], reroll: bool = False) -> None:
        """Music per the clip's choice (publish["music"]): auto, none, or a track path.
        Whatever plays, the edit is synced to it: cuts trimmed onto the beat grid, the key
        hit on a beat, the drop where the action comes back to speed."""
        pref = clip.publish.get("music") or {}
        choice = pref.get("choice", "auto")
        volume = pref.get("volume_db")
        current = next((e for e in clip.effects if e["type"] == "music"), None)
        gen = (current or {}).get("generated") or {}
        prev_mood = str(gen.get("mood") or "")
        # Recompose scores from the old 3-style composer, and synth scores once AI music is on
        old_score = bool(gen) and (
            "label" not in gen or (not gen.get("ai") and self.ai_music_ready())
        )
        if (
            current
            and not reroll
            and not old_score
            and Path(current["path"]).is_file()
            and current.get("choice", "auto") == choice
            and (volume is None or current.get("gain_db") == volume)
        ):
            # Same track: still pick up license/credit info edited since it was chosen
            same = next((t for t in tracks if t.path == current["path"]), None)
            if same is not None:
                current["credit"] = same.credit
            return
        if current:
            clip.effects.remove(current)
        clip.audio.game_gain = 1.0
        if choice == "none" or (choice == "auto" and clip.style == "meme"):
            return  # memes live on game audio unless music is picked by hand
        judgement, transcript = self._ctx
        ai = clip.publish.get("music_ai")
        if tracks and (not ai or ai.get("library") != library_key(tracks)):
            ai = pick_music(self._llm, judgement, clip.style, clip.output_duration_s, tracks)
            clip.publish = {**clip.publish, "music_ai": ai}
        # Auto: a folder track only when the AI judges it a good fit; otherwise SPHURA
        # composes an original score for the clip (nothing to download or license)
        if choice == "score" or (choice == "auto" and not (ai and ai.get("fits_well"))):
            self._score(clip, choice, judgement, transcript, prev_mood if reroll else "")
            return
        by_path = {t.path: t for t in tracks}
        track = by_path.get(ai["path"]) if choice == "auto" and ai else by_path.get(choice)
        if track is None:
            return
        beat_sync_cuts(clip, track.beat_s)
        m = plan_music(clip, tracks, track)
        if m is None:
            return
        self._finish_music(clip, m, track, choice, transcript)

    def _finish_music(
        self, clip: Clip, m: dict[str, Any], track: MusicTrack, choice: str, transcript: Any
    ) -> None:
        if clip.style == "cinematic":  # music-led: louder, the game sits under it
            m.update(gain_db=-7.0, duck=False, dip=None)
            clip.audio.game_gain = 0.45
        elif clip.style == "meme":
            m.update(gain_db=-16.0, dip=None)
        volume = (clip.publish.get("music") or {}).get("volume_db")
        if volume is not None:
            m["gain_db"] = float(volume)
        m["choice"] = choice
        m["credit"] = track.credit
        if transcript is not None:  # music makes room for dialogue
            spoken = [(s.start_s, s.end_s) for s in transcript.segments if s.text.strip()]
            m["dialogue"] = dialogue_ranges(clip, spoken)
        align_to_beats(clip, m, track)
        clip.effects.append(m)

    def ai_music_ready(self) -> bool:
        return self._ai is not None and self._analyze is not None and self._ai.ready()

    def _ai_track(
        self, clip: Clip, choice: str, judgement: Any, transcript: Any, mood: str, seed: int
    ) -> bool:
        """Generate music for this clip with the local AI model; False = use the synth."""
        assert self._ai is not None and self._analyze is not None
        label = MOODS[mood].label
        prompt = write_music_prompt(
            self._llm, judgement, clip.publish.get("game"), mood, label, seed
        )
        bpm = round(base_bpm(mood, seed))
        path = self._storage.artifact_path(
            clip.project_id, "music", f"{clip.id}_ai_{seed % 1_000_000}.wav"
        )
        self._ai.generate(prompt, clip.output_duration_s + 12, bpm, seed, path)
        track = self._analyze(path, source="SPHURA AI music (ACE-Step)",
                              license="MIT: free to use, including monetised videos")  # fmt: skip
        beat_sync_cuts(clip, track.beat_s or 60 / bpm)
        m = plan_music(clip, [], track)
        if m is None:
            return False
        tempo = 60 / track.beat_s if track.beat_s else float(bpm)
        for k in (2.0, 0.5):  # detector locked onto half or double beats: report the real tempo
            if abs(tempo / bpm - k) < 0.12:
                tempo /= k
        m["generated"] = {"mood": mood, "label": label, "bpm": round(tempo), "ai": True,
                          "prompt": prompt}  # fmt: skip
        self._finish_music(clip, m, track, choice, transcript)
        return True

    def _score(
        self, clip: Clip, choice: str, judgement: Any, transcript: Any, prev: str = "",
        force_mood: str = "",
    ) -> None:  # fmt: skip
        category = judgement.category.value if judgement and judgement.category else None
        # Picking the score again composes a new one: different seed, not the same style
        seed = _seed(clip.id + prev)
        recent = self._recent_moods.setdefault(clip.project_id, [])
        avoid = (*recent[-2:], prev) if prev else tuple(recent[-2:])
        mood = force_mood or mood_for(
            category, tuple(judgement.tags) if judgement else (), clip.style, seed, avoid
        )
        recent.append(mood)
        if self.ai_music_ready():
            try:
                if self._ai_track(clip, choice, judgement, transcript, mood, seed):
                    return
            except SphuraError as e:
                log.warning("AI music failed, using the built-in score: %s", e)
        pieces = clip.speed_pieces()
        anchor = effects_anchor(clip)

        def times() -> tuple[float, float]:
            a = clip.to_output_time(anchor) if anchor is not None else 0.0
            back = clip.to_output_time(pieces[-1][1]) if pieces else a
            return a, back

        a_out, back = times()
        bpm = fit_bpm(mood, a_out, back, seed)
        beat_sync_cuts(clip, 60 / bpm)
        pieces = clip.speed_pieces()
        anchor = effects_anchor(clip)
        a_out, back = times()
        dur = clip.output_duration_s
        path = self._storage.artifact_path(clip.project_id, "music", f"{clip.id}_score.wav")
        write_score(path, compose(ScorePlan(dur, bpm, a_out, back, mood, seed)))
        beat = 60 / bpm
        track = MusicTrack(str(path), dur + 1, back, 0.5, beat, a_out % beat)
        m = plan_music(clip, [], track)
        if m is None:
            return
        m.update(
            offset_s=0.0,
            choice=choice,
            credit="",
            generated={"mood": mood, "label": MOODS[mood].label, "bpm": round(bpm)},
        )
        if clip.style == "cinematic":
            m.update(gain_db=-7.0, duck=False, dip=None)
            clip.audio.game_gain = 0.45
        elif clip.style == "meme":
            m.update(gain_db=-16.0, dip=None)
        volume = (clip.publish.get("music") or {}).get("volume_db")
        if volume is not None:
            m["gain_db"] = float(volume)
        if transcript is not None:
            spoken = [(s.start_s, s.end_s) for s in transcript.segments if s.text.strip()]
            m["dialogue"] = dialogue_ranges(clip, spoken)
        align_to_beats(clip, m, track)  # snap accents onto the score's beats
        m["offset_s"] = 0.0  # the score is written on the clip's own timeline
        clip.effects.append(m)

    def reel_mood(self, judgement: Any, seed: int) -> tuple[str, float]:
        """Mood and tempo for a highlight reel, known before the cuts are snapped to it."""
        category = judgement.category.value if judgement and judgement.category else None
        tags = tuple(judgement.tags) if judgement else ()
        mood = mood_for(category, tags, "hype", seed)
        return mood, base_bpm(mood, seed)

    def reel_music(
        self, dest: Path, duration_s: float, mood: str, bpm: float, seed: int,
        accent_s: float, judgement: Any, game: str | None,
    ) -> dict[str, Any]:  # fmt: skip
        """A full-length soundtrack for a reel: the local AI model when it's installed,
        else the built-in composer (its big hit placed on the reel's climax)."""
        label = MOODS[mood].label
        if self.ai_music_ready():
            assert self._ai is not None and self._analyze is not None
            try:
                prompt = write_music_prompt(self._llm, judgement, game, mood, label, seed)
                self._ai.generate(prompt, duration_s + 4, round(bpm), seed, dest)
                track = self._analyze(dest)
                tempo = 60 / track.beat_s if track.beat_s else bpm
                for k in (2.0, 0.5):  # half- or double-time lock of the beat detector
                    if abs(tempo / bpm - k) < 0.12:
                        tempo /= k
                return {"path": str(dest), "bpm": tempo, "beat0_s": track.beat0_s,
                        "label": label, "mood": mood, "ai": True}  # fmt: skip
            except Exception as e:  # never lose the reel over the music
                log.warning("AI music for the reel failed, using the built-in score: %s", e)
        write_score(dest, compose(ScorePlan(duration_s + 1, bpm, accent_s, accent_s, mood, seed)))
        return {"path": str(dest), "bpm": bpm, "beat0_s": accent_s % (60 / bpm),
                "label": label, "mood": mood, "ai": False}  # fmt: skip

    def wav_for(self, project_id: str, asset_id: str) -> WavFile | None:
        """The analysis audio of a video (for effect timing), if it's still on disk."""
        return self._wav(Clip(project_id=project_id, asset_id=asset_id, title="", segments=[]))

    def set_music(self, clip_id: str, choice: str, volume_db: float | None = None) -> Clip:
        """choice: "auto", "none" or the path of a track in the music folder."""
        clip = self._clips.get(clip_id)
        if clip is None:
            raise NotFoundError("clip not found")
        tracks = self._music()
        if choice not in ("auto", "none", "score") and not any(t.path == choice for t in tracks):
            raise NotFoundError("that track isn't in your music folder")
        clip.publish = {**clip.publish, "music": {"choice": choice, "volume_db": volume_db}}
        moment = next(
            (r.candidate for r in self._queries.highlights(clip.project_id, None)
             if r.candidate.id == clip.candidate_id),
            None,
        )  # fmt: skip
        self._ctx = (
            moment.judgement if moment else None,
            self._optional(self._queries.transcript, clip.project_id),
        )
        self._apply_music(clip, tracks, reroll=choice == "score")
        self._clips.save(clip)
        return clip

    @staticmethod
    def _caption(clip: Clip) -> None:
        """On-screen text per style: setup then punchline in the meme caption bar, an
        all-caps opener for cinematic, the pop-in hook for hype. Text placed by hand in
        the timeline editor is left alone."""
        if clip.publish.get("overlays_edited"):
            return
        text = on_screen_text(clip.publish, clip.style).strip()
        total = clip.output_duration_s or clip.duration_s
        if clip.style == "meme" or clip.publish.get("caption_bar"):
            text = text or on_screen_text(clip.publish, "meme").strip()
            setup = str(clip.publish.get("setup", "")).strip()
            punch = str(clip.publish.get("punchline", "")).strip()
            anchor = effects_anchor(clip)
            k = clip.to_output_time(anchor) if anchor is not None else total * 0.65
            if "text" not in clip.publish and setup and punch and 1.5 < k < total - 0.5:
                end = max(total, 60.0)
                clip.overlays = [
                    TextOverlay(setup, 0.0, round(k - 0.1, 2), "bar"),
                    TextOverlay(punch, round(k, 2), end, "bar"),
                ]
                return
            # One line for the whole meme; the bar needs words, so fall back to the title
            text = text or str(clip.publish.get("title", ""))
            clip.overlays = [TextOverlay(text, 0.0, max(total, 60.0), "bar")] if text else []
        elif not text:
            clip.overlays = []
        elif clip.style == "cinematic":
            clip.overlays = [TextOverlay(text.upper(), 0.2, min(2.8, total), "caps")]
        else:
            clip.overlays = [TextOverlay(text, 0.0, min(3.0, total), "hook")]

    def edit_text(
        self,
        clip_id: str,
        text: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> Clip:
        """User edits to on-screen text and post copy; marked so they're never rewritten.
        Re-render afterwards to see the change."""
        clip = self._clips.get(clip_id)
        if clip is None:
            raise NotFoundError("clip not found")
        p = dict(clip.publish)
        if text is not None:
            p["text"] = text.strip()
        if title is not None:
            p["youtube_title"] = title.strip()
        if description is not None:
            p["description"] = description.strip()
        p["edited"] = True
        clip.publish = p
        self._caption(clip)
        self._clips.save(clip)
        return clip

    # ---------- timeline editor ----------

    def editor_state(self, clip_id: str) -> dict[str, Any]:
        clip = self._get(clip_id)
        asset = self._assets.get(clip.asset_id)
        if asset is None or asset.media_info is None:
            raise InvalidMediaError("source video details missing")
        v = asset.media_info.primary_video
        g = geometry(clip, v.display_width, v.display_height)
        kind = next((e["kind"] for e in clip.effects if e["type"] == "context"), "action")
        music = next((e for e in clip.effects if e["type"] == "music"), None)
        gen = (music or {}).get("generated") or {}
        return {
            "id": clip.id, "project_id": clip.project_id, "title": clip.title,
            "style": clip.style, "aspect": clip.aspect.value,
            "out_w": g.out_w, "out_h": g.out_h, "bar_h": g.bar_h,
            "source_duration": asset.media_info.duration_s,
            "segments": [[s.start_s, s.end_s] for s in clip.segments],
            "segments_out": _segments_out(clip),
            "teaser_out": round(clip.to_output_time(_teaser_len(clip)), 3),
            "duration": round(clip.output_duration_s, 3),
            "overlays": [{"text": o.text, "start_s": o.start_s, "end_s": o.end_s,
                          "style": o.style, "y": o.y, "size": o.size, "color": o.color}
                         for o in clip.overlays],
            "fx_off": list(clip.publish.get("fx_off") or []),
            "fx_kind": kind, "fx_label": LABELS.get(kind, ""), "toggles": list(TOGGLES),
            "caption_bar": bool(clip.publish.get("caption_bar")),
            "music": gen.get("label") or (Path(music["path"]).stem if music else None),
        }  # fmt: skip

    def editor_audio(
        self, clip_id: str, segments: list[list[float]] | None = None
    ) -> dict[str, Any]:
        """Waveforms for the editor's audio tracks: game audio per part (on the Short's
        timeline, slow-mo included) and the music as it plays. Unsaved cuts map 1:1."""
        clip = self._get(clip_id)
        frame, wav = 0.05, self._wav(clip)
        saved = segments is None or [[float(a), float(b)] for a, b in segments] == [
            [s.start_s, s.end_s] for s in clip.segments
        ]
        ranges = (
            clip.segments if saved else [TimeRange(float(a), float(b)) for a, b in segments or []]
        )

        def game(src: TimeRange, pre_s: float | None) -> list[float]:
            if wav is None:
                return []
            step = frame / 4
            r = wav.rms_frames(src.start_s, src.end_s, step)
            if pre_s is None:  # unsaved cuts: drawn 1:1 like the editor does
                quads = r[: len(r) // 4 * 4].reshape(-1, 4)
                return [float(v) for v in quads.max(axis=1)] if len(quads) else []
            out0 = clip.to_output_time(pre_s)
            n = max(1, int((clip.to_output_time(pre_s + src.duration_s) - out0) / frame))
            env = [-1.0] * n
            for k, v in enumerate(r):
                j = min(n - 1, int((clip.to_output_time(pre_s + k * step) - out0) / frame))
                env[j] = max(env[j], float(v))
            for j in range(n):  # slow motion stretches the audio: fill the gaps it leaves
                if env[j] < 0:
                    env[j] = env[j - 1] if j else 0.0
            return env

        teaser = next((e for e in clip.effects if e["type"] == "cold_open"), None)
        tease = TimeRange(teaser["start_s"], teaser["end_s"]) if teaser else None
        pre = tease.duration_s if tease else 0.0
        parts = []
        for seg in ranges:
            parts.append(game(seg, pre if saved else None))
            pre += seg.duration_s
        tease_env = game(tease, 0.0) if tease and saved else (game(tease, None) if tease else [])
        peak = max([v for e in [*parts, tease_env] for v in e] or [0.0]) or 1.0

        def norm(e: list[float]) -> list[float]:  # square root: closer to how loud it sounds
            return [round(min(1.0, v / peak) ** 0.5, 3) for v in e]

        return {
            "frame_s": frame,
            "teaser": norm(tease_env),
            "parts": [norm(e) for e in parts],
            "music": self._music_track(clip, frame),
        }

    def _music_track(self, clip: Clip, frame: float) -> dict[str, Any] | None:
        m = next((e for e in clip.effects if e["type"] == "music"), None)
        if m is None or not Path(m["path"]).is_file():
            return None
        path, gen = Path(m["path"]), m.get("generated") or {}
        dur, t0 = clip.output_duration_s, float(m.get("in_s", 0.0))
        env: list[float] = []
        if self._envelope is not None:
            with contextlib.suppress(Exception):  # a waveform is nice to have, never required
                env = self._envelope(path, m["offset_s"] + t0, dur - t0, frame)
        peak = max(env or [0.0]) or 1.0
        track = next((t for t in self._music() if t.path == m["path"]), None)
        if track is None and self._analyze is not None:
            with contextlib.suppress(Exception):
                track = self._analyze(path)
        bpm = gen.get("bpm") or (round(60 / track.beat_s) if track and track.beat_s else None)
        beats: list[float] = []
        if track and track.beat_s > 0 and bpm:  # beat marks, spaced by the tempo shown
            step = 60 / float(bpm)
            k = math.ceil((m["offset_s"] + t0 - track.beat0_s) / step)
            while (t := track.beat0_s + k * step - m["offset_s"]) < dur:
                beats.append(round(t, 3))
                k += 1
        return {
            "name": gen.get("label") or path.stem,
            "kind": "ai" if gen.get("ai") else "score" if gen else "track",
            "mood": gen.get("mood"), "bpm": bpm,
            "in_s": round(t0, 3), "gain_db": m.get("gain_db"), "duck": m.get("duck", True),
            "dip": m.get("dip"), "credit": m.get("credit", ""),
            "env": [round((v / peak) ** 0.5, 3) for v in env], "beats": beats,
        }  # fmt: skip

    def apply_edit(
        self, clip_id: str, segments: list[list[float]] | None,
        overlays: list[dict[str, Any]] | None, fx_off: list[str] | None,
        caption_bar: bool | None = None,
    ) -> Clip:  # fmt: skip
        """Save a timeline edit. New cuts or effect switches re-plan the effects and
        re-sync the music; text placed by hand is kept from then on."""
        clip = self._get(clip_id)
        asset = self._assets.get(clip.asset_id)
        src_len = asset.media_info.duration_s if asset and asset.media_info else 1e9
        replan = False
        if segments is not None:
            segs = [TimeRange(max(0.0, float(a)), min(src_len, float(b))) for a, b in segments]
            segs = [s for s in segs if s.end_s - s.start_s >= 0.3]
            if not segs:
                raise SphuraError("Keep at least one part of the clip (0.3 s or longer)")
            if sum(s.duration_s for s in segs) > 180:
                raise SphuraError("Shorts can be at most 3 minutes long")
            replan = [(s.start_s, s.end_s) for s in segs] != [
                (s.start_s, s.end_s) for s in clip.segments
            ]
            clip.segments = segs
            if replan:
                clip.publish = {**clip.publish, "cuts_edited": True}
        if fx_off is not None:
            off = sorted({f for f in fx_off if f in TOGGLES})
            replan = replan or off != sorted(clip.publish.get("fx_off") or [])
            clip.publish = {**clip.publish, "fx_off": off}
        if replan:
            moment = next(
                (r.candidate for r in self._queries.highlights(clip.project_id, None)
                 if r.candidate.id == clip.candidate_id),
                None,
            )  # fmt: skip
            self._ctx = (
                moment.judgement if moment else None,
                self._optional(self._queries.transcript, clip.project_id),
            )
            peak = moment.peak_s if moment else clip.segments[0].start_s
            tl = self._optional(self._queries.signals, clip.project_id)
            old_music = next((e for e in clip.effects if e["type"] == "music"), None)
            self._plan(clip, peak, tl, self._wav(clip))
            if not self._keep_music(clip, old_music):
                self._apply_music(clip, self._music())
        bar_was = bool(clip.publish.get("caption_bar"))
        if caption_bar is not None and clip.style != "meme":
            self._set_bar(clip, caption_bar)
        if overlays is not None:
            total = clip.output_duration_s
            clip.overlays = [
                TextOverlay(
                    str(o.get("text", "")).strip()[:200],
                    round(max(0.0, float(o["start_s"])), 3),
                    round(min(total, float(o["end_s"])), 3),
                    str(o.get("style", "plain")) if o.get("style") in OVERLAY_STYLES else "plain",
                    None if o.get("y") is None else min(max(float(o["y"]), 0.0), 1.0),
                    min(max(float(o.get("size") or 1.0), 0.4), 3.0),
                    str(o.get("color") or "")[:7],
                )
                for o in overlays
                if str(o.get("text", "")).strip() and float(o["end_s"]) > float(o["start_s"])
            ]
            clip.publish = {**clip.publish, "overlays_edited": True}
        elif replan:
            self._caption(clip)  # timings follow the re-planned key hit
        bar_now = bool(clip.publish.get("caption_bar"))
        if bar_now and not bar_was and not any(o.style == "bar" for o in clip.overlays):
            # Switched on: fill the new bar (setup, then punchline) and keep the user's text
            mine = [o for o in clip.overlays if o.style not in ("hook", "caps")]
            edited = clip.publish.pop("overlays_edited", None)
            self._caption(clip)
            clip.overlays = [*clip.overlays, *mine]
            if edited:
                clip.publish["overlays_edited"] = True
        elif bar_was and not bar_now:
            clip.overlays = [o for o in clip.overlays if o.style != "bar"]
        self._clips.save(clip)
        return clip

    @staticmethod
    def _set_bar(clip: Clip, on: bool) -> None:
        if bool(clip.publish.get("caption_bar")) != on:
            clip.publish = {**clip.publish, "caption_bar": on}
            clip.crop_path = []  # the footage area changes shape: re-frame the camera

    def _keep_music(self, clip: Clip, old: dict[str, Any] | None) -> bool:
        """After an edit, re-sync the same generated track instead of making a new one."""
        gen = (old or {}).get("generated") or {}
        if not old or not gen or not Path(old["path"]).is_file():
            return False
        choice = (clip.publish.get("music") or {}).get("choice", "auto")
        _, transcript = self._ctx
        if gen.get("ai") and self._analyze is not None:
            track = self._analyze(Path(old["path"]))
            if track.length_s < clip.output_duration_s + 1:
                return False  # the edit made the Short longer than the track
            beat_sync_cuts(clip, track.beat_s or 0.5)
            m = plan_music(clip, [], track)
            if m is None:
                return False
            m["generated"] = gen
            self._finish_music(clip, m, track, choice, transcript)
            return True
        if gen.get("mood") in MOODS:  # built-in score: recompose in the same style
            self._score(clip, choice, self._ctx[0], transcript, force_mood=str(gen["mood"]))
            return True
        return False

    def draft(self, clip_id: str) -> Job:
        """Quick preview render without burned-in text (the editor draws text live)."""
        clip = self._get(clip_id)
        return self._queue.enqueue(
            Job(type=JobType.RENDER, project_id=clip.project_id,
                payload={"clip_id": clip.id, "draft": True})
        )  # fmt: skip

    def draft_path(self, clip_id: str) -> Path:
        clip = self._get(clip_id)
        return self._storage.artifact_path(clip.project_id, "drafts", f"{clip.id}.mp4")

    def frame_slot(self, clip_id: str, t: float) -> tuple[Path, str, float]:
        """Cache file, source and snapped time of a filmstrip frame (every 0.5 s of the
        recording, shared by all Shorts of the stream)."""
        clip = self._get(clip_id)
        asset = self._assets.get(clip.asset_id)
        if asset is None:
            raise NotFoundError("source video not found")
        half = max(0, round(t * 2))
        path = self._storage.artifact_path(clip.project_id, "thumbs", f"strip_{half}.jpg")
        return path, asset.source_uri, half / 2

    def source_path(self, clip_id: str) -> Path:
        clip = self._get(clip_id)
        asset = self._assets.get(clip.asset_id)
        if asset is None:
            raise NotFoundError("source video not found")
        return Path(asset.source_uri)

    def _get(self, clip_id: str) -> Clip:
        clip = self._clips.get(clip_id)
        if clip is None:
            raise NotFoundError("clip not found")
        return clip

    def rerender(self, clip_id: str) -> Job:
        clip = self._clips.get(clip_id)
        if clip is None:
            raise NotFoundError("clip not found")
        folder = clip.publish.get("export_dir")
        if not folder:  # made before export folders were remembered: use the usual one
            name = self._queries.project(clip.project_id).name
            folder = str(shorts_folder(name) / clip.style)
        return self.render([clip], Path(folder))[0]

    def _frames(self, clip: Clip, peak_s: float) -> list[ImageInput]:
        return self.frames_at(clip.asset_id, peak_s)

    def frames_at(self, asset_id: str, peak_s: float) -> list[ImageInput]:
        """Three frames (build-up, the hit, aftermath) for the AI to look at."""
        asset = self._assets.get(asset_id)
        if self._sample is None or asset is None:
            return []
        times = [max(0.0, peak_s - 2.0), peak_s, peak_s + 1.5]
        try:
            jpegs = self._sample(asset.source_uri, times)
        except Exception:
            return []
        labels = ("Just before", "The key moment", "Right after")
        return [ImageInput(t, j, lab) for t, j, lab in zip(times, jpegs, labels, strict=False)]

    def _wav(self, clip: Clip) -> WavFile | None:
        asset = self._assets.get(clip.asset_id)
        mix = choose_mix_track(asset.media_info.audio) if asset and asset.media_info else None
        if mix is None:
            return None
        path = self._storage.artifact_path(clip.project_id, "proxy", f"audio_s{mix.index}.wav")
        try:
            return WavFile(path) if path.is_file() else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _optional(fn: Any, project_id: str) -> Any:
        try:
            return fn(project_id)
        except NotFoundError:
            return None


class RenderHandler:
    def __init__(
        self,
        clips: ClipRepository,
        assets: MediaAssetRepository,
        storage: StorageProvider,
        renderer: VideoRenderer,
        decoder: MediaDecoder,
        people: PeopleDetector | None = None,
    ) -> None:
        self._clips = clips
        self._assets = assets
        self._storage = storage
        self._renderer = renderer
        self._decoder = decoder
        self._people = people
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def __call__(self, ctx: JobContext) -> dict[str, Any]:
        # One render per clip at a time; a queued re-render then picks up the latest edit
        with self._lock_for(ctx.job.payload["clip_id"]):
            return self._render(ctx)

    def _lock_for(self, clip_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(clip_id, threading.Lock())

    def _render(self, ctx: JobContext) -> dict[str, Any]:
        clip = self._clips.get(ctx.job.payload["clip_id"])
        if clip is None:
            raise NotFoundError("clip not found")
        asset = self._assets.get(clip.asset_id)
        if asset is None or asset.media_info is None:
            raise InvalidMediaError("source video details missing; re-run ingest")
        mix = choose_mix_track(asset.media_info.audio)
        suffix = (
            ""
            if clip.aspect == AspectRatio.LANDSCAPE
            else "_" + clip.aspect.value.replace(":", "x")
        )
        plan = clip.publish.get("plan") or {}
        if plan.get("variant"):  # versions of one planned Short must not overwrite each other
            suffix += f"_{plan['variant']}-{slugify(str(plan.get('label', '')), 12)}"
        elif clip.style != "hype":
            suffix += f"_{clip.style}"
        name = f"{clip.rank or 0:02d}_{slugify(clip.title)}{suffix}.mp4"
        out = self._storage.artifact_path(clip.project_id, "renders", name)

        stale = clip.crop_version != reframe.VERSION
        if stale and clip.crop_path:
            clip.crop_path = []
            clip.layout = "auto"  # let the new planner choose crop vs fit again
        if clip.aspect != AspectRatio.LANDSCAPE and not clip.crop_path and clip.layout != "fit":
            ctx.report("reframe", 0.0, clip.title)
            v = asset.media_info.primary_video
            g = geometry(clip, v.display_width, v.display_height)
            plan = plan_reframe(
                clip,
                asset.source_uri,
                self._decoder,
                v.display_width / v.display_height,
                g.video_aspect,
                self._people,
            )
            clip.crop_path = plan.keyframes
            clip.crop_version = reframe.VERSION
            if clip.layout == "auto":
                clip.layout = plan.layout
            self._clips.save(clip)  # the camera path is part of the editable EDL

        if ctx.job.payload.get("draft"):
            preview = copy.copy(clip)
            preview.overlays = []  # the editor shows text live on top
            draft = self._storage.artifact_path(clip.project_id, "drafts", f"{clip.id}.mp4")
            ctx.report("render", 0.0, clip.title)
            done = self._renderer.render(
                preview, asset.source_uri, asset.media_info, mix.index if mix else None,
                draft, lambda f: ctx.report("render", f, clip.title),
            )  # fmt: skip
            return {"clip_id": clip.id, "path": str(done.path), "draft": True,
                    "duration_s": round(done.duration_s, 2)}  # fmt: skip

        ctx.report("render", 0.0, clip.title)
        result = self._renderer.render(
            clip,
            asset.source_uri,
            asset.media_info,
            mix.index if mix else None,
            out,
            lambda f: ctx.report("render", f, clip.title),
        )
        old = self._clips.output_of(clip.id)
        self._clips.set_output(clip.id, str(result.path))
        if old and Path(old) != result.path and Path(old).parent == out.parent:
            with contextlib.suppress(OSError):  # a render saved aside while the file was open
                Path(old).unlink()
        exported = None
        if export_dir := ctx.job.payload.get("export_dir"):
            dest = Path(export_dir).expanduser()
            dest.mkdir(parents=True, exist_ok=True)
            exported = str(Path(shutil.copy2(result.path, dest / name)))
            if clip.publish:  # title, description and hashtags, ready to paste
                music = next((e for e in clip.effects if e["type"] == "music"), {})
                (dest / name).with_suffix(".txt").write_text(
                    render_txt(clip.publish, music.get("credit", "")), encoding="utf-8"
                )
        return {
            "clip_id": clip.id,
            "path": str(result.path),
            "exported": exported,
            "encoder": result.encoder,
            "duration_s": round(result.duration_s, 2),
            "aspect": clip.aspect.value,
            "layout": clip.layout,
        }


def _teaser_len(clip: Clip) -> float:
    return float(sum(e["end_s"] - e["start_s"] for e in clip.effects if e["type"] == "cold_open"))


def _segments_out(clip: Clip) -> list[list[float]]:
    """Where each source part lands on the finished Short's timeline (after the teaser
    and speed changes): what the editor draws and the preview plays."""
    t, out = _teaser_len(clip), []
    for seg in clip.segments:
        a, b = t, t + seg.duration_s
        out.append([round(clip.to_output_time(a), 3), round(clip.to_output_time(b), 3)])
        t = b
    return out


def _dialogue(transcript: Any, clip: Clip) -> list[str]:
    """Lines spoken inside the clip (often cutscene dialogue that names who's who)."""
    if transcript is None:
        return []
    return [
        s.text.strip()
        for s in transcript.segments
        if any(s.end_s > seg.start_s and s.start_s < seg.end_s for seg in clip.segments)
        and s.text.strip()
    ]


def _seed(clip_id: str) -> int:
    """Stable per clip, so re-renders keep the same melody."""
    return int(clip_id[:8], 16) if all(c in "0123456789abcdef" for c in clip_id[:8]) else 7
