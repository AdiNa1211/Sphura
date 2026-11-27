"""Editor M1: undo/autosave backend (save without render, deferred re-plan, version check),
text ownership, fast drafts, the reserved preview worker, relink, and the final export
staying exactly as before. Real edit lists from God of War Ragnarök and Phantom Blade Zero
Shorts (tests/fixtures); tests that need a whole SPHURA data folder or real footage read
SPHURA_TEST_DATA / SPHURA_TEST_FOOTAGE and are skipped when those aren't there."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sphura.application.clips import draft_key
from sphura.domain.clip_codec import clip_from_dict, clip_to_dict
from sphura.domain.errors import ConflictError, SphuraError
from sphura.domain.models import Clip, Job, JobType, TextOverlay, TimeRange
from sphura.infrastructure.media.ffmpeg.render import FFmpegRenderer
from sphura.infrastructure.persistence.sqlite.repositories import _media_info_from_json
from sphura.pipeline.tracks import choose_mix_track

FIX = Path(__file__).parent / "fixtures"
REAL = json.loads((FIX / "real_clips.json").read_text())
GOLDEN = json.loads((FIX / "final_commands_pre_m1.json").read_text())
DATA = Path(os.environ.get("SPHURA_TEST_DATA", "/tmp/sd"))
FOOTAGE = Path(os.environ.get("SPHURA_TEST_FOOTAGE", "/tmp/m1ex"))
GOW_PROJECT = "3d25079368224f21ade428356444b5ca"  # God of War Ragnarök Part 12


def real_clip(prefix: str) -> tuple[Clip, Any]:
    d = next(c for c in REAL["clips"] if c["id"].startswith(prefix))
    c = clip_from_dict(json.loads(json.dumps(d)))
    info = _media_info_from_json(json.dumps(REAL["media"][c.asset_id]["media_info"]))
    return c, info


def renderer() -> FFmpegRenderer:
    tc = SimpleNamespace(ffmpeg="ffmpeg", profile=lambda: SimpleNamespace(hwaccels=("cuda",)),
                         has_filter=lambda n: True)  # fmt: skip
    return FFmpegRenderer(tc)  # type: ignore[arg-type]


def command(c: Clip, info: Any, enc: str, draft: bool = False) -> list[str]:
    mix = choose_mix_track(info.audio)
    return renderer().build_command(
        c, "SOURCE.mp4", info, mix.index if mix else None, Path("OUT.mp4"), enc,
        "TEXT.ass" if c.overlays else None, "SFX.wav", draft=draft,
    )  # fmt: skip


# ---------- final export unchanged, fast drafts ----------


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_final_export_command_is_unchanged_from_before_m1(key: str) -> None:
    clip_id, enc = key.split("/")
    c, info = real_clip(clip_id)
    assert command(c, info, enc) == GOLDEN[key]


@pytest.mark.parametrize("prefix", ["d3eef8", "0bb79c"])  # GoW Frost Guardian, PBZ combat
def test_draft_is_half_size_30fps_fast_preset(prefix: str) -> None:
    c, info = real_clip(prefix)
    final, draft = command(c, info, "h264_nvenc"), command(c, info, "h264_nvenc", draft=True)
    graph = draft[draft.index("-filter_complex") + 1]
    assert "540:960" in graph and "1080:1920" not in graph
    assert draft[draft.index("-preset") + 1] == "p1"
    assert final[final.index("-preset") + 1] == "p5"
    if info.primary_video.fps > 30.5:  # GoW is 60 fps; PBZ is already 30
        assert "fps=30" in graph
    x264 = command(c, info, "libx264", draft=True)
    assert x264[x264.index("-preset") + 1] == "ultrafast"


def test_draft_key_ignores_text_but_not_cuts_or_effects() -> None:
    c, _ = real_clip("d3eef8")
    k = draft_key(c)
    c.overlays = [*c.overlays, TextOverlay("Thor is done", 1.0, 2.0, "plain", owner="user")]
    assert draft_key(c) == k  # text is drawn live by the editor
    c.segments = [TimeRange(c.segments[0].start_s + 1, c.segments[0].end_s)]
    assert draft_key(c) != k
    c2, _ = real_clip("d3eef8")
    c2.publish = {**c2.publish, "fx_off": ["rgb"]}
    assert draft_key(c2) != k


def test_text_owner_survives_save_and_old_edit_lists_still_load() -> None:
    c, _ = real_clip("f3d4f4")
    c.overlays = [TextOverlay("Your text", 0.5, 2.0, "plain", owner="user")]
    back = clip_from_dict(clip_to_dict(c))
    assert back.overlays[0].owner == "user"
    d = clip_to_dict(c)
    d["overlays"] = [o[:7] for o in d["overlays"]]  # saved before M1: 7 fields
    assert clip_from_dict(d).overlays[0].owner == "ai"


# ---------- reserved preview worker ----------


def test_preview_render_runs_while_long_jobs_hold_every_normal_worker(tmp_path: Path) -> None:
    """Synthetic jobs (the scheduling is what's tested): two long jobs fill both normal
    workers; an editor preview still starts right away on the reserved worker."""
    from sphura.infrastructure.events.bus import InProcessEventBus
    from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase
    from sphura.infrastructure.persistence.sqlite.repositories import SqliteJobRepository
    from sphura.infrastructure.queue.local_queue import LocalJobQueue

    release, started = threading.Event(), {}

    def long_job(ctx: Any) -> dict[str, Any]:
        started[ctx.job.id] = time.monotonic()
        release.wait(10)
        return {}

    def render(ctx: Any) -> dict[str, Any]:
        started[ctx.job.id] = time.monotonic()
        return {}

    from sphura.domain.models import Project
    from sphura.infrastructure.persistence.sqlite.repositories import SqliteProjectRepository

    db = SqliteDatabase(tmp_path / "q.db")
    SqliteProjectRepository(db).add(Project(name="queue test", id="p"))
    q = LocalJobQueue(SqliteJobRepository(db), InProcessEventBus(), 2)
    q.register(JobType.ANALYZE, long_job)
    q.register(JobType.RENDER, render)
    q.start()
    try:
        long_ids = [q.enqueue(Job(type=JobType.ANALYZE, project_id="p")).id for _ in range(2)]
        time.sleep(0.5)
        final = q.enqueue(Job(type=JobType.RENDER, project_id="p", payload={"clip_id": "c"}))
        draft = q.enqueue(
            Job(type=JobType.RENDER, project_id="p", payload={"clip_id": "c", "draft": True})
        )
        deadline = time.monotonic() + 3
        while draft.id not in started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert all(i in started for i in long_ids)
        assert draft.id in started  # preview didn't wait for the long jobs
        assert final.id not in started  # a normal render still queues behind them
    finally:
        release.set()
        q.stop()


# ---------- with a real SPHURA data folder ----------

needs_data = pytest.mark.skipif(
    not (DATA / "sphura.db").is_file(), reason=f"no SPHURA data folder at {DATA}"
)


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    from sphura.config import Settings
    from sphura.container import build_local_engine

    ignore = shutil.ignore_patterns("models", "voice_cache", "voice_bench", "m1-test-excerpts")
    shutil.copytree(DATA, tmp_path / "data", ignore=ignore)
    os.environ["SPHURA_DATA_DIR"] = str(tmp_path / "data")
    return build_local_engine(Settings.from_env())


def gow_clip(engine: Any, prefix: str) -> Clip:
    return next(c for c, _ in engine.queries.clips(GOW_PROJECT) if c.id.startswith(prefix))


@needs_data
def test_autosave_stores_the_cut_and_defers_the_replan_to_the_render(engine: Any) -> None:
    clips = engine.clips
    c = gow_clip(engine, "d3eef8")  # Frost Guardian: cold open, speed ramps, music
    before = json.dumps(c.effects, sort_keys=True)
    segs = [[c.segments[0].start_s + 1.0, c.segments[0].end_s]]
    saved = clips.apply_edit(c.id, segs, None, None, plan=False, base_version=c.version)
    assert saved.version == c.version + 1
    assert [[s.start_s, s.end_s] for s in saved.segments] == segs
    assert json.dumps(saved.effects, sort_keys=True) == before  # nothing re-planned yet
    assert saved.publish.get("replan_pending")
    clips.flush_plan(c.id)  # what Update preview / Export / a re-render run first
    after = gow_clip(engine, "d3eef8")
    assert not after.publish.get("replan_pending")
    teaser = [e for e in after.effects if e["type"] == "cold_open"]
    for e in teaser:  # the cold open now comes from inside the trimmed cut
        assert segs[0][0] - 0.01 <= e["start_s"] and e["end_s"] <= segs[0][1] + 0.01


@needs_data
def test_a_save_based_on_an_old_version_is_refused(engine: Any) -> None:
    c = gow_clip(engine, "f3d4f4")
    engine.clips.apply_edit(c.id, None, None, ["rgb"], plan=False, base_version=c.version)
    with pytest.raises(ConflictError):
        engine.clips.apply_edit(c.id, None, None, [], plan=False, base_version=c.version)


@needs_data
def test_text_you_edit_is_saved_as_yours(engine: Any) -> None:
    c = gow_clip(engine, "f3d4f4")
    mine = {"text": "Thor is done talking", "start_s": 0.5, "end_s": 2.5, "owner": "user"}
    ov = [
        {**mine, "style": "plain"},
        {"text": "AI hook", "start_s": 3, "end_s": 4, "style": "hook"},
    ]
    engine.clips.apply_edit(c.id, None, ov, None, plan=False)
    st = engine.clips.editor_state(c.id)
    assert [(o["text"], o["owner"]) for o in st["overlays"]] == [
        ("Thor is done talking", "user"), ("AI hook", "ai")]  # fmt: skip


@needs_data
def test_editor_state_reports_a_missing_recording(engine: Any) -> None:
    st = engine.clips.editor_state(gow_clip(engine, "f3d4f4").id)
    assert st["source_missing"] is (not Path(st["source_path"]).is_file())
    assert st["asset_id"] and "draft_fresh" in st and "version" in st


# ---------- relink with real footage ----------

needs_footage = pytest.mark.skipif(
    not (FOOTAGE / "ex_1375.mp4").is_file() or not (FOOTAGE / "ex_1805.mp4").is_file(),
    reason=f"no real footage excerpts in {FOOTAGE}",
)


@needs_data
@needs_footage
def test_relink_accepts_the_same_recording_and_refuses_another(engine: Any, tmp_path: Path) -> None:
    rec = tmp_path / "Part 12 excerpt.mp4"
    shutil.copy(FOOTAGE / "ex_1375.mp4", rec)
    res = engine.ingest.ingest_local_file(str(rec)).asset
    moved = tmp_path / "moved" / "Part 12 excerpt.mp4"
    moved.parent.mkdir()
    rec.rename(moved)
    with pytest.raises(SphuraError, match="isn't the same recording"):
        engine.ingest.relink(res.id, str(FOOTAGE / "ex_1805.mp4"))
    a = engine.ingest.relink(res.id, str(moved))
    assert a.source_uri == str(moved.resolve())


@needs_data
def test_a_rerender_from_the_shorts_page_runs_the_pending_replan_first(engine: Any) -> None:
    c = gow_clip(engine, "d3eef8")
    segs = [[c.segments[0].start_s + 1.0, c.segments[0].end_s]]
    engine.clips.apply_edit(c.id, segs, None, None, plan=False)
    engine.clips.rerender(c.id)  # e.g. after changing the music or the voice-over
    assert not gow_clip(engine, "d3eef8").publish.get("replan_pending")


# ---------- M2: parts in any order, your speed changes, per-part sound, markers ----------


def speed_of(c: Clip) -> list[dict[str, Any]]:
    return sorted(
        (e for e in c.effects if e["type"] in ("speed", "slowmo")), key=lambda e: e["start_s"]
    )


def src_at(c: Clip, pre_s: float) -> float:
    lead = sum(e["end_s"] - e["start_s"] for e in c.effects if e["type"] == "cold_open")
    return c.to_source_time(pre_s - lead)


@needs_data
def test_parts_play_in_your_order_and_the_key_hit_is_found_in_its_part(engine: Any) -> None:
    c = gow_clip(engine, "cc56bf")  # 3 parts, cold open, slow-mo ramp on the key hit
    hit = src_at(c, speed_of(c)[1]["start_s"])
    order = [[s.start_s, s.end_s] for s in reversed(c.segments)]
    engine.clips.apply_edit(c.id, order, None, None, plan=False)
    engine.clips.flush_plan(c.id)
    after = gow_clip(engine, "cc56bf")
    # Same order (beat sync may trim a sliver off a part)
    assert all(
        abs(s.start_s - o[0]) < 1 and abs(s.end_s - o[1]) < 1
        for s, o in zip(after.segments, order, strict=True)
    )
    ramp = speed_of(after)
    assert len(ramp) == 3
    # Same moment, now near the start of the Short (ramp edges clamp there; beat sync trims)
    assert abs(src_at(after, ramp[1]["start_s"]) - hit) < 0.3


@needs_data
def test_your_speed_change_survives_a_cut_change(engine: Any) -> None:
    c = gow_clip(engine, "cc56bf")
    pieces = [dict(p) for p in speed_of(c)]
    src = src_at(c, pieces[1]["start_s"])
    pieces[1].update(factor=0.25, owner="user", src_s=src)
    engine.clips.apply_edit(c.id, None, None, None, plan=False, speed=pieces)
    saved = gow_clip(engine, "cc56bf")
    assert saved.publish.get("replan_pending")  # music and accents follow the new timing
    segs = [[s.start_s, s.end_s] for s in saved.segments]
    segs[2][1] -= 2.0  # cut the end of the last part
    engine.clips.apply_edit(c.id, segs, None, None, plan=False)
    engine.clips.flush_plan(c.id)
    after = speed_of(gow_clip(engine, "cc56bf"))
    mine = [p for p in after if p.get("owner") == "user"]
    assert len(mine) == 1 and mine[0]["factor"] == 0.25
    assert abs(src_at(gow_clip(engine, "cc56bf"), mine[0]["start_s"]) - src) < 0.05
    clash = [
        p
        for p in after
        if p is not mine[0] and p["start_s"] < mine[0]["end_s"] and p["end_s"] > mine[0]["start_s"]
    ]
    assert not clash


@needs_data
def test_a_speed_change_you_removed_stays_removed(engine: Any) -> None:
    c = gow_clip(engine, "cc56bf")
    pieces = speed_of(c)
    gone = src_at(c, pieces[0]["start_s"])
    engine.clips.apply_edit(c.id, None, None, None, plan=False, speed=pieces[1:])
    segs = [[s.start_s, s.end_s] for s in c.segments]
    segs[0][0] += 1.0
    engine.clips.apply_edit(c.id, segs, None, None, plan=False)
    engine.clips.flush_plan(c.id)
    after = gow_clip(engine, "cc56bf")
    assert all(abs(src_at(after, p["start_s"]) - gone) >= 0.3 for p in speed_of(after))


@needs_data
def test_unchanged_planner_speeds_pass_through_untouched(engine: Any) -> None:
    c = gow_clip(engine, "bccd4c")  # a near-freeze (0.08x) the editor can't set itself
    before = json.dumps(c.effects, sort_keys=True)
    out = engine.clips.apply_edit(c.id, None, None, None, plan=False, speed=speed_of(c))
    assert json.dumps(out.effects, sort_keys=True) == before
    assert not out.publish.get("replan_pending")


def test_part_volume_changes_only_its_part_and_untouched_shorts_render_as_before() -> None:
    c, info = real_clip("cc56bf")
    plain = command(c, info, "h264_nvenc")
    c.segment_audio = [{"gain_db": 0.0, "mute": False}] * 3
    assert command(c, info, "h264_nvenc") == plain  # all at defaults: same command
    c.segment_audio = [
        {"gain_db": 0.0, "mute": False},
        {"gain_db": 6.0, "mute": False},
        {"gain_db": 0, "mute": True},
    ]
    graph = command(c, info, "h264_nvenc")[
        command(c, info, "h264_nvenc").index("-filter_complex") + 1
    ]
    lead = 1  # the cold open is input 0
    assert (
        "d=0.05[a0]" in graph
        and f"volume=1.9953[a{lead + 1}]" in graph
        and f"volume=0[a{lead + 2}]" in graph
    )
    assert f"d=0.05[a{lead}]" in graph


def test_part_volume_is_stored_only_when_set() -> None:
    c, _ = real_clip("cc56bf")
    assert "segment_audio" not in clip_to_dict(c)
    assert draft_key(c) == draft_key(clip_from_dict(json.loads(json.dumps(clip_to_dict(c)))))
    c.segment_audio = [
        {"gain_db": -3.0, "mute": False},
        {"gain_db": 0.0, "mute": False},
        {"gain_db": 0.0, "mute": False},
    ]
    back = clip_from_dict(json.loads(json.dumps(clip_to_dict(c))))
    assert back.segment_audio == c.segment_audio
    assert draft_key(back) != draft_key(real_clip("cc56bf")[0])  # the preview must re-render


@needs_data
def test_volume_and_markers_save_and_a_changed_cut_from_an_older_page_resets_volume(
    engine: Any,
) -> None:
    c = gow_clip(engine, "cc56bf")
    vol = [
        {"gain_db": 4, "mute": False},
        {"gain_db": 0, "mute": True},
        {"gain_db": 0, "mute": False},
    ]
    marks = [{"src_s": c.segments[1].start_s + 1.0, "label": "the hit"}]
    out = engine.clips.apply_edit(
        c.id, None, None, None, plan=False, segment_audio=vol, markers=marks
    )
    assert out.segment_audio == [
        {"gain_db": 4.0, "mute": False},
        {"gain_db": 0.0, "mute": True},
        {"gain_db": 0.0, "mute": False},
    ]
    st = engine.clips.editor_state(c.id)
    assert st["markers"] == [{"src_s": round(marks[0]["src_s"], 3), "label": "the hit"}]
    assert st["seg_audio"] == out.segment_audio and not out.publish.get("replan_pending")
    engine.clips.apply_edit(
        c.id, [[s.start_s, s.end_s] for s in c.segments[:2]], None, None, plan=False
    )
    assert gow_clip(engine, "cc56bf").segment_audio == []  # can't tell which part kept which volume
