"""Voice Intelligence 2.0: mood transitions, segmentation, pronunciation checks by listening
back, audio measurements, consistency flags, report separation and benchmark summaries."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sphura.application.voice_bench import summarise
from sphura.application.voice_intel import VoiceIntelService
from sphura.application.voice_qa import VoiceQAService
from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase
from sphura.infrastructure.persistence.sqlite.voice import SqliteVoiceRepository
from sphura.pipeline import voice_audio as A
from sphura.pipeline import voice_intel as V

SR = 24000
P = V.Piece


@pytest.fixture
def svc(tmp_path: Path) -> VoiceIntelService:
    queries = SimpleNamespace(list_projects=lambda n: [], candidates=lambda pid: [])
    return VoiceIntelService(SqliteVoiceRepository(SqliteDatabase(tmp_path / "t.db")), queries,
                             None, None)  # fmt: skip


def tone(seconds: float, hz: float = 220.0, amp: float = 0.3, sr: int = SR) -> A.F32:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def speechy(seconds: float, hz: float = 140.0, amp: float = 0.3, seed: int = 0) -> A.F32:
    """Voiced bursts with gaps: enough like speech for level, pitch and pause measures."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = (np.sin(2 * np.pi * 4 * t) > -0.2).astype(np.float32)
    buzz = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6))
    return (amp * env * (buzz / 2 + 0.02 * rng.standard_normal(len(t)))).astype(np.float32)


# ---------- emotion transitions ----------


def test_brief_mood_detour_keeps_the_same_reference() -> None:
    groups, notes = V.smooth_moods([("neutral", 10, False), ("tense", 2, False),
                                    ("neutral", 10, False)], 0.5)  # fmt: skip
    assert groups == ["neutral"] * 3
    assert notes[0]["action"] == "kept"


def test_real_mood_change_switches_and_big_jumps_are_marked_abrupt() -> None:
    groups, notes = V.smooth_moods([("neutral", 10, False), ("excited", 12, True)], 0.5)
    assert groups == ["neutral", "intense"]
    assert notes[0]["action"] == "switched"
    _, notes = V.smooth_moods([("sad", 10, False), ("excited", 12, True)], 1.0)
    assert V.mood_distance("sad", "excited") >= V.ABRUPT
    assert notes[0]["abrupt"] is True


def test_sensitivity_controls_how_readily_moods_switch() -> None:
    beats = [("neutral", 10, False), ("mysterious", 6, False), ("neutral", 10, False)]
    low, _ = V.smooth_moods(beats, 0.0)
    high, _ = V.smooth_moods(beats, 1.0)
    assert low == ["neutral"] * 3
    assert high == ["neutral", "tense", "neutral"]


def test_narration_starts_from_the_previous_lines_reference() -> None:
    assert V.smooth_moods([("neutral", 5, False)], 0.0, start="tense")[0] == ["tense"]
    assert V.smooth_moods([("neutral", 5, False)], 0.0)[0] == ["neutral"]


# ---------- segmentation ----------


def test_sentences_with_one_voice_are_said_in_one_go() -> None:
    segs = V.group_segments([P("Thor is done talking.", "tense", "tense", "normal"),
                             P("He swings.", "tense", "tense", "normal"),
                             P("Atreus learns why.", "tense", "tense", "normal")])  # fmt: skip
    assert len(segs) == 1


def test_written_pause_splits_even_across_pieces() -> None:
    one = V.group_segments([P("Thor is done talking... and Atreus learns why.", "tense",
                              "tense", "normal")])  # fmt: skip
    two = V.group_segments([P("Thor is done talking...", "tense", "tense", "normal"),
                            P("and Atreus learns why.", "tense", "tense", "normal")])  # fmt: skip
    for segs in (one, two):
        assert [s["text"] for s in segs] == ["Thor is done talking.", "and Atreus learns why."]
        assert segs[0]["pause_after_ms"] > 500


def test_reference_change_splits_and_tiny_pieces_merge() -> None:
    segs = V.group_segments([P("A long calm opening sentence here.", "neutral", "neutral",
                               "normal"),
                             P("Run now, Atreus!", "intense", "excited", "fast")])  # fmt: skip
    assert [s["group"] for s in segs] == ["neutral", "intense"]
    merged = V.group_segments([P("Yes.", "neutral", "neutral", "fast"),
                               P("Okay.", "neutral", "neutral", "fast"),
                               P("And that was the plan all along.", "neutral", "neutral",
                                 "fast")])  # fmt: skip
    assert all(len(s["text"].split()) >= V.MIN_SEGMENT_WORDS for s in merged)


def test_long_runs_split_only_at_sentence_ends() -> None:
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(8))
    segs = V.group_segments([P(text, "neutral", "neutral", "normal")], max_words=20)
    assert len(segs) > 1
    assert all(s["text"].endswith(".") for s in segs)


def test_plan_keeps_one_reference_for_small_changes(svc: VoiceIntelService) -> None:
    svc.save_settings({"transition_sensitivity": 0.5})
    p = svc.plan("[neutral] Kratos walks into the hall and looks around slowly. [curious] "
                 "Huh. [neutral] Then he keeps walking toward the door.", "qwen_story",
                 "qwen", "en")  # fmt: skip
    assert {s["reference"] for s in p["segments"]} == {p["segments"][0]["reference"]}
    assert p["last_group"]


# ---------- pronunciation verification ----------

ATREUS = V.Term("Atreus", respell="Ah-tray-us")


def test_name_split_into_several_words_is_still_recognised() -> None:
    heard = [("Then", 0.9), ("a", 0.6), ("tray", 0.7), ("us", 0.7), ("ran.", 0.9)]
    [n] = V.align_names("Then Atreus ran.", heard, [ATREUS])
    assert n["status"] == "ok"


def test_a_different_name_is_not_accepted() -> None:
    [n] = V.align_names("Then Atreus ran.", [("Then", .9), ("Agrius", .9), ("ran.", .9)],
                        [ATREUS])  # fmt: skip
    assert n["status"] != "ok"


def test_low_confidence_or_missing_words_are_unclear_not_wrong() -> None:
    [n] = V.align_names("Then Atreus ran.", [("Then", 0.9), ("ran.", 0.9)], [ATREUS])
    assert n["status"] == "unclear"
    [n] = V.align_names("Then Atreus ran.", [("Then", .9), ("Zork", .2), ("ran.", .9)],
                        [ATREUS])  # fmt: skip
    assert n["status"] == "unclear"


def test_accepted_false_positive_counts_as_right() -> None:
    t = V.Term("Atreus", heard=["a trace"])
    assert V.name_similarity("a trace", t) == 1.0
    [n] = V.align_names("Then Atreus ran.", [("Then", .9), ("a", .8), ("trace", .8),
                                              ("ran.", .9)], [t])  # fmt: skip
    assert n["status"] == "ok"


def test_heard_list_survives_a_merge_and_variants_never_save() -> None:
    a = V.Term("Atreus", respell="Ah-tray-us", source="user", heard=["a trace"])
    merged = V.Term.from_dict(a.to_dict())
    assert merged.heard == ["a trace"]
    vs = V.respell_variants(V.Term("Jormungandr", phonetic="YOR-mun-gand"))
    assert vs and all(isinstance(v, str) for v in vs)


# ---------- audio measurements ----------


def test_lufs_matches_the_bs1770_reference_tone() -> None:
    assert A.lufs(tone(5, 997, 1.0, 48000), 48000) == pytest.approx(-3.01, abs=0.1)


def test_clipping_is_detected() -> None:
    a = np.clip(tone(2, 220, 2.0), -1, 1)
    assert any("clipping" in x for x in A.artifacts(a, SR))
    assert not any("clipping" in x for x in A.artifacts(tone(2, 220, 0.3), SR))


def test_level_match_evens_out_loudness_conservatively() -> None:
    loud, quiet, very = speechy(2, amp=0.5), speechy(2, amp=0.25), speechy(2, amp=0.01)
    out = A.level_match([loud, quiet, very], SR)
    lv = [A.speech_level(x, SR) for x in out]
    assert abs(lv[0] - lv[1]) < 1.0
    assert lv[1] - lv[2] > 10  # never boosted by more than 6 dB
    assert out[0][0] == 0.0  # faded edge


def test_long_silences_are_capped() -> None:
    a = np.concatenate([speechy(1), np.zeros(int(2.5 * SR), np.float32), speechy(1)])
    b = A.cap_silences(a, SR)
    assert len(b) < len(a) - int(1.5 * SR)


# ---------- consistency ----------


def seg(i: int, a: A.F32, emb: Any = None, speed: float = 1.0) -> dict[str, Any]:
    return {"id": f"0.{i}", "label": f"piece {i}", "metrics": A.measure(a, SR),
            "embedding": emb, "speed": speed}  # fmt: skip


def test_matching_segments_raise_no_flags() -> None:
    e = np.ones(8)
    r = A.consistency([seg(i, speechy(3, seed=i), e + 0.01 * i) for i in range(4)])
    assert r["flags"] == []
    assert r["score"] >= 90


def test_different_speaker_and_loudness_are_flagged_not_rejected() -> None:
    e1, e2 = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
    segs = [seg(0, speechy(3), e1), seg(1, speechy(3, seed=1), e1),
            seg(2, speechy(3, amp=0.03, seed=2), e2)]  # fmt: skip
    r = A.consistency(segs)
    kinds = {f["kind"] for f in r["flags"] if f["segment"] == "0.2"}
    assert {"identity", "loudness"} <= kinds
    assert len(r["segments"]) == 3  # every segment kept


def test_thresholds_are_the_users_to_change() -> None:
    segs = [seg(0, speechy(3)), seg(1, speechy(3, amp=0.1, seed=1))]
    strict = A.consistency(segs, {"loudness_spread_lu": 1.0})
    lax = A.consistency(segs, {"loudness_spread_lu": 30.0})
    assert any(f["kind"] == "loudness" for f in strict["flags"])
    assert not any(f["kind"] == "loudness" for f in lax["flags"])


def test_voice_colour_needs_enough_pieces_to_judge() -> None:
    segs = [seg(0, speechy(3)), seg(1, speechy(3, hz=300, seed=1)), seg(2, speechy(3, seed=2))]
    assert not any(f["kind"] == "timbre" for f in A.consistency(segs)["flags"])


# ---------- report and benchmark ----------


def write_wav(path: Path, a: A.F32) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def test_report_keeps_measured_recognised_and_estimates_apart(tmp_path: Path) -> None:
    intel = SimpleNamespace(settings=lambda: {"verify_pronunciation": False, "qc": {}})
    speaker = SimpleNamespace(embed=lambda a, sr: None, status=lambda: {"ready": False})
    qa = VoiceQAService(None, intel, None, speaker, tmp_path)  # type: ignore[arg-type]
    lines = [{"label": "Hook 1", "text": "Thor is done.", "index": 0,
              "path": str(write_wav(tmp_path / "a.wav", speechy(3)))},
             {"label": "Fact 3", "text": "Mjolnir was forged.", "index": 2,
              "path": str(tmp_path / "missing.wav")}]  # fmt: skip
    r = qa.report(lines, "gow", "en_michael")
    assert set(r) >= {"measured", "recognised", "estimates", "lines"}
    assert r["recognised"]["checked"] is False and r["recognised"]["names"] == []
    assert "consistency_score" in r["measured"]
    assert [ln["index"] for ln in r["lines"]] == [0, 2]  # the voice-over's own positions
    assert r["lines"][1]["missing"] is True


def test_benchmark_summary_keeps_ratings_separate_from_measurements() -> None:
    def sample(eng: str, i: int, rating: int | None) -> dict[str, Any]:
        return {"engine": eng, "latency_s": 2.0 + i, "rtf": 0.5, "lufs": -18.0 - i,
                "clipping": False, "rating": rating, "embedding": [1.0, 0.1 * i],
                "names": [{"status": "ok"}, {"status": "check"}]}  # fmt: skip

    s = summarise(
        [sample("kokoro", 0, 4), sample("kokoro", 1, None), sample("qwen_clone", 0, None)]
    )
    k = s["kokoro"]
    assert k["samples"] == 2 and k["latency_s"] == 2.5
    assert k["names_ok"] == 0.5 and k["names_checked"] == 4
    assert k["loudness_spread_lu"] == 1.0
    assert k["rating"] == 4.0 and k["rated"] == 1
    assert s["qwen_clone"]["rating"] is None and s["qwen_clone"]["consistency"] is None
