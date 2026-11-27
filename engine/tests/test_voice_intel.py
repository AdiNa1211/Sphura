"""Voice Intelligence: pronunciation matching, dictionary persistence and isolation, TTS
preprocessing, emotion-tag compatibility, script optimisation, user overrides, fallbacks."""

# ruff: noqa: RUF001  (IPA symbols on purpose)
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sphura.application.voice_intel import VoiceIntelService
from sphura.infrastructure.persistence.sqlite.db import SqliteDatabase
from sphura.infrastructure.persistence.sqlite.voice import SqliteVoiceRepository
from sphura.pipeline import narration as N
from sphura.pipeline import voice_intel as V
from sphura.ports.ai import LLMResponse, ProviderInfo

GOW = [V.Term(**d, source="builtin") for d in V.SEEDS["gow"]]


@pytest.fixture
def svc(tmp_path: Path) -> VoiceIntelService:
    queries = SimpleNamespace(list_projects=lambda n: [], candidates=lambda pid: [])
    return VoiceIntelService(SqliteVoiceRepository(SqliteDatabase(tmp_path / "t.db")), queries,
                             None, None)  # fmt: skip


# ---------- pronunciation matching ----------


def test_names_are_respelled_with_possessives_and_case() -> None:
    p = V.apply_pronunciation("Atreus's bow hit Kratos' shield. KRATOS!", GOW, phonemes=False)
    assert p.tts == "Ay-tree-us's bow hit Kraytos' shield. KRAYTOS!"


def test_words_inside_other_words_and_lowercase_words_are_left_alone() -> None:
    p = V.apply_pronunciation("Atreusian kratosx and atreus stay.", GOW, phonemes=False)
    assert p.tts == "Atreusian kratosx and atreus stay."
    assert p.changes == []


def test_spellings_share_a_pronunciation_but_nicknames_do_not() -> None:
    p = V.apply_pronunciation("Mimir and Mímir. Loki is Atreus.", GOW, phonemes=False)
    assert p.tts == "Mee-meer and Mee-meer. Loki is Ay-tree-us."


def test_phoneme_engines_get_ipa_and_keep_the_text() -> None:
    p = V.apply_pronunciation("Atreus's bow.", GOW, phonemes=True)
    assert p.tts == "Atreus's bow."
    assert p.lexicon == {"Atreus's": "eɪtɹˈiːəsɪz"}


def test_a_name_is_said_one_way_in_a_whole_script() -> None:
    worse = V.Term("Freya", respell="Free-ya", confidence=0.4, source="ai")
    better = V.Term("Freya", respell="Fray-uh", confidence=1.0, source="user", verified=True)
    p = V.apply_pronunciation("Freya. Then Freya again.", [worse, better], phonemes=False)
    assert p.tts == "Fray-uh. Then Fray-uh again."


def test_find_terms_spots_nicknames() -> None:
    assert [t.canonical for t in V.find_terms("Loki met Jotunheim", GOW)] == ["Atreus", "Jötunheim"]


# ---------- dictionary persistence, isolation, overrides ----------


def test_new_game_profile_is_seeded_and_isolated(svc: VoiceIntelService) -> None:
    gow = {t["canonical"] for t in svc.terms("gow")}
    assert "Kratos" in gow
    svc.upsert_term("pbz", {"canonical": "Soul", "respell": "Sole"})
    assert {t["canonical"] for t in svc.terms("pbz")} == {"Soul"}
    assert "Soul" not in {t["canonical"] for t in svc.terms("gow")}


def test_dictionary_survives_a_restart(tmp_path: Path) -> None:
    q = SimpleNamespace(list_projects=lambda n: [], candidates=lambda pid: [])
    a = VoiceIntelService(SqliteVoiceRepository(SqliteDatabase(tmp_path / "t.db")), q, None, None)
    a.upsert_term("gow", {"canonical": "Hræsvelgr", "respell": "Rice-vel-gur"})
    b = VoiceIntelService(SqliteVoiceRepository(SqliteDatabase(tmp_path / "t.db")), q, None, None)
    t = next(t for t in b.terms("gow") if t["canonical"] == "Hræsvelgr")
    assert t["respell"] == "Rice-vel-gur" and t["verified"] and t["source"] == "user"


def test_user_fix_wins_over_a_rescan(svc: VoiceIntelService) -> None:
    svc.upsert_term("gow", {"canonical": "Kratos", "respell": "Krah-tos"})
    svc._scan_run("gow", {"step": "", "progress": 0, "added": 0, "updated": 0})
    k = next(t for t in svc.terms("gow") if t["canonical"] == "Kratos")
    assert k["respell"] == "Krah-tos" and k["source"] == "user"


def test_merge_keeps_verified_entries() -> None:
    old = V.Term("Freya", respell="Fray-uh", verified=True, source="builtin", confidence=0.7)
    new = V.Term("Freya", respell="Free-ya", source="ai", confidence=0.9, aliases=["Freyja"])
    m = V.merge(old, new)
    assert m.respell == "Fray-uh" and "Freyja" in m.aliases


def test_review_policy_does_not_save_suggestions(tmp_path: Path) -> None:
    proj = SimpleNamespace(id="p1", name="God of War Ragnarok Part 9")
    judged = SimpleNamespace(judgement=SimpleNamespace(names=["Hræsvelgr"], characters=[]))
    q = SimpleNamespace(list_projects=lambda n: [proj], candidates=lambda pid: [judged, judged])
    s = VoiceIntelService(SqliteVoiceRepository(SqliteDatabase(tmp_path / "t.db")), q, None, None)
    s.save_settings({"auto_save": "review"})
    run: dict[str, Any] = {"step": "", "progress": 0, "added": 0, "updated": 0}
    s._scan_run("gow", run)
    assert "Hræsvelgr" not in {t["canonical"] for t in s.terms("gow")}
    assert run["suggestions"][0]["canonical"] == "Hræsvelgr"
    assert run["suggestions"][0]["needs_review"]


def test_uncertain_guesses_are_flagged() -> None:
    respell, conf = V.guess_respell("Hræsvelgr")
    assert respell and conf < 0.6
    assert V.Term("Hræsvelgr", respell=respell, confidence=conf).needs_review


# ---------- direction layer and engine compatibility ----------


def test_direction_tags_are_parsed_and_never_spoken() -> None:
    text, d = V.parse_direction('[whispering, tense, slow pace] "Something was wrong here."')
    assert text == "Something was wrong here." and d.mood == "tense" and d.pace == "slow"
    assert V.clean_spoken("He (sighs) was [beat] *really* tired") == "He was really tired"


def test_several_tags_make_several_deliveries() -> None:
    chunks = V.parse_chunks(
        "[tense] One. [deep, controlled anger] Two. [short pause, reflective] Three."
    )
    assert [(t, d.mood) for t, d in chunks] == [("One.", "tense"), ("Two.", "angry"),
                                                 ("Three.", "reflective")]  # fmt: skip
    assert chunks[2][1].pause_before_ms > 0


def test_pauses_become_segments_with_silence() -> None:
    segs = V.segments("Wait... then go — now.", V.Direction(mood="tense"))
    assert [s.text for s in segs] == ["Wait.", "then go,", "now."]
    assert segs[0].pause_after_ms > segs[1].pause_after_ms > 0


def test_plan_for_qwen_uses_a_mood_reference_and_respelling(svc: VoiceIntelService) -> None:
    p = svc.plan("[tense, slow] Atreus waited... then ran.", "qwen_story", "qwen", "en", "gow")
    assert p["tts"] == "Ay-tree-us waited... then ran."
    assert p["segments"][0]["reference"] == "tense" and p["segments"][0]["design"]
    assert "[" not in " ".join(s["text"] for s in p["segments"])
    assert p["speed"] < 1.0  # slow pace


def test_plan_for_kokoro_has_no_mood_reference_but_ipa(svc: VoiceIntelService) -> None:
    p = svc.plan("[epic] Kratos strikes.", "en_michael", "kokoro", "en", "gow")
    assert p["segments"][0]["reference"] is None
    assert p["lexicon"] == {"Kratos": "kɹˈeɪtoʊs"}


def test_turning_features_off_falls_back_to_plain_text(svc: VoiceIntelService) -> None:
    svc.save_settings({"pronunciation": False, "optimize": False, "emotion": False})
    p = svc.plan("[tense] Kratos in 2022.", "qwen_story", "qwen", "en", "gow")
    assert p["tts"] == "Kratos in 2022." and p["mood"] == "neutral"


def test_unknown_game_and_no_entries_change_nothing(svc: VoiceIntelService) -> None:
    p = svc.plan("Nobody special here.", "en_michael", "kokoro", "en", None)
    assert p["tts"] == "Nobody special here." and p["changes"] == []


def test_one_off_spelling_test_is_not_saved(svc: VoiceIntelService) -> None:
    p = svc.plan("Kratos.", "qwen_story", "qwen", "en", "gow", respell={"Kratos": "Krah-tos"})
    assert p["tts"] == "Krah-tos."
    assert next(t for t in svc.terms("gow") if t["canonical"] == "Kratos")["respell"] == "Kraytos"


# ---------- speech-friendly text ----------


def test_script_optimisation() -> None:
    text, notes = V.optimize_script("Little did he know, it is not over. He does not care.")
    assert text == "It isn't over. He doesn't care."
    assert "removed a cliché" in notes and "contractions" in notes


def test_long_sentences_are_split() -> None:
    long = ("He walked into the hall where the gods were waiting for him with their weapons "
            "drawn, and nobody in the room said a single word for a very long time.")  # fmt: skip
    text, notes = V.optimize_script(long)
    assert text.count(".") == 2 and "split a long sentence" in notes


def test_numbers_and_abbreviations_are_speakable() -> None:
    assert V.speakable("In 2022, PS5 had 1,250 HP on the 3rd try, 45% faster vs. 2018.") == (
        "In twenty twenty-two, P S five had one thousand two hundred and fifty health on the "
        "third try, forty-five percent faster versus twenty eighteen."
    )
    assert V.speakable("TYR returns", keep={"Tyr"}) == "TYR returns"


# ---------- scripts carry direction next to the text ----------


class _StubLLM:
    info = ProviderInfo("stub", True)

    def complete(self, req: Any) -> LLMResponse:
        if "ranking" in json.dumps(req.json_schema or {}):
            return LLMResponse(json.dumps({"ranking": [0]}))
        line = {"kind": "hook", "text": "Atreus is not ready for this", "fact_id": "",
                "mood": "tense", "pace": "slow"}  # fmt: skip
        return LLMResponse(json.dumps({"options": [{"angle": "a", "lines": [line]}]}))


def test_script_lines_get_a_direction_and_optimised_text(svc: VoiceIntelService) -> None:
    ctx = svc.voice_context("gow", "qwen_story", "suspense", 20, "Atreus")
    assert "Atreus" in ctx["brief"]
    out = N.write_script(_StubLLM(), "gow", None, ['"I can handle it"'], "en", 20,  # type: ignore[arg-type]
                         voice_ctx=ctx)  # fmt: skip
    ln = out["lines"][0]
    assert ln["text"] == "Atreus isn't ready for this."
    assert ln["direction"] == {"mood": "tense", "pace": "slow"}
    assert "[" not in ln["text"]
