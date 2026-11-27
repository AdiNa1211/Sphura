"""Facts the narrator may state, each from a source and written to be spoken. Narration
may only state facts from here, so Shorts never spread made-up lore. `keys` match the
moment's description and dialogue; `spoiler` facts are left out unless asked for."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    text: str
    keys: str  # regex; "" = a general fact about the game
    source: str
    spoiler: bool = False


_W = "https://godofwar.fandom.com/wiki/"
_GOWR = "https://en.wikipedia.org/wiki/God_of_War_Ragnar%C3%B6k"
_PBZ = "https://en.wikipedia.org/wiki/Phantom_Blade_Zero"

FACTS: dict[str, list[Fact]] = {
    "gow": [
        Fact(
            "thor-who",
            "Thor is the Norse god of thunder, the son of Odin and the giantess Fjörgyn.",
            r"\bthor\b",
            _W + "Thor",
        ),
        Fact(
            "thor-hammer",
            "Mjölnir was forged for Thor by the dwarves Brok and Sindri, on Odin's orders.",
            r"\bthor\b|mj[oö]lnir|hammer|brok|sindri",
            _W + "Thor",
        ),
        Fact(
            "thor-returns",
            "Thor's hammer always hits its target and flies back to his hand.",
            r"\bthor\b|mj[oö]lnir|hammer|lightning",
            _W + "Thor",
        ),
        Fact(
            "thor-mountains",
            "Mímir calls Mjölnir a superweapon, strong enough to crush a "
            "mountain in a single blow.",
            r"\bthor\b|mj[oö]lnir|hammer",
            _W + "Thor",
        ),
        Fact(
            "thor-modi",
            "After Magni died, Thor beat his own son Modi for running away from Kratos.",
            r"\bthor\b|modi|magni|\bsons?\b",
            _W + "Thor",
        ),
        Fact(
            "thor-revenge",
            "In Ragnarök, Thor and Freya both want revenge on Kratos, for Baldur, Magni and Modi.",
            r"\bthor\b|freya|revenge|vengeance|blood debt",
            _W + "Thor",
        ),
        Fact(
            "thor-serpent",
            "Thor and the World Serpent have hated each other ever since "
            "Thor massacred the giants.",
            r"\bthor\b|serpent|j[oö]rmungandr|giants?\b",
            _W + "Thor",
        ),
        Fact("thor-voice", "Thor is voiced by Ryan Hurst.", r"\bthor\b", _GOWR),
        Fact(
            "odin-villain",
            "Odin, the Allfather, is the main villain of God of War Ragnarök.",
            r"\bodin\b|allfather",
            _W + "Odin",
        ),
        Fact(
            "odin-tyr",
            "For much of the story, Odin is in disguise as Týr, the Norse god of war.",
            r"\bodin\b|\bt[yý]r\b",
            _W + "Odin",
            spoiler=True,
        ),
        Fact(
            "odin-ravens",
            "Odin's ravens, Huginn and Muninn, are his scouts and spies.",
            r"\bodin\b|raven",
            _W + "Odin",
        ),
        Fact(
            "odin-mask",
            "Odin was obsessed with a mask he believed could reveal the meaning of life.",
            r"\bodin\b|\bmask\b",
            _W + "Odin",
        ),
        Fact(
            "odin-brok",
            "Odin killed Brok with a knife when the dwarf grew suspicious of him.",
            r"\bbrok\b|\bodin\b",
            _W + "Odin",
            spoiler=True,
        ),
        Fact("odin-voice", "Odin is voiced by Richard Schiff.", r"\bodin\b|allfather", _GOWR),
        Fact(
            "atreus-loki",
            "Atreus's mother Faye named him Loki. Kratos talked her into "
            "calling him Atreus instead.",
            r"\bloki\b|atreus|\bfaye\b|calling him|name",
            _W + "Atreus",
        ),
        Fact(
            "atreus-giant",
            "Faye was a giant, which makes Atreus part giant.",
            r"\bfaye\b|giant|j[oö]tunn|atreus",
            _W + "Atreus",
        ),
        Fact(
            "atreus-age",
            "Atreus is fourteen years old in God of War Ragnarök.",
            r"atreus|\bboy\b|\bson\b",
            _W + "Atreus",
        ),
        Fact(
            "atreus-shapes",
            "Atreus can turn into a wolf, and once, in grief, into a giant bear.",
            r"atreus|wolf|bear|fenrir",
            _W + "Atreus",
        ),
        Fact(
            "atreus-angrboda",
            "In the Ironwood, the young giant Angrboda calls him Loki "
            "and says she has been expecting him.",
            r"angrboda|ironwood|\bloki\b",
            _W + "Atreus",
        ),
        Fact("atreus-voice", "Atreus is voiced by Sunny Suljic.", r"atreus|\bboy\b", _GOWR),
        Fact(
            "kratos-midgard",
            "Kratos came to Midgard to live as a man and leave his violent past behind.",
            r"kratos|\bpast\b|mistake|consequence|reminded|violence",
            _W + "Kratos",
        ),
        Fact(
            "kratos-ash",
            "Kratos's pale skin comes from the ashes of his murdered family.",
            r"kratos|\bash|family|mistake|reminded|\bpast\b",
            _W + "Kratos",
        ),
        Fact(
            "kratos-faye",
            "After Faye died, Kratos and Atreus carried her ashes to the "
            "highest peak in the Nine Realms.",
            r"\bfaye\b|ashes|\bwife\b|mother",
            _W + "Kratos",
        ),
        Fact(
            "kratos-redemption",
            "Kratos believes he chases a redemption he can never deserve.",
            r"redemption|mistake|guilt|consequence|reminded|\bpast\b",
            _W + "Kratos",
        ),
        Fact(
            "kratos-hope",
            "Kratos ends the Norse saga as a God of Hope.",
            r"kratos|\bhope\b",
            _W + "Kratos",
            spoiler=True,
        ),
        Fact("kratos-voice", "Kratos is voiced by Christopher Judge.", r"kratos", _GOWR),
        Fact(
            "freya-hunt",
            "For three years after Kratos killed her son Baldur, Freya hunted him for revenge.",
            r"freya|baldur|revenge|vengeance|justice",
            _W + "Freya",
        ),
        Fact(
            "freya-curse",
            "Fimbulwinter weakened Odin's curse on Freya, so she could fight again.",
            r"freya|curse",
            _W + "Freya",
        ),
        Fact(
            "freya-wings",
            "Freya won back her Valkyrie wings with the help of Sigrún.",
            r"freya|valkyrie|wings|sigr[uú]n",
            _W + "Freya",
        ),
        Fact("freya-voice", "Freya is voiced by Danielle Bisutti.", r"freya", _GOWR),
        Fact(
            "winter-what",
            "Fimbulwinter began after Baldur's death: three years of "
            "endless winter before Ragnarök.",
            r"snow|winter|frozen|frost|\bcold\b|\bice\b|blizzard",
            _W + "Fimbulwinter",
        ),
        Fact(
            "winter-lake",
            "During Fimbulwinter, the Lake of Nine froze solid, cutting "
            "off travel between the realms.",
            r"lake|frozen|realm",
            _W + "Fimbulwinter",
        ),
        Fact(
            "winter-magic",
            "Fimbulwinter's magic weakens curses and enchantments across the Nine Realms.",
            r"curse|magic|enchant|rune",
            _W + "Fimbulwinter",
        ),
        Fact(
            "heimdall-sight",
            "Heimdall is the god of foresight. He can see and hear across all Nine Realms.",
            r"heimdall",
            _W + "Heimdall",
        ),
        Fact(
            "heimdall-horn",
            "Heimdall guards Bifröst, the burning rainbow bridge, and "
            "carries the horn Gjallarhorn.",
            r"heimdall|bifr[oö]st|gjallarhorn|\bhorn\b",
            _W + "Heimdall",
        ),
        Fact(
            "heimdall-loki",
            "In Norse myth, Heimdall and Loki are fated to kill each other at Ragnarök.",
            r"heimdall|\bloki\b",
            _W + "Heimdall",
        ),
        Fact(
            "game-sales",
            "God of War Ragnarök sold 5.1 million copies in its first week, "
            "the fastest-selling first-party launch in PlayStation history.",
            "",
            _GOWR,
        ),
        Fact("game-15m", "It passed 15 million copies sold by November 2023.", "", _GOWR),
        Fact("game-tga", "It won six awards at The Game Awards 2022.", "", _GOWR),
        Fact(
            "game-saga",
            "Santa Monica ended the Norse saga in one game, because they "
            "didn't want to spend another five years on a single story.",
            "",
            _GOWR,
        ),
        Fact(
            "game-valhalla",
            "The Valhalla expansion is free: a roguelite epilogue to the story.",
            r"valhalla",
            _GOWR,
        ),
        Fact(
            "game-meta", "Its Metacritic score of 94 ties the best in the whole series.", "", _GOWR
        ),
    ],
    "pbz": [
        Fact("pbz-studio", "Phantom Blade Zero is made by S-GAME, a studio from China.", "", _PBZ),
        Fact(
            "pbz-release",
            "It released on October 29, 2026, after a delay from September to polish the game.",
            "",
            _PBZ,
        ),
        Fact(
            "pbz-soul",
            "You play as Soul, an elite assassin of the Order, framed for "
            "killing the Shadow Sovereign.",
            r"\bsoul\b|order|assassin|sovereign|framed",
            _PBZ,
        ),
        Fact(
            "pbz-days",
            "After a duel with Zuo Shang, Soul is revived with a Tainted Heart "
            "that leaves him only sixty-six days to live.",
            r"\bsoul\b|heart|zuo shang|\bdead\b|death|\bdie|defeat",
            _PBZ,
        ),
        Fact(
            "pbz-hollow",
            "The Flesh Sculptor offered Soul a Hollow Heart: his life back, "
            "but no emotions and no desires.",
            r"heart|flesh|sculptor",
            _PBZ,
        ),
        Fact(
            "pbz-order",
            "The Order is a powerful, secretive organization in the wulin, the martial underworld.",
            r"\border\b|sect|assassin|masked",
            _PBZ,
        ),
        Fact(
            "pbz-kungfupunk",
            "Its developers call the style kungfupunk: wuxia mixed with steampunk and cyberpunk.",
            r"mechanical|robot|machine|puppet|guardian|steam|industrial",
            _PBZ,
        ),
        Fact(
            "pbz-ming",
            "Its world is based on China around the time of the Ming dynasty.",
            r"village|ruins|temple|pier|bridge",
            _PBZ,
        ),
        Fact(
            "pbz-mocap",
            "Its fights were motion-captured from real martial artists, with "
            "Donnie Yen as a creative consultant.",
            r"combat|swordsmanship|martial|strike|sword|duel|fight|combo",
            _PBZ,
        ),
        Fact(
            "pbz-masters",
            "The team consulted masters of the Shaolin Temple and Emei Mountain schools.",
            r"martial|swordsmanship|master",
            _PBZ,
        ),
        Fact(
            "pbz-attack-defend",
            "Its combat is built on a classic martial arts idea: attack while you defend.",
            r"parry|duel|combat|strike|block",
            _PBZ,
        ),
        Fact("pbz-ue5", "Phantom Blade Zero runs on Unreal Engine 5.", "", _PBZ),
        Fact(
            "pbz-music",
            "Its soundtrack mixes traditional Chinese instruments with rock and electronic music.",
            "",
            _PBZ,
        ),
    ],
}


def relevant(
    game: str, text: str, spoilers: bool = False, limit: int = 6, context: str = "",
    general: bool = False,
) -> list[Fact]:  # fmt: skip
    """Facts about who or what is in this moment; then ones about names from just before
    it. Facts about the game itself (sales, awards) only for lore-style Shorts."""
    facts = [f for f in FACTS.get(game, []) if spoilers or not f.spoiler]
    hits = [f for f in facts if f.keys and re.search(f.keys, text, re.I)]
    near = [f for f in facts if f.keys and f not in hits and context
            and re.search(f.keys, context, re.I)]  # fmt: skip
    picked = (hits + near[:2])[:limit]
    if general and len(picked) < limit:
        picked += [f for f in facts if not f.keys][: min(2, limit - len(picked))]
    return picked


def by_id(game: str) -> dict[str, Fact]:
    return {f.id: f for f in FACTS.get(game, [])}
