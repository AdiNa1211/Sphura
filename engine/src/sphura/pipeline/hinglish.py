"""Devanagari -> Hinglish (Roman script, the way people type it): "भाई" -> "bhai", never
"brother". Common words come from a table; the rest are transliterated with schwa deletion."""

from __future__ import annotations

import re

_COMMON = {
    "है": "hai", "हैं": "hain", "में": "mein", "मैं": "main", "नहीं": "nahi", "नही": "nahi",
    "क्या": "kya", "भाई": "bhai", "यार": "yaar", "हाँ": "haan", "हां": "haan", "तो": "toh",
    "भी": "bhi", "और": "aur", "कि": "ki", "के": "ke", "का": "ka", "की": "ki", "को": "ko",
    "से": "se", "पे": "pe", "पर": "par", "ये": "ye", "यह": "yeh", "वो": "woh", "वह": "woh",
    "कर": "kar", "रहा": "raha", "रही": "rahi", "रहे": "rahe", "था": "tha", "थी": "thi",
    "थे": "the", "हूँ": "hoon", "हूं": "hoon", "हो": "ho", "अब": "ab", "चल": "chal",
    "चलो": "chalo", "ठीक": "theek", "अच्छा": "accha", "बहुत": "bahut", "क्यों": "kyun",
    "कैसे": "kaise", "कहाँ": "kahan", "कहां": "kahan", "जा": "ja", "आ": "aa", "हम": "hum",
    "तुम": "tum", "तू": "tu", "आप": "aap", "मुझे": "mujhe", "तुझे": "tujhe", "अरे": "arey",
    "बस": "bas", "एक": "ek", "दो": "do", "सही": "sahi", "कुछ": "kuch", "सब": "sab",
    "वाला": "wala", "वाली": "wali", "वाले": "wale", "ना": "na", "न": "na", "जी": "ji",
    "मत": "mat", "फिर": "phir", "इसको": "isko", "उसको": "usko", "मेरा": "mera",
    "मेरी": "meri", "तेरा": "tera", "तेरी": "teri", "ओके": "okay", "हाय": "hi", "बाय": "bye",
}  # fmt: skip

# English words Whisper writes in Devanagari inside Hindi speech
_ENGLISH = {
    "वीडियो": "video", "प्रोफेशनल": "professional", "ट्रैक": "track", "ट्रेंड": "trend",
    "गेम": "game", "प्लेयर": "player", "बॉस": "boss", "किल": "kill", "शॉट": "shot",
    "मूव": "move", "लेवल": "level", "सॉरी": "sorry", "थैंक्स": "thanks", "फोन": "phone",
    "फ़ोन": "phone", "कैमरा": "camera", "पूल": "pool", "स्लो": "slow", "मोशन": "motion",
    "एक्चुअली": "actually", "सीरियसली": "seriously", "बेस्ट": "best", "नेक्स्ट": "next",
    "प्रो": "pro", "गुड": "good", "नाइस": "nice", "वॉटर": "water", "स्विमिंग": "swimming",
    "स्टाइल": "style", "फ्रेंड": "friend", "ब्रो": "bro", "लाइक": "like", "चैनल": "channel",
    "सब्सक्राइब": "subscribe", "स्ट्रीम": "stream", "लाइव": "live", "शॉर्ट्स": "shorts",
    "रील": "reel", "रील्स": "reels", "क्लिप": "clip", "एडिट": "edit", "सेट": "set",
    "टाइम": "time", "फ्री": "free", "डाइव": "dive", "जंप": "jump", "रेस": "race",
    "स्टार्ट": "start", "फिनिश": "finish", "रेडी": "ready", "डन": "done", "वाओ": "wow",
    "ओह": "oh", "कूल": "cool", "क्रेज़ी": "crazy", "क्रेजी": "crazy", "स्पीड": "speed",
    "फास्ट": "fast", "परफेक्ट": "perfect", "मिस": "miss", "हिट": "hit", "स्कोर": "score",
    "टीम": "team", "मैच": "match", "पॉइंट": "point", "सर": "sir", "मैडम": "madam",
    "बट": "but", "सो": "so", "बिकॉज़": "because", "वेट": "wait", "लुक": "look", "एक्स": "x",
    "फिल्टर": "filter", "शूट": "shoot", "शो": "show", "फैन": "fan", "पार्टी": "party",
}  # fmt: skip

_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
    "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
    "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "फ़": "f", "ड़": "d", "ढ़": "dh",
}  # fmt: skip
_NUKTA = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "फ": "फ़", "ड": "ड़", "ढ": "ढ़"}
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai",
    "ओ": "o", "औ": "au", "ऋ": "ri", "ऑ": "o",
}  # fmt: skip
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e", "ै": "ai", "ो": "o",
    "ौ": "au", "ृ": "ri", "ॉ": "o",
}  # fmt: skip
_VIRAMA, _ANUSVARA, _CHANDRA, _VISARGA = "\u094d", "\u0902", "\u0901", "\u0903"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def romanize(text: str) -> str:
    """Every Devanagari word in text, romanized; punctuation and Latin words kept as is."""
    text = text.replace("।", ".").translate(_DIGITS)
    return re.sub(r"[ऀ-ॿ]+", lambda m: _word(m.group(0)), text)


def _word(w: str) -> str:
    if w in _COMMON:
        return _COMMON[w]
    if w in _ENGLISH:
        return _ENGLISH[w]
    # units: [consonant or None, vowel or None, "a" = inherent (deletable), nasal]
    units: list[list[str | None]] = []
    chars = list(w)
    i = 0
    while i < len(chars):
        c = chars[i]
        if c == "़" and units and units[-1][0]:
            base = next((k for k, v in _CONS.items() if v == units[-1][0]), "")
            units[-1][0] = _CONS.get(_NUKTA.get(base, ""), units[-1][0])
        elif c in _CONS:
            units.append([_CONS[c], "a", None])
        elif c in _MATRAS and units:
            units[-1][1] = _MATRAS[c]
        elif c == _VIRAMA and units:
            units[-1][1] = None
        elif c in (_ANUSVARA, _CHANDRA) and units:
            units[-1][2] = "n"
        elif c == _VISARGA and units:
            units[-1][2] = "h"
        elif c in _VOWELS:
            units.append([None, _VOWELS[c], None])
        i += 1
    _delete_schwas(units)
    out = []
    for k, (cons, vowel, nasal) in enumerate(units):
        last = k == len(units) - 1
        nxt = units[k + 1] if not last else None
        if (
            vowel == "aa"
            and cons
            and not (nxt and nxt[0] and k + 1 == len(units) - 1 and nxt[1] in ("a", None))
        ):
            vowel = "a"  # "aa" only in a closed final syllable: baat, yaar, kaam
        elif vowel == "ee" and last:
            vowel = "i"  # nahi, bhi, kabhi
        elif vowel == "oo" and last and not nasal:
            vowel = "u"
        elif vowel == "e" and last and nasal == "n":
            vowel = "ei"  # karein, lein
        if nasal == "n" and nxt and nxt[0] and nxt[0][0] in "pbm":
            nasal = "m"
        out.append((cons or "") + (vowel or "") + (nasal or ""))
    return "".join(out)


def _delete_schwas(units: list[list[str | None]]) -> None:
    """Hindi drops the inherent 'a' at the end of a word and between VC_CV."""
    if len(units) > 1 and units[-1][0] and units[-1][1] == "a" and not units[-1][2]:
        units[-1][1] = None
    for i in range(len(units) - 2, 0, -1):
        u, prev, nxt = units[i], units[i - 1], units[i + 1]
        if u[0] and u[1] == "a" and not u[2] and prev[1] and nxt[0] and nxt[1]:
            u[1] = None
