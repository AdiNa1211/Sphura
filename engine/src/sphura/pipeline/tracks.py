from __future__ import annotations

import re

from sphura.domain.models import AudioStream

_VOICE = re.compile(r"\b(mic|micro|microphone|voice|vo|commentary|narration|cam)\b", re.I)


def choose_voice_track(streams: tuple[AudioStream, ...]) -> AudioStream | None:
    """Track to transcribe: a named mic/voice track if present (OBS multi-track), else the
    default track, else the first. With one mixed track this is simply that track."""
    if not streams:
        return None
    for s in streams:
        if s.title and _VOICE.search(s.title):
            return s
    return next((s for s in streams if s.is_default), streams[0])


def choose_mix_track(streams: tuple[AudioStream, ...]) -> AudioStream | None:
    """Track for loudness/impact signals: the default (usually the full mix in OBS), else first."""
    if not streams:
        return None
    return next((s for s in streams if s.is_default), streams[0])
