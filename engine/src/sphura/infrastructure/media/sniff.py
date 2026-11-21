from __future__ import annotations

from sphura.domain.models import Container

_MATROSKA_EBML = b"\x1a\x45\xdf\xa3"
_QT_TOP_ATOMS = {b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


class MagicBytesSniffer:
    """Identifies MP4/MOV/MKV/WebM/AVI from the first bytes of the file."""

    def sniff(self, header: bytes) -> Container | None:
        if len(header) < 12:
            return None

        if header.startswith(_MATROSKA_EBML):
            # DocType element (0x42 0x82) is inside the EBML header
            idx = header.find(b"\x42\x82", 4, 64)
            if idx != -1 and b"webm" in header[idx : idx + 16]:
                return Container.WEBM
            return Container.MKV

        if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
            return Container.AVI

        box_type = header[4:8]
        if box_type == b"ftyp":
            major_brand = header[8:12]
            return Container.MOV if major_brand == b"qt  " else Container.MP4
        if box_type in _QT_TOP_ATOMS:
            # Older QuickTime files can start without an ftyp box
            return Container.MOV

        return None
