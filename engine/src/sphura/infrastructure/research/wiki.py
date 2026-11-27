"""MediaWiki reader (Fandom wikis and Wikipedia share the API): search a title, then read a
page as (section, paragraphs). Plain urllib; pages are cached by the caller."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, ClassVar

_UA = "SPHURA/1.0 (local gaming video editor; lore research)"
_TIMEOUT_S = 20


_NOISE = re.compile(r"navbox|toc|reference|mw-editsection|portable-infobox|gallery|hatnote")


class WikiError(Exception):
    pass


class WikiClient:
    def _get(self, api: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{api}?{urllib.parse.urlencode({**params, 'format': 'json', 'formatversion': 2})}"
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
                data: dict[str, Any] = json.loads(r.read().decode("utf-8"))
        except (OSError, ValueError) as e:
            raise WikiError(f"{urllib.parse.urlsplit(api).netloc}: {e}") from e
        if "error" in data:
            raise WikiError(str(data["error"].get("info") or data["error"]))
        return data

    def search(self, api: str, query: str, limit: int = 5) -> list[str]:
        data = self._get(
            api,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
            },
        )
        return [h["title"] for h in data.get("query", {}).get("search", [])]

    def page(self, api: str, title: str) -> tuple[str, str, list[tuple[str, list[str]]]]:
        """(resolved title, page url, [(section heading, paragraphs)])."""
        data = self._get(
            api,
            {
                "action": "parse",
                "page": title,
                "prop": "text",
                "redirects": 1,
                "disableeditsection": 1,
            },
        )
        p = data.get("parse") or {}
        html = p.get("text") or ""
        if isinstance(html, dict):  # formatversion 1 shape, just in case
            html = html.get("*", "")
        resolved = p.get("title") or title
        return resolved, page_url(api, resolved), sections(html)


def page_url(api: str, title: str) -> str:
    base = api.rsplit("/", 1)[0]
    base = base[:-2] if base.endswith("/w") else base  # wikipedia: /w/api.php -> /wiki/
    return f"{base}/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


class _Sections(HTMLParser):
    """Paragraphs and list items under each h2/h3, without infoboxes, tables, references."""

    _SKIP: ClassVar[set[str]] = {
        "table",
        "aside",
        "style",
        "script",
        "sup",
        "figure",
        "nav",
        "noscript",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[tuple[str, list[str]]] = [("Introduction", [])]
        self._skip = 0
        self._head: list[str] | None = None
        self._buf: list[str] | None = None
        self._tag: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = dict(attrs).get("class") or ""
        if tag in ("br", "img", "hr", "wbr"):
            return
        self._tag.append(tag)
        if self._skip or tag in self._SKIP or _NOISE.search(cls):
            self._skip += 1
            return
        if tag in ("h2", "h3"):
            self._head = []
        elif tag in ("p", "li", "blockquote") and self._buf is None:
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if not self._tag:
            return
        if tag in self._tag:  # tolerate unclosed tags
            while self._tag and self._tag.pop() != tag:
                pass
        if self._skip:
            self._skip -= 1
            return
        if tag in ("h2", "h3") and self._head is not None:
            title = re.sub(r"\s+", " ", "".join(self._head)).strip()
            self.out.append((title or "Untitled", []))
            self._head = None
        elif tag in ("p", "li", "blockquote") and self._buf is not None:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.out[-1][1].append(text)
            self._buf = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._head is not None:
            self._head.append(data)
        elif self._buf is not None:
            self._buf.append(data)


def sections(html: str) -> list[tuple[str, list[str]]]:
    parser = _Sections()
    parser.feed(html)
    parser.close()
    return [(h, ps) for h, ps in parser.out if ps]
