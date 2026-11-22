"""YouTube Data API v3 (official, read-only, API key). Only public data: search results,
video statistics and channel subscriber counts. Quota: search = 100 units, list = 1."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from sphura.domain.errors import SphuraError

log = logging.getLogger(__name__)

API = os.environ.get("SPHURA_YOUTUBE_API_BASE", "https://www.googleapis.com/youtube/v3/")
SEARCH_COST, LIST_COST = 100, 1


class YouTubeApiError(SphuraError):
    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason  # quotaExceeded, keyInvalid, forbidden, ...


class YouTubeDataClient:
    def __init__(self, api_key: str, base: str = API) -> None:
        self._key = api_key
        self._base = base
        self.quota_used = 0

    def search_shorts(
        self, query: str, order: str = "viewCount", published_after: str | None = None,
        limit: int = 50,
    ) -> list[str]:  # fmt: skip
        """Video ids for a query, short videos only (YouTube's "short" = under 4 minutes)."""
        params = {
            "part": "id", "type": "video", "videoDuration": "short", "q": query,
            "order": order, "maxResults": str(min(limit, 50)), "safeSearch": "none",
        }  # fmt: skip
        if published_after:
            params["publishedAfter"] = published_after
        data = self._get("search", params, SEARCH_COST)
        return [i["id"]["videoId"] for i in data.get("items", []) if i["id"].get("videoId")]

    def videos(self, ids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(0, len(ids), 50):
            part = {"part": "snippet,statistics,contentDetails", "id": ",".join(ids[i : i + 50])}
            out += self._get("videos", part, LIST_COST).get("items", [])
        return out

    def channels(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(ids), 50):
            part = {"part": "snippet,statistics", "id": ",".join(ids[i : i + 50])}
            for c in self._get("channels", part, LIST_COST).get("items", []):
                out[c["id"]] = c
        return out

    def _get(self, endpoint: str, params: dict[str, str], cost: int) -> dict[str, Any]:
        url = self._base + endpoint + "?" + urllib.parse.urlencode({**params, "key": self._key})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    self.quota_used += cost
                    data: dict[str, Any] = json.loads(resp.read())
                    return data
            except urllib.error.HTTPError as e:
                reason, message = _error(e)
                if e.code >= 500 and attempt < 3:
                    time.sleep(2**attempt)
                    continue
                raise YouTubeApiError(_friendly(reason, message), reason) from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < 3:
                    time.sleep(2**attempt)
                    continue
                raise YouTubeApiError(f"couldn't reach YouTube: {e}", "network") from e
        raise YouTubeApiError("YouTube didn't answer", "network")


def _error(e: urllib.error.HTTPError) -> tuple[str, str]:
    try:
        err = json.loads(e.read())["error"]
        first = (err.get("errors") or [{}])[0]
        return str(first.get("reason", "")), str(err.get("message", ""))
    except (ValueError, KeyError, TypeError):
        return "", f"HTTP {e.code}"


def _friendly(reason: str, message: str) -> str:
    return {
        "quotaExceeded": "Today's YouTube API quota is used up; it resets at midnight "
        "Pacific time.",
        "keyInvalid": "The YouTube API key isn't valid. Check it in Shorts Intelligence.",
        "accessNotConfigured": "YouTube Data API v3 isn't enabled for this key's Google "
        "Cloud project. Enable it in the API Library, then try again.",
        "ipRefererBlocked": "This key is restricted to other apps or addresses.",
    }.get(reason, f"YouTube API error: {message or reason}")
