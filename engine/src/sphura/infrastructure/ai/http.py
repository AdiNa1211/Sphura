"""Minimal JSON-over-HTTP client with retries (stdlib only, so no SDK dependency)."""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Any

from sphura.domain.errors import SphuraError

log = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class AIProviderError(SphuraError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def fatal(self) -> bool:
        """Errors retrying won't fix (bad key, unknown model): stop the whole batch."""
        return (
            self.status is not None
            and 400 <= self.status < 500
            and self.status not in _RETRY_STATUS
        )


def post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout_s: float = 180.0,
    retries: int = 4,
) -> dict[str, Any]:
    data = json.dumps(body).encode()
    hdrs = {"content-type": "application/json", **(headers or {})}
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                out: dict[str, Any] = json.loads(resp.read())
                return out
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if e.code in _RETRY_STATUS and attempt < retries:
                wait = _retry_after(e) or min(60.0, 2.0**attempt + random.random())
                log.warning("HTTP %s from %s, retrying in %.0fs", e.code, url, wait)
                time.sleep(wait)
                continue
            raise AIProviderError(f"HTTP {e.code} from {url}: {detail}", e.code) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(min(30.0, 2.0**attempt))
                continue
            raise AIProviderError(f"could not reach {url}: {e}") from e
    raise AIProviderError(f"giving up on {url}")


def get_json(url: str, timeout_s: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        out: dict[str, Any] = json.loads(resp.read())
        return out


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    try:
        return float(e.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return None
