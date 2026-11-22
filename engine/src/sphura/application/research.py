"""Shorts Intelligence, part 1: research public YouTube Shorts per game through the
official YouTube Data API, keep every metric collection, and score observed performance."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sphura.domain.errors import NotFoundError, SphuraError
from sphura.infrastructure.persistence.sqlite.research import SqliteResearchRepository
from sphura.infrastructure.research.youtube import YouTubeApiError, YouTubeDataClient
from sphura.pipeline import research as R

log = logging.getLogger(__name__)

ENV_KEY = "SPHURA_YOUTUBE_API_KEY"


class ResearchService:
    def __init__(
        self,
        repo: SqliteResearchRepository,
        data_dir: Path,
        client: Callable[[str], YouTubeDataClient] = YouTubeDataClient,
    ) -> None:
        self._repo = repo
        self._file = data_dir / "research.json"
        self._client = client
        self._lock = threading.Lock()
        self._run: dict[str, Any] | None = None  # the refresh in progress

    # --- API key -------------------------------------------------------------------------

    def _key(self) -> tuple[str | None, str]:
        try:
            key = json.loads(self._file.read_text(encoding="utf-8")).get("youtube_api_key")
            if key:
                return str(key), "settings"
        except (OSError, ValueError):
            pass
        env = os.environ.get(ENV_KEY)
        return (env, "environment") if env else (None, "")

    def key_status(self) -> dict[str, Any]:
        key, source = self._key()
        return {
            "set": bool(key),
            "masked": f"{key[:4]}…{key[-4:]}" if key and len(key) > 10 else None,
            "source": source,
        }

    def set_key(self, key: str) -> dict[str, Any]:
        """Saves the key after one cheap test call (1 quota unit) so typos show at once."""
        key = key.strip()
        if key:
            try:
                self._client(key).videos(["jNQXAC9IVRw"])  # "Me at the zoo": always public
            except YouTubeApiError as e:
                raise SphuraError(str(e)) from e
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"youtube_api_key": key}), encoding="utf-8")
        os.replace(tmp, self._file)
        return self.key_status()

    # --- refresh -------------------------------------------------------------------------

    def refresh(self, games: list[str] | None = None) -> dict[str, Any]:
        key, _ = self._key()
        if not key:
            raise SphuraError("Add a YouTube Data API key first")
        games = [g for g in (games or list(R.GAMES)) if g in R.GAMES]
        with self._lock:
            if self._run and self._run["status"] == "running":
                return self._run
            now = _now()
            self._run = {
                "id": uuid.uuid4().hex, "started_at": now, "status": "running", "games": games,
                "step": "starting", "progress": 0.0, "found": 0, "new": 0, "updated": 0,
                "skipped": 0, "quota_used": 0, "error": None,
            }  # fmt: skip
            self._repo.save_run(self._run)
            run = self._run
        threading.Thread(target=self._refresh, args=(run, key), daemon=True).start()
        return run

    def _refresh(self, run: dict[str, Any], key: str) -> None:
        client = self._client(key)
        try:
            found: dict[str, tuple[str, str]] = {}  # video id -> (game, query)
            recent = (datetime.now(UTC) - timedelta(days=R.RECENT_DAYS)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            searches = [(g, q) for g in run["games"] for q in R.GAMES[g]["queries"]]
            for i, (g, q) in enumerate(searches):
                run.update(step=f"searching: {q}", progress=0.7 * i / max(len(searches), 1))
                for vid in client.search_shorts(q) + client.search_shorts(
                    q, published_after=recent
                ):
                    found.setdefault(vid, (g, q))
                run["quota_used"] = client.quota_used
            tracked = set(self._repo.ids())
            ids = sorted(set(found) | tracked)  # also re-measure everything seen before
            run.update(step="reading video statistics", progress=0.75)
            items = {v["id"]: v for v in client.videos(ids)}
            channels = client.channels(sorted({v["snippet"]["channelId"] for v in items.values()}))
            run.update(step="saving", progress=0.9, quota_used=client.quota_used)
            now = datetime.now(UTC).isoformat(timespec="milliseconds")
            known_games = {v["video_id"]: v["game"] for v in self._repo.videos()}
            for vid, item in items.items():
                game, query = found.get(vid, ("", ""))
                game = str(known_games.get(vid) or game)
                rec = R.record(item, query) if query else _keep_queries(R.record(item, ""))
                if not game:
                    continue
                if vid not in tracked and (
                    rec["duration_s"] > R.MAX_SHORT_S or not R.is_relevant(game, item)
                ):
                    run["skipped"] += 1  # too long to be a Short, or not about this game
                    continue
                new = self._repo.upsert(vid, game, rec, now)
                ch = channels.get(item["snippet"]["channelId"])
                self._repo.snapshot(vid, now, R.stats(item, ch))
                run["new" if new else "updated"] += 1
            run["found"] = run["new"] + run["updated"]
            run.update(status="done", step="done", progress=1.0)
        except YouTubeApiError as e:
            log.warning("research refresh failed: %s", e)
            run.update(status="failed", error=str(e), reason=e.reason)
        except Exception as e:
            log.exception("research refresh failed")
            run.update(status="failed", error=f"unexpected error: {e}")
        finally:
            run.update(finished_at=_now(), quota_used=client.quota_used)
            self._repo.save_run(run)

    def status(self) -> dict[str, Any]:
        runs = self._repo.runs(10)
        done = next((r for r in runs if r["status"] == "done"), None)
        return {
            "key": self.key_status(),
            "running": self._run if self._run and self._run["status"] == "running" else None,
            "last_run": runs[0] if runs else None,
            "last_updated": done["finished_at"] if done else None,
            "runs": runs,
        }

    # --- reading -------------------------------------------------------------------------

    def _scored(self, game: str) -> list[dict[str, Any]]:
        vids = [  # rules may tighten after collection: re-check what's kept
            v
            for v in self._repo.videos(game)
            if R.mentions(game, v.get("title", ""), v.get("description", ""), v.get("tags", []))
        ]
        R.score(vids, datetime.now(UTC))
        vids.sort(key=lambda v: v.get("ops") or -1, reverse=True)
        return vids

    def overview(self) -> dict[str, Any]:
        games = {}
        total = 0
        for g, cfg in R.GAMES.items():
            vids = self._scored(g)
            total += len(vids)
            ops = [v["ops"] for v in vids if v.get("ops") is not None]
            cats: dict[str, int] = {}
            for v in vids:
                for c in v["categories"]:
                    cats[c] = cats.get(c, 0) + 1
            games[g] = {
                "name": cfg["name"],
                "count": len(vids),
                "avg_ops": round(sum(ops) / len(ops)) if ops else None,
                "categories": cats,
                "patterns": R.patterns(vids),
                "top": [_brief(v) for v in vids[:5]],
                "queries": cfg["queries"],
            }
        return {"total": total, "games": games, **self.status()}

    def videos(self, game: str, category: str | None = None) -> list[dict[str, Any]]:
        if game not in R.GAMES:
            raise NotFoundError("unknown game")
        vids = self._scored(game)
        if category:
            vids = [v for v in vids if category in v["categories"]]
        for v in vids:
            v.pop("description", None)
        return vids

    def history(self, video_id: str) -> list[dict[str, Any]]:
        return self._repo.history(video_id)


def _brief(v: dict[str, Any]) -> dict[str, Any]:
    keys = ("video_id", "url", "title", "channel", "thumbnail", "ops", "categories")
    return {k: v.get(k) for k in keys} | {"views": v["metrics"]["views"]}


def _keep_queries(rec: dict[str, Any]) -> dict[str, Any]:
    rec.pop("queries", None)  # a re-measure only: don't wipe the queries that found it
    return rec


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
