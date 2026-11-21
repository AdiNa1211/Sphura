from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from sphura.domain.events import JobProgressEvent
from sphura.ports.queue import Subscriber

log = logging.getLogger(__name__)


class InProcessEventBus:
    """Thread-safe fan-out. Subscribers must be fast; the API bridges to asyncio queues."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._lock = threading.Lock()

    def publish(self, event: JobProgressEvent) -> None:
        with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(event)
            except Exception:
                log.exception("event subscriber failed")

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return unsubscribe
