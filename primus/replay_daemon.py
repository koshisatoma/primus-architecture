"""Low-priority idle-time memory maintenance."""

import logging
import threading
import time

from primus.memory_store import memory_store


LOGGER = logging.getLogger(__name__)


class ReplayDaemon:
    def __init__(
        self,
        idle_threshold: float = 300.0,
        replay_interval: float = 600.0,
        check_interval: float = 10.0,
        store=None,
    ):
        if min(idle_threshold, replay_interval, check_interval) <= 0:
            raise ValueError('Replay intervals must be positive')
        self.idle_threshold = idle_threshold
        self.replay_interval = replay_interval
        self.check_interval = check_interval
        self.store = store
        self.last_activity = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last_replay = 0.0

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def mark_activity(self):
        with self._lock:
            self.last_activity = time.monotonic()

    def start(self):
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._replay_loop,
                daemon=True,
                name='primus-memory-replay',
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def run_once(self, now: float | None = None) -> dict[str, int]:
        store = self.store or memory_store()
        timestamp = time.time() if now is None else now
        decayed = store.decay_all(now=timestamp)
        consolidated = store.consolidate(threshold=0.7)
        return {'decayed': decayed, 'consolidated': consolidated}

    def _replay_loop(self):
        while not self._stop.wait(self.check_interval):
            current = time.monotonic()
            with self._lock:
                idle_for = current - self.last_activity
            if idle_for < self.idle_threshold:
                continue
            if current - self._last_replay < self.replay_interval:
                continue
            try:
                self.run_once()
                self._last_replay = current
            except Exception:
                LOGGER.exception('Idle memory maintenance failed')


_daemon = None
_daemon_lock = threading.Lock()


def replay_daemon() -> ReplayDaemon:
    global _daemon
    with _daemon_lock:
        if _daemon is None:
            _daemon = ReplayDaemon()
        return _daemon


def mark_activity():
    replay_daemon().mark_activity()
