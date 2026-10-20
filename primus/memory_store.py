"""Local SQLite storage for explicitly recorded conversational memories."""

import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'data' / 'primus_memory.sqlite3'


class MemoryStore:
    """Small salience-ranked store with explicit, repeatable decay."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self):
        with self._lock, self._connect() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    salience REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    emotional_valence REAL NOT NULL DEFAULT 0.0,
                    novelty REAL NOT NULL DEFAULT 0.5,
                    decay_rate REAL NOT NULL DEFAULT 0.001,
                    consolidated INTEGER NOT NULL DEFAULT 0
                )
                '''
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS memories_kind_salience '
                'ON memories(kind, salience DESC, last_accessed DESC)'
            )

    def store(
        self,
        kind: str,
        content: Any,
        salience: float = 0.5,
        emotional_valence: float = 0.0,
        novelty: float = 0.5,
    ) -> int:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError('Memory kind must be a non-empty string')
        scores = (salience, emotional_valence, novelty)
        if not all(math.isfinite(float(value)) for value in scores):
            raise ValueError('Memory scores must be finite numbers')
        serialized = json.dumps(content, ensure_ascii=False, allow_nan=False)
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                '''
                INSERT INTO memories (
                    kind, content, salience, created_at, last_accessed,
                    emotional_valence, novelty
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    kind.strip(),
                    serialized,
                    min(1.0, max(0.0, float(salience))),
                    now,
                    now,
                    min(1.0, max(-1.0, float(emotional_valence))),
                    min(1.0, max(0.0, float(novelty))),
                ),
            )
            return int(cursor.lastrowid)

    def recall(
        self,
        kind: str,
        query: str | None = None,
        limit: int = 10,
        drift: float = 0.0,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        if not math.isfinite(float(drift)) or drift < 0:
            raise ValueError('drift must be a finite non-negative number')
        # `drift` remains accepted for API compatibility. MemoryStore returns
        # faithful content; experimental reconstruction belongs in its own API.
        del drift
        terms = [term.casefold() for term in (query or '').split() if term]
        now = time.time()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                '''
                SELECT id, content, salience, created_at, last_accessed,
                       access_count, emotional_valence, novelty, consolidated
                FROM memories
                WHERE kind = ? AND salience > 0.0
                ORDER BY salience DESC, last_accessed DESC
                ''',
                (kind,),
            ).fetchall()
            results = []
            for row in rows:
                try:
                    content = json.loads(row['content'])
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Stored memory {row['id']} contains invalid JSON"
                    ) from error
                serialized = json.dumps(content, ensure_ascii=False).casefold()
                if terms and not all(term in serialized for term in terms):
                    continue
                connection.execute(
                    '''
                    UPDATE memories
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                    ''',
                    (now, row['id']),
                )
                results.append(
                    {
                        'id': row['id'],
                        'kind': kind,
                        'content': content,
                        'salience': row['salience'],
                        'age_hours': max(0.0, (now - row['created_at']) / 3600),
                        'access_count': row['access_count'] + 1,
                        'emotional_valence': row['emotional_valence'],
                        'novelty': row['novelty'],
                        'consolidated': bool(row['consolidated']),
                    }
                )
                if len(results) >= limit:
                    break
        return results

    def decay_all(self, now: float | None = None) -> int:
        current_time = time.time() if now is None else float(now)
        if not math.isfinite(current_time):
            raise ValueError('now must be a finite timestamp')
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                '''
                SELECT id, last_accessed, access_count, emotional_valence,
                       decay_rate, salience, consolidated
                FROM memories
                WHERE salience > 0.0
                '''
            ).fetchall()
            updates = []
            for row in rows:
                elapsed_hours = max(
                    0.0, (current_time - row['last_accessed']) / 3600.0
                )
                retention = (
                    1.0
                    + 0.1 * row['access_count']
                    + 0.5 * abs(row['emotional_valence'])
                )
                rate = row['decay_rate'] / retention
                if row['consolidated']:
                    rate = min(rate, 0.0001)
                salience = max(0.0, row['salience'] * math.exp(-rate * elapsed_hours))
                updates.append((salience, current_time, row['id']))
            connection.executemany(
                'UPDATE memories SET salience = ?, last_accessed = ? WHERE id = ?',
                updates,
            )
            return len(updates)

    def consolidate(self, threshold: float = 0.7) -> int:
        if not math.isfinite(float(threshold)):
            raise ValueError('threshold must be finite')
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                '''
                UPDATE memories
                SET consolidated = 1, decay_rate = 0.0001
                WHERE salience >= ? AND consolidated = 0
                ''',
                (min(1.0, max(0.0, float(threshold))),),
            )
            return cursor.rowcount

    def prune_below(
        self, threshold: float, un_consolidated_only: bool = False
    ) -> int:
        """Forget low-salience records, optionally preserving consolidated ones."""
        if not math.isfinite(float(threshold)):
            raise ValueError('threshold must be finite')
        condition = 'salience < ?'
        if un_consolidated_only:
            condition += ' AND consolidated = 0'
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f'UPDATE memories SET salience = 0 WHERE {condition}',
                (min(1.0, max(0.0, float(threshold))),),
            )
            return cursor.rowcount

    def reinforce_kind(self, kind: str, increment: float = 0.02) -> int:
        """Apply bounded salience reinforcement to a selected memory category."""
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError('Memory kind must be a non-empty string')
        if not math.isfinite(float(increment)) or increment < 0:
            raise ValueError('increment must be finite and non-negative')
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                'UPDATE memories SET salience = MIN(1.0, salience + ?) '
                'WHERE kind = ? AND salience > 0',
                (float(increment), kind.strip()),
            )
            return cursor.rowcount

    def forget(self, memory_id: int) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                'UPDATE memories SET salience = 0 WHERE id = ?',
                (int(memory_id),),
            )
            return cursor.rowcount > 0

    def close(self):
        """Kept for callers that manage store lifetimes; connections are per-call."""


_store = None
_store_lock = threading.Lock()


def memory_store() -> MemoryStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = MemoryStore()
        return _store


def reset_memory_store():
    """Reset the process singleton, primarily for isolated tests."""
    global _store
    with _store_lock:
        _store = None
