"""Structured execution history for generation sessions (#4).

Records each run's lifecycle as JSON events to disk so an agent (or a human) can
inspect what happened, without re-running anything. This is the "structured
execution history" half of #4; locality-of-reference is handled by passing the
recent window into the judge (see inference.py).
"""

import json
import os
import threading
from datetime import datetime, timezone

DEFAULT_HISTORY_PATH = os.environ.get("STRMER_HISTORY", ".strmer_history.json")


class SessionRecorder:
    """Append-only recorder of generation-session events.

    Events are written to ``path`` (JSON array) after every mutation so a crash
    never loses recorded progress. In-memory copy is kept for fast reads within
    the same process.
    """

    def __init__(self, path=DEFAULT_HISTORY_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.events = []
        self.session_id = None
        self.prompt = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    self.events = [e for e in data if isinstance(e, dict)]
            except (ValueError, OSError):
                self.events = []

    def start(self, prompt):
        with self._lock:
            self.prompt = prompt
            self.session_id = datetime.now(timezone.utc).isoformat()
            self.events.append(
                {"type": "start", "session_id": self.session_id, "prompt": prompt}
            )
            self.flush_locked()

    def record_iteration(self, iteration, signals):
        with self._lock:
            if not self.session_id:
                # Called outside a session; still persist it for debugging.
                self.start("<anonymous>")
            self.events.append(
                {
                    "type": "iteration",
                    "session_id": self.session_id,
                    "iteration": iteration,
                    "signals": signals,
                }
            )
            self.flush_locked()

    def record_event(self, kind, detail=None):
        with self._lock:
            if not self.session_id:
                self.start("<anonymous>")
            self.events.append({"type": kind, "detail": detail})
            self.flush_locked()

    def finish(self, status, action=None):
        with self._lock:
            rec = {"type": "finish", "status": status, "action": action}
            if self.session_id:
                rec["session_id"] = self.session_id
            self.events.append(rec)
            self.flush_locked()

    def flush_locked(self):
        # Caller must hold self._lock.
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.events, fh, indent=2)
        except OSError:
            pass  # history is best-effort; never fail a generation for this

    def get_history(self, limit=None):
        with self._lock:
            events = list(self.events)
            if limit is not None:
                events = events[-limit:]
            return events
