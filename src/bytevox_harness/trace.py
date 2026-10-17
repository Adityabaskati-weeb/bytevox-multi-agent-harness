"""Append-only JSONL trace writer used by the CLI and replay command."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path, run_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run_id
        self.started = 0.0
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False

    def set_started(self, monotonic_start: float) -> None:
        self.started = monotonic_start

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = self._record(event, fields)
            if self._closed:
                return record
            self._append(record)
            return record

    def finish(self, **fields: Any) -> dict[str, Any]:
        """Write the terminal event and atomically prevent late worker writes."""
        with self._lock:
            record = self._record("run_finished", fields)
            self._append(record)
            self._closed = True
            return record

    def _record(self, event: str, fields: dict[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        return {
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((__import__("time").perf_counter() - self.started) * 1000, 2)
            if self.started
            else 0.0,
            "run_id": self.run_id,
            "event": event,
            **fields,
        }

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
