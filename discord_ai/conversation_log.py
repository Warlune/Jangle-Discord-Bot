from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TestConversationLog:
    """Small rotating JSONL log for explicitly enabled test deployments."""

    def __init__(
        self,
        enabled: bool,
        path: Path,
        *,
        max_bytes: int,
        backups: int = 3,
    ) -> None:
        self.enabled = enabled
        self.path = path
        self.max_bytes = max(1024, max_bytes)
        self.backups = max(1, backups)
        self._lock = threading.Lock()

    def record(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "event": event,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
        encoded_size = len(line.encode("utf-8"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            current_size = self.path.stat().st_size if self.path.exists() else 0
            if current_size and current_size + encoded_size > self.max_bytes:
                self._rotate_locked()
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(line)

    def _rotate_locked(self) -> None:
        oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
        oldest.unlink(missing_ok=True)
        for index in range(self.backups - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        if self.path.exists():
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))
