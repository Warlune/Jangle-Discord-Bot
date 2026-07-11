from __future__ import annotations

import json
from pathlib import Path

from discord_ai.conversation_log import TestConversationLog as ConversationLog


def test_test_log_writes_structured_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "test-conversations.jsonl"
    log = ConversationLog(True, path, max_bytes=10_000)

    log.record(
        "exchange",
        mode="voice",
        user_id=42,
        user_name="Speaker",
        prompt="hello",
        response="Hi there.",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "exchange"
    assert record["mode"] == "voice"
    assert record["user_id"] == 42
    assert record["prompt"] == "hello"
    assert record["response"] == "Hi there."
    assert record["timestamp"].endswith("Z")


def test_disabled_test_log_does_not_create_a_file(tmp_path: Path) -> None:
    path = tmp_path / "disabled.jsonl"

    ConversationLog(False, path, max_bytes=10_000).record("exchange", prompt="private")

    assert not path.exists()


def test_test_log_rotates_before_appending_to_an_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "rotating.jsonl"
    log = ConversationLog(True, path, max_bytes=1024, backups=2)
    log.record("exchange", prompt="x" * 1200)

    log.record("exchange", prompt="new record")

    assert path.with_name("rotating.jsonl.1").exists()
    current = json.loads(path.read_text(encoding="utf-8"))
    assert current["prompt"] == "new record"
