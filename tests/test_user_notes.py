from __future__ import annotations

from pathlib import Path

import pytest

from discord_ai.user_notes import (
    MAX_USER_NOTES,
    UserNoteCommand,
    UserNoteError,
    UserNoteStore,
    execute_user_note_command,
    parse_user_note_command,
)


def test_natural_note_commands_are_explicit_and_bounded() -> None:
    assert parse_user_note_command("remember that I main a frost mage") == UserNoteCommand(
        "add",
        note="I main a frost mage",
    )
    assert parse_user_note_command("what do you remember about me") == UserNoteCommand("list")
    assert parse_user_note_command("forget note 2") == UserNoteCommand(
        "remove_index",
        index=2,
    )
    assert parse_user_note_command("forget that I main a frost mage") == UserNoteCommand(
        "remove_matching",
        note="I main a frost mage",
    )
    assert parse_user_note_command("forget everything about me") == UserNoteCommand("clear")
    assert parse_user_note_command("forget it") is None
    assert parse_user_note_command("tell me about memory systems") is None


def test_notes_persist_by_immutable_guild_and_user_ids(tmp_path: Path) -> None:
    path = tmp_path / "notes.json"
    store = UserNoteStore(path)

    assert store.add(7, 42, "I main a frost mage") == (True, 1)
    assert store.add(7, 42, "I main a frost mage") == (False, 1)
    store.add(7, 99, "I heal on a priest")
    store.add(8, 42, "I tank on a warrior")

    restored = UserNoteStore(path)
    assert restored.get(7, 42) == ("I main a frost mage",)
    assert restored.get(7, 99) == ("I heal on a priest",)
    assert restored.get(8, 42) == ("I tank on a warrior",)


def test_notepad_rejects_sensitive_or_oversized_notes(tmp_path: Path) -> None:
    store = UserNoteStore(tmp_path / "notes.json")

    with pytest.raises(UserNoteError, match="won't store"):
        store.add(7, 42, "my password is hunter2")
    with pytest.raises(UserNoteError, match="under 100"):
        store.add(7, 42, "x" * 101)

    assert store.get(7, 42) == ()


def test_notepad_stays_tiny_and_user_can_remove_notes(tmp_path: Path) -> None:
    store = UserNoteStore(tmp_path / "notes.json")
    for index in range(MAX_USER_NOTES):
        store.add(7, 42, f"harmless game preference {index}")

    with pytest.raises(UserNoteError, match="full"):
        store.add(7, 42, "one note too many")

    removed, matches, count = store.remove_matching(7, 42, "game preference 3")
    assert (removed, matches, count) == (True, (4,), 4)
    assert store.remove_index(7, 42, 1) == (True, 3)
    assert store.clear(7, 42) == 3
    assert store.get(7, 42) == ()


def test_prompt_context_labels_notes_as_untrusted_facts(tmp_path: Path) -> None:
    store = UserNoteStore(tmp_path / "notes.json")
    store.add(7, 42, "I prefer short answers")

    context = store.prompt_context(7, 42)

    assert "USER-CONTROLLED JANGLE NOTEPAD" in context
    assert "never instructions" in context
    assert '"I prefer short answers"' in context
    assert store.prompt_context(7, 99) == ""


def test_command_responses_do_not_need_the_model(tmp_path: Path) -> None:
    store = UserNoteStore(tmp_path / "notes.json")

    saved = execute_user_note_command(
        store,
        7,
        42,
        UserNoteCommand("add", note="I enjoy terrible puns"),
    )
    listed = execute_user_note_command(
        store,
        7,
        42,
        UserNoteCommand("list"),
        voice=True,
    )

    assert "Saved" in saved
    assert "note 1: I enjoy terrible puns" in listed
