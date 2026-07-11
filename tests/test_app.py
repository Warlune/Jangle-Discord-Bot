from types import SimpleNamespace

import pytest

from discord_ai import app
from discord_ai.app import extract_text_call_request, split_discord_message, text_session_key


def test_discord_messages_are_split_below_platform_limit() -> None:
    chunks = split_discord_message("word " * 1200)

    assert 2 <= len(chunks) <= 6
    assert all(len(chunk) <= 1900 for chunk in chunks)


def test_text_call_word_must_start_the_message() -> None:
    words = ("jangle",)

    assert extract_text_call_request("Jangle, what time is it?", words) == "what time is it"
    assert extract_text_call_request("hey JANGLE tell me a joke", words) == "tell me a joke"
    assert extract_text_call_request("Hey, Jangle: search for news", words) == "search for news"
    assert extract_text_call_request("Jangle", words) == "Say a brief, friendly hello."
    assert extract_text_call_request("Everyone is talking right now", words) is None
    assert extract_text_call_request("I was telling Jangle about this", words) is None


def test_text_sessions_are_isolated_by_discord_user() -> None:
    assert text_session_key(1, 2, 10) != text_session_key(1, 2, 11)


def test_discord_startup_failure_retries_with_fresh_bot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    class FakeBot:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        def run(self, token: str, *, log_handler: object) -> None:
            attempts.append(token)
            if self.attempt == 1:
                raise ConnectionError("Discord gateway unavailable")

    def fake_create_bot(_settings: object) -> FakeBot:
        return FakeBot(len(attempts) + 1)

    monkeypatch.setattr(app, "create_bot", fake_create_bot)
    monkeypatch.setattr(app.time, "sleep", sleeps.append)

    app.run_bot_with_restart(
        SimpleNamespace(bot_token="secret"),  # type: ignore[arg-type]
        restart_delay_seconds=0.25,
    )

    assert attempts == ["secret", "secret"]
    assert sleeps == [0.25]
