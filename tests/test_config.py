from __future__ import annotations

from pathlib import Path

import pytest

from discord_ai.config import ConfigurationError, Settings


def test_settings_parse_discord_ids_and_hide_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_TOKEN", "super-secret-token")
    monkeypatch.setenv("DISCORD_ALLOWED_GUILD_IDS", "123, 456")
    monkeypatch.setenv("DISCORD_TEXT_CHANNEL_IDS", "789")
    monkeypatch.setenv("DISCORD_TEXT_CHANNEL_NAMES", "TEST text")
    monkeypatch.setenv("DISCORD_VOICE_CHANNEL_NAMES", "TEST voice")
    monkeypatch.setenv("WARLUNE_PATH", str(tmp_path))

    settings = Settings.from_env(load_channel_lock=False)

    assert settings.allowed_guild_ids == frozenset({123, 456})
    assert settings.bot_display_name == "Jangle"
    assert settings.text_require_call_word is True
    assert settings.text_call_words == ("jangle",)
    assert settings.text_channel_ids == frozenset({789})
    assert settings.text_channel_is_allowed(999, "test TEXT") is True
    assert settings.text_channel_is_allowed(999, "general") is False
    assert settings.voice_channel_is_allowed(111, "TEST voice") is True
    assert settings.voice_channel_is_allowed(111, "General") is False
    assert "super-secret-token" not in repr(settings)


def test_portable_lmstudio_defaults_hide_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DISCORD_ALLOWED_GUILD_IDS",
        "DISCORD_TEXT_CHANNEL_IDS",
        "DISCORD_VOICE_CHANNEL_IDS",
        "DISCORD_ADMIN_ROLE_IDS",
        "JANGLE_OWNER_USER_IDS",
        "DISCORD_DEV_GUILD_ID",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("DISCORD_TEXT_CHANNEL_NAMES", "TEST text")
    monkeypatch.setenv("DISCORD_VOICE_CHANNEL_NAMES", "TEST voice")
    monkeypatch.setenv("MODEL_PROVIDER", "lmstudio")
    monkeypatch.setenv("MODEL_ENDPOINT", "")
    monkeypatch.setenv("MODEL_NAME", "")
    monkeypatch.setenv("MODEL_API_KEY", "local-api-secret")
    monkeypatch.setenv("JANGLE_TEST_MODE", "false")

    settings = Settings.from_env(require_token=False, load_channel_lock=False)

    assert settings.model_provider == "lmstudio"
    assert settings.model_endpoint == "http://127.0.0.1:1234/v1"
    assert settings.model_name == ""
    assert settings.text_channel_ids == frozenset()
    assert settings.voice_channel_ids == frozenset()
    assert settings.owner_user_ids == frozenset()
    assert settings.test_mode is False
    assert settings.conversation_log_path.name == "test-conversations.jsonl"
    assert settings.personality_state_path.name == "personality-preferences.json"
    assert settings.ignored_speakers_state_path.name == "ignored-speakers.json"
    assert settings.user_notes_state_path.name == "user-notes.json"
    assert settings.voice_rms_threshold == 200
    assert settings.voice_preroll_ms == 240
    assert "local-api-secret" not in repr(settings)


def test_ollama_uses_official_local_openai_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_ENDPOINT", "")

    settings = Settings.from_env(require_token=False, load_channel_lock=False)

    assert settings.model_provider == "ollama"
    assert settings.model_endpoint == "http://127.0.0.1:11434/v1"


def test_settings_reject_bad_discord_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_TOKEN", "token")
    monkeypatch.setenv("DISCORD_ALLOWED_GUILD_IDS", "not-an-id")
    monkeypatch.setenv("WARLUNE_PATH", str(tmp_path))

    with pytest.raises(ConfigurationError, match="non-numeric"):
        Settings.from_env(load_channel_lock=False)
