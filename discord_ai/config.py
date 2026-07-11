from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    pass


def _ids(name: str) -> frozenset[int]:
    values: set[int] = set()
    for raw in os.getenv(name, "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{name} contains a non-numeric Discord ID: {raw}") from exc
        if value <= 0:
            raise ConfigurationError(f"{name} contains an invalid Discord ID: {raw}")
        values.add(value)
    return frozenset(values)


def _names(name: str, default: str = "") -> frozenset[str]:
    return frozenset(
        value.strip().casefold()
        for value in os.getenv(name, default).split(",")
        if value.strip()
    )


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser() if raw else default


def _wake_words() -> tuple[str, ...]:
    words = tuple(
        value.strip().casefold()
        for value in os.getenv(
            "VOICE_WAKE_WORDS", "jangle,jengel,jingle,jangel"
        ).split(",")
        if value.strip()
    )
    if not words:
        raise ConfigurationError("VOICE_WAKE_WORDS must contain at least one wake word")
    return words


def _text_call_words(default: str) -> tuple[str, ...]:
    words = tuple(
        value.strip().casefold()
        for value in os.getenv("TEXT_CALL_WORDS", default).split(",")
        if value.strip()
    )
    if not words:
        raise ConfigurationError("TEXT_CALL_WORDS must contain at least one call word")
    return words


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    bot_display_name: str
    bot_prefix: str
    allowed_guild_ids: frozenset[int]
    text_channel_ids: frozenset[int]
    text_channel_names: frozenset[str]
    voice_channel_ids: frozenset[int]
    voice_channel_names: frozenset[str]
    admin_role_ids: frozenset[int]
    owner_user_ids: frozenset[int]
    dev_guild_id: int | None
    test_mode: bool
    conversation_log_path: Path
    conversation_log_max_bytes: int
    model_provider: str
    model_endpoint: str
    model_name: str
    model_api_key: str = field(repr=False)
    model_timeout_seconds: float
    internet_search_enabled: bool
    searxng_url: str
    search_max_results: int
    warlune_path: Path
    warlune_config_path: Path
    warlune_endpoint: str
    warlune_model: str
    max_concurrent_requests: int
    history_turns: int
    answer_max_chars: int
    text_require_call_word: bool
    text_call_words: tuple[str, ...]
    voice_wake_words: tuple[str, ...]
    voice_silence_ms: int
    voice_min_ms: int
    voice_max_seconds: int
    voice_rms_threshold: int
    voice_preroll_ms: int
    voice_barge_in_frames: int
    voice_followup_seconds: int
    voice_text_echo: bool
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str | None
    tts_provider: str
    tts_voice: str
    tts_rate: str
    tts_max_chars: int
    tts_state_path: Path
    personality_state_path: Path
    ignored_speakers_state_path: Path
    user_notes_state_path: Path
    music_volume: float
    music_max_seconds: int
    music_queue_max: int
    playlist_max_tracks: int
    youtube_search_timeout_seconds: int

    @classmethod
    def from_env(
        cls, *, require_token: bool = True, load_channel_lock: bool = True
    ) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        if load_channel_lock:
            load_dotenv(PROJECT_ROOT / "channels.env", override=True)
        token = os.getenv("BOT_TOKEN", "").strip()
        if require_token and not token:
            raise ConfigurationError("BOT_TOKEN is missing from .env")

        provider_aliases = {
            "warlune": "warlune",
            "lmstudio": "lmstudio",
            "lm-studio": "lmstudio",
            "lm studio": "lmstudio",
            "ollama": "ollama",
        }
        provider_raw = os.getenv("MODEL_PROVIDER", "warlune").strip().casefold()
        model_provider = provider_aliases.get(provider_raw)
        if model_provider is None:
            raise ConfigurationError("MODEL_PROVIDER must be warlune, lmstudio, or ollama")
        model_endpoint = os.getenv("MODEL_ENDPOINT", "").strip().rstrip("/")
        if not model_endpoint:
            model_endpoint = {
                "warlune": "",
                "lmstudio": "http://127.0.0.1:1234/v1",
                "ollama": "http://127.0.0.1:11434/v1",
            }[model_provider]
        if model_endpoint and not model_endpoint.startswith(("http://", "https://")):
            raise ConfigurationError("MODEL_ENDPOINT must start with http:// or https://")
        model_name = os.getenv("MODEL_NAME", "").strip()

        warlune_override = os.getenv("WARLUNE_PATH", "").strip()
        warlune_path = (
            Path(warlune_override).expanduser()
            if warlune_override
            else PROJECT_ROOT.parent / "warlune-lan-agent"
        )
        config_override = os.getenv("WARLUNE_CONFIG", "").strip()
        warlune_config_path = (
            Path(config_override).expanduser() if config_override else warlune_path / "config.json"
        )
        tts_provider = os.getenv("TTS_PROVIDER", "edge").strip().casefold()
        if tts_provider not in {"edge", "none"}:
            raise ConfigurationError("TTS_PROVIDER must be edge or none")

        dev_guild_raw = os.getenv("DISCORD_DEV_GUILD_ID", "").strip()
        dev_guild_id = int(dev_guild_raw) if dev_guild_raw else None
        if dev_guild_id is not None and dev_guild_id <= 0:
            raise ConfigurationError("DISCORD_DEV_GUILD_ID must be a positive Discord ID")

        language = os.getenv("WHISPER_LANGUAGE", "en").strip().casefold()
        display_name = os.getenv("BOT_DISPLAY_NAME", "Jangle").strip() or "Jangle"
        log_path = _path(
            "TEST_CONVERSATION_LOG",
            PROJECT_ROOT / "logs" / "test-conversations.jsonl",
        )
        if not 2 <= len(display_name) <= 32:
            raise ConfigurationError("BOT_DISPLAY_NAME must contain 2 to 32 characters")
        return cls(
            bot_token=token,
            bot_display_name=display_name,
            bot_prefix=os.getenv("BOT_PREFIX", "!").strip() or "!",
            allowed_guild_ids=_ids("DISCORD_ALLOWED_GUILD_IDS"),
            text_channel_ids=_ids("DISCORD_TEXT_CHANNEL_IDS"),
            text_channel_names=_names("DISCORD_TEXT_CHANNEL_NAMES", "TEST text"),
            voice_channel_ids=_ids("DISCORD_VOICE_CHANNEL_IDS"),
            voice_channel_names=_names("DISCORD_VOICE_CHANNEL_NAMES", "TEST voice"),
            admin_role_ids=_ids("DISCORD_ADMIN_ROLE_IDS"),
            owner_user_ids=_ids("JANGLE_OWNER_USER_IDS"),
            dev_guild_id=dev_guild_id,
            test_mode=_boolean("JANGLE_TEST_MODE", False),
            conversation_log_path=log_path.resolve(),
            conversation_log_max_bytes=_integer("TEST_LOG_MAX_MB", 10, 1, 100)
            * 1024
            * 1024,
            model_provider=model_provider,
            model_endpoint=model_endpoint,
            model_name=model_name,
            model_api_key=os.getenv("MODEL_API_KEY", "").strip(),
            model_timeout_seconds=_number(
                "MODEL_REQUEST_TIMEOUT_SECONDS",
                120.0,
                10.0,
                600.0,
            ),
            internet_search_enabled=_boolean("INTERNET_SEARCH_ENABLED", False),
            searxng_url=os.getenv("SEARXNG_URL", "http://127.0.0.1:8888").strip().rstrip("/"),
            search_max_results=_integer("SEARCH_MAX_RESULTS", 5, 1, 10),
            warlune_path=warlune_path.resolve(),
            warlune_config_path=warlune_config_path.resolve(),
            warlune_endpoint=(
                os.getenv("WARLUNE_ENDPOINT", "").strip()
                or (model_endpoint if model_provider == "warlune" else "")
            ),
            warlune_model=(
                os.getenv("WARLUNE_MODEL", "").strip()
                or (model_name if model_provider == "warlune" else "")
            ),
            max_concurrent_requests=_integer("MAX_CONCURRENT_REQUESTS", 2, 2, 8),
            history_turns=_integer("EPHEMERAL_HISTORY_TURNS", 6, 0, 20),
            answer_max_chars=_integer("ANSWER_MAX_CHARS", 6000, 500, 12000),
            text_require_call_word=_boolean("TEXT_REQUIRE_CALL_WORD", True),
            text_call_words=_text_call_words(display_name),
            voice_wake_words=_wake_words(),
            voice_silence_ms=_integer("VOICE_SILENCE_MS", 650, 300, 3000),
            voice_min_ms=_integer("VOICE_MIN_MS", 250, 100, 3000),
            voice_max_seconds=_integer("VOICE_MAX_SECONDS", 25, 3, 60),
            voice_rms_threshold=_integer("VOICE_RMS_THRESHOLD", 200, 50, 5000),
            voice_preroll_ms=_integer("VOICE_PREROLL_MS", 240, 0, 1000),
            voice_barge_in_frames=_integer("VOICE_BARGE_IN_FRAMES", 15, 3, 50),
            voice_followup_seconds=_integer("VOICE_FOLLOWUP_SECONDS", 25, 5, 90),
            voice_text_echo=_boolean("VOICE_TEXT_ECHO", False),
            whisper_model=os.getenv("WHISPER_MODEL", "base.en").strip() or "base.en",
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip().casefold() or "cpu",
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8",
            whisper_language=language or None,
            tts_provider=tts_provider,
            tts_voice=os.getenv("TTS_VOICE", "en-US-AriaNeural").strip() or "en-US-AriaNeural",
            tts_rate=os.getenv("TTS_RATE", "+5%").strip() or "+5%",
            tts_max_chars=_integer("TTS_MAX_CHARS", 600, 100, 2000),
            tts_state_path=_path(
                "TTS_STATE_PATH",
                PROJECT_ROOT / "data" / "voice-preferences.json",
            ).resolve(),
            personality_state_path=_path(
                "PERSONALITY_STATE_PATH",
                PROJECT_ROOT / "data" / "personality-preferences.json",
            ).resolve(),
            ignored_speakers_state_path=_path(
                "IGNORED_SPEAKERS_STATE_PATH",
                PROJECT_ROOT / "data" / "ignored-speakers.json",
            ).resolve(),
            user_notes_state_path=_path(
                "USER_NOTES_STATE_PATH",
                PROJECT_ROOT / "data" / "user-notes.json",
            ).resolve(),
            music_volume=_number("MUSIC_VOLUME", 0.45, 0.05, 1.0),
            music_max_seconds=_integer("MUSIC_MAX_MINUTES", 720, 1, 720) * 60,
            music_queue_max=_integer("MUSIC_QUEUE_MAX", 50, 1, 100),
            playlist_max_tracks=_integer("PLAYLIST_MAX_TRACKS", 25, 1, 50),
            youtube_search_timeout_seconds=_integer(
                "YOUTUBE_SEARCH_TIMEOUT_SECONDS", 30, 5, 60
            ),
        )

    def guild_is_allowed(self, guild_id: int | None) -> bool:
        return guild_id is not None and (
            not self.allowed_guild_ids or guild_id in self.allowed_guild_ids
        )

    def text_channel_is_allowed(self, channel_id: int | None, channel_name: str = "") -> bool:
        if not self.text_channel_ids and not self.text_channel_names:
            return True
        return (
            channel_id is not None
            and channel_id in self.text_channel_ids
            or channel_name.strip().casefold() in self.text_channel_names
        )

    def voice_channel_is_allowed(self, channel_id: int | None, channel_name: str = "") -> bool:
        if not self.voice_channel_ids and not self.voice_channel_names:
            return True
        return (
            channel_id is not None
            and channel_id in self.voice_channel_ids
            or channel_name.strip().casefold() in self.voice_channel_names
        )
