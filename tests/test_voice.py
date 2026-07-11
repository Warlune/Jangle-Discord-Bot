from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import davey
import discord
import numpy as np
from discord.ext.voice_recv import router as voice_recv_router

from discord_ai.social import GameState, QuizQuestion, SocialCommand
from discord_ai.user_notes import UserNoteStore
from discord_ai.voice import (
    ENERGY_EASTER_EGG_RESPONSE,
    FollowupState,
    IgnoredSpeakerStore,
    JangleTts,
    LocalWhisper,
    MADAM_VOICE_ID,
    MusicItem,
    PersonalityPreferenceStore,
    PERSONALITY_CHOICES,
    PcmSegment,
    PcmSegmenter,
    PocketPcmAudioSource,
    RecentVoiceExchange,
    SpokenItem,
    TemporaryFileFFmpegPCMAudio,
    VoicePreferenceStore,
    VoiceManager,
    VoiceSession,
    YouTubeMusic,
    YouTubePlaylist,
    YouTubeTrack,
    _audio_source_error,
    _bounded_audio_delay,
    _decrypt_dave_rtp,
    _music_stream_ended_too_early,
    _pocket_chunk_to_discord_pcm,
    _speech_text,
    extract_wake_request,
    find_personality_choice,
    find_voice_choice,
    install_dave_voice_receive_patch,
    is_impossible_repeated_wake_transcript,
    is_energy_easter_egg,
    is_nonverbal_interruption,
    is_admin_clear_queue_command,
    is_admin_stop_command,
    is_show_queue_command,
    member_can_administer_music,
    parse_add_music_shorthand,
    parse_add_to_queue_query,
    parse_dj_mode_command,
    parse_music_pause_command,
    parse_music_query,
    parse_music_navigation,
    parse_music_volume_command,
    parse_music_volume_level,
    parse_playlist_command,
    parse_personality_change_command,
    parse_speaker_listening_command,
    parse_voice_change_command,
    resolve_voice_members,
    select_followup_mode,
    voice_choices_text,
)
from discord_ai.warlune import parse_voice_answer


def _social_test_session() -> tuple[VoiceSession, list[str], list[tuple[str, dict[str, object]]]]:
    messages: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []

    class FakeSessions:
        def reset(self, _key: str) -> bool:
            return True

    class FakeAnswers:
        sessions = FakeSessions()

        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

        async def answer(self, _key: str, prompt: str, **_kwargs: object) -> str:
            if prompt.startswith("Begin a collaborative scenario"):
                return "A cursed portal tears open above the inn."
            if prompt.startswith("Conclude the collaborative story"):
                return "The portal seals, leaving one suspiciously warm tankard behind."
            if "story action" in prompt:
                return "The floor answers with thunder and reveals a hidden stairway."
            return "No. Question noted."

    class FakeChannel:
        id = 8
        name = "TEST voice"

        def __init__(self, members: list[object]) -> None:
            self.members = members

        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    def member(user_id: int, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=user_id,
            display_name=name,
            global_name=name,
            name=name,
            bot=False,
            roles=[],
            guild_permissions=SimpleNamespace(administrator=False, manage_guild=False),
        )

    members = [member(1, "Host"), member(2, "Guest")]
    channel = FakeChannel(members)
    session = object.__new__(VoiceSession)
    session.settings = SimpleNamespace(admin_role_ids=frozenset())
    session.answer_service = FakeAnswers()
    session.voice_client = SimpleNamespace(
        guild=SimpleNamespace(id=7, owner_id=1),
        channel=channel,
    )
    session.companion_channel = channel
    session.tts = None
    session._personality_key = "disabled"
    session._recent_exchange = None
    session._music_item_count = 0
    session._current_music_item = None
    session._dj_mode = False
    session._game_state = None
    session._poll_state = None
    session._story_state = None
    session._award_state = None
    session._party_mode_enabled_at = 0.0
    session._party_mode_deadline = 0.0
    session._party_last_reaction_at = 0.0
    session._party_mode_enabled_by = 0
    return session, messages, events


def _voiced_packet(amplitude: int = 1200) -> bytes:
    # One 20 ms Discord packet: 48 kHz, stereo, signed 16-bit PCM.
    return np.full((960, 2), amplitude, dtype=np.int16).tobytes()


def test_segmenter_emits_bounded_speech_after_silence() -> None:
    segmenter = PcmSegmenter(
        silence_ms=500,
        minimum_ms=300,
        maximum_seconds=10,
        rms_threshold=300,
    )
    for index in range(20):
        segmenter.push(42, "Speaker", _voiced_packet(), now=index * 0.02)

    segments = segmenter.pop_ready(now=1.0)

    assert len(segments) == 1
    assert segments[0].user_id == 42
    assert 0.39 <= segments[0].duration_seconds <= 0.41


def test_segmenter_ignores_short_noise() -> None:
    segmenter = PcmSegmenter(
        silence_ms=500,
        minimum_ms=300,
        maximum_seconds=10,
        rms_threshold=300,
    )
    segmenter.push(42, "Speaker", _voiced_packet(), now=0.0)

    assert segmenter.pop_ready(now=1.0) == []


def test_segmenter_prepends_quiet_audio_without_inflating_spoken_duration() -> None:
    segmenter = PcmSegmenter(
        silence_ms=100,
        minimum_ms=10,
        maximum_seconds=10,
        rms_threshold=300,
        preroll_ms=40,
    )
    quiet = _voiced_packet(100)
    segmenter.push(42, "Speaker", quiet, now=0.0)
    segmenter.push(42, "Speaker", quiet, now=0.02)
    segmenter.push(42, "Speaker", _voiced_packet(), now=0.04)

    segments = segmenter.pop_ready(now=0.2)

    assert len(segments) == 1
    assert segments[0].pcm.startswith(quiet + quiet)
    assert 0.019 <= segments[0].duration_seconds <= 0.021


def test_segmenter_preserves_trivia_buzz_in_timing_through_silence() -> None:
    segmenter = PcmSegmenter(
        silence_ms=100,
        minimum_ms=10,
        maximum_seconds=10,
        rms_threshold=300,
    )
    segmenter.push(
        42,
        "Speaker",
        _voiced_packet(),
        now=5.0,
        blocked_playback=True,
        protected_game_answer=True,
        game_window_token=7,
    )
    segmenter.push(
        42,
        "Speaker",
        _voiced_packet(0),
        now=5.02,
        game_window_token=7,
    )

    segments = segmenter.pop_ready(now=5.2)

    assert len(segments) == 1
    assert segments[0].protected_game_answer is True
    assert segments[0].game_window_token == 7
    assert segments[0].started_at == 5.0


def test_segmenter_discards_active_and_ready_audio_for_ignored_user() -> None:
    segmenter = PcmSegmenter(
        silence_ms=100,
        minimum_ms=10,
        maximum_seconds=10,
        rms_threshold=300,
    )
    segmenter.push(42, "Ignored", _voiced_packet(), now=0.0)
    segmenter.push(99, "Heard", _voiced_packet(), now=0.0)

    segmenter.discard_user(42)
    segments = segmenter.pop_ready(now=1.0)

    assert [segment.user_id for segment in segments] == [99]


def test_local_whisper_uses_single_pass_low_latency_decoding() -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, _audio: object, **kwargs: object) -> tuple[list[object], object]:
            captured.update(kwargs)
            return [SimpleNamespace(text=" hello ")], object()

    whisper = object.__new__(LocalWhisper)
    whisper.settings = SimpleNamespace(
        whisper_language="en",
        voice_wake_words=("jangle", "jingle"),
    )
    whisper._ensure_model = lambda: FakeModel()  # type: ignore[method-assign]

    result = whisper._run_transcription(np.zeros(16000, dtype=np.float32))

    assert result == "hello"
    assert captured["beam_size"] == 1
    assert captured["temperature"] == 0.0
    assert captured["condition_on_previous_text"] is False
    assert captured["without_timestamps"] is True
    assert "hotwords" not in captured
    assert captured["vad_parameters"] == {
        "threshold": 0.35,
        "min_speech_duration_ms": 80,
        "min_silence_duration_ms": 200,
        "speech_pad_ms": 240,
    }


def test_local_whisper_discards_impossible_repeated_wake_hallucination() -> None:
    class FakeModel:
        def transcribe(self, _audio: object, **_kwargs: object) -> tuple[list[object], object]:
            return [SimpleNamespace(text=", ".join(["Hey Jengel"] * 45))], object()

    whisper = object.__new__(LocalWhisper)
    whisper.settings = SimpleNamespace(
        whisper_language="en",
        voice_wake_words=("jangle", "jengel"),
    )
    whisper._ensure_model = lambda: FakeModel()  # type: ignore[method-assign]

    result = whisper._run_transcription(np.zeros(12_800, dtype=np.float32))

    assert result == ""


def test_audio_pacing_resets_instead_of_bursting_after_long_stall() -> None:
    started_at, delay, reset = _bounded_audio_delay(0.0, 100, 10.0, 0.02)

    assert reset is True
    assert round(started_at, 3) == 8.0
    assert round(delay, 3) == 0.02

    _, next_delay, next_reset = _bounded_audio_delay(started_at, 101, 10.02, 0.02)
    assert next_reset is False
    assert round(next_delay, 3) == 0.02


def test_audio_source_error_is_found_through_volume_wrapper() -> None:
    error = RuntimeError("ffmpeg failed")
    wrapped = SimpleNamespace(original=SimpleNamespace(_current_error=error))

    assert _audio_source_error(wrapped) is error


def test_pocket_chunk_is_resampled_to_discord_stereo_pcm() -> None:
    pcm = _pocket_chunk_to_discord_pcm(
        np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    )
    stereo = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)

    assert stereo.shape == (6, 2)
    assert np.array_equal(stereo[:, 0], stereo[:, 1])
    assert stereo[0, 0] == -32767
    assert stereo[-1, 0] == 32767


def test_pocket_audio_source_emits_exact_discord_frames() -> None:
    source = PocketPcmAudioSource(
        lambda: iter((np.linspace(-0.2, 0.2, 1920, dtype=np.float32),))
    )

    asyncio.run(source.wait_until_ready())
    frames = [source.read() for _ in range(4)]

    assert all(len(frame) == 3840 for frame in frames)
    assert source.read() == b""
    assert source.is_opus() is False
    source.cleanup()


def test_pocket_audio_source_cancels_and_drains_background_generation() -> None:
    stop_event = threading.Event()
    drained = threading.Event()

    def stream() -> object:
        yield np.zeros(1920, dtype=np.float32)
        stop_event.wait(1.0)
        drained.set()

    source = PocketPcmAudioSource(
        stream,  # type: ignore[arg-type]
        stop_event=stop_event,
    )
    asyncio.run(source.wait_until_ready())

    source.cleanup()

    assert stop_event.is_set()
    assert drained.wait(0.5)


def test_pocket_voice_uses_streaming_source_without_edge_rendering() -> None:
    calls: list[tuple[str, str]] = []

    class FakePocket:
        def stream(
            self,
            preset: str,
            text: str,
            _stop_event: object,
        ) -> object:
            calls.append((preset, text))
            return iter((np.zeros(1920, dtype=np.float32),))

    settings = SimpleNamespace(
        tts_voice="en-US-AriaNeural",
        tts_max_chars=600,
        tts_rate="+5%",
    )
    tts = JangleTts(settings, FakePocket(), "pocket:alba")  # type: ignore[arg-type]

    source = asyncio.run(tts.create_source("Hello from local speech."))

    assert isinstance(source, PocketPcmAudioSource)
    assert calls == [("alba", "Hello from local speech.")]
    assert tts.last_provider == "pocket"
    source.cleanup()


def test_pocket_voice_falls_back_to_edge_when_streaming_fails() -> None:
    fallback_source = object()

    class BrokenPocket:
        def stream(
            self,
            _preset: str,
            _text: str,
            _stop_event: object,
        ) -> object:
            def fail() -> object:
                raise RuntimeError("local model failed")
                yield b""  # pragma: no cover

            return fail()

    settings = SimpleNamespace(
        tts_voice="en-US-BrianNeural",
        tts_max_chars=600,
        tts_rate="+5%",
    )
    tts = JangleTts(settings, BrokenPocket(), "pocket:alba")  # type: ignore[arg-type]

    async def fake_edge_source(_text: str) -> object:
        return fallback_source

    tts._edge_source = fake_edge_source  # type: ignore[method-assign]

    assert asyncio.run(tts.create_source("Fallback test")) is fallback_source
    assert tts.last_provider == "edge-fallback"


def test_music_stream_early_end_detection_ignores_commands_and_short_media() -> None:
    track = YouTubeTrack("song", "Song", "https://youtube.test/song", "stream", 232)
    short = YouTubeTrack("clip", "Clip", "https://youtube.test/clip", "stream", 20)

    assert _music_stream_ended_too_early(track, 0.7, None) is True
    assert _music_stream_ended_too_early(track, 20.0, None) is False
    assert _music_stream_ended_too_early(track, 0.7, "next") is False
    assert _music_stream_ended_too_early(short, 0.7, None) is False


def test_youtube_resolve_refreshes_an_existing_stream_url() -> None:
    original = YouTubeTrack("song", "Song", "https://youtube.test/song", "old-stream", 180)
    refreshed = YouTubeTrack("song", "Song", "https://youtube.test/song", "new-stream", 180)
    calls: list[YouTubeTrack] = []
    music = object.__new__(YouTubeMusic)
    music.timeout_seconds = 5

    def refresh(track: YouTubeTrack) -> YouTubeTrack:
        calls.append(track)
        return refreshed

    music._resolve_sync = refresh  # type: ignore[method-assign]

    result = asyncio.run(music.resolve(original))

    assert result.stream_url == "new-stream"
    assert calls == [original]


def test_voice_manager_closes_stale_session_before_reconnect() -> None:
    state = {"closed": False, "disconnected": False, "connect_attempted": False}

    class ExistingSession:
        voice_client = SimpleNamespace(is_connected=lambda: False)

        async def close(self) -> None:
            state["closed"] = True

    class OldVoiceClient:
        async def disconnect(self, *, force: bool) -> None:
            assert force is True
            assert state["closed"] is True
            state["disconnected"] = True

    class TargetChannel:
        id = 8
        name = "TEST voice"

        async def connect(self, **_kwargs: object) -> object:
            assert state["closed"] is True
            assert state["disconnected"] is True
            state["connect_attempted"] = True
            raise RuntimeError("stop after stale-session assertions")

    async def exercise() -> None:
        manager = object.__new__(VoiceManager)
        manager.settings = SimpleNamespace(
            voice_channel_is_allowed=lambda _channel_id, _channel_name: True
        )
        manager.sessions = {7: ExistingSession()}

        async def warm() -> None:
            return None

        manager.warm = warm  # type: ignore[method-assign]
        guild = SimpleNamespace(id=7, voice_client=OldVoiceClient())
        member = SimpleNamespace(
            guild=guild,
            voice=SimpleNamespace(channel=TargetChannel()),
        )
        try:
            await manager.join(member, SimpleNamespace())
        except RuntimeError as exc:
            assert str(exc) == "stop after stale-session assertions"

    asyncio.run(exercise())

    assert state == {"closed": True, "disconnected": True, "connect_attempted": True}


def test_wake_word_extracts_only_the_request() -> None:
    assert extract_wake_request("Hey Jangle, what time is it?", ("jangle",)) == "what time is it"
    assert extract_wake_request("Jengel tell me a joke", ("jangle", "jengel")) == "tell me a joke"
    assert (
        extract_wake_request("Hey, I'm mad at you, Jangle.", ("jangle",))
        == "I'm mad at you"
    )
    assert (
        extract_wake_request("Can you, Jangle, explain that?", ("jangle",))
        == "Can you explain that"
    )
    assert extract_wake_request("No! Hey Jangle, stop!", ("jangle",)) == "stop"
    assert extract_wake_request("Hey Django, are you there?", ("jangle",)) == "are you there"
    assert extract_wake_request("Hey Jungle, listen up", ("jangle",)) == "listen up"
    assert extract_wake_request("Hey Jangle", ("jangle",)) == ""
    assert extract_wake_request("Hey Angela, listen up", ("jangle",)) is None
    assert extract_wake_request("Django is a web framework", ("jangle",)) is None
    assert extract_wake_request("This is ordinary conversation", ("jangle",)) is None


def test_impossible_repeated_wake_transcripts_are_rejected_by_audio_duration() -> None:
    repeated = ", ".join(["Hey Jengel"] * 45)

    assert is_impossible_repeated_wake_transcript(repeated, ("jangle", "jengel"), 0.8)
    assert not is_impossible_repeated_wake_transcript(
        "Hey Jengel, Hey Jengel, Hey Jengel",
        ("jangle", "jengel"),
        2.0,
    )
    assert not is_impossible_repeated_wake_transcript(
        "Hey Jangle, can you hear me",
        ("jangle",),
        1.0,
    )


def test_laughter_only_transcripts_are_nonverbal_interruptions() -> None:
    assert is_nonverbal_interruption("Hahahaha.") is True
    assert is_nonverbal_interruption("Heh heh.") is True
    assert is_nonverbal_interruption("[laughter]") is True
    assert is_nonverbal_interruption("No, stop for a second") is False


def test_spoken_answer_omits_single_and_multi_source_citations() -> None:
    assert _speech_text("Answer [1] with more facts [2, 4].", 200) == "Answer with more facts."


def test_barge_in_stops_playback_and_clears_queued_speech() -> None:
    class FakeVoiceClient:
        def __init__(self) -> None:
            self.playing = True
            self.stop_count = 0

        def is_playing(self) -> bool:
            return self.playing

        def stop_playing(self) -> None:
            self.playing = False
            self.stop_count += 1

    interrupted: list[tuple[str, str]] = []

    class FakeSessions:
        def mark_assistant_interrupted(self, key: str, response: str) -> bool:
            interrupted.append((key, response))
            return True

    class FakeAnswers:
        sessions = FakeSessions()

        def record_event(self, _event: str, **_fields: object) -> None:
            return None

    async def exercise() -> tuple[int, int, bool]:
        session = object.__new__(VoiceSession)
        session.voice_client = FakeVoiceClient()
        session.answer_service = FakeAnswers()
        session.playback_queue = asyncio.Queue()
        session._barge_in_pending = True
        session._current_spoken_item = SpokenItem(
            "playing answer", "voice:1", "playing answer"
        )
        await session.playback_queue.put(
            SpokenItem("queued answer", "voice:2", "queued answer")
        )

        session._interrupt_playback()

        return (
            session.voice_client.stop_count,
            session.playback_queue.qsize(),
            session._barge_in_pending,
        )

    assert asyncio.run(exercise()) == (1, 0, False)
    assert interrupted == [
        ("voice:1", "playing answer"),
        ("voice:2", "queued answer"),
    ]


def test_only_answer_owner_is_marked_as_a_possible_voice_barge_in() -> None:
    scheduled: list[tuple[object, tuple[object, ...]]] = []
    pushed: list[dict[str, object]] = []

    class FakeLoop:
        def call_soon_threadsafe(self, callback: object, *args: object) -> None:
            scheduled.append((callback, args))

    class FakeSegmenter:
        def push(self, *_args: object, **kwargs: object) -> None:
            pushed.append(kwargs)

    session = object.__new__(VoiceSession)
    session._closed = False
    session.decoded_audio_received = True
    session.settings = SimpleNamespace(voice_rms_threshold=300, voice_barge_in_frames=2)
    session.voice_client = SimpleNamespace(is_playing=lambda: True)
    session._current_spoken_item = SpokenItem("answer", user_id=42)
    session._barge_frames = {}
    session._barge_in_pending = False
    session._loop = FakeLoop()
    session.segmenter = FakeSegmenter()
    packet = SimpleNamespace(pcm=_voiced_packet())

    foreign_user = SimpleNamespace(id=99, display_name="Other", bot=False)
    session._on_audio(foreign_user, packet)
    session._on_audio(foreign_user, packet)

    assert scheduled == []
    assert pushed[-1]["foreign_playback"] is True
    assert pushed[-1]["owner_barge_in"] is False

    owner = SimpleNamespace(id=42, display_name="Owner", bot=False)
    session._on_audio(owner, packet)
    session._on_audio(owner, packet)

    assert scheduled == []
    assert pushed[-1]["owner_barge_in"] is True
    assert pushed[-1]["foreign_playback"] is False


def test_noninterruptible_game_speech_blocks_every_speaker() -> None:
    pushed: list[dict[str, object]] = []

    class FakeSegmenter:
        def push(self, *_args: object, **kwargs: object) -> None:
            pushed.append(kwargs)

    session = object.__new__(VoiceSession)
    session._closed = False
    session.decoded_audio_received = True
    session.settings = SimpleNamespace(voice_rms_threshold=300, voice_barge_in_frames=2)
    session.voice_client = SimpleNamespace(is_playing=lambda: True)
    session._current_spoken_item = SpokenItem(
        "protected game question",
        user_id=42,
        interruptible=False,
    )
    session._barge_frames = {}
    session._barge_in_pending = False
    session.segmenter = FakeSegmenter()
    packet = SimpleNamespace(pcm=_voiced_packet())

    session._on_audio(SimpleNamespace(id=42, display_name="Host", bot=False), packet)
    session._on_audio(SimpleNamespace(id=99, display_name="Guest", bot=False), packet)

    assert len(pushed) == 2
    assert all(item["blocked_playback"] is True for item in pushed)
    assert all(item["protected_game_answer"] is False for item in pushed)
    assert all(item["owner_barge_in"] is False for item in pushed)
    assert all(item["foreign_playback"] is False for item in pushed)


def test_trivia_answer_during_question_does_not_interrupt_playback() -> None:
    pushed: list[dict[str, object]] = []

    class FakeSegmenter:
        def push(self, *_args: object, **kwargs: object) -> None:
            pushed.append(kwargs)

    session = object.__new__(VoiceSession)
    session._closed = False
    session.decoded_audio_received = True
    session.settings = SimpleNamespace(voice_rms_threshold=300, voice_barge_in_frames=2)
    session.voice_client = SimpleNamespace(is_playing=lambda: True)
    session._current_spoken_item = SpokenItem(
        "protected trivia question",
        user_id=1,
        interruptible=False,
        game_answers_allowed=True,
        game_window_token=7,
    )
    session._barge_frames = {}
    session._barge_in_pending = False
    session.segmenter = FakeSegmenter()

    session._on_audio(
        SimpleNamespace(id=2, display_name="Guest", bot=False),
        SimpleNamespace(pcm=_voiced_packet()),
    )
    session._on_audio(
        SimpleNamespace(id=2, display_name="Guest", bot=False),
        SimpleNamespace(pcm=_voiced_packet(0)),
    )

    assert pushed[0] == {
        "owner_barge_in": False,
        "foreign_playback": False,
        "blocked_playback": True,
        "protected_game_answer": True,
        "game_window_token": 7,
    }
    assert pushed[1]["blocked_playback"] is False
    assert pushed[1]["protected_game_answer"] is False
    assert pushed[1]["game_window_token"] == 7


def test_blocked_game_speech_skips_whisper_transcription() -> None:
    session, _messages, events = _social_test_session()
    session._ignored_user_ids = set()

    class FailingStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            raise AssertionError("protected game speech must not reach Whisper")

    session.stt = FailingStt()

    asyncio.run(
        session._handle_segment(
            PcmSegment(
                2,
                "Guest",
                b"pcm",
                1.0,
                blocked_playback=True,
            )
        )
    )

    assert events == [
        (
            "voice_noninterruptible_speech_ignored",
            {
                "guild_id": 7,
                "channel_id": 8,
                "user_id": 2,
                "user_name": "Guest",
                "audio_duration_ms": 1000,
            },
        )
    ]


def test_ignored_speaker_audio_is_dropped_before_segmentation() -> None:
    pushed: list[int] = []

    class FakeSegmenter:
        def push(self, user_id: int, *_args: object, **_kwargs: object) -> None:
            pushed.append(user_id)

    session = object.__new__(VoiceSession)
    session._closed = False
    session._ignored_user_ids = {99}
    session.decoded_audio_received = True
    session.segmenter = FakeSegmenter()
    packet = SimpleNamespace(pcm=_voiced_packet())

    session._on_audio(SimpleNamespace(id=99, display_name="Ignored", bot=False), packet)

    assert pushed == []


def test_dave_audio_is_decrypted_before_opus_decode() -> None:
    calls: list[tuple[int, object, bytes]] = []

    class FakeDaveSession:
        ready = True

        def decrypt(self, user_id: int, media_type: object, payload: bytes) -> bytes:
            calls.append((user_id, media_type, payload))
            return b"plain opus"

    voice_client = SimpleNamespace(
        _connection=SimpleNamespace(
            dave_protocol_version=1,
            dave_session=FakeDaveSession(),
        ),
        _get_id_from_ssrc=lambda _ssrc: 42,
    )
    router = SimpleNamespace(sink=SimpleNamespace(voice_client=voice_client))
    packet = SimpleNamespace(ssrc=1234, decrypted_data=b"encrypted opus")

    _decrypt_dave_rtp(router, packet)

    assert packet.decrypted_data == b"plain opus"
    assert calls[0][0] == 42
    assert calls[0][1] == davey.MediaType.audio
    assert calls[0][2] == b"encrypted opus"


def test_unknown_dave_speaker_frame_is_dropped_without_crashing() -> None:
    session = SimpleNamespace(ready=True)
    voice_client = SimpleNamespace(
        _connection=SimpleNamespace(dave_protocol_version=1, dave_session=session),
        _get_id_from_ssrc=lambda _ssrc: None,
    )
    router = SimpleNamespace(sink=SimpleNamespace(voice_client=voice_client))
    packet = SimpleNamespace(ssrc=1234, decrypted_data=b"encrypted opus")
    install_dave_voice_receive_patch()

    voice_recv_router.PacketRouter.feed_rtp(router, packet)


def test_voice_control_marker_is_silent_and_opens_a_reply_turn() -> None:
    answer, expects_reply = parse_voice_answer("Knock knock. [[AWAIT_REPLY]]")

    assert answer == "Knock knock."
    assert expects_reply is True


def test_question_mark_without_control_marker_does_not_open_a_reply_turn() -> None:
    answer, expects_reply = parse_voice_answer("Would you like anything else?")

    assert answer == "Would you like anything else?"
    assert expects_reply is False


def test_followup_window_is_limited_to_the_same_speaker() -> None:
    session = object.__new__(VoiceSession)
    state = FollowupState(time.monotonic() + 10, "clarification", 1)
    session._followups = {42: state}

    assert session._consume_followup(99) is None
    assert session._consume_followup(42) == state
    assert session._consume_followup(42) is None


def test_wake_only_call_uses_short_acknowledgement_and_opens_one_reply() -> None:
    events: list[str] = []

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "Hey Jangle"

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

        async def answer(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("A wake-only call must not invoke the model")

    async def exercise() -> tuple[SpokenItem, FollowupState | None]:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            voice_wake_words=("jangle",),
            voice_followup_seconds=25,
        )
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8),
            is_playing=lambda: False,
        )
        session.text_echo = False
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._ignored_user_ids = set()
        session._followups = {}
        session._voice_choice_deadlines = {}
        session._personality_choice_deadlines = {}
        session._music_query_deadlines = {}
        session._playlist_query_deadlines = {}
        session._dj_mode = False
        session._game_state = None
        session._poll_state = None
        session._story_state = None
        session._award_state = None

        await session._handle_segment(PcmSegment(42, "Speaker", b"pcm", 0.8))
        return session.playback_queue.get_nowait(), session._followups.get(42)

    spoken, followup = asyncio.run(exercise())

    assert spoken.text == "Yeah?"
    assert followup is not None and followup.remaining_turns == 1
    assert events == ["voice_wake_acknowledged"]


def test_complete_answer_cannot_open_a_turn_just_because_model_added_marker() -> None:
    assert (
        select_followup_mode(
            "tell me a joke",
            "Why did the scarecrow win? Because he was outstanding in his field.",
            True,
        )
        is None
    )


def test_interactive_turn_is_bounded_to_two_wake_free_replies() -> None:
    session = object.__new__(VoiceSession)
    session.settings = SimpleNamespace(voice_followup_seconds=25)

    first = session._next_followup(
        "tell me a knock-knock joke",
        "Knock knock.",
        True,
        None,
    )
    assert first is not None and first.remaining_turns == 2
    second = session._next_followup("who's there", "Orange.", True, first)
    assert second is not None and second.remaining_turns == 1
    assert session._next_followup("orange who", "Orange you glad!", True, second) is None


def test_lookup_acknowledgement_plays_before_the_answer() -> None:
    captured: dict[str, object] = {}

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "Jangle, what is the weather today?"

    class FakeAnswers:
        sessions = SimpleNamespace()

        def will_search(self, _prompt: str, **_kwargs: object) -> bool:
            return True

        async def answer(self, _key: str, prompt: str, **kwargs: object) -> str:
            captured["prompt"] = prompt
            captured.update(kwargs)
            return "It is sunny and 72 degrees."

        def record_event(self, _event: str, **_fields: object) -> None:
            return None

    async def exercise() -> tuple[str, str]:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            voice_wake_words=("jangle",),
            voice_followup_seconds=25,
        )
        session.text_echo = False
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._followups = {}
        session._recent_exchange = None
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            is_playing=lambda: False,
            channel=SimpleNamespace(
                id=8,
                members=[SimpleNamespace(display_name="Speaker", bot=False)],
            ),
        )

        await session._handle_segment(PcmSegment(42, "Speaker", b"pcm", 1.25))
        return (
            session.playback_queue.get_nowait().text,
            session.playback_queue.get_nowait().text,
        )

    acknowledgement, answer = asyncio.run(exercise())

    assert acknowledgement == "Let me look that up."
    assert answer == "Speaker, It is sunny and 72 degrees."
    assert captured["prompt"] == "what is the weather today"
    assert captured["voice"] is True
    assert captured["log_context"]["user_id"] == 42  # type: ignore[index]
    assert captured["log_context"]["turn_mode"] == "wake_word"  # type: ignore[index]
    assert "Current speaker display name: Speaker" in str(captured["runtime_context"])
    assert captured["personality_prompt"] == ""


def test_energy_easter_egg_bypasses_model_with_exact_response() -> None:
    events: list[str] = []

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "Hey Jangle, energeee"

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

        def will_search(self, *_args: object, **_kwargs: object) -> bool:
            raise AssertionError("The Easter egg must bypass search routing")

        async def answer(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("The Easter egg must bypass the model")

    async def exercise() -> SpokenItem:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(voice_wake_words=("jangle",))
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8),
            is_playing=lambda: False,
        )
        session.text_echo = False
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._ignored_user_ids = set()
        session._followups = {}
        session._voice_choice_deadlines = {}
        session._personality_choice_deadlines = {}
        session._music_query_deadlines = {}
        session._playlist_query_deadlines = {}
        session._dj_mode = False

        await session._handle_segment(PcmSegment(42, "Speaker", b"pcm", 1.0))
        return session.playback_queue.get_nowait()

    response = asyncio.run(exercise())

    assert is_energy_easter_egg("energeee") is True
    assert is_energy_easter_egg("Energy.") is True
    assert is_energy_easter_egg("energy levels") is False
    assert response.text == ENERGY_EASTER_EGG_RESPONSE
    assert events == ["voice_energy_easter_egg"]


def test_runtime_context_carries_recent_cross_speaker_exchange() -> None:
    session = object.__new__(VoiceSession)
    session.user_notes = None
    session.voice_client = SimpleNamespace(
        guild=SimpleNamespace(id=7),
        channel=SimpleNamespace(
            members=[
                SimpleNamespace(display_name="First", bot=False),
                SimpleNamespace(display_name="Second", bot=False),
            ]
        )
    )
    session._recent_exchange = RecentVoiceExchange(
        session_key="voice:1",
        user_id=1,
        user_name="First",
        prompt="give me a five-day forecast",
        answer="Today will be sunny.",
        created_at=time.monotonic(),
    )

    context = session._runtime_context(PcmSegment(2, "Second", b"pcm", 1.0))

    assert "Current speaker display name: Second" in context
    assert "Request: give me a five-day forecast" in context
    assert "Jangle response (delivered): Today will be sunny." in context


def test_voice_note_command_updates_only_the_current_users_notepad(tmp_path: Path) -> None:
    messages: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []

    class FakeChannel:
        id = 8

        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    class FakeAnswers:
        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

    async def exercise() -> bool:
        session = object.__new__(VoiceSession)
        session.user_notes = UserNoteStore(tmp_path / "notes.json")
        session.answer_service = FakeAnswers()
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8),
        )
        session.companion_channel = FakeChannel()
        session.tts = None
        session._dj_mode = False

        handled = await session._handle_control_request(
            "remember that I main a frost mage",
            PcmSegment(42, "Mage", b"pcm", 1.0),
            pending_control=None,
            transcript="Hey Jangle, remember that I main a frost mage",
            transcription_ms=45,
        )
        assert session.user_notes.get(7, 42) == ("I main a frost mage",)
        assert session.user_notes.get(7, 99) == ()
        return handled

    assert asyncio.run(exercise()) is True
    assert messages == ["**Jangle:** Saved. Your Jangle notepad now has 1 of 5 notes."]
    assert events[0][0] == "user_note_command"
    assert "note" not in events[0][1]
    assert "transcript" not in events[0][1]


def test_foreign_normal_speech_during_playback_is_ignored_after_transcription() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "ordinary conversation without the call word"

    class FakeAnswers:
        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

        async def answer(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("Foreign normal speech must not reach the model")

    async def exercise() -> None:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(voice_wake_words=("jangle",))
        session._followups = {}
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8),
        )
        await session._handle_segment(
            PcmSegment(99, "Other", b"pcm", 1.0, foreign_playback=True)
        )

    asyncio.run(exercise())

    assert events[0][0] == "voice_foreign_speech_ignored"
    assert events[0][1]["user_id"] == 99


def test_owner_laughter_during_answer_does_not_interrupt_or_reach_model() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "Heh heh."

    class FakeAnswers:
        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

        async def answer(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("Laughter must not reach the model")

    class FakeVoiceClient:
        def __init__(self) -> None:
            self.stop_count = 0
            self.guild = SimpleNamespace(id=7)
            self.channel = SimpleNamespace(id=8)

        def is_playing(self) -> bool:
            return True

        def stop_playing(self) -> None:
            self.stop_count += 1

    async def exercise() -> FakeVoiceClient:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(voice_wake_words=("jangle",))
        session._followups = {}
        session.voice_client = FakeVoiceClient()

        await session._handle_segment(
            PcmSegment(42, "Laughing User", b"pcm", 1.2, owner_barge_in=True)
        )
        return session.voice_client

    voice_client = asyncio.run(exercise())

    assert voice_client.stop_count == 0
    assert [event for event, _fields in events] == ["voice_nonverbal_interruption_ignored"]
    assert "transcript" not in events[0][1]


def test_foreign_wake_word_can_take_over_current_answer() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    captured: dict[str, object] = {}

    class FakeVoiceClient:
        def __init__(self) -> None:
            self.playing = True
            self.stop_count = 0
            self.guild = SimpleNamespace(id=7)
            self.channel = SimpleNamespace(
                id=8,
                members=[
                    SimpleNamespace(display_name="Owner", bot=False),
                    SimpleNamespace(display_name="Other", bot=False),
                ],
            )

        def is_playing(self) -> bool:
            return self.playing

        def stop_playing(self) -> None:
            self.playing = False
            self.stop_count += 1

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "Hey, I'm mad at you, Jangle"

    class FakeSessions:
        def mark_assistant_interrupted(self, _key: str, _response: str) -> bool:
            return True

    class FakeAnswers:
        sessions = FakeSessions()

        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

        def will_search(self, _prompt: str, **_kwargs: object) -> bool:
            return False

        async def answer(self, _key: str, prompt: str, **kwargs: object) -> str:
            captured.update({"prompt": prompt, **kwargs})
            return "I switched to you."

    async def exercise() -> tuple[FakeVoiceClient, SpokenItem]:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            voice_wake_words=("jangle",),
            voice_followup_seconds=25,
        )
        session.text_echo = False
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._followups = {}
        session._barge_in_pending = False
        session._recent_exchange = None
        session._current_spoken_item = SpokenItem(
            "Owner answer",
            session_key="voice:owner",
            history_response="Owner answer",
            user_id=42,
        )
        session.voice_client = FakeVoiceClient()

        await session._handle_segment(
            PcmSegment(99, "Other", b"pcm", 1.0, foreign_playback=True)
        )
        return session.voice_client, session.playback_queue.get_nowait()

    voice_client, queued = asyncio.run(exercise())

    assert voice_client.stop_count == 1
    assert captured["prompt"] == "I'm mad at you"
    assert queued.user_id == 99
    assert queued.text == "Other, I switched to you."
    interruption = next(fields for event, fields in events if event == "voice_response_interrupted")
    assert interruption["response_owner_user_id"] == 42
    assert interruption["interrupter_user_id"] == 99
    assert interruption["interruption_reason"] == "wake_word_takeover"


def test_temporary_speech_file_is_deleted_after_ffmpeg_cleanup(
    monkeypatch: object, tmp_path: Path
) -> None:
    path = tmp_path / "speech.mp3"
    path.write_bytes(b"audio")
    events: list[str] = []

    def fake_cleanup(_source: object) -> None:
        events.append("ffmpeg released")

    monkeypatch.setattr(discord.FFmpegPCMAudio, "cleanup", fake_cleanup)  # type: ignore[attr-defined]
    source = object.__new__(TemporaryFileFFmpegPCMAudio)
    source.temporary_path = path

    source.cleanup()

    assert events == ["ffmpeg released"]
    assert not path.exists()


def test_voice_and_music_commands_are_parsed_without_hijacking_other_requests() -> None:
    bare = parse_voice_change_command("change voice")
    selected = parse_voice_change_command("change your voice to Brian")
    bare_personality = parse_personality_change_command("change personality")
    selected_personality = parse_personality_change_command(
        "switch your personality to Savage"
    )
    enabled_mode = parse_personality_change_command("enable Madam mode")
    brutal_mode = parse_personality_change_command("enable Brutal mode")
    guided_mode = parse_personality_change_command("enable mode")
    disabled_mode = parse_personality_change_command("disable mode")
    named_disabled_mode = parse_personality_change_command("turn Madam mode off")
    conversational_mode = parse_personality_change_command(
        "enable or change personality to Savage"
    )

    assert bare is not None and bare.choice_text is None
    assert selected is not None and selected.choice_text == "Brian"
    assert bare_personality is not None and bare_personality.choice_text is None
    assert selected_personality is not None
    assert selected_personality.choice_text == "Savage"
    assert enabled_mode is not None and enabled_mode.choice_text == "Madam"
    assert brutal_mode is not None and brutal_mode.choice_text == "Brutal"
    assert guided_mode is not None and guided_mode.choice_text is None
    assert disabled_mode is not None and disabled_mode.choice_text == "disabled"
    assert named_disabled_mode is not None
    assert named_disabled_mode.choice_text == "disabled"
    assert conversational_mode is not None
    assert conversational_mode.choice_text == "Savage"
    assert parse_voice_change_command("use the internet to answer") is None
    assert parse_personality_change_command("tell me about personality types") is None
    assert find_voice_choice("Brian voice").edge_voice == "en-US-BrianNeural"  # type: ignore[union-attr]
    assert find_voice_choice("Kokoro Heart") is None
    assert find_voice_choice("Emma").edge_voice == "en-US-EmmaNeural"  # type: ignore[union-attr]
    assert find_voice_choice("Pocket Alba").edge_voice == "pocket:alba"  # type: ignore[union-attr]
    assert find_voice_choice("Caro Davy").provider == "pocket"  # type: ignore[union-attr]
    choices = voice_choices_text()
    assert "Brian" in choices
    assert "Pocket Alba" in choices
    assert "Kokoro" not in choices
    assert find_personality_choice("unforgiving savage mode").key == "savage"  # type: ignore[union-attr]
    assert find_personality_choice("normal mode").key == "disabled"  # type: ignore[union-attr]
    assert find_personality_choice("Madame personality").key == "madam"  # type: ignore[union-attr]
    assert find_personality_choice("max troll mode").key == "brutal"  # type: ignore[union-attr]
    assert find_personality_choice("raid leader") is None
    assert find_personality_choice("bard personality") is None
    assert parse_music_query("play me the song Africa by Toto on YouTube") == "Africa by Toto"
    assert parse_music_query("could you play some low fire") == "lo-fi"
    assert parse_music_query("play a song") == ""
    assert parse_music_query("tell me about Africa by Toto") is None
    assert parse_add_to_queue_query("add Africa by Toto to the queue") == "Africa by Toto"
    assert parse_add_to_queue_query("could you put Africa by Toto in the queue please") == "Africa by Toto"
    assert parse_add_to_queue_query("add Africa Bikoto to Q") == "Africa Bikoto"
    assert parse_add_to_queue_query("ad cat girl ASMR to queue") == "cat girl ASMR"
    assert parse_add_to_queue_query("admin clear Q") is None
    assert parse_add_to_queue_query("clear cue") is None
    assert parse_add_music_shorthand("add Mac Morrison") == "Mac Morrison"
    assert parse_playlist_command("find me a playlist called 80s hits").query == "80s hits"  # type: ignore[union-attr]
    assert parse_playlist_command("add playlist jazz to queue").query == "jazz"  # type: ignore[union-attr]
    assert parse_playlist_command("play a low-five playlist").query == "lo-fi"  # type: ignore[union-attr]
    assert parse_playlist_command("Play, Deathcore playlist").query == "Deathcore"  # type: ignore[union-attr]
    assert parse_playlist_command("fine playlist 80s hits").query == "80s hits"  # type: ignore[union-attr]
    assert parse_playlist_command("could you play me a low five music playlist please").query == "lo-fi music"  # type: ignore[union-attr]
    assert parse_music_navigation("skip song") == "next"
    assert parse_music_navigation("previous") == "previous"
    assert parse_dj_mode_command("enable D.J. mode") is True
    assert parse_dj_mode_command("DJ mode off") is False
    assert parse_music_pause_command("can you pause the music") == "pause"
    assert parse_music_pause_command("continue music") == "resume"
    assert parse_music_volume_level("set volume 50") == 50
    assert parse_music_volume_level("set the music volume to seventy five percent") == 75
    assert parse_music_volume_level("volume 100%") == 100
    assert parse_music_volume_level("set volume 125") == 125
    assert parse_music_volume_level("volume up") is None
    assert parse_music_volume_command("volume up please") == 1
    assert parse_music_volume_command("turn it down") == -1
    assert is_admin_stop_command("Admin, stop the music") is True
    assert is_admin_stop_command("stop") is True
    assert is_admin_stop_command("stop the music") is True
    assert is_admin_stop_command("stop asking questions") is False
    assert is_admin_clear_queue_command("Admin, clear the queue") is True
    assert is_admin_clear_queue_command("admin clear Q") is True
    assert is_admin_clear_queue_command("administrator clear cue") is True
    assert is_admin_clear_queue_command("clear queue") is True
    assert is_admin_clear_queue_command("clear the queue") is True
    assert is_admin_clear_queue_command("clear Q") is True
    assert is_admin_clear_queue_command("admin clear queue and stop") is False
    assert is_show_queue_command("show queue") is True
    assert is_show_queue_command("show cue") is True
    assert is_show_queue_command("could you show me the Q") is True
    assert is_show_queue_command("what's in the queue?") is True


def test_owner_listening_commands_and_member_names_are_resolved_conservatively() -> None:
    stop = parse_speaker_listening_command("stop listening to James Royce Ryan")
    start = parse_speaker_listening_command("start listening to James Royce Ryan again")
    ignored = parse_speaker_listening_command("ignore obj")
    unignored = parse_speaker_listening_command("unignore obj")
    listen_all = parse_speaker_listening_command("listen to all")
    listen_everyone = parse_speaker_listening_command(
        "start listening to everyone again"
    )

    assert stop is not None and stop.listening_enabled is False
    assert stop.target_text == "James Royce Ryan"
    assert start is not None and start.listening_enabled is True
    assert start.target_text == "James Royce Ryan"
    assert ignored is not None and ignored.listening_enabled is False
    assert unignored is not None and unignored.listening_enabled is True
    assert listen_all is not None and listen_all.listening_enabled is True
    assert listen_all.target_text == "all"
    assert listen_everyone is not None and listen_everyone.target_text == "everyone"
    assert parse_speaker_listening_command("stop talking about James") is None

    james = SimpleNamespace(
        id=10,
        display_name="James Royce Ryan",
        global_name="James Ryan",
        name="jrr",
        bot=False,
    )
    warlune = SimpleNamespace(
        id=11,
        display_name="Warlune",
        global_name=None,
        name="warlune_owner",
        bot=False,
    )
    bot = SimpleNamespace(id=12, display_name="Jangle", name="jangle", bot=True)

    assert resolve_voice_members("James", [james, warlune, bot]) == [james]
    assert resolve_voice_members("war loon", [james, warlune, bot]) == [warlune]
    assert resolve_voice_members("missing person", [james, warlune, bot]) == []


def test_ignored_speaker_store_persists_only_discord_ids(tmp_path: Path) -> None:
    path = tmp_path / "ignored-speakers.json"
    store = IgnoredSpeakerStore(path)

    store.set_ignored(7, 99, True)

    assert IgnoredSpeakerStore(path).get(7) == frozenset({99})
    store.set_ignored(7, 100, True)
    assert store.clear_guild(7) == 2
    assert store.get(7) == frozenset()
    store.set_ignored(7, 99, True)
    store.set_ignored(7, 99, False)
    assert IgnoredSpeakerStore(path).get(7) == frozenset()


def test_warlune_owner_can_stop_and_resume_listening_to_voice_member(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    discarded: list[int] = []
    reset_keys: list[str] = []

    class FakeSessions:
        def reset(self, key: str) -> bool:
            reset_keys.append(key)
            return True

    class FakeAnswers:
        sessions = FakeSessions()

        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

    class FakeSegmenter:
        def discard_user(self, user_id: int) -> None:
            discarded.append(user_id)

    async def exercise() -> tuple[list[SpokenItem], IgnoredSpeakerStore]:
        owner = SimpleNamespace(id=42, display_name="Warlune", name="warlune", bot=False)
        target = SimpleNamespace(
            id=99,
            display_name="James Royce Ryan",
            name="james_ryan",
            bot=False,
        )
        store = IgnoredSpeakerStore(tmp_path / "ignored.json")
        session = object.__new__(VoiceSession)
        session.settings = SimpleNamespace(owner_user_ids=frozenset({42}))
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8, members=[owner, target]),
        )
        session.ignored_speakers = store
        session._ignored_user_ids = set()
        session.segmenter = FakeSegmenter()
        session.answer_service = FakeAnswers()
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._barge_frames = {}
        session._followups = {}
        session._voice_choice_deadlines = {}
        session._personality_choice_deadlines = {}
        session._music_query_deadlines = {}
        session._music_query_queue_only = {}
        session._playlist_query_deadlines = {}
        session._current_spoken_item = None
        session._recent_exchange = None
        segment = PcmSegment(42, "Warlune", b"pcm", 1.0)

        await session._handle_speaker_listening(
            parse_speaker_listening_command("stop listening to James"),  # type: ignore[arg-type]
            segment,
            "Jangle, stop listening to James",
            70,
        )
        stopped = session.playback_queue.get_nowait()
        assert 99 in session._ignored_user_ids

        await session._handle_speaker_listening(
            parse_speaker_listening_command("start listening to James again"),  # type: ignore[arg-type]
            segment,
            "Jangle, start listening to James again",
            65,
        )
        resumed = session.playback_queue.get_nowait()
        assert 99 not in session._ignored_user_ids
        return [stopped, resumed], store

    confirmations, store = asyncio.run(exercise())

    assert confirmations[0].text.startswith("Okay. I will stop listening to James Royce Ryan")
    assert confirmations[1].text == "I am listening to James Royce Ryan again."
    assert store.get(7) == frozenset()
    assert discarded == [99]
    assert reset_keys == ["voice:7:8:99"]
    assert [fields["listening_enabled"] for _event, fields in events] == [False, True]


def test_non_owner_cannot_change_speaker_listening_state(tmp_path: Path) -> None:
    async def exercise() -> tuple[str, frozenset[int]]:
        guest = SimpleNamespace(id=10, display_name="Guest", name="guest", bot=False)
        target = SimpleNamespace(id=99, display_name="Target", name="target", bot=False)
        store = IgnoredSpeakerStore(tmp_path / "ignored.json")
        session = object.__new__(VoiceSession)
        session.settings = SimpleNamespace(owner_user_ids=frozenset({42}))
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8, members=[guest, target]),
        )
        session.ignored_speakers = store
        session.answer_service = SimpleNamespace()
        session.tts = object()
        session.playback_queue = asyncio.Queue()

        await session._handle_speaker_listening(
            parse_speaker_listening_command("stop listening to Target"),  # type: ignore[arg-type]
            PcmSegment(10, "Guest", b"pcm", 1.0),
            "Jangle, stop listening to Target",
            50,
        )
        return session.playback_queue.get_nowait().text, store.get(7)

    confirmation, ignored = asyncio.run(exercise())

    assert confirmation == "That command is reserved for the configured bot owner."
    assert ignored == frozenset()


def test_warlune_owner_can_resume_listening_to_everyone_without_name_lookup(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class FakeAnswers:
        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

    async def exercise() -> tuple[str, IgnoredSpeakerStore, set[int]]:
        store = IgnoredSpeakerStore(tmp_path / "ignored.json")
        store.set_ignored(7, 99, True)
        store.set_ignored(7, 100, True)
        session = object.__new__(VoiceSession)
        session.settings = SimpleNamespace(owner_user_ids=frozenset({42}))
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8, members=[]),
        )
        session.ignored_speakers = store
        session._ignored_user_ids = {99, 100}
        session.answer_service = FakeAnswers()
        session.tts = object()
        session.playback_queue = asyncio.Queue()

        await session._handle_speaker_listening(
            parse_speaker_listening_command("listen to all"),  # type: ignore[arg-type]
            PcmSegment(42, "Warlune", b"pcm", 1.0),
            "Jangle, listen to all",
            45,
        )
        return (
            session.playback_queue.get_nowait().text,
            store,
            session._ignored_user_ids,
        )

    confirmation, store, ignored = asyncio.run(exercise())

    assert confirmation == "I am listening to everyone again."
    assert ignored == set()
    assert store.get(7) == frozenset()
    assert events[0][0] == "voice_speaker_listening_reset"
    assert events[0][1]["cleared_speaker_count"] == 2


def test_voice_preference_store_persists_only_supported_choices(tmp_path: Path) -> None:
    path = tmp_path / "voice-preferences.json"
    store = VoicePreferenceStore(path)

    store.set(7, "en-US-BrianNeural")
    store.set(8, "pocket:alba")

    assert VoicePreferenceStore(path).get(7, "en-US-AriaNeural") == "en-US-BrianNeural"
    assert VoicePreferenceStore(path).get(8, "en-US-AriaNeural") == "pocket:alba"


def test_personality_preference_store_persists_plugin_owned_choice(
    tmp_path: Path,
) -> None:
    path = tmp_path / "personality-preferences.json"
    store = PersonalityPreferenceStore(path)

    store.set(7, "savage")
    store.remember_voice_before_madam(7, "en-US-BrianNeural")

    assert PersonalityPreferenceStore(path).get(7) == "savage"
    assert PersonalityPreferenceStore(path).get(8) == "disabled"
    assert (
        PersonalityPreferenceStore(path).restore_voice_after_madam(
            7,
            "en-US-AriaNeural",
        )
        == "en-US-BrianNeural"
    )


def test_personality_mode_prompts_keep_themes_scoped() -> None:
    disabled = PERSONALITY_CHOICES["disabled"].system_prompt
    savage = PERSONALITY_CHOICES["savage"].system_prompt
    madam = PERSONALITY_CHOICES["madam"].system_prompt
    brutal = PERSONALITY_CHOICES["brutal"].system_prompt

    assert disabled == ""
    assert "WoW" in savage
    assert "WoW" not in madam
    assert "WoW" not in brutal
    assert all(
        trait in madam
        for trait in (
            "authoritative yet nurturing",
            "cool-headed and diplomatic",
            "business acumen",
            "discerning",
        )
    )
    assert "Curse freely" in brutal
    assert "dark adult humor" in brutal
    assert "internet memes" in brutal


def test_admin_voice_command_changes_personality_and_logs_it(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    cleared_prefixes: list[str] = []

    class FakeSessions:
        def reset_prefix(self, prefix: str) -> int:
            cleared_prefixes.append(prefix)
            return 1

    class FakeAnswers:
        sessions = FakeSessions()

        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

    async def exercise() -> tuple[VoiceSession, PersonalityPreferenceStore, SpokenItem]:
        session = object.__new__(VoiceSession)
        store = PersonalityPreferenceStore(tmp_path / "personalities.json")
        session.personality_preferences = store
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            admin_role_ids=frozenset(),
            voice_followup_seconds=25,
        )
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7, owner_id=42),
            channel=SimpleNamespace(id=8, members=[]),
        )
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._personality_choice_deadlines = {}

        await session._handle_personality_change(
            "Savage",
            PcmSegment(42, "Guild Master", b"pcm", 1.0),
            "Hey Jangle, change personality to Savage",
            75,
        )
        return session, store, session.playback_queue.get_nowait()

    session, store, confirmation = asyncio.run(exercise())

    assert session._personality_key == "savage"
    assert store.get(7) == "savage"
    assert confirmation.text == "Savage mode enabled. Bad parses are now admissible evidence."
    assert events[0][0] == "voice_personality_changed"
    assert events[0][1]["personality_key"] == "savage"
    assert events[0][1]["cleared_session_count"] == 2
    assert cleared_prefixes == ["voice:7:", "text:7:"]


def test_non_admin_cannot_change_personality(tmp_path: Path) -> None:
    class FakeAnswers:
        def record_event(self, _event: str, **_fields: object) -> None:
            raise AssertionError("A denied personality change must not be logged as successful")

    async def exercise() -> SpokenItem:
        session = object.__new__(VoiceSession)
        session.personality_preferences = PersonalityPreferenceStore(
            tmp_path / "personalities.json"
        )
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            admin_role_ids=frozenset(),
            voice_followup_seconds=25,
        )
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7, owner_id=1),
            channel=SimpleNamespace(id=8, members=[]),
        )
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._personality_choice_deadlines = {}

        await session._handle_personality_change(
            "Savage",
            PcmSegment(99, "Guest", b"pcm", 1.0),
            "Hey Jangle, change personality to Savage",
            75,
        )
        return session.playback_queue.get_nowait()

    confirmation = asyncio.run(exercise())

    assert confirmation.text == "Only a server administrator can change my mode."


def test_madam_mode_switches_to_female_voice_and_restores_previous_voice(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class FakeSessions:
        def reset_prefix(self, _prefix: str) -> int:
            return 0

    class FakeAnswers:
        sessions = FakeSessions()

        def record_event(self, event: str, **fields: object) -> None:
            events.append((event, fields))

    class FakeTts:
        def __init__(self) -> None:
            self.voice = "en-US-BrianNeural"

        def set_voice(self, voice: str) -> None:
            self.voice = voice

    async def exercise() -> tuple[
        VoiceSession,
        VoicePreferenceStore,
        PersonalityPreferenceStore,
        list[SpokenItem],
    ]:
        session = object.__new__(VoiceSession)
        voice_store = VoicePreferenceStore(tmp_path / "voices.json")
        mode_store = PersonalityPreferenceStore(tmp_path / "modes.json")
        voice_store.set(7, "en-US-BrianNeural")
        session.voice_preferences = voice_store
        session.personality_preferences = mode_store
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            admin_role_ids=frozenset(),
            voice_followup_seconds=25,
            tts_voice="en-US-AriaNeural",
        )
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7, owner_id=42),
            channel=SimpleNamespace(id=8, members=[]),
        )
        session.tts = FakeTts()
        session.playback_queue = asyncio.Queue()
        session._personality_key = "disabled"
        session._personality_choice_deadlines = {}
        segment = PcmSegment(42, "Guild Master", b"pcm", 1.0)

        await session._handle_personality_change(
            "Madam",
            segment,
            "Hey Jangle, enable Madam mode",
            70,
        )
        madam_confirmation = session.playback_queue.get_nowait()
        await session._handle_voice_change(
            "Brian",
            segment,
            "Hey Jangle, change voice to Brian",
            60,
        )
        blocked_voice_change = session.playback_queue.get_nowait()
        await session._handle_personality_change(
            "disabled",
            segment,
            "Hey Jangle, disable mode",
            65,
        )
        default_confirmation = session.playback_queue.get_nowait()
        return (
            session,
            voice_store,
            mode_store,
            [madam_confirmation, blocked_voice_change, default_confirmation],
        )

    session, voice_store, mode_store, confirmations = asyncio.run(exercise())

    assert confirmations[0].text == (
        "Madam mode enabled. Mind the house rules, darlings, and we will get along beautifully."
    )
    assert confirmations[1].text == (
        "Madam mode uses the Michelle voice automatically. Disable the mode before "
        "changing voices."
    )
    assert confirmations[2].text == "Personality modes disabled. Back to plain Jangle."
    assert session._personality_key == "disabled"
    assert mode_store.get(7) == "disabled"
    assert session.tts.voice == "en-US-BrianNeural"
    assert voice_store.get(7, "en-US-AriaNeural") == "en-US-BrianNeural"
    changes = [fields for event, fields in events if event == "voice_personality_changed"]
    assert changes[0]["changed_edge_voice"] == MADAM_VOICE_ID
    assert changes[1]["changed_edge_voice"] == "en-US-BrianNeural"


def test_music_admin_accepts_server_permissions_or_allowlisted_roles() -> None:
    guild = SimpleNamespace(owner_id=10)
    ordinary = SimpleNamespace(
        id=20,
        guild_permissions=SimpleNamespace(administrator=False, manage_guild=False),
        roles=[SimpleNamespace(id=30)],
    )
    manager = SimpleNamespace(
        id=21,
        guild_permissions=SimpleNamespace(administrator=False, manage_guild=True),
        roles=[],
    )

    assert member_can_administer_music(ordinary, guild, frozenset({30})) is True
    assert member_can_administer_music(manager, guild, frozenset()) is True
    assert member_can_administer_music(ordinary, guild, frozenset()) is False


def test_barge_in_preserves_current_and_queued_music() -> None:
    class FakeVoiceClient:
        def __init__(self) -> None:
            self.stop_count = 0

        def is_playing(self) -> bool:
            return True

        def stop_playing(self) -> None:
            self.stop_count += 1

    async def exercise() -> tuple[int, list[object]]:
        track = YouTubeTrack("query", "Song", "https://youtube.test/watch", "stream", 180)
        music = MusicItem(track, 42, "Speaker")
        session = object.__new__(VoiceSession)
        session.voice_client = FakeVoiceClient()
        session.playback_queue = asyncio.Queue()
        session.answer_service = SimpleNamespace()
        session._current_music_item = music
        session._current_spoken_item = None
        session._barge_in_pending = True
        await session.playback_queue.put(SpokenItem("queued speech"))
        await session.playback_queue.put(music)

        session._interrupt_playback(42, "owner_voice")

        return session.voice_client.stop_count, list(session.playback_queue._queue)

    stop_count, queued = asyncio.run(exercise())

    assert stop_count == 0
    assert len(queued) == 1 and isinstance(queued[0], MusicItem)


def test_music_commands_require_dj_mode_before_searching_or_queueing() -> None:
    async def exercise() -> tuple[bool, list[str], list[str]]:
        session, _messages, events = _social_test_session()
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._music_query_deadlines = {}
        session._music_query_queue_only = {}
        session._playlist_query_deadlines = {}
        segment = PcmSegment(1, "Host", b"pcm", 1.0)

        handled = await session._handle_control_request(
            "play Africa by Toto",
            segment,
            pending_control=None,
            transcript="Hey Jangle, play Africa by Toto",
            transcription_ms=50,
        )
        spoken = [
            item.text
            for item in session.playback_queue._queue
            if isinstance(item, SpokenItem)
        ]
        return handled, [event for event, _fields in events], spoken

    handled, events, spoken = asyncio.run(exercise())

    assert handled is True
    assert events == ["voice_music_requires_dj_mode"]
    assert spoken == [
        "DJ mode is off. Say Hey Jangle, enable DJ mode before using music commands."
    ]


def test_disabling_dj_mode_stops_music_and_clears_the_queue() -> None:
    async def exercise() -> tuple[VoiceSession, int, list[tuple[str, dict[str, object]]]]:
        session, _messages, events = _social_test_session()
        channel = session.voice_client.channel

        class FakeVoiceClient:
            guild = SimpleNamespace(id=7, owner_id=1)

            def __init__(self) -> None:
                self.channel = channel
                self.playing = True
                self.stop_count = 0

            def is_playing(self) -> bool:
                return self.playing

            def is_paused(self) -> bool:
                return False

            def stop_playing(self) -> None:
                self.playing = False
                self.stop_count += 1

        track = YouTubeTrack("query", "Song", "https://youtube.test/watch", "stream", 180)
        session.voice_client = FakeVoiceClient()
        session.playback_queue = asyncio.Queue()
        session._current_music_item = MusicItem(track, 1, "Host")
        session._music_item_count = 2
        session._music_generation = 0
        session._music_queue_revision = 0
        session._music_history = [track]
        session._music_history_cursor = 0
        session._music_query_deadlines = {}
        session._music_query_queue_only = {}
        session._playlist_query_deadlines = {}
        session._followups = {}
        session._voice_choice_deadlines = {}
        session._personality_choice_deadlines = {}
        session._dj_mode = True
        await session.playback_queue.put(MusicItem(track, 2, "Guest"))

        await session._handle_dj_mode(
            False,
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, DJ mode off",
            50,
        )
        return session, session.voice_client.stop_count, events

    session, stop_count, events = asyncio.run(exercise())

    assert session._dj_mode is False
    assert stop_count == 1
    assert all(not isinstance(item, MusicItem) for item in session.playback_queue._queue)
    assert events[-1][0] == "voice_dj_mode_changed"
    assert events[-1][1]["music_stopped"] is True


def test_play_voice_command_searches_youtube_and_bypasses_model() -> None:
    captured: dict[str, object] = {}
    events: list[str] = []

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "Hey Jangle, play Africa by Toto"

    class FakeYouTube:
        async def search(self, query: str) -> YouTubeTrack:
            captured["query"] = query
            return YouTubeTrack(
                query,
                "Toto - Africa",
                "https://www.youtube.com/watch?v=test",
                "https://stream.test/audio",
                295,
            )

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

        def will_search(self, *_args: object, **_kwargs: object) -> bool:
            raise AssertionError("Music commands must bypass the model search path")

        async def answer(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("Music commands must bypass the model")

    class FakeChannel:
        id = 8
        members: list[object] = []

        async def send(self, text: str, **_kwargs: object) -> None:
            captured["status"] = text

    async def exercise() -> list[object]:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.youtube_music = FakeYouTube()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            voice_wake_words=("jangle",),
            voice_followup_seconds=25,
            music_queue_max=5,
        )
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7, owner_id=42),
            channel=FakeChannel(),
            is_playing=lambda: False,
        )
        session.companion_channel = FakeChannel()
        session.text_echo = False
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._followups = {}
        session._voice_choice_deadlines = {}
        session._music_query_deadlines = {}
        session._music_item_count = 0
        session._music_generation = 0
        session._music_lookup_lock = asyncio.Lock()
        session._current_music_item = None
        session._recent_exchange = None
        session._dj_mode = True

        await session._handle_segment(PcmSegment(42, "Speaker", b"pcm", 1.0))
        return list(session.playback_queue._queue)

    queued = asyncio.run(exercise())

    assert captured["query"] == "Africa by Toto"
    assert "voice_music_search" in events
    assert "voice_music_queued" in events
    assert isinstance(queued[0], SpokenItem)
    assert queued[0].text == "Searching YouTube for Africa by Toto."
    assert isinstance(queued[1], MusicItem)


def test_admin_stop_stops_music_but_ordinary_stop_does_not_match() -> None:
    events: list[str] = []

    class FakeVoiceClient:
        def __init__(self) -> None:
            self.playing = True
            self.stop_count = 0
            self.guild = SimpleNamespace(id=7, owner_id=42)
            self.channel = SimpleNamespace(id=8, members=[])

        def is_playing(self) -> bool:
            return self.playing

        def stop_playing(self) -> None:
            self.playing = False
            self.stop_count += 1

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

    async def exercise() -> tuple[int, list[object]]:
        track = YouTubeTrack("query", "Song", "https://youtube.test/watch", "stream", 180)
        current = MusicItem(track, 11, "Requester")
        queued = MusicItem(track, 12, "Other")
        session = object.__new__(VoiceSession)
        session.voice_client = FakeVoiceClient()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(admin_role_ids=frozenset())
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._current_music_item = current
        session._music_item_count = 2
        session._music_generation = 0
        session._music_query_deadlines = {}
        await session.playback_queue.put(queued)

        await session._handle_admin_stop(PcmSegment(42, "Owner", b"pcm", 1.0), "stop", 50)
        return session.voice_client.stop_count, list(session.playback_queue._queue)

    stop_count, queued = asyncio.run(exercise())

    assert stop_count == 1
    assert "voice_music_stopped" in events
    assert all(not isinstance(item, MusicItem) for item in queued)
    assert any(isinstance(item, SpokenItem) and item.text == "Music stopped." for item in queued)


def test_playlist_command_queues_flat_tracks_without_model_calls() -> None:
    events: list[str] = []
    messages: list[str] = []
    tracks = (
        YouTubeTrack("One", "Song One", "https://youtube.test/one", "", 120),
        YouTubeTrack("Two", "Song Two", "https://youtube.test/two", "", 130),
    )

    class FakeYouTube:
        async def search_playlist(self, query: str) -> YouTubePlaylist:
            assert query == "80s hits"
            return YouTubePlaylist("Best 80s", "https://youtube.test/playlist", tracks)

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

    class FakeChannel:
        id = 8

        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    async def exercise() -> list[object]:
        session = object.__new__(VoiceSession)
        session.youtube_music = FakeYouTube()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            voice_followup_seconds=25,
            music_queue_max=50,
        )
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7, owner_id=42),
            channel=SimpleNamespace(id=8),
        )
        session.companion_channel = FakeChannel()
        session.tts = object()
        session.playback_queue = asyncio.Queue(maxsize=70)
        session._playlist_query_deadlines = {}
        session._music_item_count = 0
        session._music_generation = 0
        session._music_lookup_lock = asyncio.Lock()
        session._current_music_item = None

        await session._handle_playlist_request(
            "80s hits",
            PcmSegment(42, "Speaker", b"pcm", 1.0),
            "Hey Jangle, find playlist 80s hits",
            50,
        )
        return list(session.playback_queue._queue)

    queued = asyncio.run(exercise())

    assert isinstance(queued[0], SpokenItem)
    assert [item.track.title for item in queued[1:] if isinstance(item, MusicItem)] == [
        "Song One",
        "Song Two",
    ]
    assert "voice_playlist_search" in events
    assert "voice_playlist_queued" in events
    assert any("Queued 2 tracks" in message for message in messages)


def test_non_admin_cannot_play_music_or_add_playlists() -> None:
    events: list[str] = []

    class FakeYouTube:
        async def search(self, _query: str) -> YouTubeTrack:
            raise AssertionError("A denied play request must not search YouTube")

        async def search_playlist(self, _query: str) -> YouTubePlaylist:
            raise AssertionError("A denied playlist request must not search YouTube")

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

    member = SimpleNamespace(
        id=42,
        guild_permissions=SimpleNamespace(administrator=False, manage_guild=False),
        roles=[],
    )

    async def exercise() -> list[object]:
        session = object.__new__(VoiceSession)
        session.youtube_music = FakeYouTube()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(admin_role_ids=frozenset())
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7, owner_id=1),
            channel=SimpleNamespace(id=8, members=[member]),
        )
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._current_music_item = None
        session._music_item_count = 0
        segment = PcmSegment(42, "Listener", b"pcm", 1.0)

        await session._handle_music_request(
            "Africa by Toto",
            segment,
            "Hey Jangle, play Africa by Toto",
            50,
        )
        await session._handle_playlist_request(
            "80s hits",
            segment,
            "Hey Jangle, play 80s hits playlist",
            50,
        )
        return list(session.playback_queue._queue)

    queued = asyncio.run(exercise())

    assert events == ["voice_music_play_denied", "voice_playlist_denied"]
    assert [item.text for item in queued if isinstance(item, SpokenItem)] == [
        "Only a server administrator can start music. You can add a song to the queue or "
        "show the queue.",
        "Only a server administrator can add playlists. You can add one song to the queue "
        "or show the queue.",
    ]


def test_previous_command_prepends_history_and_stops_current_track() -> None:
    events: list[str] = []
    messages: list[str] = []
    first = YouTubeTrack("one", "Song One", "https://youtube.test/one", "stream1", 120)
    current_track = YouTubeTrack(
        "two", "Song Two", "https://youtube.test/two", "stream2", 130
    )
    next_track = YouTubeTrack("three", "Song Three", "https://youtube.test/three", "", 140)

    class FakeVoiceClient:
        def __init__(self) -> None:
            self.guild = SimpleNamespace(id=7, owner_id=42)
            self.channel = SimpleNamespace(id=8, members=[])
            self.stop_count = 0

        def is_playing(self) -> bool:
            return True

        def stop_playing(self) -> None:
            self.stop_count += 1

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

    class FakeChannel:
        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    async def exercise() -> tuple[FakeVoiceClient, list[object], int]:
        session = object.__new__(VoiceSession)
        session.voice_client = FakeVoiceClient()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(admin_role_ids=frozenset())
        session.companion_channel = FakeChannel()
        session.tts = object()
        session.playback_queue = asyncio.Queue(maxsize=20)
        session._current_music_item = MusicItem(current_track, 11, "Requester")
        session._music_item_count = 2
        session._music_generation = 0
        session._music_history = [first, current_track]
        session._music_history_cursor = 1
        session._music_finish_reason = None
        await session.playback_queue.put(MusicItem(next_track, 12, "Other"))

        await session._handle_music_navigation(
            "previous",
            PcmSegment(42, "Owner", b"pcm", 1.0),
            "Hey Jangle, previous",
            40,
        )
        return (
            session.voice_client,
            list(session.playback_queue._queue),
            session._music_item_count,
        )

    voice_client, queued, count = asyncio.run(exercise())

    assert voice_client.stop_count == 1
    assert isinstance(queued[0], MusicItem)
    assert queued[0].track.title == "Song One"
    assert queued[0].history_index == 0
    assert count == 3
    assert "voice_music_navigation" in events
    assert any("Going back" in message for message in messages)


def test_dj_mode_discards_wake_word_questions_before_the_model() -> None:
    events: list[str] = []

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            return "Hey Jangle, energeee"

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

        def will_search(self, *_args: object, **_kwargs: object) -> bool:
            raise AssertionError("DJ mode questions must not reach search routing")

        async def answer(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("DJ mode questions must not reach the model")

    async def exercise() -> None:
        session = object.__new__(VoiceSession)
        session.stt = FakeStt()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(voice_wake_words=("jangle",))
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8),
            is_playing=lambda: False,
        )
        session.text_echo = False
        session.playback_queue = asyncio.Queue()
        session._followups = {}
        session._voice_choice_deadlines = {}
        session._music_query_deadlines = {}
        session._playlist_query_deadlines = {}
        session._dj_mode = True

        await session._handle_segment(PcmSegment(42, "Speaker", b"pcm", 1.0))

    asyncio.run(exercise())

    assert events == ["voice_dj_question_ignored"]


def test_dj_mode_disables_owner_barge_in_detection() -> None:
    scheduled: list[object] = []
    pushed: list[dict[str, object]] = []

    class FakeLoop:
        def call_soon_threadsafe(self, callback: object, *_args: object) -> None:
            scheduled.append(callback)

    class FakeSegmenter:
        def push(self, *_args: object, **kwargs: object) -> None:
            pushed.append(kwargs)

    session = object.__new__(VoiceSession)
    session.settings = SimpleNamespace(voice_rms_threshold=300, voice_barge_in_frames=1)
    session.voice_client = SimpleNamespace(is_playing=lambda: True)
    session._current_spoken_item = SpokenItem("DJ status", user_id=42)
    session._current_music_item = None
    session._barge_frames = {}
    session._barge_in_pending = False
    session._loop = FakeLoop()
    session._closed = False
    session._dj_mode = True
    session.decoded_audio_received = True
    session.segmenter = FakeSegmenter()
    user = SimpleNamespace(id=42, display_name="Owner", bot=False)
    data = SimpleNamespace(pcm=_voiced_packet())

    session._on_audio(user, data)

    assert scheduled == []
    assert pushed[0]["owner_barge_in"] is False


def test_add_shorthand_queues_during_music_without_spoken_confirmation() -> None:
    messages: list[str] = []
    captured: dict[str, object] = {}
    current_track = YouTubeTrack(
        "current", "Current Song", "https://youtube.test/current", "stream", 180
    )

    class FakeYouTube:
        async def search(self, query: str) -> YouTubeTrack:
            captured["query"] = query
            return YouTubeTrack(
                query,
                "Mark Morrison - Return of the Mack",
                "https://youtube.test/next",
                "next-stream",
                226,
            )

    class FakeAnswers:
        def record_event(self, _event: str, **_fields: object) -> None:
            return None

    class FakeChannel:
        id = 8

        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    async def exercise() -> tuple[bool, list[object]]:
        session = object.__new__(VoiceSession)
        session.youtube_music = FakeYouTube()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(
            voice_followup_seconds=25,
            music_queue_max=50,
        )
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8),
        )
        session.companion_channel = FakeChannel()
        session.tts = object()
        session.playback_queue = asyncio.Queue(maxsize=70)
        session._dj_mode = True
        session._current_music_item = MusicItem(current_track, 1, "First")
        session._music_item_count = 1
        session._music_generation = 0
        session._music_lookup_lock = asyncio.Lock()
        session._voice_choice_deadlines = {}
        session._music_query_deadlines = {}
        session._playlist_query_deadlines = {}

        handled = await session._handle_control_request(
            "add Mac Morrison",
            PcmSegment(42, "Speaker", b"pcm", 1.0),
            pending_control=None,
            transcript="Hey Jangle, add Mac Morrison",
            transcription_ms=50,
        )
        return handled, list(session.playback_queue._queue)

    handled, queued = asyncio.run(exercise())

    assert handled is True
    assert captured["query"] == "Mac Morrison"
    assert len(queued) == 1 and isinstance(queued[0], MusicItem)
    assert not any(isinstance(item, SpokenItem) for item in queued)
    assert any("Finding Mac Morrison" in message for message in messages)


def test_pause_resume_and_volume_change_active_music() -> None:
    messages: list[str] = []
    events: list[str] = []
    track = YouTubeTrack("song", "Song", "https://youtube.test/song", "stream", 180)

    class SilentSource(discord.AudioSource):
        def read(self) -> bytes:
            return b""

        def is_opus(self) -> bool:
            return False

    class FakeVoiceClient:
        def __init__(self) -> None:
            self.guild = SimpleNamespace(id=7, owner_id=42)
            self.channel = SimpleNamespace(id=8, members=[])
            self.paused = False
            self.source = discord.PCMVolumeTransformer(SilentSource(), volume=0.45)

        def is_playing(self) -> bool:
            return not self.paused

        def is_paused(self) -> bool:
            return self.paused

        def pause(self) -> None:
            self.paused = True

        def resume(self) -> None:
            self.paused = False

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

    class FakeChannel:
        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    async def exercise() -> tuple[FakeVoiceClient, float]:
        session = object.__new__(VoiceSession)
        session.voice_client = FakeVoiceClient()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(admin_role_ids=frozenset(), music_volume=0.45)
        session.companion_channel = FakeChannel()
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._current_music_item = MusicItem(track, 1, "Requester")
        session._music_item_count = 1
        session._music_volume = 0.45
        segment = PcmSegment(42, "Owner", b"pcm", 1.0)

        await session._handle_music_pause("pause", segment, "pause", 30)
        assert session.voice_client.is_paused()
        await session._handle_music_volume(1, segment, "volume up", 30)
        await session._handle_music_volume_level(37, segment, "set volume 37", 30)
        await session._handle_music_volume_level(125, segment, "set volume 125", 30)
        await session._handle_music_pause("resume", segment, "resume", 30)
        return session.voice_client, session._music_volume

    voice_client, volume = asyncio.run(exercise())

    assert voice_client.is_paused() is False
    assert volume == 0.37
    assert voice_client.source.volume == 0.37
    assert events == [
        "voice_music_pause_changed",
        "voice_music_volume_changed",
        "voice_music_volume_changed",
        "voice_music_pause_changed",
    ]
    assert any("Music paused" in message for message in messages)
    assert any("Music volume 65 percent" in message for message in messages)
    assert any("Music volume 37 percent" in message for message in messages)
    assert any("number from 0 through 100" in message for message in messages)


def test_admin_stop_terminates_paused_music() -> None:
    track = YouTubeTrack("song", "Song", "https://youtube.test/song", "stream", 180)

    class PausedVoiceClient:
        def __init__(self) -> None:
            self.stop_count = 0

        def is_playing(self) -> bool:
            return False

        def is_paused(self) -> bool:
            return True

        def stop_playing(self) -> None:
            self.stop_count += 1

    session = object.__new__(VoiceSession)
    session.voice_client = PausedVoiceClient()
    session.playback_queue = asyncio.Queue()
    session._current_music_item = MusicItem(track, 1, "Requester")
    session._music_item_count = 1
    session._music_generation = 0
    session._music_query_deadlines = {}
    session._playlist_query_deadlines = {}
    session._music_history = [track]
    session._music_history_cursor = 0

    stopped = session._stop_music()

    assert stopped is True
    assert session.voice_client.stop_count == 1


def test_admin_clear_queue_keeps_current_music_and_queued_speech() -> None:
    events: list[str] = []
    messages: list[str] = []
    track = YouTubeTrack("song", "Current Song", "https://youtube.test/current", "stream", 180)
    queued_one = YouTubeTrack("one", "Queued One", "https://youtube.test/one", "", 120)
    queued_two = YouTubeTrack("two", "Queued Two", "https://youtube.test/two", "", 240)

    class FakeVoiceClient:
        def __init__(self) -> None:
            self.guild = SimpleNamespace(id=7, owner_id=42)
            self.channel = SimpleNamespace(id=8, members=[])
            self.stop_count = 0

        def is_playing(self) -> bool:
            return True

        def is_paused(self) -> bool:
            return False

        def stop_playing(self) -> None:
            self.stop_count += 1

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

    class FakeChannel:
        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    async def exercise() -> tuple[FakeVoiceClient, VoiceSession, list[object]]:
        session = object.__new__(VoiceSession)
        session.voice_client = FakeVoiceClient()
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(admin_role_ids=frozenset(), music_volume=0.5)
        session.companion_channel = FakeChannel()
        session.tts = object()
        session.playback_queue = asyncio.Queue()
        session._current_music_item = MusicItem(track, 1, "Requester")
        session._music_item_count = 3
        session._music_generation = 4
        session._music_queue_revision = 2
        session._music_query_deadlines = {42: time.monotonic() + 10}
        session._playlist_query_deadlines = {42: time.monotonic() + 10}
        await session.playback_queue.put(SpokenItem("keep this speech"))
        await session.playback_queue.put(MusicItem(queued_one, 2, "One"))
        await session.playback_queue.put(MusicItem(queued_two, 3, "Two"))

        await session._handle_admin_clear_queue(
            PcmSegment(42, "Owner", b"pcm", 1.0),
            "clear queue",
            40,
        )
        return session.voice_client, session, list(session.playback_queue._queue)

    voice_client, session, queued = asyncio.run(exercise())

    assert voice_client.stop_count == 0
    assert session._current_music_item is not None
    assert session._music_item_count == 1
    assert session._music_generation == 4
    assert session._music_queue_revision == 3
    assert queued == [SpokenItem("keep this speech")]
    assert events == ["voice_music_queue_cleared"]
    assert any("Cleared 2 queued tracks" in message for message in messages)


def test_show_queue_posts_current_and_up_next_without_speaking() -> None:
    events: list[str] = []
    messages: list[str] = []
    current = YouTubeTrack("current", "Current Song", "https://youtube.test/current", "", 180)
    upcoming = YouTubeTrack("next", "Next Song", "https://youtube.test/next", "", 245)

    class FakeAnswers:
        def record_event(self, event: str, **_fields: object) -> None:
            events.append(event)

    class FakeChannel:
        async def send(self, text: str, **_kwargs: object) -> None:
            messages.append(text)

    async def exercise() -> list[object]:
        session = object.__new__(VoiceSession)
        session.voice_client = SimpleNamespace(
            guild=SimpleNamespace(id=7),
            channel=SimpleNamespace(id=8),
            is_paused=lambda: True,
        )
        session.answer_service = FakeAnswers()
        session.settings = SimpleNamespace(music_volume=0.5)
        session.companion_channel = FakeChannel()
        session.playback_queue = asyncio.Queue()
        session._current_music_item = MusicItem(current, 1, "First")
        session._music_volume = 0.6
        await session.playback_queue.put(MusicItem(upcoming, 2, "Second"))

        await session._handle_show_queue(
            PcmSegment(42, "Listener", b"pcm", 1.0),
            "show queue",
            35,
        )
        return list(session.playback_queue._queue)

    queued = asyncio.run(exercise())

    assert len(queued) == 1 and isinstance(queued[0], MusicItem)
    assert events == ["voice_music_queue_shown"]
    assert len(messages) == 1
    assert "Current Song" in messages[0]
    assert "paused" in messages[0]
    assert "Next Song" in messages[0]
    assert "60%" in messages[0]


def test_voice_trivia_scores_first_correct_speaker_and_advances() -> None:
    session, messages, events = _social_test_session()

    async def exercise() -> None:
        await session._handle_game_command(
            SocialCommand("game", "start", mode="trivia", argument="wow"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start WoW trivia",
            40,
        )
        state = session._game_state
        assert state is not None
        assert state.current_question is not None
        answer = state.current_question.answers[0]  # type: ignore[union-attr]
        await session._handle_game_input(
            answer,
            PcmSegment(2, "Guest", b"pcm", 1.0),
            35,
        )
        assert state.scores == {}
        assert state.round_number == 1
        await session._handle_game_input(
            "definitely wrong",
            PcmSegment(1, "Host", b"pcm", 1.0),
            35,
        )

    asyncio.run(exercise())

    assert session._game_state is not None
    assert session._game_state.scores == {2: 1}
    assert session._game_state.round_number == 2
    assert any("Correct, Guest" in message for message in messages)
    assert events[0][0] == "voice_game_started"


def test_voice_poll_accepts_wake_free_votes_and_posts_results() -> None:
    session, messages, events = _social_test_session()

    async def exercise() -> None:
        await session._handle_poll_command(
            SocialCommand(
                "poll",
                "start",
                argument="What should we run?",
                options=("raid", "keys"),
            ),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start poll raid or keys",
            40,
        )
        assert session._activity_input_kind(1, "raid") == "poll"
        await session._handle_poll_input(
            "raid",
            PcmSegment(1, "Host", b"pcm", 1.0),
            20,
        )
        await session._handle_poll_input(
            "option two",
            PcmSegment(2, "Guest", b"pcm", 1.0),
            25,
        )

    asyncio.run(exercise())

    assert session._poll_state is None
    assert any("Voice poll results" in message for message in messages)
    assert any("tie between raid and keys" in message for message in messages)
    assert [event for event, _fields in events].count("voice_poll_vote") == 2


def test_story_mode_enforces_speaker_id_turn_order() -> None:
    session, messages, _events = _social_test_session()

    async def exercise() -> None:
        await session._handle_story_command(
            SocialCommand("story", "start", argument="a cursed raid portal"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start story mode about a cursed raid portal",
            50,
        )
        assert session._activity_input_kind(1, "I charge through") == "story"
        assert session._activity_input_kind(2, "I interrupt") is None
        await session._handle_story_input(
            "I charge through the portal",
            PcmSegment(1, "Host", b"pcm", 1.0),
            30,
        )

    asyncio.run(exercise())

    assert session._story_state is not None
    assert session._story_current_participant().user_id == 2  # type: ignore[union-attr]
    assert any("Guest, what do you do?" in message for message in messages)


def test_jangle_awards_resolve_discord_names_and_announce_tie() -> None:
    session, messages, events = _social_test_session()

    async def exercise() -> None:
        await session._handle_award_command(
            SocialCommand("award", "start", argument="most likely to stand in fire"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start an award for most likely to stand in fire",
            40,
        )
        await session._handle_award_input(
            "I nominate Guest",
            PcmSegment(1, "Host", b"pcm", 1.0),
            25,
        )
        await session._handle_award_input(
            "I nominate Host",
            PcmSegment(2, "Guest", b"pcm", 1.0),
            25,
        )

    asyncio.run(exercise())

    assert session._award_state is None
    assert any("tied between Guest and Host" in message for message in messages)
    assert [event for event, _fields in events].count("voice_award_nomination") == 2


def test_party_mode_requires_admin_to_start_but_anyone_can_stop() -> None:
    session, messages, events = _social_test_session()

    async def exercise() -> None:
        await session._handle_party_mode_command(
            SocialCommand("party", "start", duration_minutes=15),
            PcmSegment(2, "Guest", b"pcm", 1.0),
            20,
        )
        assert session._party_mode_active() is False
        await session._handle_party_mode_command(
            SocialCommand("party", "start", duration_minutes=15),
            PcmSegment(1, "Host", b"pcm", 1.0),
            20,
        )
        assert session._party_mode_active() is True
        await session._handle_party_mode_command(
            SocialCommand("party", "stop"),
            PcmSegment(2, "Guest", b"pcm", 1.0),
            20,
        )

    asyncio.run(exercise())

    assert session._party_mode_active() is False
    assert any("Only a server administrator" in message for message in messages)
    assert any("Party mode enabled for 15 minutes" in message for message in messages)
    assert events[-1][0] == "voice_party_mode_disabled"


def test_handle_segment_accepts_answer_spoken_during_trivia_question() -> None:
    session, messages, _events = _social_test_session()

    class FakeStt:
        async def transcribe(self, _segment: PcmSegment) -> str:
            state = session._game_state
            assert state is not None
            question = state.current_question
            assert isinstance(question, QuizQuestion)
            return question.answers[0]

    async def exercise() -> None:
        session.stt = FakeStt()
        session.settings.voice_wake_words = ("jangle",)
        session.text_echo = False
        session._ignored_user_ids = set()
        session._followups = {}
        session._voice_choice_deadlines = {}
        session._personality_choice_deadlines = {}
        session._music_query_deadlines = {}
        session._music_query_queue_only = {}
        session._playlist_query_deadlines = {}
        await session._handle_game_command(
            SocialCommand("game", "start", mode="trivia", argument="general"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start general trivia",
            30,
        )
        state = session._game_state
        assert state is not None
        await session._handle_segment(
            PcmSegment(
                2,
                "Guest",
                b"pcm",
                1.0,
                blocked_playback=True,
                protected_game_answer=True,
                game_window_token=state.window_token,
                started_at=10.0,
            )
        )
        await session._handle_game_input(
            "definitely wrong",
            PcmSegment(1, "Host", b"pcm", 1.0),
            25,
        )

    asyncio.run(exercise())

    assert session._game_state is not None
    assert session._game_state.scores == {2: 1}
    assert any("Correct, Guest" in message for message in messages)


def test_trivia_awards_fastest_correct_even_if_processed_second() -> None:
    session, messages, _events = _social_test_session()
    state = GameState(
        "trivia",
        "general",
        1,
        "Host",
        round_number=1,
        rounds_total=1,
        current_question=QuizQuestion("Largest planet?", ("Jupiter",)),
        accepting_answers=True,
        eligible_user_ids={1, 2},
    )
    session._game_state = state

    async def exercise() -> None:
        await session._handle_activity_input(
            "game",
            "Jupiter",
            PcmSegment(2, "Guest", b"pcm", 1.0, started_at=20.0),
            25,
        )
        await session._handle_activity_input(
            "game",
            "Jupiter",
            PcmSegment(1, "Host", b"pcm", 1.0, started_at=10.0),
            25,
        )

    asyncio.run(exercise())

    assert state.scores == {1: 1}
    assert session._game_state is None
    assert sum("Correct" in message for message in messages) == 1
    assert any("Host. You were fastest" in message for message in messages)


def test_trivia_rejects_audio_from_an_old_question_window() -> None:
    session, _messages, events = _social_test_session()
    state = GameState(
        "trivia",
        "general",
        1,
        "Host",
        round_number=1,
        current_question=QuizQuestion("Largest planet?", ("Jupiter",)),
        accepting_answers=True,
        eligible_user_ids={1, 2},
        window_token=4,
    )
    session._game_state = state

    asyncio.run(
        session._handle_game_input(
            "Jupiter",
            PcmSegment(2, "Guest", b"pcm", 1.0, game_window_token=3),
            20,
        )
    )

    assert state.attempted_user_ids == set()
    assert any(event == "voice_game_stale_submission_ignored" for event, _ in events)


def test_game_gives_each_player_one_answer_per_attempt() -> None:
    session, _messages, events = _social_test_session()
    state = GameState(
        "trivia",
        "general",
        1,
        "Host",
        round_number=1,
        current_question=QuizQuestion("Largest planet?", ("Jupiter",)),
        accepting_answers=True,
        eligible_user_ids={1, 2},
    )
    session._game_state = state

    async def exercise() -> None:
        await session._handle_game_input(
            "Jupiter",
            PcmSegment(1, "Host", b"pcm", 1.0),
            20,
        )
        await session._handle_game_input(
            "Mars",
            PcmSegment(1, "Host", b"pcm", 1.0),
            20,
        )

    asyncio.run(exercise())

    assert state.attempted_user_ids == {1}
    assert state.correct_user_ids == {1}
    assert state.scores == {}
    assert any(event == "voice_game_duplicate_submission_ignored" for event, _ in events)


def test_trivia_repeats_once_then_reveals_and_advances() -> None:
    session, messages, _events = _social_test_session()

    async def exercise() -> None:
        await session._handle_game_command(
            SocialCommand("game", "start", mode="trivia", argument="general"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start general trivia",
            30,
        )
        state = session._game_state
        assert state is not None
        first_question = state.current_question
        for user_id, name in ((1, "Host"), (2, "Guest")):
            await session._handle_game_input(
                "wrong answer",
                PcmSegment(user_id, name, b"pcm", 1.0),
                20,
            )
        assert state.answer_attempt == 2
        assert state.round_number == 1
        assert state.current_question is first_question
        assert state.attempted_user_ids == set()
        for user_id, name in ((1, "Host"), (2, "Guest")):
            await session._handle_game_input(
                "still wrong",
                PcmSegment(user_id, name, b"pcm", 1.0),
                20,
            )
        assert state.round_number == 2
        assert state.current_question is not first_question
        session._cancel_game_timer()

    asyncio.run(exercise())

    assert any("Second try for everyone" in message for message in messages)
    assert any("No correct answers after two tries" in message for message in messages)


def test_would_you_rather_collects_one_vote_from_everyone() -> None:
    session, messages, _events = _social_test_session()

    async def exercise() -> None:
        await session._handle_game_command(
            SocialCommand("game", "start", mode="would"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start Would You Rather",
            30,
        )
        state = session._game_state
        assert state is not None
        await session._handle_game_input("A", PcmSegment(1, "Host", b"pcm", 1.0), 20)
        await session._handle_game_input("B", PcmSegment(1, "Host", b"pcm", 1.0), 20)
        assert state.round_number == 1
        await session._handle_game_input("B", PcmSegment(2, "Guest", b"pcm", 1.0), 20)
        assert state.round_number == 2
        session._cancel_game_timer()

    asyncio.run(exercise())

    assert any("got 1" in message and "Results:" in message for message in messages)


def test_twenty_questions_cycles_after_each_player_gets_one_turn() -> None:
    session, messages, events = _social_test_session()

    async def exercise() -> None:
        await session._handle_game_command(
            SocialCommand("game", "start", mode="twenty"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start Twenty Questions",
            30,
        )
        state = session._game_state
        assert state is not None
        await session._handle_game_input(
            "Is it alive?",
            PcmSegment(1, "Host", b"pcm", 1.0),
            20,
        )
        await session._handle_game_input(
            "Is it blue?",
            PcmSegment(1, "Host", b"pcm", 1.0),
            20,
        )
        assert state.question_count == 1
        await session._handle_game_input(
            "Can it fit in a room?",
            PcmSegment(2, "Guest", b"pcm", 1.0),
            20,
        )
        assert state.question_count == 2
        assert state.attempted_user_ids == set()
        assert state.accepting_answers is True
        session._cancel_game_timer()

    asyncio.run(exercise())

    assert any("Everyone gets one question or guess again" in message for message in messages)
    assert any(event == "voice_game_duplicate_submission_ignored" for event, _ in events)


def test_game_hint_preserves_submissions_and_reopens_window() -> None:
    session, messages, _events = _social_test_session()

    async def exercise() -> None:
        await session._handle_game_command(
            SocialCommand("game", "start", mode="riddle"),
            PcmSegment(1, "Host", b"pcm", 1.0),
            "Hey Jangle, start riddles",
            30,
        )
        state = session._game_state
        assert state is not None
        await session._handle_game_input(
            "wrong answer",
            PcmSegment(1, "Host", b"pcm", 1.0),
            20,
        )
        await session._handle_game_command(
            SocialCommand("game", "hint"),
            PcmSegment(2, "Guest", b"pcm", 1.0),
            "Hey Jangle, hint",
            15,
        )
        assert state.attempted_user_ids == {1}
        assert state.accepting_answers is True
        assert state.hint_used is True
        session._cancel_game_timer()

    asyncio.run(exercise())

    assert any("Hint:" in message for message in messages)


def test_active_game_blocks_side_commands_from_normal_handlers() -> None:
    session, _messages, events = _social_test_session()
    session._game_state = GameState(
        "trivia",
        "general",
        1,
        "Host",
        round_number=1,
        current_question=QuizQuestion("Largest planet?", ("Jupiter",)),
        accepting_answers=True,
        eligible_user_ids={1, 2},
    )

    async def exercise() -> bool:
        return await session._handle_control_request(
            "change voice to Aria",
            PcmSegment(1, "Host", b"pcm", 1.0),
            pending_control=None,
            transcript="Hey Jangle, change voice to Aria",
            transcription_ms=20,
        )

    assert asyncio.run(exercise()) is True
    assert any(event == "voice_game_side_command_ignored" for event, _ in events)
