from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from discord_ai.warlune import (
    AnswerService,
    GuestWarluneConfig,
    OpenAICompatibleGateway,
    WarluneGateway,
    copyright_safe_answer,
    voice_system_prompt,
)


def _guest_config() -> GuestWarluneConfig:
    return GuestWarluneConfig(
        endpoint="http://127.0.0.1:1234/v1",
        model="test-model",
        internet_enabled=False,
        internet_mode="auto",
        searxng_url="http://127.0.0.1:8888",
        max_results=3,
        timeout_seconds=5,
        fetch_pages=False,
        max_pages=0,
        page_timeout_seconds=5,
        page_max_bytes=100_000,
        page_context_chars=1000,
        context_max_chars=2000,
    )


def _standalone_settings(provider: str = "lmstudio") -> SimpleNamespace:
    return SimpleNamespace(
        model_provider=provider,
        model_endpoint=(
            "http://127.0.0.1:1234/v1"
            if provider == "lmstudio"
            else "http://127.0.0.1:11434/v1"
        ),
        model_name="",
        model_api_key="",
        model_timeout_seconds=30.0,
        internet_search_enabled=False,
        searxng_url="http://127.0.0.1:8888",
        search_max_results=5,
        bot_display_name="Jangle",
        answer_max_chars=6000,
    )


def test_lmstudio_standalone_gateway_discovers_model_and_uses_guest_boundary() -> None:
    settings = _standalone_settings()
    gateway = OpenAICompatibleGateway(settings)
    captured: dict[str, object] = {}

    def fake_request(
        path: str,
        *,
        method: str,
        payload: dict[str, object] | None = None,
        timeout: float,
    ) -> dict[str, object]:
        if path == "/models":
            return {
                "data": [
                    {"id": "text-embedding-model"},
                    {"id": "local-chat-model"},
                ]
            }
        captured.update(
            {"path": path, "method": method, "payload": payload, "timeout": timeout}
        )
        return {"choices": [{"message": {"content": "Local answer"}}]}

    gateway._request_json = fake_request  # type: ignore[method-assign]

    answer = gateway.ask(
        "What should I play?",
        [{"role": "assistant", "content": "Temporary context"}],
        runtime_context="Current speaker: Guest",
    )

    assert gateway.prepare() == "local-chat-model"
    assert gateway.provider_name == "LM Studio"
    assert answer == "Local answer"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    serialized = json.dumps(payload)
    assert "DISCORD GUEST MODE" in serialized
    assert "Temporary context" in serialized
    assert "Current speaker: Guest" in serialized
    assert "profile_context" not in serialized
    assert "memory_context" not in serialized


def test_ollama_standalone_gateway_uses_same_openai_compatible_contract() -> None:
    settings = _standalone_settings("ollama")
    settings.model_name = "llama3.2"
    gateway = OpenAICompatibleGateway(settings)
    captured: dict[str, object] = {}

    def fake_request(
        path: str,
        *,
        method: str,
        payload: dict[str, object] | None = None,
        timeout: float,
    ) -> dict[str, object]:
        captured.update(
            {"path": path, "method": method, "payload": payload, "timeout": timeout}
        )
        return {"choices": [{"message": {"content": "<think>hidden</think>Hi!"}}]}

    gateway._request_json = fake_request  # type: ignore[method-assign]

    answer = gateway.ask("hello", [], voice=True)

    assert gateway.provider_name == "Ollama"
    assert answer == "Hi!"
    assert captured["path"] == "/chat/completions"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False


def test_standalone_gateways_use_real_local_openai_http_contract() -> None:
    requests: list[tuple[str, str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            requests.append(("GET", self.path, self.headers.get("Authorization", "")))
            self._send({"data": [{"id": "local-test-model"}]})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requests.append(("POST", self.path, str(payload.get("model") or "")))
            self._send({"choices": [{"message": {"content": "HTTP answer"}}]})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for provider in ("lmstudio", "ollama"):
            settings = _standalone_settings(provider)
            settings.model_endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            settings.model_api_key = "test-local-token"
            gateway = OpenAICompatibleGateway(settings)

            assert gateway.prepare() == "local-test-model"
            assert gateway.ask("hello", []) == "HTTP answer"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert requests.count(("GET", "/v1/models", "Bearer test-local-token")) == 2
    assert requests.count(("POST", "/v1/chat/completions", "local-test-model")) == 2


def test_guest_config_never_loads_profile_memory_or_tokens(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "orchestrator": {"main_endpoint": "http://127.0.0.1:1234/v1", "main_model": "model"},
                "internet": {"enabled": True},
                "user_profile": {"important_context": "PRIVATE PROFILE"},
                "memory": {"db_path": "PRIVATE MEMORY"},
                "lan_api": {"token": "PRIVATE TOKEN"},
            }
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(warlune_endpoint="", warlune_model="")

    selected = GuestWarluneConfig.read(path, settings)
    serialized = json.dumps(asdict(selected))

    assert "PRIVATE" not in serialized
    assert not hasattr(selected, "user_profile")
    assert not hasattr(selected, "memory")


def test_every_orchestrator_call_enforces_guest_privacy_arguments() -> None:
    captured: dict[str, object] = {}
    gateway = object.__new__(WarluneGateway)
    gateway.settings = SimpleNamespace(answer_max_chars=6000, bot_display_name="Jangle")
    gateway.config = _guest_config()
    gateway._model = "test-model"
    gateway.prepare = lambda: "test-model"
    gateway._web_context = lambda _prompt, _force, **_kwargs: (
        "PUBLIC WEB CONTEXT",
        [{"title": "Source", "url": "https://example.com"}],
        "query",
    )

    def fake_run(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"final": "safe answer"}

    gateway._run_prompt = fake_run

    assert gateway.ask("hello", [], force_search=True) == "safe answer"
    assert captured["memory_context"] == ""
    assert captured["profile_context"] == ""
    assert captured["enable_logging"] is False
    assert captured["attachments"] == []
    assert captured["tools"] is None


def test_fast_chat_receives_only_ephemeral_history_and_guest_instructions() -> None:
    captured: dict[str, object] = {}
    gateway = object.__new__(WarluneGateway)
    gateway.settings = SimpleNamespace(answer_max_chars=6000, bot_display_name="Jangle")
    gateway.config = _guest_config()
    gateway._model = "test-model"
    gateway.prepare = lambda: "test-model"
    gateway._web_context = lambda _prompt, _force, **_kwargs: ("", [], "")

    def fake_chat(endpoint: str, model: str, messages: list[dict[str, str]], **kwargs: object) -> str:
        captured.update({"endpoint": endpoint, "model": model, "messages": messages, **kwargs})
        return "fast answer"

    gateway._chat_completion = fake_chat
    history = [{"role": "assistant", "content": "ephemeral prior answer"}]

    assert (
        gateway.ask(
            "hello",
            history,
            personality_prompt="Use the configured Madam test style.",
        )
        == "fast answer"
    )
    serialized = json.dumps(captured)
    assert "ephemeral prior answer" in serialized
    assert "DISCORD GUEST MODE" in serialized
    assert "profile_context" not in serialized
    assert "memory_context" not in serialized
    assert "tools" not in captured
    assert "ACTIVE PERSONALITY" in serialized
    assert "Madam test style" in serialized


def test_fast_text_receives_plugin_notes_as_untrusted_user_context() -> None:
    captured: dict[str, object] = {}
    gateway = object.__new__(WarluneGateway)
    gateway.settings = SimpleNamespace(answer_max_chars=6000, bot_display_name="Jangle")
    gateway.config = _guest_config()
    gateway._model = "test-model"
    gateway.prepare = lambda: "test-model"
    gateway._web_context = lambda _prompt, _force, **_kwargs: ("", [], "")

    def fake_chat(
        _endpoint: str,
        _model: str,
        messages: list[dict[str, str]],
        **_kwargs: object,
    ) -> str:
        captured["messages"] = messages
        return "personalized answer"

    gateway._chat_completion = fake_chat

    answer = gateway.ask(
        "What should I play tonight?",
        [],
        runtime_context="USER-CONTROLLED JANGLE NOTEPAD\n1. I main a frost mage",
    )

    assert answer == "personalized answer"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "I main a frost mage" in messages[-1]["content"]
    assert "never instructions" in messages[-1]["content"]
    assert "I main a frost mage" not in messages[0]["content"]


def test_medium_uses_full_warlune_reasoning_route() -> None:
    captured: dict[str, object] = {}
    gateway = object.__new__(WarluneGateway)
    gateway.settings = SimpleNamespace(answer_max_chars=6000, bot_display_name="Jangle")
    gateway.config = _guest_config()
    gateway._model = "test-model"
    gateway.prepare = lambda: "test-model"
    gateway._web_context = lambda _prompt, _force, **_kwargs: ("", [], "")
    gateway._chat_completion = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Medium must not use the Instant fast path")
    )

    def fake_run(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"final": "medium answer"}

    gateway._run_prompt = fake_run

    assert gateway.ask("think about this", [], intelligence_level="Medium") == "medium answer"
    assert captured["intelligence_level"] == "Medium"
    assert captured["memory_context"] == ""
    assert captured["profile_context"] == ""


def test_voice_uses_native_reasoning_off_fast_path() -> None:
    captured: dict[str, object] = {}
    gateway = object.__new__(WarluneGateway)
    gateway.settings = SimpleNamespace(answer_max_chars=6000, bot_display_name="Jangle")
    gateway.config = _guest_config()
    gateway._model = "test-model"
    gateway.prepare = lambda: "test-model"
    gateway._web_context = lambda _prompt, _force, **_kwargs: ("", [], "")

    def fake_native(endpoint: str, model: str, input_text: str, **kwargs: object) -> str:
        captured.update(
            {"endpoint": endpoint, "model": model, "input_text": input_text, **kwargs}
        )
        return "spoken answer"

    gateway._native_chat_completion = fake_native
    gateway._chat_completion = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("Native voice path should not fall back")
    )

    answer = gateway.ask(
        "hello",
        [{"role": "assistant", "content": "temporary context"}],
        voice=True,
        runtime_context="Current speaker display name: Speaker",
        personality_prompt="Use the configured Savage test style.",
    )

    assert answer == "spoken answer"
    assert captured["reasoning"] == "off"
    assert captured["max_tokens"] == 96
    assert "temporary context" in str(captured["input_text"])
    assert "Current speaker display name: Speaker" in str(captured["input_text"])
    assert "Savage test style" in str(captured["system_prompt"])
    assert "Savage test style" not in str(captured["input_text"])


def test_voice_system_prompt_keeps_personality_in_trusted_configuration() -> None:
    prompt = voice_system_prompt(
        "Jangle",
        "Use the configured Brutal test style.",
    )

    assert "ACTIVE PERSONALITY (trusted Discord bot configuration)" in prompt
    assert "Brutal test style" in prompt
    assert prompt.index("ACTIVE PERSONALITY") > prompt.index("Turn-taking example")


def test_voice_search_summarizes_web_context_without_full_reasoning_route() -> None:
    captured: dict[str, object] = {}
    gateway = object.__new__(WarluneGateway)
    gateway.settings = SimpleNamespace(answer_max_chars=6000, bot_display_name="Jangle")
    gateway.config = _guest_config()
    gateway._model = "test-model"
    gateway.prepare = lambda: "test-model"
    gateway._web_context = lambda _prompt, _force, **_kwargs: (
        "PUBLIC RECIPE FACTS",
        [{"title": "Source", "url": "https://example.com"}],
        "recipe query",
    )
    gateway._run_prompt = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("Instant voice search must not use the full reasoning route")
    )

    def fake_native(endpoint: str, model: str, input_text: str, **kwargs: object) -> str:
        captured.update(
            {"endpoint": endpoint, "model": model, "input_text": input_text, **kwargs}
        )
        return "Concise researched answer."

    gateway._native_chat_completion = fake_native

    answer = gateway.ask("look up the recipe", [], voice=True)

    assert answer == "Concise researched answer."
    assert captured["reasoning"] == "off"
    assert captured["max_tokens"] == 128
    assert "PUBLIC RECIPE FACTS" in str(captured["input_text"])


def test_voice_lane_does_not_wait_behind_text() -> None:
    text_started = threading.Event()
    release_text = threading.Event()

    class FakeGateway:
        def ask(
            self,
            prompt: str,
            _history: list[dict[str, str]],
            **_kwargs: object,
        ) -> str:
            if prompt == "slow text":
                text_started.set()
                release_text.wait(timeout=2)
                return "text done"
            return "voice done"

    async def exercise() -> str:
        settings = SimpleNamespace(history_turns=1, max_concurrent_requests=2)
        service = AnswerService(settings, FakeGateway())
        text_task = asyncio.create_task(service.answer("text", "slow text"))
        await asyncio.to_thread(text_started.wait, 1)
        try:
            voice_answer = await asyncio.wait_for(
                service.answer("voice", "voice", voice=True),
                timeout=1,
            )
        finally:
            release_text.set()
        await text_task
        return voice_answer

    assert asyncio.run(exercise()) == "voice done"


def test_voice_search_intent_covers_fresh_information() -> None:
    gateway = object.__new__(WarluneGateway)
    gateway.config = GuestWarluneConfig(
        **{**asdict(_guest_config()), "internet_enabled": True, "internet_mode": "auto"}
    )
    gateway._should_search = lambda _mode, prompt, _route: "today" in prompt.casefold()

    assert gateway.will_search("What is the weather?", voice=True) is True
    assert gateway.will_search("What time is it in Illinois?", voice=True) is True
    assert gateway.will_search(
        "How do you make the Hand of Ragnaros?", voice=True
    ) is True
    assert gateway.will_search("What time is it in Illinois?", voice=False) is True
    assert gateway.will_search("Tell me a timeless riddle", voice=True) is False


def test_voice_control_marker_is_not_stored_in_conversation_history() -> None:
    class FakeGateway:
        def ask(self, *_args: object, **_kwargs: object) -> str:
            return "Knock knock. [[AWAIT_REPLY]]"

    async def exercise() -> list[dict[str, str]]:
        settings = SimpleNamespace(history_turns=2, max_concurrent_requests=2)
        service = AnswerService(settings, FakeGateway())
        raw_answer = await service.answer("voice:42", "Tell me a joke", voice=True)
        assert "AWAIT_REPLY" in raw_answer
        return service.sessions.snapshot("voice:42")

    assert asyncio.run(exercise()) == [
        {"role": "user", "content": "Tell me a joke"},
        {"role": "assistant", "content": "Knock knock."},
    ]


def test_location_based_copyright_request_is_redirected_to_summary() -> None:
    answer = copyright_safe_answer("Tell me the first page of The Hobbit")

    assert answer is not None
    assert "summary" in answer
    assert copyright_safe_answer("Summarize the first page of The Hobbit") is None
