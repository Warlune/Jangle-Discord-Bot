from __future__ import annotations

import asyncio

from discord_ai.sessions import EphemeralSessionStore


def test_separate_session_lanes_can_run_concurrently() -> None:
    async def exercise() -> int:
        store = EphemeralSessionStore(history_turns=2)
        active = 0
        maximum_active = 0

        async def answer(_history: list[dict[str, str]]) -> str:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            return "done"

        await asyncio.gather(
            store.run("text:one", "hello", answer),
            store.run("voice:one", "hello", answer),
        )
        return maximum_active

    assert asyncio.run(exercise()) == 2


def test_history_is_bounded_and_resettable() -> None:
    async def exercise() -> EphemeralSessionStore:
        store = EphemeralSessionStore(history_turns=1)

        async def answer(_history: list[dict[str, str]]) -> str:
            return "answer"

        await store.run("lane", "first", answer)
        await store.run("lane", "second", answer)
        return store

    store = asyncio.run(exercise())
    assert store.snapshot("lane") == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer"},
    ]
    assert store.reset("lane") is True
    assert store.snapshot("lane") == []


def test_interrupted_assistant_response_is_not_kept_as_fully_heard() -> None:
    async def exercise() -> EphemeralSessionStore:
        store = EphemeralSessionStore(history_turns=2)

        async def answer(_history: list[dict[str, str]]) -> str:
            return "A long answer the user interrupted"

        await store.run("voice:42", "question", answer)
        return store

    store = asyncio.run(exercise())

    assert store.mark_assistant_interrupted(
        "voice:42", "A long answer the user interrupted"
    ) is True
    assert store.snapshot("voice:42")[-1] == {
        "role": "assistant",
        "content": "[Assistant response was interrupted before completion.]",
    }
