from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable


History = list[dict[str, str]]
AnswerFunction = Callable[[History], Awaitable[str]]
ResponseTransform = Callable[[str], str]


@dataclass
class _Session:
    messages: History


class EphemeralSessionStore:
    """In-memory conversation history. Nothing here is written to disk."""

    def __init__(self, history_turns: int) -> None:
        self.history_turns = max(0, history_turns)
        self._sessions: dict[str, _Session] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def run(
        self,
        key: str,
        prompt: str,
        answer: AnswerFunction,
        *,
        response_for_history: ResponseTransform | None = None,
    ) -> str:
        async with self._locks[key]:
            history = [dict(item) for item in self._sessions.get(key, _Session([])).messages]
            response = await answer(history)
            if self.history_turns:
                stored_response = (
                    response_for_history(response) if response_for_history is not None else response
                )
                history.extend(
                    (
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": stored_response},
                    )
                )
                history = history[-self.history_turns * 2 :]
                self._sessions[key] = _Session(history)
            return response

    def reset(self, key: str) -> bool:
        return self._sessions.pop(key, None) is not None

    def reset_prefix(self, prefix: str) -> int:
        keys = [key for key in self._sessions if key.startswith(prefix)]
        for key in keys:
            self._sessions.pop(key, None)
        return len(keys)

    def snapshot(self, key: str) -> History:
        return [dict(item) for item in self._sessions.get(key, _Session([])).messages]

    def mark_assistant_interrupted(self, key: str, response: str) -> bool:
        session = self._sessions.get(key)
        if session is None:
            return False
        for index in range(len(session.messages) - 1, -1, -1):
            message = session.messages[index]
            if message.get("role") != "assistant":
                continue
            if str(message.get("content") or "") != response:
                continue
            session.messages[index] = {
                "role": "assistant",
                "content": "[Assistant response was interrupted before completion.]",
            }
            return True
        return False
