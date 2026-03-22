from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_call_id: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class ChatSession:
    """Manages conversation history for an agent chat session.

    Thread-safe: all mutations to the message list are guarded by a lock.
    """

    def __init__(self, system_prompt: str = "", max_history: int = 100) -> None:
        self.system_prompt = system_prompt
        self.max_history = max_history
        self._messages: list[Message] = []
        self._lock = threading.Lock()

    def add_user(self, content: str) -> Message:
        msg = Message(role="user", content=content)
        with self._lock:
            self._messages.append(msg)
            self._trim()
        return msg

    def add_assistant(
        self, content: str, tool_calls: list[dict] | None = None
    ) -> Message:
        msg = Message(role="assistant", content=content, tool_calls=tool_calls or [])
        with self._lock:
            self._messages.append(msg)
            self._trim()
        return msg

    def add_tool_result(self, tool_call_id: str, content: str) -> Message:
        msg = Message(role="tool", content=content, tool_call_id=tool_call_id)
        with self._lock:
            self._messages.append(msg)
            self._trim()
        return msg

    def get_messages(self) -> list[dict]:
        """Return messages in LLM provider format (list of dicts).

        Note: system prompt is NOT included here — it is passed separately
        to the LLM provider via its ``system`` parameter to avoid duplication.
        """
        with self._lock:
            snapshot = list(self._messages)

        result: list[dict] = []
        for m in snapshot:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            result.append(entry)
        return result

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    @property
    def last_user_message(self) -> str:
        with self._lock:
            for msg in reversed(self._messages):
                if msg.role == "user":
                    return msg.content
        return ""

    @property
    def last_assistant_message(self) -> str:
        with self._lock:
            for msg in reversed(self._messages):
                if msg.role == "assistant" and msg.content:
                    return msg.content
        return ""

    def get_summary_context(self) -> str:
        """Return a compact summary of the conversation for knowledge extraction.

        Includes: user questions, assistant conclusions.
        Excludes: system prompts, tool results, empty messages.
        """
        with self._lock:
            snapshot = list(self._messages)

        parts: list[str] = []
        for m in snapshot:
            if m.role == "user" and m.content:
                parts.append(f"User: {m.content}")
            elif m.role == "assistant" and m.content:
                parts.append(f"Assistant: {m.content}")
        return "\n".join(parts)

    @property
    def turn_count(self) -> int:
        """Number of user messages in this session."""
        with self._lock:
            return sum(1 for m in self._messages if m.role == "user")

    def _trim(self) -> None:
        """Trim history to max_history. Caller must hold self._lock."""
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history :]
