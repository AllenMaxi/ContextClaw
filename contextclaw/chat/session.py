from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_call_id: str = ""
    tool_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_call_id": self.tool_call_id,
            "tool_calls": list(self.tool_calls),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=str(data.get("role", "")),
            content=str(data.get("content", "")),
            timestamp=float(data.get("timestamp", time.time())),
            tool_call_id=str(data.get("tool_call_id", "")),
            tool_calls=list(data.get("tool_calls", [])),
        )


class ChatSession:
    """Manages conversation history for an agent chat session.

    Thread-safe: all mutations to the message list are guarded by a lock.
    """

    def __init__(self, system_prompt: str = "", max_history: int = 0) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            snapshot = list(self._messages)
        return {
            "system_prompt": self.system_prompt,
            "max_history": self.max_history,
            "messages": [message.to_dict() for message in snapshot],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        system_prompt: str = "",
        max_history: int | None = None,
    ) -> ChatSession:
        session = cls(
            system_prompt=system_prompt or str(data.get("system_prompt", "")),
            max_history=(
                int(max_history)
                if max_history is not None
                else int(data.get("max_history", 0))
            ),
        )
        session._messages = [
            Message.from_dict(item)
            for item in list(data.get("messages", []))
            if isinstance(item, dict)
        ]
        session._trim()
        return session

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        system_prompt: str = "",
        max_history: int | None = None,
    ) -> ChatSession:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Session checkpoint must contain a JSON object")
        return cls.from_dict(data, system_prompt=system_prompt, max_history=max_history)

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
        if self.max_history > 0 and len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history :]
