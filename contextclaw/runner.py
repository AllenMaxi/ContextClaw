from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .chat.session import ChatSession
from .config.agent_config import AgentConfig
from .providers.protocol import LLMProvider, LLMResponse, ToolCall
from .tools.manager import ToolManager

logger = logging.getLogger(__name__)

# Defaults — override via AgentRunner constructor kwargs
_DEFAULT_MAX_TURNS = 20
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
_DEFAULT_MIN_CALL_INTERVAL = 0.5  # seconds between LLM calls (rate limit)


@dataclass(slots=True)
class Event:
    """Event emitted by the runner during execution."""

    type: str  # "text" | "tool_call" | "tool_result" | "error" | "done" | "knowledge_recalled"
    data: dict[str, Any] = field(default_factory=dict)


class AgentRunner:
    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        sandbox: Any | None = None,
        tools: ToolManager | None = None,
        knowledge: Any | None = None,
        policy: Any | None = None,
        *,
        max_turns: int = _DEFAULT_MAX_TURNS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_base_delay: float = _DEFAULT_RETRY_BASE_DELAY,
        min_call_interval: float = _DEFAULT_MIN_CALL_INTERVAL,
    ) -> None:
        self.config = config
        self.provider = provider
        self.sandbox = sandbox
        self.tools = tools or ToolManager()
        self.knowledge = knowledge
        self.policy = policy
        self._max_turns = max_turns
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._min_call_interval = min_call_interval
        self._last_call_time: float = 0.0
        self._total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        # Load SOUL.md system prompt
        self._system_prompt = ""
        if config.soul_path and config.soul_path.exists():
            from .config.soul import load_soul

            soul = load_soul(config.soul_path)
            self._system_prompt = soul.body

        self.session = ChatSession(system_prompt=self._system_prompt)

    @property
    def total_usage(self) -> dict[str, int]:
        """Cumulative token usage across all provider calls this session."""
        return dict(self._total_usage)

    # ------------------------------------------------------------------
    # Provider call with retry
    # ------------------------------------------------------------------

    async def _call_provider(self) -> LLMResponse:
        """Call the LLM provider with exponential backoff retry.

        Retries on transient errors (ConnectionError, TimeoutError, OSError).
        Raises immediately on non-transient errors (ValueError, TypeError).
        """
        # Rate limiting: enforce minimum interval between LLM calls
        elapsed = time.monotonic() - self._last_call_time
        if self._min_call_interval > 0 and elapsed < self._min_call_interval:
            await asyncio.sleep(self._min_call_interval - elapsed)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                self._last_call_time = time.monotonic()
                response = await asyncio.to_thread(
                    self.provider.complete,
                    self.session.get_messages(),
                    self.tools.list_tools(),
                    self._system_prompt,
                )
                # Track token usage
                for key in ("input_tokens", "output_tokens"):
                    self._total_usage[key] += response.usage.get(key, 0)
                return response
            except (ConnectionError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    delay = self._retry_base_delay * (2**attempt)
                    logger.warning(
                        "Provider call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Provider call failed after %d attempts: %s",
                        self._max_retries,
                        exc,
                    )
            except (ValueError, TypeError) as exc:
                # Non-transient — don't retry
                logger.error("Provider call failed with non-transient error: %s", exc)
                raise

        raise ConnectionError(
            f"Provider unreachable after {self._max_retries} attempts"
        ) from last_exc

    # ------------------------------------------------------------------
    # Tool validation
    # ------------------------------------------------------------------

    def _validate_tool_call(self, tc: ToolCall) -> str | None:
        """Validate a tool call from the LLM. Returns error message or None."""
        known = self.tools.get_tool(tc.name)
        if known is None and tc.name != "shell_execute":
            # shell_execute is handled specially, always valid
            return f"Unknown tool '{tc.name}'. Available: {[t['name'] for t in self.tools.list_tools()]}"
        if not isinstance(tc.arguments, dict):
            return f"Tool '{tc.name}' arguments must be a dict, got {type(tc.arguments).__name__}"
        return None

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    async def run(self, message: str) -> AsyncIterator[Event]:
        """Run a single user message through the ReAct loop.

        1. Add user message to session
        2. Recall knowledge from ContextGraph if available
        3. ReAct loop: LLM -> tool calls -> execute -> feed results back
        4. Auto-store significant outputs to ContextGraph
        5. Yield events throughout
        """
        self.session.add_user(message)

        # Recall relevant knowledge using richer context after first turn
        if self.knowledge and self.knowledge.auto_recall:
            recall_query = message
            last_assistant = self._get_last_assistant_content()
            if last_assistant:
                recall_query = f"{last_assistant}\n{message}"

            try:
                memories = self.knowledge.recall(recall_query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Knowledge recall failed: %s", exc)
                memories = []

            if memories:
                yield Event(type="knowledge_recalled", data={"memories": memories})
                context = "\n".join(f"- {m.get('content', '')}" for m in memories)
                self.session.add_user(f"[Recalled knowledge]\n{context}")

        # ReAct loop
        for turn in range(self._max_turns):
            logger.debug("ReAct turn %d/%d", turn + 1, self._max_turns)

            try:
                response = await self._call_provider()
            except (ConnectionError, TimeoutError) as exc:
                logger.error("Provider unavailable: %s", exc)
                yield Event(type="error", data={"message": f"Provider error: {exc}"})
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected provider error: %s", exc)
                yield Event(type="error", data={"message": f"Provider error: {exc}"})
                return

            if response.content:
                yield Event(type="text", data={"content": response.content})

            if not response.tool_calls:
                # No tool calls — we're done
                self.session.add_assistant(response.content)

                # Auto-store to ContextGraph
                if self.knowledge and response.content:
                    try:
                        self.knowledge.store(response.content)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Knowledge store failed: %s", exc)

                yield Event(type="done", data={"content": response.content})
                return

            # Record assistant turn with tool calls
            self.session.add_assistant(
                response.content,
                tool_calls=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
            )

            for tc in response.tool_calls:
                yield Event(
                    type="tool_call",
                    data={"id": tc.id, "name": tc.name, "arguments": tc.arguments},
                )

                # Validate tool call
                validation_error = self._validate_tool_call(tc)
                if validation_error:
                    logger.warning("Invalid tool call: %s", validation_error)
                    self.session.add_tool_result(tc.id, f"Error: {validation_error}")
                    yield Event(
                        type="tool_result",
                        data={"id": tc.id, "result": f"Error: {validation_error}"},
                    )
                    continue

                # Policy check
                if self.policy:
                    decision = self.policy.check_tool(tc.name)
                    if decision == "block":
                        result = f"Tool '{tc.name}' is blocked by policy."
                        logger.info("Blocked tool call: %s", tc.name)
                        self.session.add_tool_result(tc.id, result)
                        yield Event(
                            type="tool_result", data={"id": tc.id, "result": result}
                        )
                        continue

                # Execute tool
                result = await self._execute_tool(tc)
                self.session.add_tool_result(tc.id, result)
                yield Event(type="tool_result", data={"id": tc.id, "result": result})

        yield Event(type="error", data={"message": "Max turns reached"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_last_assistant_content(self) -> str:
        """Return the last assistant message content, or empty string."""
        return self.session.last_assistant_message

    async def close_session(self) -> list[dict]:
        """Summarize and store session knowledge to ContextGraph.

        Call this when the chat session ends (user exits, server shuts down).
        Returns list of stored memories, empty if nothing worth storing.
        """
        if not self.knowledge or not self.knowledge.agent_id:
            return []

        if self.session.turn_count < 2:
            return []

        context = self.session.get_summary_context()
        if not context.strip():
            return []

        try:
            stored = self.knowledge.summarize_and_store(
                conversation_context=context,
                provider=self.provider,
                agent_name=self.config.name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Session summarization failed: %s", exc)
            return []

        logger.info("Stored %d memories from session", len(stored))
        return stored

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call. For shell_execute, use sandbox. Otherwise return a stub."""
        if tool_call.name == "shell_execute" and self.sandbox:
            command = tool_call.arguments.get("command", "")
            if not command:
                return "Error: shell_execute requires a 'command' argument"
            try:
                result = await self.sandbox.execute(command)
            except Exception as exc:  # noqa: BLE001
                logger.error("Sandbox execution error: %s", exc)
                return f"Error executing command: {exc}"
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr] {result.stderr}"
            if result.timed_out:
                output += "\n[timed out]"
            return output

        # Extensible — specific tool implementations would be wired in here
        return json.dumps(
            {
                "status": "ok",
                "tool": tool_call.name,
                "note": "Tool execution not implemented",
            }
        )
