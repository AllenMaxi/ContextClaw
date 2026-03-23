from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .chat.session import ChatSession
from .config.agent_config import AgentConfig
from .providers.protocol import LLMProvider, LLMResponse, ToolCall
from .tools.manager import ToolDefinition, ToolManager

logger = logging.getLogger(__name__)

# Defaults — override via AgentRunner constructor kwargs
_DEFAULT_MAX_TURNS = 20
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
_DEFAULT_MIN_CALL_INTERVAL = 0.5  # seconds between LLM calls (rate limit)

_BUILTIN_TOOL_ALIASES = {
    "read_file": "filesystem_read",
    "write_file": "filesystem_write",
    "ls": "filesystem_list",
    "execute": "shell_execute",
}


@dataclass(slots=True)
class Event:
    """Event emitted by the runner during execution."""

    type: str  # "text" | "tool_call" | "tool_result" | "error" | "done" | "knowledge_recalled"
    data: dict[str, Any] = field(default_factory=dict)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)

    def text(self) -> str:
        return "\n".join(self.parts)


class AgentRunner:
    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        sandbox: Any | None = None,
        tools: ToolManager | None = None,
        knowledge: Any | None = None,
        policy: Any | None = None,
        tool_approver: Any | None = None,
        provider_factory: Any | None = None,
        delegation_depth: int = 0,
        max_delegation_depth: int = 2,
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
        self.tool_approver = tool_approver
        self.provider_factory = provider_factory
        self.delegation_depth = delegation_depth
        self.max_delegation_depth = max_delegation_depth
        self._max_turns = max_turns
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._min_call_interval = min_call_interval
        self._last_call_time: float = 0.0
        self._total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        # Load SOUL.md system prompt plus optional skills appendix
        self._system_prompt = ""
        if config.soul_path and config.soul_path.exists():
            from .config.soul import load_soul

            soul = load_soul(config.soul_path)
            self._system_prompt = soul.body

        from .config.skills import render_skills_prompt

        skills_prompt = render_skills_prompt(config.skills_path)
        if skills_prompt:
            if self._system_prompt:
                self._system_prompt += "\n\n"
            self._system_prompt += skills_prompt

        self._checkpoint_path = config.checkpoint_path
        self.session = self._load_or_create_session()
        self._subagents = self._discover_subagents()
        self._register_task_tool_if_needed()

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
        if known is None and self._canonical_tool_name(tc.name) != "shell_execute":
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
        self._save_checkpoint()

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
                self._save_checkpoint()

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
                self._save_checkpoint()

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
            self._save_checkpoint()

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
                    decision = self.policy.check_tool(
                        self._canonical_tool_name(tc.name)
                    )
                    if decision == "block":
                        result = f"Tool '{tc.name}' is blocked by policy."
                        logger.info("Blocked tool call: %s", tc.name)
                        self.session.add_tool_result(tc.id, result)
                        yield Event(
                            type="tool_result", data={"id": tc.id, "result": result}
                        )
                        continue
                    if decision == "confirm":
                        approved = await self._request_tool_approval(tc)
                        if approved is None:
                            result = (
                                f"Tool '{tc.name}' requires approval, but no approver "
                                f"is configured."
                            )
                            self.session.add_tool_result(tc.id, result)
                            yield Event(
                                type="tool_result",
                                data={"id": tc.id, "result": result},
                            )
                            continue
                        if not approved:
                            result = f"Tool '{tc.name}' was denied by the operator."
                            self.session.add_tool_result(tc.id, result)
                            yield Event(
                                type="tool_result",
                                data={"id": tc.id, "result": result},
                            )
                            continue

                # Execute tool
                result = await self._execute_tool(tc)
                self.session.add_tool_result(tc.id, result)
                self._save_checkpoint()
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
        self._save_checkpoint()
        return stored

    async def _request_tool_approval(self, tool_call: ToolCall) -> bool | None:
        """Return approval decision for a tool call.

        Returns:
            True when approved, False when denied, None when no approver exists.
        """
        if self.tool_approver is None:
            return None

        decision = self.tool_approver(tool_call)
        if inspect.isawaitable(decision):
            decision = await decision
        return bool(decision)

    def _workspace_path(self, raw_path: str, *, allow_missing: bool = False) -> Path:
        """Resolve a path inside the agent workspace.

        Relative paths are anchored to the workspace. Absolute paths must still
        stay within the workspace root.
        """
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.config.workspace / candidate
        resolved = candidate.resolve(strict=not allow_missing)

        workspace_root = self.config.workspace.resolve()
        try:
            resolved.relative_to(workspace_root)
        except ValueError as exc:
            raise PermissionError(
                "Access denied: filesystem tools may only operate inside the agent workspace."
            ) from exc

        if self.policy and not self.policy.check_path(resolved):
            raise PermissionError("Access denied by policy.")

        return resolved

    def _truncate_text(self, text: str, limit: int = 12000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"

    def _canonical_tool_name(self, name: str) -> str:
        return _BUILTIN_TOOL_ALIASES.get(name, name)

    def _relative_to_workspace(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.config.workspace.resolve()))
        except ValueError:
            return str(path)

    def _load_or_create_session(self) -> ChatSession:
        if self._checkpoint_path and self._checkpoint_path.exists():
            try:
                payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    usage = payload.get("total_usage", {})
                    if isinstance(usage, dict):
                        for key in ("input_tokens", "output_tokens"):
                            self._total_usage[key] = int(usage.get(key, 0))
                    session_data = payload.get("session", {})
                    if isinstance(session_data, dict):
                        return ChatSession.from_dict(
                            session_data,
                            system_prompt=self._system_prompt,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load checkpoint %s: %s", self._checkpoint_path, exc
                )
        return ChatSession(system_prompt=self._system_prompt)

    def _save_checkpoint(self) -> None:
        if not self._checkpoint_path:
            return
        payload = {
            "session": self.session.to_dict(),
            "total_usage": self._total_usage,
            "saved_at": time.time(),
        }
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _discover_subagents(self) -> dict[str, AgentConfig]:
        registry: dict[str, AgentConfig] = {}
        subagents_path = self.config.subagents_path
        if (
            subagents_path is None
            or not subagents_path.exists()
            or not subagents_path.is_dir()
        ):
            return registry

        for child in sorted(subagents_path.iterdir()):
            if not child.is_dir():
                continue
            config_file = child / "config.yaml"
            soul_file = child / "SOUL.md"
            if not config_file.exists() and not soul_file.exists():
                continue
            try:
                subconfig = AgentConfig.from_dir(child)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load subagent from %s: %s", child, exc)
                continue
            registry[subconfig.name] = subconfig
        return registry

    def _subagent_descriptions(self) -> list[str]:
        descriptions: list[str] = []
        for name, subconfig in sorted(self._subagents.items()):
            summary = subconfig.name
            if subconfig.soul_path and subconfig.soul_path.exists():
                try:
                    from .config.soul import load_soul

                    soul = load_soul(subconfig.soul_path)
                    if soul.extra.get("description"):
                        summary = soul.extra["description"]
                    elif soul.body:
                        summary = soul.body.splitlines()[0].strip()
                except Exception:  # noqa: BLE001
                    pass
            descriptions.append(f"{name}: {summary}")
        return descriptions

    def _register_task_tool_if_needed(self) -> None:
        if not self._subagents or self.tools.get_tool("task") is not None:
            return
        description = (
            "Delegate work to a specialized sub-agent with isolated context. "
            "Available subagents: " + "; ".join(self._subagent_descriptions())
        )
        self.tools.register(
            ToolDefinition(
                name="task",
                description=description,
                parameters={
                    "type": "object",
                    "properties": {
                        "subagent": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["subagent", "prompt"],
                },
            )
        )

    def _execute_builtin_tool(self, tool_call: ToolCall) -> str:
        tool_name = self._canonical_tool_name(tool_call.name)

        if tool_name == "filesystem_read":
            path = self._workspace_path(tool_call.arguments.get("path", ""))
            if not path.is_file():
                return f"Error: '{path}' is not a file"
            content = path.read_text(encoding="utf-8", errors="replace")
            offset = tool_call.arguments.get("offset")
            limit = tool_call.arguments.get("limit")
            if offset is not None or limit is not None:
                try:
                    start = max(0, int(offset or 0))
                    span = max(0, int(limit or len(content)))
                except (TypeError, ValueError):
                    return "Error: read_file offset and limit must be integers"
                window = content[start : start + span]
                return json.dumps(
                    {
                        "path": self._relative_to_workspace(path),
                        "offset": start,
                        "limit": span,
                        "content": self._truncate_text(window),
                    },
                    ensure_ascii=True,
                )
            return self._truncate_text(content)

        if tool_name == "filesystem_write":
            path = self._workspace_path(
                tool_call.arguments.get("path", ""),
                allow_missing=True,
            )
            content = tool_call.arguments.get("content", "")
            if path.exists() and path.is_dir():
                return f"Error: '{path}' is a directory"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
            return f"Wrote {len(str(content))} characters to {path}"

        if tool_name == "filesystem_list":
            path_arg = tool_call.arguments.get("path", ".")
            path = self._workspace_path(path_arg)
            if not path.is_dir():
                return f"Error: '{path}' is not a directory"
            entries = []
            for entry in sorted(
                path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            ):
                entries.append(
                    {
                        "name": entry.name,
                        "type": "directory" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else None,
                    }
                )
            return json.dumps(entries, ensure_ascii=True, indent=2)

        if tool_name == "edit_file":
            path = self._workspace_path(tool_call.arguments.get("path", ""))
            if not path.is_file():
                return f"Error: '{path}' is not a file"
            old_text = str(tool_call.arguments.get("old_text", ""))
            new_text = str(tool_call.arguments.get("new_text", ""))
            if not old_text:
                return "Error: edit_file requires a non-empty 'old_text'"
            original = path.read_text(encoding="utf-8", errors="replace")
            match_count = original.count(old_text)
            if match_count == 0:
                return "Error: edit_file could not find 'old_text' in the target file"
            replace_all = bool(tool_call.arguments.get("replace_all", False))
            replacements = match_count if replace_all else 1
            updated = (
                original.replace(old_text, new_text)
                if replace_all
                else original.replace(old_text, new_text, 1)
            )
            path.write_text(updated, encoding="utf-8")
            return json.dumps(
                {
                    "path": self._relative_to_workspace(path),
                    "replacements": replacements,
                },
                ensure_ascii=True,
            )

        if tool_name == "glob":
            path_arg = str(tool_call.arguments.get("path", ".") or ".")
            pattern = str(tool_call.arguments.get("pattern", "**/*") or "**/*").strip()
            if not pattern:
                return "Error: glob requires a non-empty 'pattern'"
            if Path(pattern).is_absolute():
                return "Error: glob patterns must be workspace-relative"
            root = self._workspace_path(path_arg)
            if not root.is_dir():
                return f"Error: '{root}' is not a directory"
            workspace_root = self.config.workspace.resolve()
            glob_matches: list[dict[str, Any]] = []
            glob_truncated = False
            for match in sorted(root.glob(pattern)):
                resolved = match.resolve()
                try:
                    resolved.relative_to(workspace_root)
                except ValueError:
                    continue
                glob_matches.append(
                    {
                        "path": self._relative_to_workspace(resolved),
                        "type": "directory" if resolved.is_dir() else "file",
                    }
                )
                if len(glob_matches) >= 200:
                    glob_truncated = True
                    break
            return json.dumps(
                {
                    "path": self._relative_to_workspace(root),
                    "pattern": pattern,
                    "matches": glob_matches,
                    "truncated": glob_truncated,
                },
                ensure_ascii=True,
            )

        if tool_name == "grep":
            pattern = str(tool_call.arguments.get("pattern", "")).strip()
            if not pattern:
                return "Error: grep requires a non-empty 'pattern'"
            path_arg = str(tool_call.arguments.get("path", ".") or ".")
            include = str(tool_call.arguments.get("include", "**/*") or "**/*").strip()
            case_sensitive = bool(tool_call.arguments.get("case_sensitive", False))
            root = self._workspace_path(path_arg)
            if not root.exists():
                return f"Error: '{root}' does not exist"
            try:
                regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
            except re.error as exc:
                return f"Error: invalid grep pattern: {exc}"

            if root.is_file():
                files = [root]
            else:
                files = [
                    entry for entry in sorted(root.glob(include)) if entry.is_file()
                ]

            grep_matches: list[dict[str, Any]] = []
            grep_truncated = False
            for file_path in files:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        grep_matches.append(
                            {
                                "path": self._relative_to_workspace(file_path),
                                "line_number": line_number,
                                "line": line,
                            }
                        )
                        if len(grep_matches) >= 50:
                            grep_truncated = True
                            break
                if grep_truncated:
                    break

            return json.dumps(
                {
                    "pattern": pattern,
                    "path": self._relative_to_workspace(root),
                    "matches": grep_matches,
                    "truncated": grep_truncated,
                },
                ensure_ascii=True,
            )

        if tool_name == "write_todos":
            items = tool_call.arguments.get("items", [])
            if not isinstance(items, list) or not items:
                return "Error: write_todos requires a non-empty 'items' array"
            normalized_items = [
                str(item).strip() for item in items if str(item).strip()
            ]
            if not normalized_items:
                return "Error: write_todos requires at least one non-empty item"
            heading = (
                str(tool_call.arguments.get("heading", "Task Plan")).strip()
                or "Task Plan"
            )
            path_arg = str(tool_call.arguments.get("path", ".contextclaw/todos.md"))
            path = self._workspace_path(path_arg, allow_missing=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"# {heading}", "", "Generated by ContextClaw.", ""]
            lines.extend(f"- [ ] {item}" for item in normalized_items)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return json.dumps(
                {
                    "path": str(path),
                    "item_count": len(normalized_items),
                },
                ensure_ascii=True,
            )

        if tool_name == "read_todos":
            path_arg = str(
                tool_call.arguments.get("path", ".contextclaw/todos.md")
                or ".contextclaw/todos.md"
            )
            path = self._workspace_path(path_arg, allow_missing=True)
            if not path.exists():
                return json.dumps(
                    {
                        "path": self._relative_to_workspace(path),
                        "exists": False,
                        "items": [],
                        "content": "",
                    },
                    ensure_ascii=True,
                )
            content = path.read_text(encoding="utf-8", errors="replace")
            items = []
            for line in content.splitlines():
                if line.startswith("- [ ] ") or line.startswith("- [x] "):
                    items.append(line[6:])
            return json.dumps(
                {
                    "path": self._relative_to_workspace(path),
                    "exists": True,
                    "items": items,
                    "content": self._truncate_text(content, limit=4000),
                },
                ensure_ascii=True,
            )

        if tool_name == "web_fetch":
            url = str(tool_call.arguments.get("url", "")).strip()
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                return "Error: web_fetch only supports http and https URLs"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "ContextClaw/0.1 (+https://github.com/AllenMaxi/ContextGraph)"
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read(25000)
                final_url = resp.geturl()
                content_type = resp.headers.get(
                    "Content-Type", "application/octet-stream"
                )
            text = raw.decode("utf-8", errors="replace")
            if "html" in content_type:
                parser = _HTMLTextExtractor()
                parser.feed(text)
                text = parser.text() or text
            return json.dumps(
                {
                    "url": final_url,
                    "content_type": content_type,
                    "content": self._truncate_text(text, limit=8000),
                },
                ensure_ascii=True,
            )

        if tool_name == "web_search":
            query = str(tool_call.arguments.get("query", "")).strip()
            if not query:
                return "Error: web_search requires a non-empty 'query'"
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                }
            )
            url = f"https://api.duckduckgo.com/?{params}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "ContextClaw/0.1 (+https://github.com/AllenMaxi/ContextGraph)"
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))

            results: list[dict[str, str]] = []
            if payload.get("AbstractText"):
                results.append(
                    {
                        "title": payload.get("Heading") or query,
                        "snippet": payload.get("AbstractText", ""),
                        "url": payload.get("AbstractURL", ""),
                    }
                )

            for topic in payload.get("RelatedTopics", []):
                if isinstance(topic, dict) and "Topics" in topic:
                    for nested in topic.get("Topics", []):
                        if nested.get("Text"):
                            results.append(
                                {
                                    "title": nested.get("Text", "").split(" - ", 1)[0],
                                    "snippet": nested.get("Text", ""),
                                    "url": nested.get("FirstURL", ""),
                                }
                            )
                elif isinstance(topic, dict) and topic.get("Text"):
                    results.append(
                        {
                            "title": topic.get("Text", "").split(" - ", 1)[0],
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                        }
                    )
                if len(results) >= 5:
                    break

            return json.dumps(
                {
                    "query": query,
                    "engine": "duckduckgo_instant_answer",
                    "results": results[:5],
                },
                ensure_ascii=True,
            )

        raise ValueError(f"Tool '{tool_call.name}' is not implemented")

    async def _execute_task_tool(self, tool_call: ToolCall) -> str:
        if self.delegation_depth >= self.max_delegation_depth:
            return "Error: max task delegation depth reached"

        subagent_name = str(tool_call.arguments.get("subagent", "")).strip()
        prompt = str(tool_call.arguments.get("prompt", "")).strip()
        if not subagent_name or not prompt:
            return "Error: task requires 'subagent' and 'prompt'"
        if subagent_name not in self._subagents:
            available = ", ".join(sorted(self._subagents))
            return f"Error: unknown subagent '{subagent_name}'. Available: {available}"

        subconfig = self._subagents[subagent_name]

        from .runtime import (
            create_knowledge,
            create_policy,
            create_provider,
            create_sandbox,
            create_tools,
        )

        provider = (
            self.provider_factory(subconfig)
            if self.provider_factory is not None
            else create_provider(subconfig)
        )
        sandbox = create_sandbox(subconfig)
        tools = await create_tools(subconfig)
        knowledge = create_knowledge(subconfig)
        policy = create_policy(subconfig)

        subrunner = AgentRunner(
            config=subconfig,
            provider=provider,
            sandbox=sandbox,
            tools=tools,
            knowledge=knowledge,
            policy=policy,
            tool_approver=self.tool_approver,
            provider_factory=self.provider_factory,
            delegation_depth=self.delegation_depth + 1,
            max_delegation_depth=self.max_delegation_depth,
            min_call_interval=self._min_call_interval,
        )

        final_text = ""
        tool_events = 0
        stored = []
        try:
            if sandbox:
                await sandbox.start()
            async for event in subrunner.run(prompt):
                if event.type == "text":
                    final_text = event.data.get("content", final_text)
                elif event.type == "done":
                    final_text = event.data.get("content", final_text)
                elif event.type == "tool_call":
                    tool_events += 1
                elif event.type == "error":
                    final_text = event.data.get("message", final_text)
            stored = await subrunner.close_session()
        finally:
            if sandbox:
                await sandbox.stop()
            await tools.stop_all()

        return json.dumps(
            {
                "subagent": subagent_name,
                "result": final_text,
                "tool_calls": tool_events,
                "stored_memories": len(stored),
            },
            ensure_ascii=True,
        )

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call."""
        tool_name = self._canonical_tool_name(tool_call.name)

        if tool_name == "shell_execute" and self.sandbox:
            command = tool_call.arguments.get("command", "")
            if not command:
                return f"Error: {tool_call.name} requires a 'command' argument"
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

        if tool_name == "shell_execute":
            return f"Error: {tool_call.name} requires a configured sandbox"

        if tool_call.name == "task":
            return await self._execute_task_tool(tool_call)

        if self.tools.is_mcp_tool(tool_call.name):
            try:
                return await self.tools.call_mcp_tool(
                    tool_call.name, tool_call.arguments
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("MCP tool execution error for %s: %s", tool_call.name, exc)
                return f"Error executing MCP tool '{tool_call.name}': {exc}"

        try:
            return await asyncio.to_thread(self._execute_builtin_tool, tool_call)
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool execution error for %s: %s", tool_call.name, exc)
            return f"Error executing tool '{tool_call.name}': {exc}"
