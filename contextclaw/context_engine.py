from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from .chat.session import ChatSession

DEFAULT_MEMORY_POLICY: dict[str, Any] = {
    "mode": "review_queue",
    "context_window_tokens": 24000,
    "reserve_tokens": 4000,
    "warn_threshold": 0.70,
    "compact_threshold": 0.85,
    "force_threshold": 0.92,
    "recent_messages_target_tokens": 6000,
    "keep_recent_turns": 2,
    "working_memory_target_tokens": 1500,
    "max_section_items": 10,
    "message_preview_chars": 220,
    "durable_recall_target_tokens": 1500,
    "durable_recall_max_memories": 6,
    "durable_recall_summary_only": True,
}


def estimate_text_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    # Conservative heuristic: roughly 4 chars per token with a small floor.
    return max(1, math.ceil(len(stripped) / 4))


def estimate_payload_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return estimate_text_tokens(value)
    return estimate_text_tokens(json.dumps(value, ensure_ascii=True, sort_keys=True))


def normalize_memory_policy(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_MEMORY_POLICY)
    if policy:
        merged.update(policy)

    def _normalized_ratio(value: Any, fallback: float) -> float:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return fallback
        if 0.0 < ratio < 1.0:
            return ratio
        return fallback

    merged["warn_threshold"] = _normalized_ratio(
        merged.get("warn_threshold"), DEFAULT_MEMORY_POLICY["warn_threshold"]
    )
    merged["compact_threshold"] = _normalized_ratio(
        merged.get("compact_threshold"), DEFAULT_MEMORY_POLICY["compact_threshold"]
    )
    merged["force_threshold"] = _normalized_ratio(
        merged.get("force_threshold"), DEFAULT_MEMORY_POLICY["force_threshold"]
    )
    for key in (
        "context_window_tokens",
        "reserve_tokens",
        "recent_messages_target_tokens",
        "keep_recent_turns",
        "working_memory_target_tokens",
        "max_section_items",
        "message_preview_chars",
        "durable_recall_target_tokens",
        "durable_recall_max_memories",
    ):
        try:
            merged[key] = int(merged.get(key, DEFAULT_MEMORY_POLICY[key]))
        except (TypeError, ValueError):
            merged[key] = int(DEFAULT_MEMORY_POLICY[key])
    durable_recall_summary_only = merged.get(
        "durable_recall_summary_only",
        DEFAULT_MEMORY_POLICY["durable_recall_summary_only"],
    )
    if isinstance(durable_recall_summary_only, bool):
        merged["durable_recall_summary_only"] = durable_recall_summary_only
    elif isinstance(durable_recall_summary_only, str):
        lowered = durable_recall_summary_only.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            merged["durable_recall_summary_only"] = True
        elif lowered in {"false", "0", "no", "off"}:
            merged["durable_recall_summary_only"] = False
        else:
            merged["durable_recall_summary_only"] = DEFAULT_MEMORY_POLICY[
                "durable_recall_summary_only"
            ]
    else:
        merged["durable_recall_summary_only"] = DEFAULT_MEMORY_POLICY[
            "durable_recall_summary_only"
        ]

    if merged["reserve_tokens"] >= merged["context_window_tokens"]:
        merged["reserve_tokens"] = min(
            DEFAULT_MEMORY_POLICY["reserve_tokens"],
            max(512, merged["context_window_tokens"] // 4),
        )

    thresholds = (
        merged["warn_threshold"],
        merged["compact_threshold"],
        merged["force_threshold"],
    )
    if not (thresholds[0] < thresholds[1] < thresholds[2] < 1.0):
        merged["warn_threshold"] = DEFAULT_MEMORY_POLICY["warn_threshold"]
        merged["compact_threshold"] = DEFAULT_MEMORY_POLICY["compact_threshold"]
        merged["force_threshold"] = DEFAULT_MEMORY_POLICY["force_threshold"]

    return merged


class ContextController:
    def __init__(self, workspace: Path, *, memory_policy: dict[str, Any] | None = None):
        self.workspace = workspace.resolve()
        self.runtime_dir = self.workspace / ".contextclaw"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.policy = normalize_memory_policy(memory_policy)
        self.working_memory_path = self.runtime_dir / "working_memory.json"
        self.pending_compact_path = self.runtime_dir / "pending_compact.json"
        self.compactions_dir = self.runtime_dir / "compactions"
        self.compactions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_working_memory(self) -> dict[str, Any] | None:
        return self._read_json(self.working_memory_path)

    def load_pending_compact(self) -> dict[str, Any] | None:
        return self._read_json(self.pending_compact_path)

    def reject_pending_compact(self) -> dict[str, Any]:
        existed = self.pending_compact_path.exists()
        if existed:
            self.pending_compact_path.unlink()
        return {"rejected": existed, "pending_path": str(self.pending_compact_path)}

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def _render_working_memory(self, payload: dict[str, Any] | None) -> str:
        if not payload:
            return ""
        sections = payload.get("sections", {})
        if not isinstance(sections, dict):
            return ""
        lines = [
            "## Working Memory",
            "Use this as compacted context from earlier turns. Prefer it over replaying stale history.",
            "",
        ]
        current_goal = str(sections.get("current_goal", "")).strip()
        if current_goal:
            lines.append(f"Current Goal: {current_goal}")
            lines.append("")
        for key in (
            "constraints",
            "important_user_requests",
            "assistant_conclusions",
            "tool_outcomes",
            "open_questions",
            "pinned_items",
        ):
            items = sections.get(key, [])
            if not isinstance(items, list) or not items:
                continue
            heading = key.replace("_", " ").title()
            lines.append(f"{heading}:")
            for item in items:
                item_text = str(item).strip()
                if item_text:
                    lines.append(f"- {item_text}")
            lines.append("")
        return "\n".join(line for line in lines if line is not None).strip()

    def compose_system_prompt(
        self,
        base_prompt: str,
        working_memory: dict[str, Any] | None = None,
        extra_context_text: str = "",
    ) -> str:
        rendered_memory = self._render_working_memory(
            working_memory if working_memory is not None else self.load_working_memory()
        )
        sections = [
            section.strip()
            for section in (base_prompt, rendered_memory, extra_context_text)
            if section and section.strip()
        ]
        return "\n\n".join(sections)

    def _message_token_estimate(self, message: dict[str, Any]) -> int:
        total = estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
        total += estimate_payload_tokens(message.get("tool_calls", []))
        total += estimate_text_tokens(str(message.get("tool_call_id", "")))
        return total

    def _session_hash(self, messages: list[dict[str, Any]]) -> str:
        encoded = json.dumps(messages, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _budget_status(self, total_tokens: int, available_tokens: int) -> str:
        if available_tokens <= 0:
            return "over_budget"
        ratio = total_tokens / available_tokens
        if ratio >= 1.0:
            return "over_budget"
        if ratio >= self.policy["force_threshold"]:
            return "force_compact"
        if ratio >= self.policy["compact_threshold"]:
            return "suggest_compact"
        if ratio >= self.policy["warn_threshold"]:
            return "warn"
        return "healthy"

    def inspect_state(
        self,
        session_payload: dict[str, Any],
        *,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        extra_context_text: str = "",
    ) -> dict[str, Any]:
        session_data = session_payload.get("session", {})
        messages = list(session_data.get("messages", []))
        working_memory = self.load_working_memory()
        rendered_working_memory = self._render_working_memory(working_memory)

        base_system_tokens = estimate_text_tokens(system_prompt)
        working_memory_tokens = estimate_text_tokens(rendered_working_memory)
        extra_context_tokens = estimate_text_tokens(extra_context_text)
        message_tokens = sum(
            self._message_token_estimate(message)
            for message in messages
            if isinstance(message, dict)
        )
        tools_tokens = estimate_payload_tokens(tools or [])
        total_tokens = (
            base_system_tokens
            + working_memory_tokens
            + extra_context_tokens
            + message_tokens
            + tools_tokens
        )
        available_tokens = (
            self.policy["context_window_tokens"] - self.policy["reserve_tokens"]
        )
        status = self._budget_status(total_tokens, available_tokens)
        ratio = total_tokens / available_tokens if available_tokens > 0 else 1.0
        return {
            "status": status,
            "ratio": round(ratio, 4),
            "context_window_tokens": self.policy["context_window_tokens"],
            "reserve_tokens": self.policy["reserve_tokens"],
            "available_tokens": available_tokens,
            "total_tokens": total_tokens,
            "base_system_tokens": base_system_tokens,
            "working_memory_tokens": working_memory_tokens,
            "extra_context_tokens": extra_context_tokens,
            "message_tokens": message_tokens,
            "tools_tokens": tools_tokens,
            "message_count": len(messages),
            "working_memory_path": str(self.working_memory_path),
            "pending_compact_path": str(self.pending_compact_path),
            "working_memory_exists": bool(working_memory),
            "session_hash": self._session_hash(
                [message for message in messages if isinstance(message, dict)]
            ),
        }

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _message_preview(self, message: dict[str, Any]) -> str:
        role = str(message.get("role", "")).strip() or "message"
        content = " ".join(str(message.get("content", "")).split())
        limit = self.policy["message_preview_chars"]
        if len(content) > limit:
            content = content[: limit - 3] + "..."
        if not content and message.get("tool_calls"):
            content = json.dumps(message.get("tool_calls", []), ensure_ascii=True)
        return f"{role.title()}: {content}".strip()

    def _dedupe_items(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            normalized = " ".join(item.split()).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result[: self.policy["max_section_items"]]

    def _suffix_start_for_recent_messages(self, messages: list[dict[str, Any]]) -> int:
        if not messages:
            return 0

        keep_turns = max(1, self.policy["keep_recent_turns"])
        user_indexes = [
            index
            for index, message in enumerate(messages)
            if str(message.get("role", "")) == "user"
        ]
        turn_start = 0
        if user_indexes:
            turn_start = user_indexes[max(0, len(user_indexes) - keep_turns)]

        token_target = max(512, self.policy["recent_messages_target_tokens"])
        token_total = 0
        token_start = len(messages) - 1
        for index in range(len(messages) - 1, -1, -1):
            token_total += self._message_token_estimate(messages[index])
            token_start = index
            if token_total >= token_target and index < len(messages) - 1:
                break

        return min(max(turn_start, token_start), len(messages))

    def _coerce_working_memory_sections(
        self,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sections = payload.get("sections", {}) if isinstance(payload, dict) else {}
        if not isinstance(sections, dict):
            sections = {}
        result = {
            "current_goal": str(sections.get("current_goal", "")).strip(),
            "constraints": list(sections.get("constraints", []))
            if isinstance(sections.get("constraints", []), list)
            else [],
            "important_user_requests": list(sections.get("important_user_requests", []))
            if isinstance(sections.get("important_user_requests", []), list)
            else [],
            "assistant_conclusions": list(sections.get("assistant_conclusions", []))
            if isinstance(sections.get("assistant_conclusions", []), list)
            else [],
            "tool_outcomes": list(sections.get("tool_outcomes", []))
            if isinstance(sections.get("tool_outcomes", []), list)
            else [],
            "open_questions": list(sections.get("open_questions", []))
            if isinstance(sections.get("open_questions", []), list)
            else [],
            "pinned_items": list(sections.get("pinned_items", []))
            if isinstance(sections.get("pinned_items", []), list)
            else [],
        }
        return result

    def preview_compact(
        self,
        session_payload: dict[str, Any],
        *,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        extra_context_text: str = "",
        reason: str = "manual",
        persist: bool = True,
    ) -> dict[str, Any]:
        session_data = session_payload.get("session", {})
        messages = [
            message
            for message in list(session_data.get("messages", []))
            if isinstance(message, dict)
        ]
        before = self.inspect_state(
            session_payload,
            system_prompt=system_prompt,
            tools=tools,
            extra_context_text=extra_context_text,
        )

        if not messages:
            preview = {
                "can_apply": False,
                "reason": "No session messages to compact.",
                "budget_before": before,
                "budget_after": before,
                "session_hash": before["session_hash"],
                "trimmed_message_count": 0,
                "trimmed_previews": [],
                "kept_message_count": 0,
                "sections": self._coerce_working_memory_sections(
                    self.load_working_memory()
                ),
                "working_memory_preview": self._render_working_memory(
                    self.load_working_memory()
                ),
                "status": before["status"],
            }
            if persist:
                self._write_json(self.pending_compact_path, preview)
            return preview

        keep_from = self._suffix_start_for_recent_messages(messages)
        compacted_messages = messages[:keep_from]
        kept_messages = messages[keep_from:]
        existing = self._coerce_working_memory_sections(self.load_working_memory())

        user_requests = list(existing["important_user_requests"])
        assistant_conclusions = list(existing["assistant_conclusions"])
        tool_outcomes = list(existing["tool_outcomes"])
        constraints = list(existing["constraints"])
        open_questions = list(existing["open_questions"])

        for message in compacted_messages:
            role = str(message.get("role", ""))
            preview = self._message_preview(message)
            if role == "user":
                user_requests.append(preview)
                lowered = preview.lower()
                if any(
                    keyword in lowered
                    for keyword in ("must", "cannot", "can't", "never", "always")
                ):
                    constraints.append(preview)
            elif role == "assistant":
                assistant_conclusions.append(preview)
            elif role == "tool":
                tool_outcomes.append(preview)

        latest_user_messages = [
            " ".join(str(message.get("content", "")).split())
            for message in reversed(messages)
            if str(message.get("role", "")) == "user"
            and str(message.get("content", "")).strip()
        ]
        current_goal = (
            latest_user_messages[0][: self.policy["message_preview_chars"]]
            if latest_user_messages
            else existing["current_goal"]
        )
        if (
            latest_user_messages
            and messages
            and str(messages[-1].get("role", "")) == "user"
        ):
            open_questions.insert(
                0, latest_user_messages[0][: self.policy["message_preview_chars"]]
            )

        sections = {
            "current_goal": current_goal,
            "constraints": self._dedupe_items(constraints),
            "important_user_requests": self._dedupe_items(user_requests),
            "assistant_conclusions": self._dedupe_items(assistant_conclusions),
            "tool_outcomes": self._dedupe_items(tool_outcomes),
            "open_questions": self._dedupe_items(open_questions),
            "pinned_items": self._dedupe_items(list(existing["pinned_items"])),
        }

        working_memory = {
            "schema_version": 1,
            "updated_at": time.time(),
            "reason": reason,
            "source_session_hash": before["session_hash"],
            "trimmed_message_count": len(compacted_messages),
            "kept_message_count": len(kept_messages),
            "sections": sections,
        }

        rendered_preview = self._render_working_memory(working_memory)
        after_session_payload = {
            "session": {
                **session_data,
                "messages": kept_messages,
            }
        }
        after = self.inspect_state(
            after_session_payload,
            system_prompt=system_prompt,
            tools=tools,
            extra_context_text=extra_context_text,
        )
        after["working_memory_tokens"] = estimate_text_tokens(rendered_preview)
        after["total_tokens"] = (
            after["base_system_tokens"]
            + after["working_memory_tokens"]
            + after["extra_context_tokens"]
            + after["message_tokens"]
            + after["tools_tokens"]
        )
        after["status"] = self._budget_status(
            after["total_tokens"], after["available_tokens"]
        )
        after["ratio"] = round(
            after["total_tokens"] / after["available_tokens"]
            if after["available_tokens"] > 0
            else 1.0,
            4,
        )

        preview = {
            "schema_version": 1,
            "can_apply": True,
            "reason": reason,
            "session_hash": before["session_hash"],
            "budget_before": before,
            "budget_after": after,
            "trimmed_message_count": len(compacted_messages),
            "trimmed_previews": [
                self._message_preview(message) for message in compacted_messages[-20:]
            ],
            "kept_message_count": len(kept_messages),
            "sections": sections,
            "working_memory_preview": rendered_preview,
            "status": after["status"],
        }
        if persist:
            self._write_json(self.pending_compact_path, preview)
        return preview

    def _artifact_markdown(
        self,
        *,
        preview: dict[str, Any],
        artifact_path: Path,
    ) -> str:
        lines = [
            "# Context Compaction Artifact",
            "",
            f"Artifact: {artifact_path.name}",
            f"Reason: {preview['reason']}",
            f"Created At: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
            "",
            "## Budget",
            "",
            f"- Before: {preview['budget_before']['total_tokens']} / {preview['budget_before']['available_tokens']} ({preview['budget_before']['status']})",
            f"- After: {preview['budget_after']['total_tokens']} / {preview['budget_after']['available_tokens']} ({preview['budget_after']['status']})",
            f"- Trimmed Messages: {preview['trimmed_message_count']}",
            f"- Kept Messages: {preview['kept_message_count']}",
            "",
            "## Working Memory",
            "",
            preview["working_memory_preview"] or "_empty_",
            "",
            "## Compacted Message Previews",
            "",
        ]
        previews = preview.get("trimmed_previews", [])
        if previews:
            lines.extend(f"- {item}" for item in previews)
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)

    def apply_compact(
        self,
        session_payload: dict[str, Any],
        *,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        extra_context_text: str = "",
        reason: str = "manual",
        preview: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current_hash = self.inspect_state(
            session_payload,
            system_prompt=system_prompt,
            tools=tools,
            extra_context_text=extra_context_text,
        )["session_hash"]
        candidate = preview or self.load_pending_compact()
        if candidate is None or candidate.get("session_hash") != current_hash:
            candidate = self.preview_compact(
                session_payload,
                system_prompt=system_prompt,
                tools=tools,
                extra_context_text=extra_context_text,
                reason=reason,
                persist=False,
            )
        if not candidate.get("can_apply"):
            raise ValueError(candidate.get("reason", "Compaction cannot be applied"))

        session_data = session_payload.get("session", {})
        messages = [
            message
            for message in list(session_data.get("messages", []))
            if isinstance(message, dict)
        ]
        keep_from = len(messages) - int(candidate["kept_message_count"])
        keep_from = max(0, keep_from)
        kept_messages = messages[keep_from:]

        artifact_name = (
            f"compact_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_"
            f"{candidate['session_hash'][:8]}.md"
        )
        artifact_path = self.compactions_dir / artifact_name
        artifact_path.write_text(
            self._artifact_markdown(preview=candidate, artifact_path=artifact_path),
            encoding="utf-8",
        )

        working_memory = {
            "schema_version": 1,
            "updated_at": time.time(),
            "artifact_path": str(artifact_path),
            "reason": reason,
            "source_session_hash": candidate["session_hash"],
            "trimmed_message_count": candidate["trimmed_message_count"],
            "kept_message_count": candidate["kept_message_count"],
            "sections": candidate["sections"],
        }
        self._write_json(self.working_memory_path, working_memory)
        if self.pending_compact_path.exists():
            self.pending_compact_path.unlink()

        updated_session = {
            **session_data,
            "messages": kept_messages,
        }
        result = {
            "schema_version": 1,
            "reason": reason,
            "artifact_path": str(artifact_path),
            "working_memory_path": str(self.working_memory_path),
            "working_memory": working_memory,
            "session": updated_session,
            "budget_before": candidate["budget_before"],
            "budget_after": candidate["budget_after"],
            "trimmed_message_count": candidate["trimmed_message_count"],
            "kept_message_count": candidate["kept_message_count"],
        }
        return result

    # ------------------------------------------------------------------
    # Convenience helpers for ChatSession / checkpoints
    # ------------------------------------------------------------------

    def session_payload_from_chat(
        self,
        session: ChatSession,
        *,
        total_usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return {
            "session": session.to_dict(),
            "total_usage": total_usage or {},
            "saved_at": time.time(),
        }

    def apply_compact_to_chat_session(
        self,
        session: ChatSession,
        *,
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
        extra_context_text: str = "",
        total_usage: dict[str, int] | None = None,
        reason: str = "manual",
        preview: dict[str, Any] | None = None,
    ) -> tuple[ChatSession, dict[str, Any]]:
        payload = self.session_payload_from_chat(session, total_usage=total_usage)
        result = self.apply_compact(
            payload,
            system_prompt=system_prompt,
            tools=tools,
            extra_context_text=extra_context_text,
            reason=reason,
            preview=preview,
        )
        updated_session = ChatSession.from_dict(
            result["session"],
            system_prompt=session.system_prompt,
            max_history=session.max_history,
        )
        return updated_session, result
