from __future__ import annotations

from pathlib import Path

from contextclaw.chat.session import ChatSession
from contextclaw.context_engine import ContextController, normalize_memory_policy


def _seed_session() -> ChatSession:
    session = ChatSession(system_prompt="Base system prompt.", max_history=0)
    for index in range(3):
        session.add_user(f"User request {index}: " + ("x" * 220))
        session.add_assistant(f"Assistant conclusion {index}: " + ("y" * 180))
    session.add_tool_result("tool_1", "Command output: build completed successfully.")
    return session


def test_chat_session_does_not_trim_when_max_history_is_zero() -> None:
    session = ChatSession(max_history=0)
    for index in range(150):
        session.add_user(f"message {index}")

    assert len(session.get_messages()) == 150


def test_context_controller_preview_and_apply_compaction(tmp_path: Path) -> None:
    controller = ContextController(
        tmp_path,
        memory_policy={
            "recent_messages_target_tokens": 300,
            "keep_recent_turns": 1,
            "message_preview_chars": 100,
        },
    )
    session = _seed_session()

    preview = controller.preview_compact(
        controller.session_payload_from_chat(session),
        system_prompt="Base system prompt.",
        tools=[{"name": "filesystem_read"}],
        reason="test_preview",
    )

    assert preview["can_apply"] is True
    assert preview["trimmed_message_count"] > 0
    assert controller.pending_compact_path.exists()
    assert "## Working Memory" in preview["working_memory_preview"]

    updated_session, result = controller.apply_compact_to_chat_session(
        session,
        system_prompt="Base system prompt.",
        tools=[{"name": "filesystem_read"}],
        reason="test_apply",
    )

    assert len(updated_session.get_messages()) == result["kept_message_count"]
    assert controller.pending_compact_path.exists() is False
    assert controller.working_memory_path.exists()
    assert Path(result["artifact_path"]).exists()
    assert "## Working Memory" in controller.compose_system_prompt("Base")


def test_memory_policy_normalizes_durable_recall_settings() -> None:
    policy = normalize_memory_policy(
        {
            "durable_recall_target_tokens": "2048",
            "durable_recall_max_memories": "4",
            "durable_recall_summary_only": "false",
        }
    )

    assert policy["durable_recall_target_tokens"] == 2048
    assert policy["durable_recall_max_memories"] == 4
    assert policy["durable_recall_summary_only"] is False
