from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import contextclaw.studio.daemon as studio_daemon
from contextclaw.chat.session import ChatSession
from contextclaw.providers.protocol import LLMResponse, ToolCall
from contextclaw.studio.daemon import create_app
from contextclaw.studio.service import StudioService


class ScriptedProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        del messages, tools, system
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="done")


def _wait_for_run(service: StudioService, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = service.get_run(run_id)
        if run and run["status"] in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not finish before timeout")


def _wait_for_no_handles(service: StudioService, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with service._lock:
            if not service._handles:
                return
        time.sleep(0.05)
    raise AssertionError("StudioService still has active run handles after timeout")


def test_studio_service_records_shared_run_state(tmp_path: Path):
    def provider_factory(_config) -> ScriptedProvider:
        return ScriptedProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="mem_1",
                            name="memory_propose",
                            arguments={
                                "content": "Ship ContextClaw Studio from the repo root.",
                                "metadata": {"type": "decision"},
                            },
                        ),
                        ToolCall(
                            id="doc_1",
                            name="docs_propose",
                            arguments={
                                "path": "docs/decisions/studio.md",
                                "content": "# Studio\n\nStudio owns the shared control plane.\n",
                                "summary": "Record the studio architecture.",
                            },
                        ),
                    ],
                ),
                LLMResponse(content="Studio plan captured."),
            ]
        )

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})

    run = service.start_run(
        "Capture the studio decisions.",
        source="cli",
        agent_name="orchestrator",
    )
    result = _wait_for_run(service, run["id"])
    events = service.list_run_events(run["id"])
    memory = service.list_memory_proposals()
    docs = service.list_docs_proposals()

    assert result["status"] == "completed"
    assert any(event["type"] == "message.user" for event in events)
    assert any(event["type"] == "tool.call" for event in events)
    assert any(event["type"] == "memory.proposed" for event in events)
    assert any(event["type"] == "docs.proposed" for event in events)
    assert memory[0]["content"] == "Ship ContextClaw Studio from the repo root."
    assert docs[0]["path"] == "docs/decisions/studio.md"


def test_studio_daemon_bootstraps_project(tmp_path: Path):
    client = TestClient(create_app(StudioService()))

    response = client.post(
        "/projects/init",
        json={
            "root": str(tmp_path),
            "entry_agent": "orchestrator",
            "provider": "openai",
        },
    )
    assert response.status_code == 200
    assert (tmp_path / "Workflow.md").exists()

    agents = client.get("/agents")
    workflow = client.get("/workflow")

    assert agents.status_code == 200
    assert agents.json()[0]["name"] == "orchestrator"
    assert workflow.status_code == 200
    assert workflow.json()["config"]["entry_agent"] == "orchestrator"


def test_studio_status_reports_health_and_project_state(tmp_path: Path):
    service = StudioService()
    client = TestClient(create_app(service))

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "contextclaw-studio"
    assert response.json()["project_open"] is False

    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    response = client.get("/status")
    assert response.json()["project_open"] is True


def test_studio_dashboard_falls_back_to_inline_ui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # Point _frontend_dist_dir to an empty directory so the SPA is not found
    monkeypatch.setattr(
        studio_daemon, "_frontend_dist_dir", lambda: tmp_path / "no_dist"
    )

    client = TestClient(create_app(StudioService()))

    response = client.get("/studio")

    assert response.status_code == 200
    assert "ContextClaw Studio" in response.text
    assert "Run Prompt" in response.text


def test_studio_dashboard_serves_built_frontend_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body><div id='root'>React Studio</div></body></html>",
        encoding="utf-8",
    )
    (assets / "app.js").write_text("console.log('studio');", encoding="utf-8")
    monkeypatch.setattr(studio_daemon, "_frontend_dist_dir", lambda: dist)

    client = TestClient(create_app(StudioService()))

    response = client.get("/studio")
    asset = client.get("/studio/assets/app.js")

    assert response.status_code == 200
    assert "React Studio" in response.text
    assert asset.status_code == 200
    assert "console.log('studio');" in asset.text


def test_studio_live_subscribers_receive_streamed_events(tmp_path: Path):
    import queue as _queue

    def provider_factory(_config) -> ScriptedProvider:
        return ScriptedProvider([LLMResponse(content="streamed result")])

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})
    subscription_id, event_queue = service.subscribe_events()
    try:
        run = service.start_run(
            "Stream this run.",
            source="cli",
            agent_name="orchestrator",
        )

        streamed = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                event = event_queue.get(timeout=0.5)
            except _queue.Empty:
                continue
            if event["type"] == "run.started":
                streamed = event
                break

        assert streamed is not None
        assert streamed["run_id"] == run["id"]
        assert streamed["type"] == "run.started"
        assert streamed["payload"]["agent"] == "orchestrator"

        _wait_for_run(service, run["id"], timeout=8.0)
    finally:
        service.unsubscribe_events(subscription_id)
        service.shutdown(timeout=3.0)


def test_studio_context_endpoints_preview_and_apply_compaction(tmp_path: Path):
    service = StudioService()
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    (tmp_path / "AGENTS.md").write_text(
        "# Project Memory\n\nStudio owns approvals.\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "agents" / "orchestrator"
    (workspace / "MEMORY.md").write_text(
        "# Agent Memory\n\nThe orchestrator prefers concise plans.\n",
        encoding="utf-8",
    )
    checkpoint_path = workspace / ".contextclaw" / "session.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    session = ChatSession(max_history=0)
    for index in range(3):
        session.add_user(f"Decision {index}: " + ("x" * 180))
        session.add_assistant(f"Answer {index}: " + ("y" * 140))
    checkpoint_path.write_text(
        json.dumps({"session": session.to_dict(), "total_usage": {}}, indent=2),
        encoding="utf-8",
    )

    client = TestClient(create_app(service))

    context = client.get("/context")
    preview = client.post(
        "/compact/preview",
        json={"agent_name": "orchestrator", "reason": "test_preview"},
    )
    applied = client.post(
        "/compact/apply",
        json={"agent_name": "orchestrator", "reason": "test_apply"},
    )
    rejected = client.post(
        "/compact/reject",
        json={"agent_name": "orchestrator"},
    )

    assert context.status_code == 200
    assert context.json()["agent"] == "orchestrator"
    assert len(context.json()["memory_files"]) == 2
    assert preview.status_code == 200
    assert preview.json()["preview"]["trimmed_message_count"] > 0
    assert applied.status_code == 200
    assert Path(applied.json()["result"]["artifact_path"]).exists()
    assert rejected.status_code == 200
    assert rejected.json()["rejected"] is False


def test_sync_memory_proposal_preserves_nested_memory_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class FakeBridge:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def store(self, content: str, **kwargs):
            captured["content"] = content
            captured["kwargs"] = kwargs
            return {"memory": {"memory_id": "mem_synced"}}

    monkeypatch.setattr("contextclaw.studio.service.ContextGraphBridge", FakeBridge)

    service = StudioService()
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent(
        "orchestrator",
        config_updates={
            "cg_url": "http://contextgraph.local",
            "agent_id": "agt_test",
        },
    )
    layout, journal = service._ensure_project()
    proposal = journal.create_memory_proposal(
        proposal_id="mem_nested",
        run_id="run_1",
        project_root=layout.root,
        agent_name="orchestrator",
        content="Keep the blue-green rollout plan.",
        metadata={
            "metadata": {
                "source": "session_summary",
                "memory_kind": "summary",
                "summary": "Blue-green rollout",
                "tags": ["session-summary", "deploy"],
                "importance_score": 0.6,
            },
            "evidence": ["note:1"],
            "citations": ["ticket:42"],
        },
    )

    synced = service.sync_memory_proposal(proposal["id"])

    assert synced is not None
    assert synced["status"] == "synced"
    assert synced["synced_memory_id"] == "mem_synced"
    assert captured["content"] == "Keep the blue-green rollout plan."
    assert captured["kwargs"] == {
        "metadata": {"source": "session_summary"},
        "evidence": ["note:1"],
        "citations": ["ticket:42"],
        "memory_kind": "summary",
        "summary": "Blue-green rollout",
        "tags": ["session-summary", "deploy"],
        "importance_score": 0.6,
    }


@pytest.mark.parametrize("exc_type", [ConnectionError, TimeoutError, OSError])
def test_sync_memory_proposal_propagates_transient_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exc_type: type[Exception],
):
    class FailingBridge:
        def __init__(self, **kwargs):
            del kwargs

        def store(self, content: str, **kwargs):
            del content, kwargs
            raise exc_type("ContextGraph temporarily unavailable")

    monkeypatch.setattr("contextclaw.studio.service.ContextGraphBridge", FailingBridge)

    service = StudioService()
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent(
        "orchestrator",
        config_updates={
            "cg_url": "http://contextgraph.local",
            "agent_id": "agt_transient",
        },
    )
    layout, journal = service._ensure_project()
    proposal = journal.create_memory_proposal(
        proposal_id=f"mem_{exc_type.__name__.lower()}",
        run_id="run_transient",
        project_root=layout.root,
        agent_name="orchestrator",
        content="Keep deployment notes handy.",
        metadata={},
    )

    with pytest.raises(exc_type, match="temporarily unavailable"):
        service.sync_memory_proposal(proposal["id"])

    unchanged = journal.get_memory_proposal(proposal["id"])
    assert unchanged is not None
    assert unchanged["status"] == "pending_review"
    assert unchanged["synced_memory_id"] == ""


def test_sync_contextgraph_reports_mixed_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class MixedBridge:
        def __init__(self, **kwargs):
            del kwargs

        def store(self, content: str, **kwargs):
            del kwargs
            if content == "Transient sync":
                raise ConnectionError("ContextGraph is temporarily unavailable")
            return {"memory": {"memory_id": f"mem_{content.lower().replace(' ', '_')}"}}

    monkeypatch.setattr("contextclaw.studio.service.ContextGraphBridge", MixedBridge)

    service = StudioService()
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.create_agent(name="unlinked", provider="openai")
    service.update_agent(
        "orchestrator",
        config_updates={
            "cg_url": "http://contextgraph.local",
            "agent_id": "agt_bulk_sync",
        },
    )

    layout, journal = service._ensure_project()
    synced_proposal = journal.create_memory_proposal(
        proposal_id="mem_ok",
        run_id="run_bulk",
        project_root=layout.root,
        agent_name="orchestrator",
        content="Sync me",
        metadata={},
    )
    transient_proposal = journal.create_memory_proposal(
        proposal_id="mem_transient",
        run_id="run_bulk",
        project_root=layout.root,
        agent_name="orchestrator",
        content="Transient sync",
        metadata={},
    )
    rejected_proposal = journal.create_memory_proposal(
        proposal_id="mem_rejected",
        run_id="run_bulk",
        project_root=layout.root,
        agent_name="unlinked",
        content="Needs ContextGraph link",
        metadata={},
    )

    result = service.sync_contextgraph()

    assert {item["id"] for item in result["synced"]} == {synced_proposal["id"]}
    failed = {item["id"]: item["error"] for item in result["failed"]}
    assert transient_proposal["id"] in failed
    assert "temporarily unavailable" in failed[transient_proposal["id"]]
    assert rejected_proposal["id"] in failed
    assert "not linked to ContextGraph" in failed[rejected_proposal["id"]]

    synced_row = journal.get_memory_proposal(synced_proposal["id"])
    transient_row = journal.get_memory_proposal(transient_proposal["id"])
    rejected_row = journal.get_memory_proposal(rejected_proposal["id"])
    assert synced_row is not None and synced_row["status"] == "synced"
    assert transient_row is not None and transient_row["status"] == "pending_review"
    assert rejected_row is not None and rejected_row["status"] == "pending_review"


def test_studio_approval_timeout_records_timeout_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    policy_path = tmp_path / "approval_policy.yaml"
    policy_path.write_text(
        "permissions:\n  tools:\n    require_confirm:\n      - memory_propose\n",
        encoding="utf-8",
    )

    def provider_factory(_config) -> ScriptedProvider:
        return ScriptedProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="mem_timeout",
                            name="memory_propose",
                            arguments={
                                "content": "Remember this only if approved.",
                                "metadata": {"type": "fact"},
                            },
                        )
                    ],
                ),
                LLMResponse(content="Approval handled."),
            ]
        )

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent(
        "orchestrator",
        config_updates={"policy_path": str(policy_path)},
    )

    original_wait = service._wait_for_approval

    def short_timeout(layout, run_id, agent_name, tool_call):
        return original_wait(
            layout,
            run_id,
            agent_name,
            tool_call,
            timeout=0.05,
        )

    monkeypatch.setattr(service, "_wait_for_approval", short_timeout)

    run = service.start_run(
        "Propose memory with approval.",
        source="cli",
        agent_name="orchestrator",
    )
    result = _wait_for_run(service, run["id"], timeout=8.0)

    assert result["status"] == "completed"
    approvals = service.list_approvals()
    assert len(approvals) == 1
    approval = approvals[0]
    assert approval["status"] == "denied"
    assert approval["resolution"] == "timeout"

    events = service.list_run_events(run["id"])
    requested = next(event for event in events if event["type"] == "approval.requested")
    resolved = next(event for event in events if event["type"] == "approval.resolved")
    tool_results = [event for event in events if event["type"] == "tool.result"]

    assert requested["payload"]["status"] == "pending"
    assert resolved["payload"]["status"] == "denied"
    assert resolved["payload"]["resolution"] == "timeout"
    assert resolved["payload"]["resolution_reason"] == "timeout"
    assert resolved["payload"]["timeout_seconds"] == pytest.approx(0.05)
    assert any(
        "denied by the operator" in event["payload"]["result"] for event in tool_results
    )
    assert service.list_memory_proposals() == []
    service.shutdown(timeout=3.0)


def test_studio_memory_file_endpoints_write_and_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class FakeBridge:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def store(self, content: str, **kwargs):
            captured["content"] = content
            captured["kwargs"] = kwargs
            return {"memory": {"memory_id": "mem_file_synced"}}

    monkeypatch.setattr("contextclaw.studio.service.ContextGraphBridge", FakeBridge)

    service = StudioService()
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent(
        "orchestrator",
        config_updates={
            "cg_url": "http://contextgraph.local",
            "agent_id": "agt_memory_file",
        },
    )
    client = TestClient(create_app(service))

    written = client.put(
        "/memory-files/content",
        json={
            "scope": "agent",
            "agent_name": "orchestrator",
            "content": "# Agent Memory\n\nKeep notes terse.\n",
        },
    )
    assert written.status_code == 200
    assert written.json()["filename"] == "MEMORY.md"
    first_revision_id = written.json()["current_revision_id"]

    rewritten = client.put(
        "/memory-files/content",
        json={
            "scope": "agent",
            "agent_name": "orchestrator",
            "content": "# Agent Memory\n\nKeep notes detailed.\n",
        },
    )
    assert rewritten.status_code == 200
    second_revision_id = rewritten.json()["current_revision_id"]
    assert second_revision_id != first_revision_id

    listed = client.get("/memory-files?agent_name=orchestrator")
    assert listed.status_code == 200
    memory_entry = next(
        item
        for item in listed.json()
        if item["scope"] == "agent" and item["filename"] == "MEMORY.md"
    )
    assert memory_entry["revision_count"] >= 2

    revisions = client.get(
        "/memory-files/revisions",
        params={
            "scope": "agent",
            "agent_name": "orchestrator",
        },
    )
    assert revisions.status_code == 200
    revision_ids = [item["id"] for item in revisions.json()["revisions"]]
    assert first_revision_id in revision_ids
    assert second_revision_id in revision_ids

    first_revision = client.get(
        "/memory-files/revision",
        params={
            "scope": "agent",
            "agent_name": "orchestrator",
            "revision_id": first_revision_id,
        },
    )
    assert first_revision.status_code == 200
    assert "Keep notes terse." in first_revision.json()["content"]

    synced = client.post(
        "/memory-files/sync",
        json={
            "scope": "agent",
            "agent_name": "orchestrator",
            "revision_id": first_revision_id,
        },
    )
    assert synced.status_code == 200
    assert synced.json()["synced_memory_id"] == "mem_file_synced"
    assert captured["content"] == "# Agent Memory\n\nKeep notes terse."
    assert captured["kwargs"]["memory_kind"] == "artifact"
    assert captured["kwargs"]["metadata"]["revision_id"] == first_revision_id
    assert captured["kwargs"]["metadata"]["source_revision_id"] == ""

    synced_again = client.post(
        "/memory-files/sync",
        json={
            "scope": "agent",
            "agent_name": "orchestrator",
            "revision_id": first_revision_id,
        },
    )
    assert synced_again.status_code == 200
    assert synced_again.json()["already_synced"] is True

    restored = client.post(
        "/memory-files/restore",
        json={
            "scope": "agent",
            "agent_name": "orchestrator",
            "revision_id": first_revision_id,
        },
    )
    assert restored.status_code == 200
    assert restored.json()["restored_from_revision_id"] == first_revision_id
    assert "Keep notes terse." in restored.json()["content"]


def test_studio_run_cancel_emits_cancelled_event(tmp_path: Path):
    """Cancelling a run should record a run.cancelled event, not run.failed.

    Strategy: the provider keeps the run alive by requesting a tool on every
    turn.  We cancel once several turns have been observed, which exercises
    the cooperative cancel path between emitted events without depending on
    executor-thread timing.
    """
    turn_count = {"value": 0}

    class LoopingProvider:
        def complete(self, messages, tools, system=""):
            del messages, tools, system
            turn_count["value"] += 1
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"t{turn_count['value']}",
                        name="memory_propose",
                        arguments={
                            "content": f"turn {turn_count['value']}",
                            "metadata": {},
                        },
                    )
                ],
            )

    def provider_factory(_config):
        return LoopingProvider()

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})

    run = service.start_run(
        "Do a multi-turn task.",
        source="cli",
        agent_name="orchestrator",
    )
    deadline = time.time() + 5.0
    while turn_count["value"] < 3 and time.time() < deadline:
        time.sleep(0.05)
    assert turn_count["value"] >= 3, "Provider should have reached multiple turns"

    cancelled = service.cancel_run(run["id"])
    assert cancelled is not None
    assert cancelled["status"] == "cancelling"

    result = _wait_for_run(service, run["id"], timeout=8.0)
    assert result["status"] == "cancelled"

    events = service.list_run_events(run["id"])
    event_types = [e["type"] for e in events]
    assert "run.cancel_requested" in event_types
    # The runtime must emit a distinct run.cancelled event (not run.failed)
    assert "run.cancelled" in event_types, (
        f"Expected run.cancelled event but got: {event_types}"
    )
    assert "run.failed" not in event_types, (
        "Cancellation should not produce run.failed events"
    )
    service.shutdown(timeout=3.0)


def test_studio_service_shutdown_cleans_up(tmp_path: Path):
    """shutdown() should stop the event loop and join the thread."""

    def provider_factory(_config):
        return ScriptedProvider([LLMResponse(content="done")])

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})

    run = service.start_run(
        "Quick task.",
        source="cli",
        agent_name="orchestrator",
    )
    _wait_for_run(service, run["id"])

    # Shutdown should not hang or raise
    service.shutdown(timeout=5.0)
    assert service._loop is None or not service._loop.is_running()
    assert service._loop_thread is None

    # Loop should restart on next run
    run2 = service.start_run(
        "After restart.",
        source="cli",
        agent_name="orchestrator",
    )
    result2 = _wait_for_run(service, run2["id"])
    assert result2["status"] == "completed"
    service.shutdown(timeout=3.0)


def test_studio_shutdown_cancels_active_run_and_cleans_handles(tmp_path: Path):
    import threading as _threading

    entered = _threading.Event()
    release = _threading.Event()

    class BlockingProvider:
        def complete(self, messages, tools, system=""):
            del messages, tools, system
            entered.set()
            release.wait(timeout=5.0)
            return LLMResponse(content="late completion")

    def provider_factory(_config):
        return BlockingProvider()

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})

    run = service.start_run(
        "Block until shutdown.", source="cli", agent_name="orchestrator"
    )
    assert entered.wait(timeout=5.0), "Provider did not start before shutdown"

    service.shutdown(timeout=5.0)
    release.set()

    result = _wait_for_run(service, run["id"], timeout=8.0)
    event_types = [event["type"] for event in service.list_run_events(run["id"])]
    _wait_for_no_handles(service, timeout=3.0)

    assert result["status"] == "cancelled"
    assert "run.cancelled" in event_types
    assert "run.failed" not in event_types
    assert service._loop is None or not service._loop.is_running()
    assert service._loop_thread is None


def test_studio_shutdown_cancels_all_active_runs_and_restarts(tmp_path: Path):
    import threading as _threading

    entered_lock = _threading.Lock()
    all_entered = _threading.Event()
    release = _threading.Event()
    provider_count = {"value": 0}
    entered_count = {"value": 0}

    class BlockingProvider:
        def complete(self, messages, tools, system=""):
            del messages, tools, system
            with entered_lock:
                entered_count["value"] += 1
                if entered_count["value"] >= 2:
                    all_entered.set()
            release.wait(timeout=5.0)
            return LLMResponse(content="late completion")

    class QuickProvider:
        def complete(self, messages, tools, system=""):
            del messages, tools, system
            return LLMResponse(content="restarted")

    def provider_factory(_config):
        provider_count["value"] += 1
        if provider_count["value"] <= 2:
            return BlockingProvider()
        return QuickProvider()

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})

    run_a = service.start_run("Run A", source="cli", agent_name="orchestrator")
    run_b = service.start_run("Run B", source="cli", agent_name="orchestrator")
    assert all_entered.wait(timeout=5.0), (
        "Both runs should become active before shutdown"
    )

    service.shutdown(timeout=5.0)
    release.set()

    result_a = _wait_for_run(service, run_a["id"], timeout=8.0)
    result_b = _wait_for_run(service, run_b["id"], timeout=8.0)
    _wait_for_no_handles(service, timeout=3.0)

    assert result_a["status"] == "cancelled"
    assert result_b["status"] == "cancelled"
    assert service._loop is None or not service._loop.is_running()
    assert service._loop_thread is None

    restarted = service.start_run(
        "After shutdown", source="cli", agent_name="orchestrator"
    )
    restarted_result = _wait_for_run(service, restarted["id"], timeout=8.0)

    assert restarted_result["status"] == "completed"
    service.shutdown(timeout=3.0)


def test_studio_concurrent_runs_on_shared_loop(tmp_path: Path):
    """Multiple runs must truly overlap on the shared event loop.

    We prove overlap by using a threading.Barrier that requires both provider
    calls to arrive before either can return.  If runs were sequential, the
    barrier would timeout and raise BrokenBarrierError.
    """
    import threading as _threading

    # Barrier for 2 parties — both providers must enter before either proceeds
    barrier = _threading.Barrier(2, timeout=5.0)
    timestamps: dict[str, float] = {}

    class OverlapProvider:
        def __init__(self, label: str) -> None:
            self._label = label

        def complete(self, messages, tools, system=""):
            del messages, tools, system
            timestamps[f"{self._label}_enter"] = time.time()
            barrier.wait()  # blocks until BOTH runs are inside
            timestamps[f"{self._label}_exit"] = time.time()
            return LLMResponse(content=f"done-{self._label}")

    call_index = {"value": 0}

    def provider_factory(_config):
        call_index["value"] += 1
        return OverlapProvider(f"run{call_index['value']}")

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})

    run_a = service.start_run("Task A.", source="cli", agent_name="orchestrator")
    run_b = service.start_run("Task B.", source="cli", agent_name="orchestrator")

    result_a = _wait_for_run(service, run_a["id"], timeout=10.0)
    result_b = _wait_for_run(service, run_b["id"], timeout=10.0)

    assert result_a["status"] == "completed"
    assert result_b["status"] == "completed"
    # Verify both were inside provider.complete at the same time.
    # The barrier is the definitive proof: if it released without
    # BrokenBarrierError, both coroutines were alive concurrently.
    assert "run1_enter" in timestamps and "run2_enter" in timestamps, (
        f"Both providers must have been called, got: {list(timestamps)}"
    )
    assert "run1_exit" in timestamps and "run2_exit" in timestamps, (
        f"Both providers must have completed, got: {list(timestamps)}"
    )
    # Both ran on the same loop thread
    assert service._loop_thread is not None
    assert service._loop is not None and service._loop.is_running()
    service.shutdown(timeout=3.0)


def test_studio_shared_loop_stress_smoke(tmp_path: Path):
    def provider_factory(_config):
        return ScriptedProvider([LLMResponse(content="done")])

    service = StudioService(provider_factory=provider_factory)
    service.init_project(tmp_path, entry_agent="orchestrator", provider="openai")
    service.update_agent("orchestrator", config_updates={"policy_path": ""})

    runs = [
        service.start_run(
            f"Stress task {index}",
            source="cli",
            agent_name="orchestrator",
        )
        for index in range(6)
    ]
    results = [_wait_for_run(service, run["id"], timeout=10.0) for run in runs]
    _wait_for_no_handles(service, timeout=3.0)

    assert all(result["status"] == "completed" for result in results)
    assert service._loop_thread is not None
    assert service._loop is not None and service._loop.is_running()
    service.shutdown(timeout=3.0)


def test_frontend_dist_resolves_to_valid_html():
    """The daemon must find a frontend dist directory with index.html.

    This catches packaging regressions where studio-ui/dist is not
    included in the wheel or the resolution path is broken.
    """
    dist = studio_daemon._frontend_dist_dir()
    assert dist.exists(), f"Frontend dist not found at {dist}"
    index = dist / "index.html"
    assert index.exists(), f"index.html not found in {dist}"
    content = index.read_text(encoding="utf-8")
    assert "<html" in content.lower(), "index.html does not look like HTML"
    assets = dist / "assets"
    assert assets.exists(), f"assets/ directory not found in {dist}"
