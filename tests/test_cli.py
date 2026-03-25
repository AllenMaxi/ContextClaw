from __future__ import annotations

import argparse
import json
from pathlib import Path

import contextclaw.cli as cli
import contextclaw.knowledge as knowledge_module
from contextclaw.chat.session import ChatSession
from contextclaw.memory_files import list_memory_file_revisions


def _make_agent_workspace(base: Path, name: str = "demo-agent") -> Path:
    workspace = base / name
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "config.yaml").write_text(
        "name: demo-agent\nprovider: openai\ntools: filesystem,planning\n",
        encoding="utf-8",
    )
    return workspace


def test_cmd_link_writes_contextgraph_settings(monkeypatch, tmp_path: Path, capsys):
    _make_agent_workspace(tmp_path)
    monkeypatch.setattr(cli, "AGENTS_DIR", tmp_path)

    args = argparse.Namespace(
        name="demo-agent",
        cg_url="http://localhost:8420",
        api_key="env:CONTEXTGRAPH_API_KEY",
        register=False,
        org_id="default",
        capability=[],
    )

    cli.cmd_link(args)

    config_text = (tmp_path / "demo-agent" / "config.yaml").read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "cg_url: http://localhost:8420" in config_text
    assert "cg_api_key: env:CONTEXTGRAPH_API_KEY" in config_text
    assert (
        "Linked 'demo-agent' to ContextGraph at http://localhost:8420" in captured.out
    )


def test_cmd_link_registers_agent_and_persists_agent_id(
    monkeypatch, tmp_path: Path, capsys
):
    _make_agent_workspace(tmp_path)
    monkeypatch.setattr(cli, "AGENTS_DIR", tmp_path)

    class FakeBridge:
        def __init__(self, cg_url: str, api_key: str, agent_id: str = "") -> None:
            self.cg_url = cg_url
            self.api_key = api_key
            self.agent_id = agent_id

        def register(
            self, name: str, org_id: str, capabilities: list[str] | None = None
        ) -> str:
            assert self.cg_url == "http://localhost:8420"
            assert self.api_key == "plain-test-key"
            assert name == "demo-agent"
            assert org_id == "acme"
            assert capabilities == ["research", "memory"]
            self.api_key = "issued-agent-key"
            return "agt_demo_123"

    monkeypatch.setattr(
        knowledge_module, "ContextGraphBridge", FakeBridge, raising=False
    )

    args = argparse.Namespace(
        name="demo-agent",
        cg_url="http://localhost:8420",
        api_key="plain-test-key",
        register=True,
        org_id="acme",
        capability=["research", "memory"],
    )

    cli.cmd_link(args)

    config_text = (tmp_path / "demo-agent" / "config.yaml").read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "cg_url: http://localhost:8420" in config_text
    assert "cg_api_key: ${CONTEXTGRAPH_AGENT_KEY}" in config_text
    assert "agent_id: agt_demo_123" in config_text
    assert "Registered 'demo-agent' with ContextGraph as agt_demo_123" in captured.out
    assert "export CONTEXTGRAPH_AGENT_KEY='issued-agent-key'" in captured.err


def test_cmd_status_reports_registration_pending(monkeypatch, tmp_path: Path, capsys):
    workspace = _make_agent_workspace(tmp_path)
    (workspace / "config.yaml").write_text(
        "name: demo-agent\nprovider: openai\ncg_url: http://localhost:8420\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "AGENTS_DIR", tmp_path)

    cli.cmd_status(argparse.Namespace(name="demo-agent"))

    captured = capsys.readouterr()
    assert "ContextGraph: linked" in captured.out
    assert (
        "Agent ID: none (run `cclaw link ... --register` to enable ContextGraph recall/store)"
        in captured.out
    )


def test_cmd_connectors_install_generates_catalog_state(
    monkeypatch, tmp_path: Path, capsys
):
    _make_agent_workspace(tmp_path)
    monkeypatch.setattr(cli, "AGENTS_DIR", tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    cli.cmd_connectors_install(
        argparse.Namespace(name="demo-agent", connector_id="github")
    )

    workspace = tmp_path / "demo-agent"
    catalog_text = (workspace / ".contextclaw" / "catalog.yaml").read_text(
        encoding="utf-8"
    )
    generated = json.loads(
        (workspace / ".contextclaw" / "generated" / "mcp_servers.json").read_text(
            encoding="utf-8"
        )
    )
    captured = capsys.readouterr()
    assert "- github" in catalog_text
    assert generated["servers"][0]["name"] == "github"
    assert "Missing env vars for 'github': GITHUB_TOKEN" in captured.out


def test_cmd_skills_install_auto_installs_required_connector(
    monkeypatch, tmp_path: Path, capsys
):
    _make_agent_workspace(tmp_path)
    monkeypatch.setattr(cli, "AGENTS_DIR", tmp_path)

    cli.cmd_skills_install(
        argparse.Namespace(
            name="demo-agent",
            skill_id="github-maintainer",
            no_deps=False,
        )
    )

    workspace = tmp_path / "demo-agent"
    catalog_text = (workspace / ".contextclaw" / "catalog.yaml").read_text(
        encoding="utf-8"
    )
    captured = capsys.readouterr()
    assert "- github" in catalog_text
    assert "- github-maintainer" in catalog_text
    assert (
        workspace / "skills" / "packages" / "github-maintainer" / "SKILL.md"
    ).exists()
    assert "Auto-installed connectors: github" in captured.out


def test_cmd_status_reports_catalog_install_state(monkeypatch, tmp_path: Path, capsys):
    _make_agent_workspace(tmp_path)
    monkeypatch.setattr(cli, "AGENTS_DIR", tmp_path)
    cli.cmd_connectors_install(
        argparse.Namespace(name="demo-agent", connector_id="github")
    )
    capsys.readouterr()

    cli.cmd_status(argparse.Namespace(name="demo-agent"))

    captured = capsys.readouterr()
    assert "Installed Connectors: github" in captured.out
    assert "Catalog Sync: up to date" in captured.out
    assert "Generated MCP Registry:" in captured.out


def test_cmd_create_uses_project_agents_dir(monkeypatch, tmp_path: Path, capsys):
    project_root = tmp_path / "project"
    legacy_agents = tmp_path / "legacy-agents"
    monkeypatch.setattr(cli, "AGENTS_DIR", legacy_agents)

    cli.cmd_project_init(
        argparse.Namespace(
            root=str(project_root),
            entry_agent="orchestrator",
            provider="openai",
        )
    )
    capsys.readouterr()
    monkeypatch.chdir(project_root)

    cli.cmd_create(
        argparse.Namespace(name="reviewer", template="coding", provider="openai")
    )

    captured = capsys.readouterr()
    assert (project_root / "agents" / "reviewer" / "config.yaml").exists()
    assert not (legacy_agents / "reviewer").exists()
    assert "Created agent 'reviewer'" in captured.out


def test_cmd_context_and_compact_commands_use_project_state(
    monkeypatch, tmp_path: Path, capsys
):
    project_root = tmp_path / "project"
    cli.cmd_project_init(
        argparse.Namespace(
            root=str(project_root),
            entry_agent="orchestrator",
            provider="openai",
        )
    )
    capsys.readouterr()
    monkeypatch.chdir(project_root)

    (project_root / "AGENTS.md").write_text(
        "# Project Memory\n\nAlways prefer safe rollouts.\n",
        encoding="utf-8",
    )
    workspace = project_root / "agents" / "orchestrator"
    (workspace / "MEMORY.md").write_text(
        "# Agent Memory\n\nThis agent keeps release notes short.\n",
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

    cli.cmd_compact_preview(argparse.Namespace(name=None, reason="test_preview"))
    preview_output = capsys.readouterr().out
    assert "Compaction Preview:" in preview_output

    cli.cmd_context_show(argparse.Namespace(name=None))
    context_output = capsys.readouterr().out
    assert "Agent: orchestrator" in context_output
    assert "Pending Compact:" in context_output
    assert "Always-Loaded Memory Files:" in context_output
    assert "AGENTS.md" in context_output
    assert "MEMORY.md" in context_output

    cli.cmd_compact_apply(argparse.Namespace(name=None, reason="test_apply"))
    apply_output = capsys.readouterr().out
    assert "Applied compaction:" in apply_output


def test_cmd_memory_file_show_write_and_sync(monkeypatch, tmp_path: Path, capsys):
    project_root = tmp_path / "project"
    cli.cmd_project_init(
        argparse.Namespace(
            root=str(project_root),
            entry_agent="orchestrator",
            provider="openai",
        )
    )
    capsys.readouterr()
    monkeypatch.chdir(project_root)

    cli.cmd_memory_file_write(
        argparse.Namespace(
            scope="project",
            name=None,
            filename="",
            content="# Project Memory\n\nShip with review gates.\n",
            file=None,
            append=False,
        )
    )
    write_output = capsys.readouterr().out
    assert "Updated project memory file AGENTS.md." in write_output

    cli.cmd_memory_file_show(
        argparse.Namespace(scope="project", name=None, filename="", revision_id="")
    )
    show_output = capsys.readouterr().out
    assert "Ship with review gates." in show_output

    cli.cmd_memory_file_write(
        argparse.Namespace(
            scope="project",
            name=None,
            filename="",
            content="# Project Memory\n\nUse release candidates for demos.\n",
            file=None,
            append=False,
        )
    )
    capsys.readouterr()

    revisions = list_memory_file_revisions(
        project_root / "agents" / "orchestrator",
        scope="project",
    )
    oldest_revision_id = revisions["revisions"][-1]["id"]

    cli.cmd_memory_file_history(
        argparse.Namespace(scope="project", name=None, filename="")
    )
    history_output = capsys.readouterr().out
    assert oldest_revision_id in history_output
    assert "Current Revision:" in history_output

    cli.cmd_memory_file_show(
        argparse.Namespace(
            scope="project",
            name=None,
            filename="",
            revision_id=oldest_revision_id,
        )
    )
    revision_output = capsys.readouterr().out
    assert "Ship with review gates." in revision_output

    cli.cmd_memory_file_restore(
        argparse.Namespace(
            scope="project",
            revision_id=oldest_revision_id,
            name=None,
            filename="",
        )
    )
    restore_output = capsys.readouterr().out
    assert oldest_revision_id in restore_output

    class FakeService:
        def sync_memory_file(
            self,
            *,
            scope,
            agent_name=None,
            filename="",
            revision_id="",
        ):
            return {
                "scope": scope,
                "filename": filename or "AGENTS.md",
                "synced_memory_id": "mem_file_1",
                "revision": {"id": revision_id or "rev_latest"},
                "already_synced": bool(revision_id),
            }

        def reject_compact(self, agent_name=None):
            return {"rejected": False}

    monkeypatch.setattr(
        cli, "_require_project_service", lambda root=None: (FakeService(), None)
    )
    cli.cmd_memory_file_sync(
        argparse.Namespace(
            scope="project",
            name=None,
            filename="",
            revision_id=oldest_revision_id,
        )
    )
    sync_output = capsys.readouterr().out
    assert "mem_file_1" in sync_output
    assert oldest_revision_id in sync_output
    assert "already synced" in sync_output.lower()

    cli.cmd_compact_reject(argparse.Namespace(name=None))
    reject_output = capsys.readouterr().out
    assert "No pending compaction preview was present." in reject_output
