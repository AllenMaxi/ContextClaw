from __future__ import annotations

import argparse
import json
from pathlib import Path

import contextclaw.cli as cli
import contextclaw.knowledge as knowledge_module


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
