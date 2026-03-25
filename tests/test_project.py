from __future__ import annotations

from pathlib import Path

from contextclaw.project import (
    ensure_project,
    find_project_root,
    get_project_layout,
    scaffold_agent_workspace,
)


def test_ensure_project_and_scaffold_agent_workspace(tmp_path: Path):
    layout = ensure_project(tmp_path, entry_agent="orchestrator")
    workspace = scaffold_agent_workspace(
        layout,
        name="orchestrator",
        template="coding",
        provider="openai",
    )

    assert layout.workflow_path.exists()
    assert layout.agents_dir.exists()
    assert layout.runtime_dir.exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (workspace / "SOUL.md").exists()
    assert (workspace / "MEMORY.md").exists()
    assert (workspace / "policy.yaml").exists()
    assert (workspace / "mcp_servers.json").exists()
    config_text = (workspace / "config.yaml").read_text(encoding="utf-8")
    assert "project_agents_path: .." in config_text
    assert "policy_path: policy.yaml" in config_text


def test_find_project_root_from_nested_workspace(tmp_path: Path):
    layout = ensure_project(tmp_path, entry_agent="orchestrator")
    workspace = scaffold_agent_workspace(layout, name="reviewer")
    nested = workspace / "skills"

    discovered = find_project_root(nested)
    resolved_layout = get_project_layout(nested)

    assert discovered == tmp_path.resolve()
    assert resolved_layout is not None
    assert resolved_layout.root == tmp_path.resolve()
