from __future__ import annotations

import json
from pathlib import Path

import pytest
from contextclaw.catalog_engine import (
    CatalogState,
    catalog_sync_required,
    generated_mcp_path,
    generated_policy_path,
    load_connector_specs,
    load_skill_specs,
    read_catalog_lock,
    sync_agent_catalog,
    write_catalog_state,
)


def test_builtin_catalog_contains_curated_connectors_and_skills():
    connectors = load_connector_specs()
    skills = load_skill_specs()

    assert {
        "filesystem",
        "web",
        "shell",
        "github",
        "playwright",
        "notion",
        "slack",
        "contextgraph-mcp",
    }.issubset(connectors)
    assert {
        "research",
        "coding",
        "code-review",
        "qa-triage",
        "docs-writer",
        "launch-marketing",
        "memory-governor",
        "github-maintainer",
        "notion-knowledge-base",
        "playwright-debugger",
    }.issubset(skills)


def test_load_connector_specs_rejects_invalid_type(tmp_path: Path):
    catalog_root = tmp_path / "catalog"
    manifest = catalog_root / "connectors" / "broken" / "connector.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "id: broken\n"
        "version: 1.0.0\n"
        "display_name: Broken\n"
        "description: Broken connector.\n"
        "stability: preview\n"
        "tags: [broken]\n"
        "type: invalid\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid connector type"):
        load_connector_specs(catalog_root)


def test_load_skill_specs_rejects_invalid_asset_dir(tmp_path: Path):
    catalog_root = tmp_path / "catalog"
    package_dir = catalog_root / "skills" / "broken-skill"
    package_dir.mkdir(parents=True)
    (package_dir / "skill.yaml").write_text(
        "id: broken-skill\n"
        "version: 1.0.0\n"
        "display_name: Broken Skill\n"
        "description: Broken package.\n"
        "stability: preview\n"
        "tags: [broken]\n"
        "entrypoint: SKILL.md\n"
        "requires_connectors: []\n"
        "asset_dirs: [notes]\n",
        encoding="utf-8",
    )
    (package_dir / "SKILL.md").write_text("Broken", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported asset_dirs"):
        load_skill_specs(catalog_root)


def test_sync_agent_catalog_generates_lock_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = tmp_path / "agent"
    workspace.mkdir()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    state = CatalogState(
        connectors=["filesystem", "github"],
        skills=["github-maintainer", "docs-writer"],
    )
    write_catalog_state(workspace, state)

    result = sync_agent_catalog(workspace)

    assert result.missing_env == {"github": ["GITHUB_TOKEN"]}
    generated_mcp = generated_mcp_path(workspace)
    generated_policy = generated_policy_path(workspace)
    assert generated_mcp.exists()
    assert generated_policy.exists()

    mcp_data = json.loads(generated_mcp.read_text(encoding="utf-8"))
    assert mcp_data["servers"] == [
        {
            "name": "github",
            "command": ["python3", "-m", "contextclaw.catalog_mcp_server", "github"],
        }
    ]

    policy_text = generated_policy.read_text(encoding="utf-8")
    assert "mcp__github__*" in policy_text
    assert (
        workspace / "skills" / "packages" / "github-maintainer" / "SKILL.md"
    ).exists()
    assert (
        workspace
        / "skills"
        / "packages"
        / "docs-writer"
        / "templates"
        / "readme-checklist.md"
    ).exists()

    lock = read_catalog_lock(workspace)
    assert [item["id"] for item in lock["connectors"]] == ["filesystem", "github"]
    assert [item["id"] for item in lock["skills"]] == [
        "docs-writer",
        "github-maintainer",
    ]


def test_sync_agent_catalog_tracks_missing_connector_dependencies(tmp_path: Path):
    workspace = tmp_path / "agent"
    workspace.mkdir()
    write_catalog_state(workspace, CatalogState(skills=["github-maintainer"]))

    result = sync_agent_catalog(workspace)

    assert result.missing_connector_dependencies == {"github-maintainer": ["github"]}


def test_sync_agent_catalog_removes_orphaned_packaged_skills(tmp_path: Path):
    workspace = tmp_path / "agent"
    workspace.mkdir()
    write_catalog_state(workspace, CatalogState(skills=["docs-writer"]))
    sync_agent_catalog(workspace)
    packaged = workspace / "skills" / "packages" / "docs-writer"
    assert packaged.exists()

    write_catalog_state(workspace, CatalogState())
    sync_agent_catalog(workspace)

    assert not packaged.exists()


def test_catalog_sync_required_tracks_stale_state(tmp_path: Path):
    workspace = tmp_path / "agent"
    workspace.mkdir()

    write_catalog_state(workspace, CatalogState(connectors=["filesystem"]))
    assert catalog_sync_required(workspace) is True

    sync_agent_catalog(workspace)
    assert catalog_sync_required(workspace) is False

    write_catalog_state(workspace, CatalogState(connectors=["filesystem", "web"]))
    assert catalog_sync_required(workspace) is True
