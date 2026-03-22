"""Tests for AgentConfig and SOUL.md parsing."""

from __future__ import annotations

from pathlib import Path

from contextclaw.config.agent_config import (
    AgentConfig,
    _resolve_config_path,
    _resolve_env,
)
from contextclaw.config.skills import load_skills, render_skills_prompt
from contextclaw.config.soul import SoulConfig, load_soul

# ---------------------------------------------------------------------------
# AgentConfig defaults
# ---------------------------------------------------------------------------


def test_agent_config_defaults():
    config = AgentConfig(name="myagent", workspace=Path("/tmp"))
    assert config.provider == "claude"
    assert config.model == ""
    assert config.soul_path is None
    assert config.policy_path is None
    assert config.sandbox_type == "process"
    assert config.tools == []
    assert config.cg_url == ""
    assert config.cg_api_key == ""
    assert config.agent_id == ""


def test_agent_config_explicit_fields():
    config = AgentConfig(
        name="researcher",
        workspace=Path("/workspace"),
        provider="openai",
        model="gpt-4o",
        sandbox_type="docker",
        tools=["filesystem", "web"],
        cg_url="http://localhost:8000",
        cg_api_key="secret",
        agent_id="agent-abc",
    )
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.tools == ["filesystem", "web"]
    assert config.cg_url == "http://localhost:8000"


# ---------------------------------------------------------------------------
# AgentConfig.from_yaml
# ---------------------------------------------------------------------------


def test_from_yaml_basic(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "name: test-agent\n"
        f"workspace: {tmp_path}\n"
        "provider: openai\n"
        "model: gpt-4o\n"
        "sandbox_type: process\n"
        "tools: filesystem, web\n"
        "cg_url: http://cg.local\n"
        "cg_api_key: key123\n"
        "agent_id: agent-xyz\n",
        encoding="utf-8",
    )
    config = AgentConfig.from_yaml(config_file)
    assert config.name == "test-agent"
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.tools == ["filesystem", "web"]
    assert config.cg_url == "http://cg.local"
    assert config.cg_api_key == "key123"
    assert config.agent_id == "agent-xyz"


def test_from_yaml_defaults_when_fields_missing(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("name: minimal\n", encoding="utf-8")
    config = AgentConfig.from_yaml(config_file)
    assert config.name == "minimal"
    assert config.provider == "claude"
    assert config.tools == []
    assert config.soul_path is None
    assert config.policy_path is None


def test_from_yaml_tools_empty_string(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("name: agent\ntools: \n", encoding="utf-8")
    config = AgentConfig.from_yaml(config_file)
    assert config.tools == []


def test_from_yaml_soul_path(tmp_path: Path):
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text("# Soul", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"name: agent\nsoul_path: {soul_file}\n", encoding="utf-8")
    config = AgentConfig.from_yaml(config_file)
    assert config.soul_path == soul_file


def test_from_yaml_policy_path(tmp_path: Path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("permissions:\n  tools:\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"name: agent\npolicy_path: {policy_file}\n", encoding="utf-8"
    )
    config = AgentConfig.from_yaml(config_file)
    assert config.policy_path == policy_file


def test_from_yaml_skills_path(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "plan.md").write_text("Use checklists.", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("name: agent\nskills_path: skills\n", encoding="utf-8")

    config = AgentConfig.from_yaml(config_file)

    assert config.skills_path == skills_dir.resolve()


def test_from_yaml_resolves_relative_paths_against_config_dir(tmp_path: Path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    soul_file = tmp_path / "relative" / "SOUL.md"
    soul_file.parent.mkdir()
    soul_file.write_text("# Soul", encoding="utf-8")
    policy_file = tmp_path / "policies" / "default.yaml"
    policy_file.parent.mkdir()
    policy_file.write_text("permissions:\n  tools:\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "name: relative-agent\nworkspace: workspace\nsoul_path: relative/SOUL.md\npolicy_path: policies/default.yaml\n",
        encoding="utf-8",
    )

    config = AgentConfig.from_yaml(config_file)

    assert config.workspace == workspace_dir.resolve()
    assert config.soul_path == soul_file.resolve()
    assert config.policy_path == policy_file.resolve()


# ---------------------------------------------------------------------------
# AgentConfig.from_dir
# ---------------------------------------------------------------------------


def test_from_dir_with_config_yaml(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "name: dir-agent\nprovider: claude\n",
        encoding="utf-8",
    )
    config = AgentConfig.from_dir(tmp_path)
    assert config.name == "dir-agent"
    assert config.workspace == tmp_path.resolve()


def test_from_dir_auto_discovers_soul_md(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("name: agent-with-soul\n", encoding="utf-8")
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text("You are a helpful assistant.", encoding="utf-8")
    config = AgentConfig.from_dir(tmp_path)
    assert config.soul_path == soul_file.resolve()


def test_from_dir_no_config_yaml_fallback(tmp_path: Path):
    """When config.yaml is absent, fall back to directory name as agent name."""
    config = AgentConfig.from_dir(tmp_path)
    assert config.name == tmp_path.name
    assert config.workspace == tmp_path.resolve()


def test_from_dir_no_soul_md_soul_path_none(tmp_path: Path):
    """When SOUL.md is absent, soul_path stays None."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("name: agent\n", encoding="utf-8")
    config = AgentConfig.from_dir(tmp_path)
    assert config.soul_path is None


def test_from_dir_auto_discovers_skills_dir(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "review.md").write_text("Review carefully.", encoding="utf-8")

    config = AgentConfig.from_dir(tmp_path)

    assert config.skills_path == skills_dir.resolve()


def test_from_dir_explicit_soul_path_not_overridden(tmp_path: Path):
    """An explicit soul_path in config.yaml should not be replaced by SOUL.md auto-discovery."""
    explicit_soul = tmp_path / "custom_soul.md"
    explicit_soul.write_text("Custom soul.", encoding="utf-8")
    soul_auto = tmp_path / "SOUL.md"
    soul_auto.write_text("Auto soul.", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"name: agent\nsoul_path: {explicit_soul}\n", encoding="utf-8"
    )
    config = AgentConfig.from_dir(tmp_path)
    assert config.soul_path == explicit_soul


def test_load_skills_from_directory(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "plan.md").write_text("Break work into steps.", encoding="utf-8")
    nested = skills_dir / "research"
    nested.mkdir()
    (nested / "compare.md").write_text("Compare sources carefully.", encoding="utf-8")

    skills = load_skills(skills_dir)

    assert [label for label, _ in skills] == ["plan.md", "research/compare.md"]


def test_render_skills_prompt(tmp_path: Path):
    skill_file = tmp_path / "skill.md"
    skill_file.write_text("Always verify with tests.", encoding="utf-8")

    prompt = render_skills_prompt(skill_file)

    assert "Additional skills are available" in prompt
    assert "[Skill: skill.md]" in prompt
    assert "Always verify with tests." in prompt


def test_resolve_config_path_resolves_relative_and_absolute_paths(tmp_path: Path):
    relative = _resolve_config_path("nested/file.txt", tmp_path)
    absolute_target = (tmp_path / "absolute.txt").resolve()
    absolute = _resolve_config_path(str(absolute_target), tmp_path)

    assert relative == (tmp_path / "nested" / "file.txt").resolve()
    assert absolute == absolute_target


# ---------------------------------------------------------------------------
# load_soul
# ---------------------------------------------------------------------------


def test_load_soul_with_frontmatter(tmp_path: Path):
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text(
        "---\n"
        "name: Research Assistant\n"
        "role: researcher\n"
        "tone: professional\n"
        "verbosity: concise\n"
        "---\n"
        "\n"
        "You are a research assistant.\n",
        encoding="utf-8",
    )
    soul = load_soul(soul_file)
    assert soul.name == "Research Assistant"
    assert soul.role == "researcher"
    assert soul.tone == "professional"
    assert soul.verbosity == "concise"
    assert "You are a research assistant." in soul.body


def test_load_soul_no_frontmatter(tmp_path: Path):
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text("Just a body without frontmatter.\n", encoding="utf-8")
    soul = load_soul(soul_file)
    assert soul.name == ""
    assert soul.role == ""
    assert soul.body == "Just a body without frontmatter."


def test_load_soul_missing_closing_delimiter(tmp_path: Path):
    """A file with opening --- but no closing --- is treated as body-only."""
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text(
        "---\nname: Broken\nrole: tester\nThis never closes.\n",
        encoding="utf-8",
    )
    soul = load_soul(soul_file)
    # No closing delimiter → entire file is body
    assert soul.name == ""
    assert "---" in soul.body or "Broken" in soul.body


def test_load_soul_extra_frontmatter_fields(tmp_path: Path):
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text(
        "---\n"
        "name: Agent\n"
        "role: assistant\n"
        "tone: casual\n"
        "verbosity: verbose\n"
        "custom_field: some_value\n"
        "---\n"
        "Body here.\n",
        encoding="utf-8",
    )
    soul = load_soul(soul_file)
    assert soul.extra.get("custom_field") == "some_value"
    assert soul.body == "Body here."


def test_load_soul_empty_body(tmp_path: Path):
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text(
        "---\nname: Agent\n---\n",
        encoding="utf-8",
    )
    soul = load_soul(soul_file)
    assert soul.name == "Agent"
    assert soul.body == ""


def test_soul_config_defaults():
    soul = SoulConfig()
    assert soul.name == ""
    assert soul.role == ""
    assert soul.tone == ""
    assert soul.verbosity == ""
    assert soul.extra == {}
    assert soul.body == ""


# ---------------------------------------------------------------------------
# _resolve_env
# ---------------------------------------------------------------------------


def test_resolve_env_plain_string():
    assert _resolve_env("hello") == "hello"


def test_resolve_env_dollar_brace_syntax(monkeypatch):
    monkeypatch.setenv("TEST_CG_KEY", "secret123")
    assert _resolve_env("${TEST_CG_KEY}") == "secret123"


def test_resolve_env_dollar_brace_missing(monkeypatch):
    monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
    assert _resolve_env("${NONEXISTENT_VAR_XYZ}") == ""


def test_resolve_env_env_colon_syntax(monkeypatch):
    monkeypatch.setenv("TEST_CG_KEY2", "val456")
    assert _resolve_env("env:TEST_CG_KEY2") == "val456"


def test_resolve_env_empty_with_fallback(monkeypatch):
    monkeypatch.setenv("CONTEXTGRAPH_API_KEY", "fallback_val")
    assert _resolve_env("", env_fallback="CONTEXTGRAPH_API_KEY") == "fallback_val"


def test_resolve_env_empty_no_fallback():
    assert _resolve_env("") == ""


# ---------------------------------------------------------------------------
# AgentConfig.from_yaml — env var credential resolution
# ---------------------------------------------------------------------------


def test_from_yaml_resolves_env_var_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY_FOR_CG", "resolved_key")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "name: env-agent\ncg_url: http://cg.local\ncg_api_key: ${TEST_API_KEY_FOR_CG}\n",
        encoding="utf-8",
    )
    config = AgentConfig.from_yaml(config_file)
    assert config.cg_api_key == "resolved_key"


def test_from_yaml_falls_back_to_contextgraph_api_key_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CONTEXTGRAPH_API_KEY", "env_fallback_key")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "name: env-agent\ncg_url: http://cg.local\n",
        encoding="utf-8",
    )
    config = AgentConfig.from_yaml(config_file)
    assert config.cg_api_key == "env_fallback_key"
