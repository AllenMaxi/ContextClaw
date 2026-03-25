from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _parse_simple_yaml(text: str) -> dict[str, str]:
    """Parse simple key: value YAML (no nesting, no lists)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.strip()
            # Strip inline comments
            if " #" in value:
                value = value[: value.index(" #")].rstrip()
            result[key.strip()] = value
    return result


def _resolve_env(value: str, env_fallback: str = "") -> str:
    """Resolve a config value that may reference an environment variable.

    Supports:
    - ``${VAR_NAME}`` — substituted with os.environ.get("VAR_NAME", "")
    - ``env:VAR_NAME`` — substituted with os.environ.get("VAR_NAME", "")
    - plain strings — returned as-is

    If *env_fallback* is provided, also check that env var when the value
    itself is empty.
    """
    if not value:
        return os.environ.get(env_fallback, "") if env_fallback else ""

    # ${VAR_NAME} syntax
    if value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return os.environ.get(var_name, "")

    # env:VAR_NAME syntax
    if value.startswith("env:"):
        var_name = value[4:]
        return os.environ.get(var_name, "")

    return value


def _resolve_config_path(value: str, base_dir: Path) -> Path:
    """Resolve a config path relative to the config file directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass
class AgentConfig:
    name: str
    workspace: Path
    provider: str = "claude"  # claude | openai | ollama
    model: str = ""  # empty = provider default
    soul_path: Path | None = None  # path to SOUL.md
    skills_path: Path | None = None  # path to a skills directory or markdown file
    policy_path: Path | None = None  # path to policy YAML
    mcp_servers_path: Path | None = None  # path to an MCP registry JSON file
    subagents_path: Path | None = None  # path to sub-agent workspaces
    project_agents_path: Path | None = None  # path to sibling project agents
    checkpoint_path: Path | None = None  # path to session checkpoint JSON
    sandbox_type: str = "process"  # docker | process | none
    tools: list[str] = field(default_factory=list)  # tool bundle names
    cg_url: str = ""  # ContextGraph server URL
    cg_api_key: str = ""  # ContextGraph API key
    agent_id: str = ""  # ContextGraph agent ID

    @classmethod
    def from_yaml(cls, path: Path) -> AgentConfig:
        """Load config from a YAML file (config.yaml in agent workspace)."""
        raw = _parse_simple_yaml(path.read_text(encoding="utf-8"))
        base_dir = path.parent.resolve()

        workspace_raw = raw.get("workspace", "")
        workspace = (
            _resolve_config_path(workspace_raw, base_dir) if workspace_raw else base_dir
        )

        tools_raw = raw.get("tools", "")
        tools = (
            [t.strip() for t in tools_raw.split(",") if t.strip()] if tools_raw else []
        )

        soul_raw = raw.get("soul_path", "")
        soul_path = _resolve_config_path(soul_raw, base_dir) if soul_raw else None

        skills_raw = raw.get("skills_path", "")
        skills_path = _resolve_config_path(skills_raw, base_dir) if skills_raw else None

        mcp_raw = raw.get("mcp_servers_path", "")
        mcp_servers_path = _resolve_config_path(mcp_raw, base_dir) if mcp_raw else None

        subagents_raw = raw.get("subagents_path", "")
        subagents_path = (
            _resolve_config_path(subagents_raw, base_dir) if subagents_raw else None
        )

        project_agents_raw = raw.get("project_agents_path", "")
        project_agents_path = (
            _resolve_config_path(project_agents_raw, base_dir)
            if project_agents_raw
            else None
        )

        checkpoint_raw = raw.get("checkpoint_path", "")
        checkpoint_path = (
            _resolve_config_path(checkpoint_raw, base_dir) if checkpoint_raw else None
        )

        policy_raw = raw.get("policy_path", "")
        policy_path = _resolve_config_path(policy_raw, base_dir) if policy_raw else None

        # Resolve credentials — support ${ENV_VAR} and env:ENV_VAR syntax,
        # with fallback to well-known environment variable names
        cg_api_key = _resolve_env(raw.get("cg_api_key", ""))
        if not cg_api_key:
            cg_api_key = os.environ.get("CONTEXTGRAPH_AGENT_KEY", "") or os.environ.get(
                "CONTEXTGRAPH_API_KEY", ""
            )

        return cls(
            name=raw.get("name", path.parent.name),
            workspace=workspace,
            provider=raw.get("provider", "claude"),
            model=raw.get("model", ""),
            soul_path=soul_path,
            skills_path=skills_path,
            policy_path=policy_path,
            mcp_servers_path=mcp_servers_path,
            subagents_path=subagents_path,
            project_agents_path=project_agents_path,
            checkpoint_path=checkpoint_path,
            sandbox_type=raw.get("sandbox_type", "process"),
            tools=tools,
            cg_url=raw.get("cg_url", ""),
            cg_api_key=cg_api_key,
            agent_id=raw.get("agent_id", ""),
        )

    @classmethod
    def from_dir(cls, workspace: Path) -> AgentConfig:
        """Load config from an agent workspace directory.

        Looks for config.yaml; sets soul_path to SOUL.md if present.
        Falls back to a minimal config derived from the directory name.
        """
        config_file = workspace / "config.yaml"
        if config_file.exists():
            config = cls.from_yaml(config_file)
            # Always anchor workspace to the resolved directory
            config.workspace = workspace.resolve()
        else:
            config = cls(name=workspace.name, workspace=workspace.resolve())

        # Auto-discover SOUL.md when soul_path is not explicitly set
        if config.soul_path is None:
            soul_file = workspace / "SOUL.md"
            if soul_file.exists():
                config.soul_path = soul_file.resolve()

        # Auto-discover a skills directory when not explicitly set
        if config.skills_path is None:
            skills_dir = workspace / "skills"
            if skills_dir.exists() and skills_dir.is_dir():
                config.skills_path = skills_dir.resolve()

        if config.mcp_servers_path is None:
            mcp_file = workspace / "mcp_servers.json"
            if mcp_file.exists():
                config.mcp_servers_path = mcp_file.resolve()

        if config.subagents_path is None:
            subagents_dir = workspace / "subagents"
            if subagents_dir.exists() and subagents_dir.is_dir():
                config.subagents_path = subagents_dir.resolve()

        if config.checkpoint_path is None:
            config.checkpoint_path = (
                workspace / ".contextclaw" / "session.json"
            ).resolve()

        return config
