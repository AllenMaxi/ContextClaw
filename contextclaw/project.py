from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .simple_yaml import dump_yaml
from .workflow import WorkflowConfig, load_workflow, write_workflow

PROJECT_RUNTIME_DIR = ".contextclaw"
WORKFLOW_FILENAME = "Workflow.md"
PROJECT_MEMORY_FILENAME = "AGENTS.md"
AGENT_MEMORY_FILENAME = "MEMORY.md"
DEFAULT_TEMPLATE_BODY = {
    "default": "You are a helpful assistant.",
    "research": "You are a research assistant that finds, validates, and synthesizes information.",
    "coding": "You are a coding assistant that helps write, review, and debug code.",
}

DEFAULT_PROJECT_MEMORY_TEXT = (
    "# Project Memory\n\n"
    "Use this file for project-wide durable context that every agent should see.\n\n"
    "## Suggested Sections\n\n"
    "- Architecture notes\n"
    "- Team conventions\n"
    "- Constraints and non-goals\n"
    "- Important decisions\n"
)


@dataclass(frozen=True)
class ProjectLayout:
    root: Path
    workflow_path: Path
    agents_dir: Path
    runtime_dir: Path


def find_project_root(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / WORKFLOW_FILENAME).exists() or (candidate / "agents").is_dir():
            return candidate
    return None


def get_project_layout(start: Path | None = None) -> ProjectLayout | None:
    root = find_project_root(start)
    if root is None:
        return None
    workflow_path = root / WORKFLOW_FILENAME
    agents_dir_name = "agents"
    if workflow_path.exists():
        config, _ = load_workflow(workflow_path)
        agents_dir_name = config.agents_dir or "agents"
    agents_dir = (root / agents_dir_name).resolve()
    runtime_dir = (root / PROJECT_RUNTIME_DIR).resolve()
    return ProjectLayout(
        root=root,
        workflow_path=workflow_path,
        agents_dir=agents_dir,
        runtime_dir=runtime_dir,
    )


def ensure_project(root: Path, *, entry_agent: str = "orchestrator") -> ProjectLayout:
    root = root.resolve()
    runtime_dir = root / PROJECT_RUNTIME_DIR
    runtime_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = root / WORKFLOW_FILENAME
    if workflow_path.exists():
        config, _ = load_workflow(workflow_path)
    else:
        config = WorkflowConfig(entry_agent=entry_agent)
        write_workflow(workflow_path, config)
    project_memory_path = root / PROJECT_MEMORY_FILENAME
    if not project_memory_path.exists():
        project_memory_path.write_text(DEFAULT_PROJECT_MEMORY_TEXT, encoding="utf-8")
    agents_dir = (root / (config.agents_dir or "agents")).resolve()
    agents_dir.mkdir(parents=True, exist_ok=True)
    return ProjectLayout(
        root=root,
        workflow_path=workflow_path,
        agents_dir=agents_dir,
        runtime_dir=runtime_dir.resolve(),
    )


def default_policy_text(workspace: Path) -> str:
    return dump_yaml(
        {
            "permissions": {
                "tools": {
                    "require_confirm": [
                        "shell_execute",
                        "execute",
                        "memory_propose",
                        "docs_propose",
                    ],
                    "blocked": [],
                },
                "filesystem": {"allowed": [str(workspace.resolve())]},
            },
            "sandbox": {"type": "process"},
        }
    )


def scaffold_agent_workspace(
    layout: ProjectLayout,
    *,
    name: str,
    template: str = "default",
    provider: str = "claude",
    include_project_agents: bool = True,
) -> Path:
    workspace = layout.agents_dir / name
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "skills").mkdir(parents=True, exist_ok=True)
    (workspace / "subagents").mkdir(parents=True, exist_ok=True)
    (workspace / ".contextclaw").mkdir(parents=True, exist_ok=True)

    config_path = workspace / "config.yaml"
    if not config_path.exists():
        config_payload = {
            "name": name,
            "provider": provider,
            "sandbox_type": "process",
            "tools": "filesystem,web,shell,planning",
            "policy_path": "policy.yaml",
            "mcp_servers_path": "mcp_servers.json",
            "checkpoint_path": ".contextclaw/session.json",
        }
        if include_project_agents:
            config_payload["project_agents_path"] = ".."
        config_path.write_text(
            dump_yaml(config_payload) + "\n",
            encoding="utf-8",
        )

    soul_path = workspace / "SOUL.md"
    if not soul_path.exists():
        body = DEFAULT_TEMPLATE_BODY.get(template, DEFAULT_TEMPLATE_BODY["default"])
        soul_path.write_text(
            (
                f"---\nname: {name}\nrole: {template}\ntone: professional\n"
                f"verbosity: concise\n---\n\n{body}\n"
            ),
            encoding="utf-8",
        )

    memory_path = workspace / AGENT_MEMORY_FILENAME
    if not memory_path.exists():
        memory_path.write_text(
            (
                "# Agent Memory\n\n"
                f"This file is always loaded for the `{name}` agent.\n\n"
                "Use it for durable agent-specific context such as preferences, "
                "constraints, and recent operating notes.\n"
            ),
            encoding="utf-8",
        )

    policy_path = workspace / "policy.yaml"
    if not policy_path.exists():
        policy_path.write_text(default_policy_text(workspace) + "\n", encoding="utf-8")

    mcp_path = workspace / "mcp_servers.json"
    if not mcp_path.exists():
        mcp_path.write_text(
            json.dumps({"servers": []}, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    return workspace


def resolve_agent_workspace(
    name: str,
    *,
    project_layout: ProjectLayout | None,
    legacy_agents_dir: Path,
) -> Path:
    if project_layout is not None:
        return project_layout.agents_dir / name
    return legacy_agents_dir / name
