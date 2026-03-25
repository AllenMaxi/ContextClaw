from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .context_engine import DEFAULT_MEMORY_POLICY, normalize_memory_policy
from .simple_yaml import dump_yaml, parse_yaml

_FRONTMATTER_DELIM = "---"


@dataclass
class WorkflowConfig:
    version: int = 1
    entry_agent: str = "orchestrator"
    agents_dir: str = "agents"
    routing_rules: list[dict[str, Any]] = field(default_factory=list)
    delegation_rules: list[dict[str, Any]] = field(default_factory=list)
    approval_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "queue",
            "require": ["shell", "mcp", "repo_write", "memory", "docs"],
        }
    )
    memory_policy: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_MEMORY_POLICY)
    )
    docs_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "review_queue",
            "roots": ["docs", "docs/runbooks", "docs/decisions"],
        }
    )
    contextgraph: dict[str, Any] = field(
        default_factory=lambda: {"required": False, "sync_mode": "deferred"}
    )

    def to_frontmatter(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "entry_agent": self.entry_agent,
            "agents_dir": self.agents_dir,
            "routing_rules": self.routing_rules,
            "delegation_rules": self.delegation_rules,
            "approval_policy": self.approval_policy,
            "memory_policy": self.memory_policy,
            "docs_policy": self.docs_policy,
            "contextgraph": self.contextgraph,
        }


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    stripped = text.lstrip()
    if not stripped.startswith(_FRONTMATTER_DELIM):
        return {}, text

    lines = stripped.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return {}, text

    frontmatter_lines: list[str] = []
    body_index = 1
    while body_index < len(lines):
        line = lines[body_index]
        if line.strip() == _FRONTMATTER_DELIM:
            body_index += 1
            break
        frontmatter_lines.append(line)
        body_index += 1

    body = "\n".join(lines[body_index:]).strip()
    raw = parse_yaml("\n".join(frontmatter_lines))
    return raw if isinstance(raw, dict) else {}, body


def workflow_from_dict(data: dict[str, Any]) -> WorkflowConfig:
    return WorkflowConfig(
        version=int(data.get("version", 1) or 1),
        entry_agent=str(data.get("entry_agent", "orchestrator") or "orchestrator"),
        agents_dir=str(data.get("agents_dir", "agents") or "agents"),
        routing_rules=[
            item
            for item in list(data.get("routing_rules", []))
            if isinstance(item, dict)
        ],
        delegation_rules=[
            item
            for item in list(data.get("delegation_rules", []))
            if isinstance(item, dict)
        ],
        approval_policy=dict(data.get("approval_policy", {}) or {}),
        memory_policy=normalize_memory_policy(
            dict(data.get("memory_policy", {}) or {})
        ),
        docs_policy=dict(data.get("docs_policy", {}) or {}),
        contextgraph=dict(data.get("contextgraph", {}) or {}),
    )


def load_workflow(path: Path) -> tuple[WorkflowConfig, str]:
    raw, body = _split_frontmatter(path.read_text(encoding="utf-8"))
    return workflow_from_dict(raw), body


def validate_workflow(
    config: WorkflowConfig, *, project_root: Path | None = None
) -> list[str]:
    issues: list[str] = []
    if not config.entry_agent.strip():
        issues.append("entry_agent is required")
    agents_dir = Path(config.agents_dir)
    if agents_dir.is_absolute():
        issues.append("agents_dir must be relative to the project root")
    for index, rule in enumerate(config.routing_rules):
        agent = str(rule.get("agent", "")).strip()
        keywords = [str(item).strip() for item in list(rule.get("keywords", []))]
        if not agent:
            issues.append(f"routing_rules[{index}] is missing agent")
        if not keywords:
            issues.append(f"routing_rules[{index}] must include at least one keyword")
        if project_root is not None and agent:
            candidate = project_root / config.agents_dir / agent
            if not candidate.exists():
                issues.append(
                    f"routing_rules[{index}] points to missing agent workspace '{agent}'"
                )
    docs_roots = [
        str(item).strip() for item in list(config.docs_policy.get("roots", []))
    ]
    if not docs_roots:
        issues.append("docs_policy.roots must not be empty")
    memory_policy = normalize_memory_policy(config.memory_policy)
    if config.memory_policy != memory_policy:
        issues.append(
            "memory_policy contains invalid values and will be normalized at runtime"
        )
    if memory_policy["context_window_tokens"] <= memory_policy["reserve_tokens"]:
        issues.append("memory_policy.context_window_tokens must exceed reserve_tokens")
    return issues


def route_prompt(
    config: WorkflowConfig, prompt: str
) -> tuple[str, dict[str, Any] | None]:
    lowered = prompt.lower()
    for rule in config.routing_rules:
        keywords = [str(item).lower() for item in list(rule.get("keywords", []))]
        match_mode = str(rule.get("match", "any")).strip().lower() or "any"
        if not keywords:
            continue
        matches = [
            bool(re.search(r"\b" + re.escape(keyword) + r"\b", lowered))
            for keyword in keywords
        ]
        if (match_mode == "all" and all(matches)) or (
            match_mode != "all" and any(matches)
        ):
            agent = str(rule.get("agent", "")).strip()
            if agent:
                return agent, rule
    return config.entry_agent, None


def default_workflow_body() -> str:
    return (
        "# ContextClaw Workflow\n\n"
        "This file is the project router contract for ContextClaw Studio.\n\n"
        "## Routing Notes\n\n"
        "- Add `routing_rules` keywords in the frontmatter to route obvious tasks directly.\n"
        "- Leave `entry_agent` as `orchestrator` when you want the orchestrator agent to decide.\n"
        "- Keep agent-specific personality, tools, and skills in `agents/<name>/`.\n"
    )


def render_workflow_markdown(config: WorkflowConfig, body: str | None = None) -> str:
    rendered_body = body if body is not None else default_workflow_body()
    frontmatter = dump_yaml(config.to_frontmatter())
    return f"{_FRONTMATTER_DELIM}\n{frontmatter}\n{_FRONTMATTER_DELIM}\n\n{rendered_body.strip()}\n"


def write_workflow(path: Path, config: WorkflowConfig, body: str | None = None) -> Path:
    path.write_text(render_workflow_markdown(config, body=body), encoding="utf-8")
    return path
