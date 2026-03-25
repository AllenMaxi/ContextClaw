from __future__ import annotations

from pathlib import Path

from contextclaw.context_engine import DEFAULT_MEMORY_POLICY
from contextclaw.workflow import (
    WorkflowConfig,
    load_workflow,
    route_prompt,
    validate_workflow,
    workflow_from_dict,
    write_workflow,
)


def test_workflow_round_trip_and_routing(tmp_path: Path):
    workflow_path = tmp_path / "Workflow.md"
    config = WorkflowConfig(
        entry_agent="orchestrator",
        routing_rules=[
            {
                "agent": "researcher",
                "keywords": ["research", "paper"],
                "match": "any",
            }
        ],
        docs_policy={"mode": "review_queue", "roots": ["docs", "docs/decisions"]},
    )

    write_workflow(workflow_path, config, body="# Router\n\nProject workflow.\n")
    loaded, body = load_workflow(workflow_path)
    selected, matched = route_prompt(loaded, "Please research the paper trail")

    assert loaded.entry_agent == "orchestrator"
    assert "Project workflow." in body
    assert selected == "researcher"
    assert matched == config.routing_rules[0]


def test_validate_workflow_reports_missing_agent_workspace(tmp_path: Path):
    (tmp_path / "agents" / "orchestrator").mkdir(parents=True, exist_ok=True)
    config = WorkflowConfig(
        entry_agent="orchestrator",
        routing_rules=[{"agent": "missing-agent", "keywords": ["route"]}],
        docs_policy={"mode": "review_queue", "roots": ["docs"]},
    )

    issues = validate_workflow(config, project_root=tmp_path)

    assert any("missing agent workspace 'missing-agent'" in issue for issue in issues)


def test_workflow_normalizes_invalid_memory_policy_values() -> None:
    config = workflow_from_dict({"memory_policy": {"compact_threshold": 25}})

    assert (
        config.memory_policy["compact_threshold"]
        == DEFAULT_MEMORY_POLICY["compact_threshold"]
    )
