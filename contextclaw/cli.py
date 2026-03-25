from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from .catalog_engine import (
    catalog_lock_path,
    catalog_state_path,
    catalog_sync_required,
    connector_bundles_from_lock,
    generated_mcp_path,
    generated_policy_path,
    installed_connectors_from_lock,
    installed_skills_from_lock,
    load_connector_specs,
    load_skill_specs,
    missing_connector_dependencies_from_lock,
    missing_env_from_lock,
    read_catalog_state,
    sync_agent_catalog,
    validate_connector_prerequisites,
    write_catalog_state,
)
from .chat.session import ChatSession
from .context_engine import ContextController
from .memory_files import (
    list_memory_file_revisions as list_memory_file_revisions_entry,
    list_tracked_memory_files,
    read_memory_file_revision as read_memory_file_revision_entry,
    render_memory_files_prompt,
    restore_memory_file_revision as restore_memory_file_revision_entry,
)
from .memory_files import read_memory_file as read_memory_file_entry
from .memory_files import write_memory_file as write_memory_file_entry
from .project import (
    ProjectLayout,
    get_project_layout,
    resolve_agent_workspace,
    scaffold_agent_workspace,
)
from .tools.manager import ToolManager

AGENTS_DIR = Path.home() / ".contextclaw" / "agents"


def _active_project_layout(start: Path | None = None) -> ProjectLayout | None:
    return get_project_layout(start or Path.cwd())


def _legacy_layout() -> ProjectLayout:
    root = AGENTS_DIR.parent
    return ProjectLayout(
        root=root,
        workflow_path=root / "Workflow.md",
        agents_dir=AGENTS_DIR,
        runtime_dir=root / ".contextclaw",
    )


def _rewrite_config(
    lines: list[str], updates: dict[str, str], remove_keys: set[str] | None = None
) -> list[str]:
    remove_keys = remove_keys or set()
    managed_keys = set(updates) | set(remove_keys)
    filtered = [
        line
        for line in lines
        if not any(line.startswith(f"{key}:") for key in managed_keys)
    ]
    for key, value in updates.items():
        filtered.append(f"{key}: {value}")
    return filtered


def _resolve_workspace_for_name(name: str) -> tuple[Path, ProjectLayout | None]:
    layout = _active_project_layout()
    workspace = resolve_agent_workspace(
        name,
        project_layout=layout,
        legacy_agents_dir=AGENTS_DIR,
    )
    return workspace, layout


def _require_agent_workspace(name: str) -> tuple[Path, ProjectLayout | None]:
    workspace, layout = _resolve_workspace_for_name(name)
    if not workspace.exists():
        print(f"Agent '{name}' not found.", file=sys.stderr)
        sys.exit(1)
    return workspace, layout


def _require_project_service(root: Path | None = None):
    from .studio.service import StudioService

    layout = get_project_layout(root or Path.cwd())
    if layout is None:
        print(
            "No ContextClaw project found here. Run `cclaw project init` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    service = StudioService()
    service.open_project(layout.root, initialize=False)
    return service, layout


def _format_items(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _format_mapping(mapping: dict[str, list[str]]) -> str:
    if not mapping:
        return "none"
    parts = []
    for key in sorted(mapping):
        values = ", ".join(mapping[key])
        parts.append(f"{key} ({values})")
    return "; ".join(parts)


def _status_from_lock(
    workspace: Path,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    return (
        installed_connectors_from_lock(workspace),
        installed_skills_from_lock(workspace),
        missing_env_from_lock(workspace),
    )


def _print_connector_summary(connector_id: str, missing_env: list[str]) -> None:
    if missing_env:
        joined = ", ".join(missing_env)
        print(f"Missing env vars for '{connector_id}': {joined}")


def _print_sync_summary(name: str, workspace: Path) -> None:
    connectors, skills, missing_env = _status_from_lock(workspace)
    print(f"Synchronized catalog for '{name}'.")
    print(f"Installed connectors: {_format_items(connectors)}")
    print(f"Installed skills: {_format_items(skills)}")
    print(f"Missing env: {_format_mapping(missing_env)}")
    print(
        "Generated MCP Registry: "
        f"{generated_mcp_path(workspace) if generated_mcp_path(workspace).exists() else 'none'}"
    )
    print(
        "Generated Policy: "
        f"{generated_policy_path(workspace) if generated_policy_path(workspace).exists() else 'none'}"
    )


def _load_catalog_or_exit(kind: str) -> dict[str, Any]:
    try:
        if kind == "connectors":
            return load_connector_specs()
        return load_skill_specs()
    except ValueError as exc:
        print(f"Catalog error: {exc}", file=sys.stderr)
        sys.exit(1)


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _build_system_prompt(config: Any) -> str:
    system_prompt = ""
    if config.soul_path and config.soul_path.exists():
        from .config.soul import load_soul

        system_prompt = load_soul(config.soul_path).body

    from .config.skills import render_skills_prompt

    skills_prompt = render_skills_prompt(config.skills_path)
    if skills_prompt:
        if system_prompt:
            system_prompt += "\n\n"
        system_prompt += skills_prompt
    memory_prompt, _ = render_memory_files_prompt(config.workspace)
    sections = [item.strip() for item in (system_prompt, memory_prompt) if item.strip()]
    return "\n\n".join(sections)


def _tool_payloads_for_config(config: Any) -> tuple[list[dict[str, Any]], list[str]]:
    tools = ToolManager()
    bundle_names: list[str] = []
    for bundle_name in config.tools + connector_bundles_from_lock(config.workspace):
        if bundle_name not in bundle_names:
            bundle_names.append(bundle_name)
    for bundle_name in bundle_names:
        tools.register_bundle(bundle_name)
    dynamic_registries: list[str] = []
    if config.mcp_servers_path and config.mcp_servers_path.exists():
        dynamic_registries.append(str(config.mcp_servers_path))
    generated_mcp = generated_mcp_path(config.workspace)
    if generated_mcp.exists():
        dynamic_registries.append(str(generated_mcp))
    return tools.list_tools(), dynamic_registries


def _legacy_context_state(
    name: str,
) -> tuple[Any, ContextController, dict[str, Any], str]:
    workspace, _ = _require_agent_workspace(name)

    from .config import AgentConfig

    config = AgentConfig.from_dir(workspace)
    system_prompt = _build_system_prompt(config)
    controller = ContextController(workspace)
    checkpoint_path = config.checkpoint_path
    payload: dict[str, Any] | None = None
    if checkpoint_path and checkpoint_path.exists():
        try:
            raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            raw = {}
        if isinstance(raw, dict):
            payload = raw
    if payload is None:
        payload = {
            "session": ChatSession(
                system_prompt=system_prompt, max_history=0
            ).to_dict(),
            "total_usage": {},
        }
    return config, controller, payload, system_prompt


def _print_context_report(context: dict[str, Any]) -> None:
    budget = context["budget"]
    print(f"Agent: {context['agent']}")
    print(f"Workspace: {context['workspace']}")
    print(
        "Token Budget: "
        f"{budget['total_tokens']} / {budget['available_tokens']} "
        f"({budget['status']}, ratio={budget['ratio']})"
    )
    print(
        "Messages: "
        f"{budget['message_count']}  "
        f"Working Memory: {'present' if budget['working_memory_exists'] else 'empty'}"
    )
    pending = context.get("pending_compact")
    if pending:
        print(
            "Pending Compact: "
            f"trim={pending.get('trimmed_message_count', 0)} "
            f"keep={pending.get('kept_message_count', 0)} "
            f"after={pending.get('budget_after', {}).get('status', 'unknown')}"
        )
    else:
        print("Pending Compact: none")
    working_memory = context.get("working_memory") or {}
    sections = (
        working_memory.get("sections", {}) if isinstance(working_memory, dict) else {}
    )
    current_goal = str(sections.get("current_goal", "")).strip()
    if current_goal:
        print(f"Current Goal: {current_goal}")
    dynamic_registries = context.get("dynamic_tool_registries", [])
    if dynamic_registries:
        print("Tool Estimate: static bundle snapshot (dynamic MCP tools not expanded)")
    memory_files = context.get("memory_files", [])
    if memory_files:
        print("Always-Loaded Memory Files:")
        for item in memory_files:
            print(
                f"- {item.get('scope', 'memory')}: "
                f"{item.get('filename', '')} "
                f"({item.get('token_estimate', 0)} tokens, "
                f"{item.get('revision_count', 0)} revisions)"
            )


def _print_compact_preview(preview: dict[str, Any]) -> None:
    print(
        "Compaction Preview: "
        f"trim={preview['trimmed_message_count']} "
        f"keep={preview['kept_message_count']} "
        f"status={preview['budget_after']['status']}"
    )
    print(
        "Budget After: "
        f"{preview['budget_after']['total_tokens']} / "
        f"{preview['budget_after']['available_tokens']}"
    )
    if preview.get("trimmed_previews"):
        print("Trimmed History:")
        for item in preview["trimmed_previews"]:
            print(f"- {_preview(item, limit=120)}")
    rendered = str(preview.get("working_memory_preview", "")).strip()
    if rendered:
        print("\nWorking Memory Preview:")
        print(rendered)


def _print_run_event(event: dict[str, Any]) -> None:
    payload = event.get("payload", {})
    event_type = event.get("type", "")
    if event_type == "message.assistant":
        content = str(payload.get("content", "")).strip()
        if content:
            print(f"\nAssistant: {content}")
    elif event_type == "tool.call":
        print(f"  [calling {payload.get('name', 'tool')}...]")
    elif event_type == "tool.result":
        result = str(payload.get("result", ""))
        lowered = result.lower()
        if any(term in lowered for term in ("error", "blocked", "denied", "approval")):
            print(f"  [tool result: {_preview(result)}]")
    elif event_type == "memory.proposed":
        print(f"  [memory proposed: {payload.get('id', '')}]")
    elif event_type == "memory.recalled":
        print(
            "  [memory recalled: "
            f"{payload.get('count', len(payload.get('memories', [])))} "
            f"items via {payload.get('mode', 'claims')}]"
        )
    elif event_type == "docs.proposed":
        print(f"  [docs proposed: {payload.get('path', '')}]")
    elif event_type == "context.budget":
        print(
            "  [context "
            f"{payload.get('status', 'unknown')}: "
            f"{payload.get('total_tokens', 0)}/"
            f"{payload.get('available_tokens', 0)} tokens]"
        )
    elif event_type == "context.compact.preview":
        print(
            "  [compact preview: "
            f"trim {payload.get('trimmed_message_count', 0)} "
            f"keep {payload.get('kept_message_count', 0)}]"
        )
    elif event_type == "context.compacted":
        suffix = ""
        if payload.get("stored_memory_id"):
            suffix = f" -> {payload.get('stored_memory_id')}"
        print(f"  [context compacted: {payload.get('artifact_path', '')}{suffix}]")
    elif event_type == "run.failed":
        print(f"\n[Run failed: {payload.get('error', 'unknown error')}]")


def _handle_pending_approvals(
    service: Any,
    run_id: str,
    prompted: set[str],
) -> None:
    for approval in service.list_approvals(pending_only=True):
        if approval["run_id"] != run_id or approval["id"] in prompted:
            continue
        prompted.add(approval["id"])
        args_preview = json.dumps(approval["arguments"], ensure_ascii=True)
        prompt = (
            f"Approve tool '{approval['tool_name']}' "
            f"with arguments {args_preview}? [y/N]: "
        )
        answer = input(prompt)
        service.resolve_approval(
            approval["id"],
            approved=answer.strip().lower() in {"y", "yes"},
        )


def _wait_for_run(service: Any, run_id: str) -> dict[str, Any] | None:
    seen_events: set[int] = set()
    prompted_approvals: set[str] = set()
    while True:
        for event in service.list_run_events(run_id):
            event_id = int(event.get("id", 0))
            if event_id in seen_events:
                continue
            seen_events.add(event_id)
            _print_run_event(event)

        _handle_pending_approvals(service, run_id, prompted_approvals)
        run = service.get_run(run_id)
        if run and run["status"] in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.1)


def cmd_project_init(args: argparse.Namespace) -> None:
    from .studio.service import StudioService

    root = Path(args.root).resolve()
    studio = StudioService()
    layout = get_project_layout(root)
    if layout is not None:
        result = studio.open_project(layout.root, initialize=False)
    else:
        result = studio.init_project(
            root,
            entry_agent=args.entry_agent,
            provider=args.provider,
        )
    print(f"Initialized project at {result['root']}")
    print(f"Workflow: {result['workflow_path']}")
    print(f"Agents: {result['agents_dir']}")


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new agent workspace."""
    layout = _active_project_layout()
    include_project_agents = layout is not None
    workspace = scaffold_agent_workspace(
        layout or _legacy_layout(),
        name=args.name,
        template=args.template or "default",
        provider=args.provider or "claude",
        include_project_agents=include_project_agents,
    )
    print(f"Created agent '{args.name}' at {workspace}")


def cmd_start(args: argparse.Namespace) -> None:
    """Start the agent runtime (sandbox)."""
    workspace, _ = _require_agent_workspace(args.name)

    from .config import AgentConfig

    config = AgentConfig.from_dir(workspace)

    print(f"Starting agent '{args.name}' with {config.sandbox_type} sandbox...")
    if config.sandbox_type == "docker":
        from .sandbox.docker import DockerSandbox

        sandbox = DockerSandbox(workspace=config.workspace)
        asyncio.run(sandbox.start())
        print(f"Agent '{args.name}' started in Docker container.")
    elif config.sandbox_type == "process":
        print(f"Agent '{args.name}' ready (process sandbox, no persistent daemon).")
    else:
        print(f"Agent '{args.name}' ready (no sandbox).")


def _chat_direct(workspace: Path, agent_name: str) -> None:
    from .config import AgentConfig
    from .runtime import (
        create_knowledge,
        create_policy,
        create_provider,
        create_sandbox,
        create_tools_sync,
    )
    from .runner import AgentRunner

    config = AgentConfig.from_dir(workspace)
    provider = create_provider(config)
    sandbox = create_sandbox(config)
    tools = create_tools_sync(config)
    knowledge = create_knowledge(config)
    policy = create_policy(config)

    runner = AgentRunner(
        config=config,
        provider=provider,
        sandbox=sandbox,
        tools=tools,
        knowledge=knowledge,
        policy=policy,
        tool_approver=_cli_tool_approver,
        provider_factory=create_provider,
    )

    print(f"Chatting with '{agent_name}'. Type 'exit' or Ctrl+C to quit.\n")

    async def chat_loop():
        if sandbox:
            await sandbox.start()
        try:
            while True:
                try:
                    user_input = input("You: ")
                except EOFError:
                    break
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue

                async for event in runner.run(user_input):
                    if event.type == "text":
                        print(f"\nAssistant: {event.data['content']}")
                    elif event.type == "tool_call":
                        print(f"  [calling {event.data['name']}...]")
                    elif event.type == "context_budget":
                        print(
                            "  [context "
                            f"{event.data.get('status', 'unknown')}: "
                            f"{event.data.get('total_tokens', 0)}/"
                            f"{event.data.get('available_tokens', 0)} tokens]"
                        )
                    elif event.type == "knowledge_recalled":
                        print(
                            "  [memory recalled: "
                            f"{event.data.get('count', len(event.data.get('memories', [])))} "
                            f"items via {event.data.get('mode', 'claims')}]"
                        )
                    elif event.type == "compact_preview":
                        print(
                            "  [compact preview: "
                            f"trim {event.data.get('trimmed_message_count', 0)} "
                            f"keep {event.data.get('kept_message_count', 0)}]"
                        )
                    elif event.type == "context_compacted":
                        suffix = ""
                        if event.data.get("stored_memory_id"):
                            suffix = f" -> {event.data.get('stored_memory_id')}"
                        print(
                            "  [context compacted: "
                            f"{event.data.get('artifact_path', '')}{suffix}]"
                        )
                    elif event.type == "error":
                        print(f"\n[Error: {event.data['message']}]")
                print()
        finally:
            stored = await runner.close_session()
            if stored:
                print(f"\n[Stored {len(stored)} memories to ContextGraph]")
            if sandbox:
                await sandbox.stop()
            await tools.stop_all()

    asyncio.run(chat_loop())


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat with an agent."""
    workspace, layout = _require_agent_workspace(args.name)
    if layout is None:
        _chat_direct(workspace, args.name)
        return

    service, _ = _require_project_service(layout.root)
    print(
        f"Chatting with '{args.name}' in project mode. "
        "Desktop Studio will see the same runs.\n"
    )
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue
        run = service.start_run(user_input, source="cli", agent_name=args.name)
        _wait_for_run(service, run["id"])
        print()


async def _cli_tool_approver(tool_call: Any) -> bool:
    """Ask the operator to approve a tool call."""
    args_preview = json.dumps(tool_call.arguments, ensure_ascii=True)
    prompt = f"Approve tool '{tool_call.name}' with arguments {args_preview}? [y/N]: "
    answer = await asyncio.to_thread(input, prompt)
    return answer.strip().lower() in {"y", "yes"}


def cmd_status(args: argparse.Namespace) -> None:
    """Check agent status."""
    workspace, layout = _require_agent_workspace(args.name)

    from .config import AgentConfig

    config = AgentConfig.from_dir(workspace)
    desired_state = read_catalog_state(workspace)
    sync_required = catalog_sync_required(workspace)
    installed_connectors, installed_skills, missing_env = _status_from_lock(workspace)
    generated_mcp = generated_mcp_path(workspace)
    generated_policy = generated_policy_path(workspace)

    if layout is not None:
        print(f"Project: {layout.root}")
    print(f"Agent: {config.name}")
    print(f"Workspace: {config.workspace}")
    print(f"Provider: {config.provider}")
    print(f"Sandbox: {config.sandbox_type}")
    print(f"Tools: {', '.join(config.tools) if config.tools else 'none'}")
    print(f"Desired Connectors: {_format_items(desired_state.connectors)}")
    print(f"Desired Skills: {_format_items(desired_state.skills)}")
    print(f"Installed Connectors: {_format_items(installed_connectors)}")
    print(f"Installed Skills: {_format_items(installed_skills)}")
    print(
        "Catalog State: "
        f"{catalog_state_path(workspace) if catalog_state_path(workspace).exists() else 'none'}"
    )
    print(
        "Catalog Lock: "
        f"{catalog_lock_path(workspace) if catalog_lock_path(workspace).exists() else 'none'}"
    )
    if catalog_state_path(workspace).exists():
        print(f"Catalog Sync: {'required' if sync_required else 'up to date'}")
    else:
        print("Catalog Sync: not initialized")
    print(f"Missing Env: {_format_mapping(missing_env)}")
    print(
        "Missing Connector Deps: "
        f"{_format_mapping(missing_connector_dependencies_from_lock(workspace))}"
    )
    print(f"ContextGraph: {'linked' if config.cg_url else 'not linked'}")
    print(f"Skills: {config.skills_path if config.skills_path else 'none'}")
    print(
        f"MCP Registry: {config.mcp_servers_path if config.mcp_servers_path else 'none'}"
    )
    print(
        f"Generated MCP Registry: {generated_mcp if generated_mcp.exists() else 'none'}"
    )
    print(
        f"Generated Policy: {generated_policy if generated_policy.exists() else 'none'}"
    )
    print(f"Subagents: {config.subagents_path if config.subagents_path else 'none'}")
    print(f"Checkpoint: {config.checkpoint_path if config.checkpoint_path else 'none'}")
    if config.agent_id:
        print(f"Agent ID: {config.agent_id}")
    elif config.cg_url:
        print(
            "Agent ID: none (run `cclaw link ... --register` to enable ContextGraph recall/store)"
        )


def cmd_connectors_list(args: argparse.Namespace) -> None:
    del args
    specs = _load_catalog_or_exit("connectors")
    for connector_id in sorted(specs):
        spec = specs[connector_id]
        tags = f" [{', '.join(spec.tags)}]" if spec.tags else ""
        print(f"{spec.id}: {spec.display_name} ({spec.type}, {spec.stability}){tags}")
        print(f"  {spec.description}")


def cmd_connectors_info(args: argparse.Namespace) -> None:
    specs = _load_catalog_or_exit("connectors")
    spec = specs.get(args.connector_id)
    if spec is None:
        print(f"Unknown connector '{args.connector_id}'.", file=sys.stderr)
        sys.exit(1)
    print(f"ID: {spec.id}")
    print(f"Name: {spec.display_name}")
    print(f"Version: {spec.version}")
    print(f"Type: {spec.type}")
    print(f"Stability: {spec.stability}")
    print(f"Description: {spec.description}")
    print(f"Tags: {_format_items(spec.tags)}")
    print(f"Bundles: {_format_items(spec.bundles)}")
    print(f"Tools Exposed: {_format_items(spec.tools_exposed)}")
    print(f"Required Env: {_format_items(sorted(spec.required_env))}")
    print(f"Prerequisites: {_format_items(sorted(spec.prerequisites))}")
    if spec.mcp is not None:
        print(f"MCP Server: {spec.mcp.name}")
        print(f"MCP Command: {' '.join(spec.mcp.command)}")


def cmd_connectors_install(args: argparse.Namespace) -> None:
    workspace, _ = _require_agent_workspace(args.name)
    connector_specs = _load_catalog_or_exit("connectors")
    spec = connector_specs.get(args.connector_id)
    if spec is None:
        print(f"Unknown connector '{args.connector_id}'.", file=sys.stderr)
        sys.exit(1)

    state = read_catalog_state(workspace)
    already_installed = spec.id in state.connectors
    if not already_installed:
        missing_prereq = validate_connector_prerequisites([spec.id], connector_specs)
        if missing_prereq.get(spec.id):
            joined = ", ".join(missing_prereq[spec.id])
            print(
                f"Cannot install connector '{spec.id}': missing prerequisites: {joined}",
                file=sys.stderr,
            )
            sys.exit(1)
        state.connectors.append(spec.id)
    write_catalog_state(workspace, state)

    try:
        result = sync_agent_catalog(workspace)
    except ValueError as exc:
        print(f"Catalog error: {exc}", file=sys.stderr)
        sys.exit(1)

    if already_installed:
        print(f"Connector '{spec.id}' is already installed in '{args.name}'.")
    else:
        print(f"Installed connector '{spec.id}' into '{args.name}'.")
    _print_connector_summary(spec.id, result.missing_env.get(spec.id, []))
    _print_sync_summary(args.name, workspace)


def cmd_connectors_remove(args: argparse.Namespace) -> None:
    workspace, _ = _require_agent_workspace(args.name)
    connector_specs = _load_catalog_or_exit("connectors")
    if args.connector_id not in connector_specs:
        print(f"Unknown connector '{args.connector_id}'.", file=sys.stderr)
        sys.exit(1)

    state = read_catalog_state(workspace)
    state.connectors = [
        connector_id
        for connector_id in state.connectors
        if connector_id != args.connector_id
    ]
    write_catalog_state(workspace, state)
    sync_agent_catalog(workspace)
    print(f"Removed connector '{args.connector_id}' from '{args.name}'.")
    _print_sync_summary(args.name, workspace)


def cmd_connectors_sync(args: argparse.Namespace) -> None:
    workspace, _ = _require_agent_workspace(args.name)
    sync_agent_catalog(workspace)
    _print_sync_summary(args.name, workspace)


def cmd_skills_list(args: argparse.Namespace) -> None:
    del args
    specs = _load_catalog_or_exit("skills")
    for skill_id in sorted(specs):
        spec = specs[skill_id]
        tags = f" [{', '.join(spec.tags)}]" if spec.tags else ""
        print(f"{spec.id}: {spec.display_name} ({spec.stability}){tags}")
        print(f"  {spec.description}")


def cmd_skills_info(args: argparse.Namespace) -> None:
    specs = _load_catalog_or_exit("skills")
    spec = specs.get(args.skill_id)
    if spec is None:
        print(f"Unknown skill '{args.skill_id}'.", file=sys.stderr)
        sys.exit(1)
    print(f"ID: {spec.id}")
    print(f"Name: {spec.display_name}")
    print(f"Version: {spec.version}")
    print(f"Stability: {spec.stability}")
    print(f"Description: {spec.description}")
    print(f"Tags: {_format_items(spec.tags)}")
    print(f"Requires Connectors: {_format_items(spec.requires_connectors)}")
    print(f"Asset Dirs: {_format_items(spec.asset_dirs)}")
    print(f"Entrypoint: {spec.entrypoint}")


def cmd_skills_install(args: argparse.Namespace) -> None:
    workspace, _ = _require_agent_workspace(args.name)
    connector_specs = _load_catalog_or_exit("connectors")
    skill_specs = _load_catalog_or_exit("skills")
    spec = skill_specs.get(args.skill_id)
    if spec is None:
        print(f"Unknown skill '{args.skill_id}'.", file=sys.stderr)
        sys.exit(1)

    state = read_catalog_state(workspace)
    auto_installed: list[str] = []
    if not args.no_deps:
        for connector_id in spec.requires_connectors:
            if connector_id not in state.connectors:
                auto_installed.append(connector_id)
                state.connectors.append(connector_id)
        missing_prereq = validate_connector_prerequisites(
            auto_installed, connector_specs
        )
        if missing_prereq:
            problems = []
            for connector_id, commands in sorted(missing_prereq.items()):
                problems.append(f"{connector_id} ({', '.join(commands)})")
            print(
                "Cannot install skill because required connectors are missing prerequisites: "
                + "; ".join(problems),
                file=sys.stderr,
            )
            sys.exit(1)
    if spec.id not in state.skills:
        state.skills.append(spec.id)
    write_catalog_state(workspace, state)

    try:
        result = sync_agent_catalog(workspace)
    except ValueError as exc:
        print(f"Catalog error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Installed skill '{spec.id}' into '{args.name}'.")
    if auto_installed:
        print(f"Auto-installed connectors: {', '.join(sorted(auto_installed))}")
    if args.no_deps and result.missing_connector_dependencies.get(spec.id):
        print(
            f"Missing connector dependencies for '{spec.id}': "
            + ", ".join(result.missing_connector_dependencies[spec.id])
        )
    _print_sync_summary(args.name, workspace)


def cmd_skills_remove(args: argparse.Namespace) -> None:
    workspace, _ = _require_agent_workspace(args.name)
    skill_specs = _load_catalog_or_exit("skills")
    if args.skill_id not in skill_specs:
        print(f"Unknown skill '{args.skill_id}'.", file=sys.stderr)
        sys.exit(1)
    state = read_catalog_state(workspace)
    state.skills = [skill_id for skill_id in state.skills if skill_id != args.skill_id]
    write_catalog_state(workspace, state)
    sync_agent_catalog(workspace)
    print(f"Removed skill '{args.skill_id}' from '{args.name}'.")
    _print_sync_summary(args.name, workspace)


def cmd_skills_sync(args: argparse.Namespace) -> None:
    workspace, _ = _require_agent_workspace(args.name)
    sync_agent_catalog(workspace)
    _print_sync_summary(args.name, workspace)


def cmd_link(args: argparse.Namespace) -> None:
    """Link agent to ContextGraph."""
    workspace, _ = _require_agent_workspace(args.name)

    config_path = workspace / "config.yaml"
    lines = config_path.read_text().splitlines() if config_path.exists() else []
    updates = {"cg_url": args.cg_url}

    if args.api_key.startswith("${") or args.api_key.startswith("env:"):
        updates["cg_api_key"] = args.api_key
    else:
        updates["cg_api_key"] = "${CONTEXTGRAPH_API_KEY}"
        print(
            "WARNING: For security, set CONTEXTGRAPH_API_KEY in your environment "
            "instead of storing the key in config.yaml.\n"
            f"  export CONTEXTGRAPH_API_KEY='{args.api_key}'",
            file=sys.stderr,
        )

    lines = _rewrite_config(lines, updates)

    if args.register:
        from .config.agent_config import _resolve_env
        from .knowledge import ContextGraphBridge

        resolved_api_key = _resolve_env(
            args.api_key, env_fallback="CONTEXTGRAPH_API_KEY"
        )
        if not resolved_api_key:
            print(
                "ContextGraph registration requires an API key or an environment variable reference that resolves now.",
                file=sys.stderr,
            )
            sys.exit(1)

        capabilities = args.capability or []
        bridge = ContextGraphBridge(cg_url=args.cg_url, api_key=resolved_api_key)
        try:
            agent_id = bridge.register(
                name=args.name,
                org_id=args.org_id,
                capabilities=capabilities,
            )
        except Exception as exc:  # noqa: BLE001
            config_path.write_text("\n".join(lines) + "\n")
            print(
                f"Linked '{args.name}' to ContextGraph at {args.cg_url}",
                file=sys.stderr,
            )
            print(f"Failed to register agent with ContextGraph: {exc}", file=sys.stderr)
            sys.exit(1)

        registration_updates = {"agent_id": agent_id}
        if bridge.api_key:
            registration_updates["cg_api_key"] = "${CONTEXTGRAPH_AGENT_KEY}"
            print(
                "WARNING: ContextGraph issued a dedicated agent API key. "
                "Set it in your environment before chatting:\n"
                f"  export CONTEXTGRAPH_AGENT_KEY='{bridge.api_key}'",
                file=sys.stderr,
            )
        lines = _rewrite_config(lines, registration_updates)
        print(f"Registered '{args.name}' with ContextGraph as {agent_id}")

    config_path.write_text("\n".join(lines) + "\n")
    print(f"Linked '{args.name}' to ContextGraph at {args.cg_url}")


def cmd_runs_list(args: argparse.Namespace) -> None:
    del args
    service, _ = _require_project_service()
    runs = service.list_runs()
    if not runs:
        print("No runs recorded.")
        return
    for run in runs:
        print(
            f"{run['id']}  {run['status']}  {run['selected_agent']}  "
            f"{_preview(run['prompt'], limit=80)}"
        )


def cmd_approvals_list(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    approvals = service.list_approvals(pending_only=args.pending_only)
    if not approvals:
        print("No approvals found.")
        return
    for approval in approvals:
        print(
            f"{approval['id']}  {approval['status']}  {approval['tool_name']}  "
            f"run={approval['run_id']}"
        )


def cmd_approval_resolve(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.resolve_approval(args.approval_id, approved=args.approved)
    if result is None:
        print(f"Approval '{args.approval_id}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"{result['status'].capitalize()} approval '{args.approval_id}'.")


def cmd_memory_list(args: argparse.Namespace) -> None:
    del args
    service, _ = _require_project_service()
    proposals = service.list_memory_proposals()
    if not proposals:
        print("No memory proposals.")
        return
    for proposal in proposals:
        pin = " pinned" if proposal["pinned"] else ""
        print(
            f"{proposal['id']}  {proposal['status']}{pin}  "
            f"{proposal['agent_name']}  {_preview(proposal['content'], 90)}"
        )


def cmd_memory_pin(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.pin_memory(args.proposal_id, pinned=not args.unpin)
    if result is None:
        print(f"Memory proposal '{args.proposal_id}' not found.", file=sys.stderr)
        sys.exit(1)
    state = "unpinned" if args.unpin else "pinned"
    print(f"{state.capitalize()} memory proposal '{args.proposal_id}'.")


def cmd_memory_sync(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.sync_memory_proposal(args.proposal_id)
    if result is None:
        print(f"Memory proposal '{args.proposal_id}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Synced memory proposal '{args.proposal_id}'.")


def cmd_memory_reject(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.discard_memory(args.proposal_id)
    if result is None:
        print(f"Memory proposal '{args.proposal_id}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Rejected memory proposal '{args.proposal_id}'.")


def cmd_memory_compact(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.compact_memory(args.proposal_ids, summary=args.summary)
    print(f"Created compacted memory proposal '{result['id']}'.")


def cmd_memory_file_list(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        items = service.list_memory_files(agent_name=args.name)
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        workspace, _ = _require_agent_workspace(args.name)
        items = list_tracked_memory_files(workspace)
    if not items:
        print("No memory files found.")
        return
    for item in items:
        print(
            f"{item['scope']}  {item['filename']}  "
            f"{item['token_estimate']} tokens  "
            f"{item.get('revision_count', 0)} revisions  {item['path']}"
        )


def cmd_memory_file_show(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        if args.revision_id:
            result = service.read_memory_file_revision(
                scope=args.scope,
                revision_id=args.revision_id,
                agent_name=args.name,
                filename=args.filename,
            )
        else:
            result = service.read_memory_file(
                scope=args.scope,
                agent_name=args.name,
                filename=args.filename,
            )
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        workspace, _ = _require_agent_workspace(args.name)
        if args.scope != "agent":
            print(
                "Only agent-scoped memory files are supported outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.revision_id:
            result = read_memory_file_revision_entry(
                workspace,
                scope=args.scope,
                revision_id=args.revision_id,
                filename=args.filename,
            )
        else:
            result = read_memory_file_entry(
                workspace,
                scope=args.scope,
                filename=args.filename,
            )
    print(f"Scope: {result['scope']}")
    print(f"File: {result['filename']}")
    print(f"Path: {result['path']}")
    print(f"Tokens: {result['token_estimate']}")
    if result.get("current_revision_id"):
        print(f"Current Revision: {result['current_revision_id']}")
    if result.get("revision", {}).get("id"):
        print(f"Selected Revision: {result['revision']['id']}")
    print()
    print(result["content"])


def cmd_memory_file_write(args: argparse.Namespace) -> None:
    content = args.content
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    if content is None:
        print("Provide --content or --file.", file=sys.stderr)
        sys.exit(1)
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        result = service.write_memory_file(
            scope=args.scope,
            content=content,
            agent_name=args.name,
            filename=args.filename,
            append=args.append,
        )
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        workspace, _ = _require_agent_workspace(args.name)
        if args.scope != "agent":
            print(
                "Only agent-scoped memory files are supported outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        result = write_memory_file_entry(
            workspace,
            scope=args.scope,
            content=content,
            filename=args.filename,
            append=args.append,
        )
    print(f"Updated {result['scope']} memory file {result['filename']}.")
    print(f"Path: {result['path']}")


def cmd_memory_file_sync(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.sync_memory_file(
        scope=args.scope,
        agent_name=args.name,
        filename=args.filename,
        revision_id=args.revision_id,
    )
    print(
        "Synced memory file: "
        f"{result['scope']} {result['filename']} -> {result['synced_memory_id'] or 'unknown'}"
    )
    revision = result.get("revision") or {}
    if revision.get("id"):
        print(f"Revision: {revision['id']}")
    if result.get("already_synced"):
        print("Status: already synced")


def cmd_memory_file_history(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        result = service.list_memory_file_revisions(
            scope=args.scope,
            agent_name=args.name,
            filename=args.filename,
        )
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        workspace, _ = _require_agent_workspace(args.name)
        if args.scope != "agent":
            print(
                "Only agent-scoped memory files are supported outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        result = list_memory_file_revisions_entry(
            workspace,
            scope=args.scope,
            filename=args.filename,
        )
    print(f"Scope: {result['scope']}")
    print(f"File: {result['filename']}")
    print(f"Path: {result['path']}")
    print(f"Current Revision: {result.get('current_revision_id') or 'none'}")
    print(f"Last Synced Revision: {result.get('last_synced_revision_id') or 'none'}")
    revisions = result.get("revisions", [])
    if not revisions:
        print("No revisions recorded.")
        return
    print()
    for item in revisions:
        flags = []
        if item.get("current"):
            flags.append("current")
        if item.get("synced_memory_id"):
            flags.append(f"synced={item['synced_memory_id']}")
        if item.get("source_revision_id"):
            flags.append(f"from={item['source_revision_id']}")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        print(
            f"{item['id']}  {item.get('action', 'unknown')}  "
            f"{item.get('token_estimate', 0)} tokens{suffix}"
        )


def cmd_memory_file_restore(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        result = service.restore_memory_file(
            scope=args.scope,
            revision_id=args.revision_id,
            agent_name=args.name,
            filename=args.filename,
        )
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        workspace, _ = _require_agent_workspace(args.name)
        if args.scope != "agent":
            print(
                "Only agent-scoped memory files are supported outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        result = restore_memory_file_revision_entry(
            workspace,
            scope=args.scope,
            revision_id=args.revision_id,
            filename=args.filename,
        )
    print(
        "Restored memory file: "
        f"{result['scope']} {result['filename']} "
        f"from {result['restored_from_revision_id']}"
    )
    print(f"Current Revision: {result.get('current_revision_id') or 'none'}")


def cmd_context_show(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        context = service.get_context(agent_name=args.name)
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        config, controller, payload, system_prompt = _legacy_context_state(args.name)
        tool_payloads, dynamic_registries = _tool_payloads_for_config(config)
        context = {
            "agent": config.name,
            "workspace": str(config.workspace),
            "budget": controller.inspect_state(
                payload,
                system_prompt=system_prompt,
                tools=tool_payloads,
            ),
            "working_memory": controller.load_working_memory(),
            "pending_compact": controller.load_pending_compact(),
            "dynamic_tool_registries": dynamic_registries,
            "memory_files": list_tracked_memory_files(config.workspace),
        }
    _print_context_report(context)


def cmd_compact_preview(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        result = service.preview_compact(
            agent_name=args.name,
            reason=args.reason,
        )
        preview = result["preview"]
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        config, controller, payload, system_prompt = _legacy_context_state(args.name)
        tool_payloads, _ = _tool_payloads_for_config(config)
        preview = controller.preview_compact(
            payload,
            system_prompt=system_prompt,
            tools=tool_payloads,
            reason=args.reason,
        )
    _print_compact_preview(preview)


def cmd_compact_apply(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        result = service.apply_compact(
            agent_name=args.name,
            reason=args.reason,
        )["result"]
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        config, controller, payload, system_prompt = _legacy_context_state(args.name)
        tool_payloads, _ = _tool_payloads_for_config(config)
        session = ChatSession.from_dict(
            payload.get("session", {}),
            system_prompt=system_prompt,
            max_history=0,
        )
        updated_session, result = controller.apply_compact_to_chat_session(
            session,
            system_prompt=system_prompt,
            tools=tool_payloads,
            total_usage=payload.get("total_usage", {}),
            reason=args.reason,
        )
        checkpoint_path = config.checkpoint_path
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    {
                        **payload,
                        "session": updated_session.to_dict(),
                        "saved_at": time.time(),
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
    print(
        "Applied compaction: "
        f"artifact={result['artifact_path']} "
        f"working_memory={result['working_memory_path']}"
    )


def cmd_compact_reject(args: argparse.Namespace) -> None:
    layout = _active_project_layout()
    if layout is not None:
        service, _ = _require_project_service(layout.root)
        result = service.reject_compact(agent_name=args.name)
    else:
        if not args.name:
            print(
                "An agent name is required outside project mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        _, controller, _, _ = _legacy_context_state(args.name)
        result = controller.reject_pending_compact()
    if result["rejected"]:
        print("Rejected pending compaction preview.")
    else:
        print("No pending compaction preview was present.")


def cmd_docs_list(args: argparse.Namespace) -> None:
    del args
    service, _ = _require_project_service()
    proposals = service.list_docs_proposals()
    if not proposals:
        print("No docs proposals.")
        return
    for proposal in proposals:
        print(
            f"{proposal['id']}  {proposal['status']}  "
            f"{proposal['path']}  {_preview(proposal['summary'], 80)}"
        )


def cmd_docs_apply(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.apply_docs_proposal(args.proposal_id)
    if result is None:
        print(f"Docs proposal '{args.proposal_id}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Applied docs proposal '{args.proposal_id}' to {result['path']}.")


def cmd_docs_reject(args: argparse.Namespace) -> None:
    service, _ = _require_project_service()
    result = service.reject_docs_proposal(args.proposal_id)
    if result is None:
        print(f"Docs proposal '{args.proposal_id}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Rejected docs proposal '{args.proposal_id}'.")


def cmd_studio_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        print(
            "uvicorn is required for `cclaw studio serve`. "
            "Install the `studio` extra first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from .studio.daemon import create_app
    from .studio.service import StudioService

    root = Path(args.root).resolve()
    service = StudioService()
    layout = get_project_layout(root)
    if layout is not None:
        service.open_project(layout.root, initialize=False)
    elif args.init:
        service.init_project(root, entry_agent=args.entry_agent, provider=args.provider)
    else:
        print(
            "No ContextClaw project found at that path. Pass --init to scaffold one.",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run(create_app(service), host=args.host, port=args.port)


def main() -> None:
    from .logging_config import setup_logging

    parser = argparse.ArgumentParser(
        prog="cclaw", description="ContextClaw — Knowledge-aware agent orchestrator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="Create a new agent workspace")
    p.add_argument("name")
    p.add_argument(
        "--template", default="default", choices=["default", "research", "coding"]
    )
    p.add_argument(
        "--provider", default="claude", choices=["claude", "openai", "ollama"]
    )
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("start", help="Start the agent runtime")
    p.add_argument("name")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("chat", help="Interactive chat with an agent")
    p.add_argument("name")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("status", help="Check agent status")
    p.add_argument("name")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("link", help="Link agent to ContextGraph")
    p.add_argument("name")
    p.add_argument("--cg-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument(
        "--register",
        action="store_true",
        help="Register the agent with ContextGraph now and persist agent_id",
    )
    p.add_argument(
        "--org-id",
        default="default",
        help="ContextGraph org_id to use when --register is set",
    )
    p.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Capability to register with ContextGraph (repeatable)",
    )
    p.set_defaults(func=cmd_link)

    project = sub.add_parser("project", help="Initialize and manage project-local mode")
    project_sub = project.add_subparsers(dest="project_command", required=True)

    p = project_sub.add_parser("init", help="Initialize Workflow.md and agents/")
    p.add_argument("--root", default=".")
    p.add_argument("--entry-agent", default="orchestrator")
    p.add_argument(
        "--provider", default="claude", choices=["claude", "openai", "ollama"]
    )
    p.set_defaults(func=cmd_project_init)

    studio = sub.add_parser("studio", help="Run the local desktop/web control plane")
    studio_sub = studio.add_subparsers(dest="studio_command", required=True)

    p = studio_sub.add_parser("serve", help="Serve the Studio daemon and dashboard")
    p.add_argument("--root", default=".")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--init", action="store_true")
    p.add_argument("--entry-agent", default="orchestrator")
    p.add_argument(
        "--provider", default="claude", choices=["claude", "openai", "ollama"]
    )
    p.set_defaults(func=cmd_studio_serve)

    runs = sub.add_parser("runs", help="Inspect project runs")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)

    p = runs_sub.add_parser("list", help="List project runs")
    p.set_defaults(func=cmd_runs_list)

    approvals = sub.add_parser("approvals", help="Inspect or resolve approvals")
    approvals_sub = approvals.add_subparsers(dest="approvals_command", required=True)

    p = approvals_sub.add_parser("list", help="List approvals")
    p.add_argument("--pending-only", action="store_true")
    p.set_defaults(func=cmd_approvals_list)

    p = approvals_sub.add_parser("approve", help="Approve a pending approval")
    p.add_argument("approval_id")
    p.set_defaults(func=cmd_approval_resolve, approved=True)

    p = approvals_sub.add_parser("deny", help="Deny a pending approval")
    p.add_argument("approval_id")
    p.set_defaults(func=cmd_approval_resolve, approved=False)

    memory = sub.add_parser("memory", help="Inspect and manage memory proposals")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)

    p = memory_sub.add_parser("list", help="List memory proposals")
    p.set_defaults(func=cmd_memory_list)

    p = memory_sub.add_parser("pin", help="Pin a memory proposal")
    p.add_argument("proposal_id")
    p.add_argument("--unpin", action="store_true")
    p.set_defaults(func=cmd_memory_pin)

    p = memory_sub.add_parser("sync", help="Sync a memory proposal to ContextGraph")
    p.add_argument("proposal_id")
    p.set_defaults(func=cmd_memory_sync)

    p = memory_sub.add_parser("reject", help="Reject a memory proposal")
    p.add_argument("proposal_id")
    p.set_defaults(func=cmd_memory_reject)

    p = memory_sub.add_parser("compact", help="Compact multiple memory proposals")
    p.add_argument("proposal_ids", nargs="+")
    p.add_argument("--summary", default="")
    p.set_defaults(func=cmd_memory_compact)

    memory_file = sub.add_parser(
        "memory-file", help="Inspect, edit, and sync always-loaded memory files"
    )
    memory_file_sub = memory_file.add_subparsers(
        dest="memory_file_command", required=True
    )

    p = memory_file_sub.add_parser("list", help="List discovered memory files")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_memory_file_list)

    p = memory_file_sub.add_parser("show", help="Show a memory file")
    p.add_argument("scope", choices=["user", "project", "agent"])
    p.add_argument("name", nargs="?")
    p.add_argument("--filename", default="")
    p.add_argument("--revision-id", default="")
    p.set_defaults(func=cmd_memory_file_show)

    p = memory_file_sub.add_parser(
        "history", help="Show revision history for a memory file"
    )
    p.add_argument("scope", choices=["user", "project", "agent"])
    p.add_argument("name", nargs="?")
    p.add_argument("--filename", default="")
    p.set_defaults(func=cmd_memory_file_history)

    p = memory_file_sub.add_parser("write", help="Write a memory file")
    p.add_argument("scope", choices=["user", "project", "agent"])
    p.add_argument("name", nargs="?")
    p.add_argument("--filename", default="")
    p.add_argument("--content")
    p.add_argument("--file")
    p.add_argument("--append", action="store_true")
    p.set_defaults(func=cmd_memory_file_write)

    p = memory_file_sub.add_parser(
        "sync", help="Sync a memory file to ContextGraph as an artifact memory"
    )
    p.add_argument("scope", choices=["user", "project", "agent"])
    p.add_argument("name", nargs="?")
    p.add_argument("--filename", default="")
    p.add_argument("--revision-id", default="")
    p.set_defaults(func=cmd_memory_file_sync)

    p = memory_file_sub.add_parser(
        "restore", help="Restore a memory file to a previous revision"
    )
    p.add_argument("scope", choices=["user", "project", "agent"])
    p.add_argument("name", nargs="?")
    p.add_argument("--filename", default="")
    p.add_argument("--revision-id", required=True)
    p.set_defaults(func=cmd_memory_file_restore)

    context = sub.add_parser(
        "context", help="Inspect token budget and working memory state"
    )
    context_sub = context.add_subparsers(dest="context_command", required=True)

    p = context_sub.add_parser("show", help="Show current context state")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_context_show)

    compact = sub.add_parser(
        "compact", help="Preview, apply, or reject session compaction"
    )
    compact_sub = compact.add_subparsers(dest="compact_command", required=True)

    p = compact_sub.add_parser("preview", help="Preview a context compaction")
    p.add_argument("name", nargs="?")
    p.add_argument("--reason", default="manual_preview")
    p.set_defaults(func=cmd_compact_preview)

    p = compact_sub.add_parser("apply", help="Apply a pending or fresh compaction")
    p.add_argument("name", nargs="?")
    p.add_argument("--reason", default="manual_apply")
    p.set_defaults(func=cmd_compact_apply)

    p = compact_sub.add_parser("reject", help="Reject a pending compaction preview")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_compact_reject)

    docs = sub.add_parser("docs", help="Inspect and manage docs proposals")
    docs_sub = docs.add_subparsers(dest="docs_command", required=True)

    p = docs_sub.add_parser("list", help="List docs proposals")
    p.set_defaults(func=cmd_docs_list)

    p = docs_sub.add_parser("apply", help="Apply a docs proposal")
    p.add_argument("proposal_id")
    p.set_defaults(func=cmd_docs_apply)

    p = docs_sub.add_parser("reject", help="Reject a docs proposal")
    p.add_argument("proposal_id")
    p.set_defaults(func=cmd_docs_reject)

    connectors = sub.add_parser(
        "connectors", help="Browse and manage first-party connector catalog entries"
    )
    connectors_sub = connectors.add_subparsers(dest="connectors_command", required=True)

    p = connectors_sub.add_parser("list", help="List available connectors")
    p.set_defaults(func=cmd_connectors_list)

    p = connectors_sub.add_parser("info", help="Show connector details")
    p.add_argument("connector_id")
    p.set_defaults(func=cmd_connectors_info)

    p = connectors_sub.add_parser(
        "install", help="Install a connector into an agent workspace"
    )
    p.add_argument("name")
    p.add_argument("connector_id")
    p.set_defaults(func=cmd_connectors_install)

    p = connectors_sub.add_parser(
        "remove", help="Remove a connector from an agent workspace"
    )
    p.add_argument("name")
    p.add_argument("connector_id")
    p.set_defaults(func=cmd_connectors_remove)

    p = connectors_sub.add_parser(
        "sync", help="Resolve connector state and regenerate agent artifacts"
    )
    p.add_argument("name")
    p.set_defaults(func=cmd_connectors_sync)

    skills = sub.add_parser(
        "skills", help="Browse and manage first-party packaged skills"
    )
    skills_sub = skills.add_subparsers(dest="skills_command", required=True)

    p = skills_sub.add_parser("list", help="List available packaged skills")
    p.set_defaults(func=cmd_skills_list)

    p = skills_sub.add_parser("info", help="Show packaged skill details")
    p.add_argument("skill_id")
    p.set_defaults(func=cmd_skills_info)

    p = skills_sub.add_parser(
        "install", help="Install a packaged skill into an agent workspace"
    )
    p.add_argument("name")
    p.add_argument("skill_id")
    p.add_argument(
        "--no-deps",
        action="store_true",
        help="Do not auto-install required first-party connectors",
    )
    p.set_defaults(func=cmd_skills_install)

    p = skills_sub.add_parser(
        "remove", help="Remove a packaged skill from an agent workspace"
    )
    p.add_argument("name")
    p.add_argument("skill_id")
    p.set_defaults(func=cmd_skills_remove)

    p = skills_sub.add_parser(
        "sync", help="Resolve packaged skill state and regenerate agent artifacts"
    )
    p.add_argument("name")
    p.set_defaults(func=cmd_skills_sync)

    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--json-logs", action="store_true", help="Output structured JSON logs"
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level, structured=args.json_logs)
    args.func(args)


if __name__ == "__main__":
    main()
