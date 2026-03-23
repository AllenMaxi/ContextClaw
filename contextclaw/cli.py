from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .catalog_engine import (
    catalog_lock_path,
    catalog_state_path,
    catalog_sync_required,
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

AGENTS_DIR = Path.home() / ".contextclaw" / "agents"


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


def _require_agent_workspace(name: str) -> Path:
    workspace = AGENTS_DIR / name
    if not workspace.exists():
        print(f"Agent '{name}' not found.", file=sys.stderr)
        sys.exit(1)
    return workspace


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
        f"Generated MCP Registry: {generated_mcp_path(workspace) if generated_mcp_path(workspace).exists() else 'none'}"
    )
    print(
        f"Generated Policy: {generated_policy_path(workspace) if generated_policy_path(workspace).exists() else 'none'}"
    )


def _load_catalog_or_exit(kind: str) -> dict[str, Any]:
    try:
        if kind == "connectors":
            return load_connector_specs()
        return load_skill_specs()
    except ValueError as exc:
        print(f"Catalog error: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new agent workspace."""
    workspace = AGENTS_DIR / args.name
    workspace.mkdir(parents=True, exist_ok=True)

    # Copy template SOUL.md + config.yaml
    template = args.template or "default"
    provider = args.provider or "claude"

    # Write config.yaml
    config = {
        "name": args.name,
        "provider": provider,
        "sandbox_type": "process",
        "tools": "filesystem,web,shell,planning",
    }
    config_text = "\n".join(f"{k}: {v}" for k, v in config.items())
    (workspace / "config.yaml").write_text(config_text)

    # Write default SOUL.md based on template
    soul_templates = {
        "default": "You are a helpful assistant.",
        "research": "You are a research assistant that finds, validates, and synthesizes information.",
        "coding": "You are a coding assistant that helps write, review, and debug code.",
    }
    soul_body = soul_templates.get(template, soul_templates["default"])
    soul_text = f"---\nname: {args.name}\nrole: {template}\ntone: professional\nverbosity: concise\n---\n\n{soul_body}\n"
    (workspace / "SOUL.md").write_text(soul_text)

    print(f"Created agent '{args.name}' at {workspace}")


def cmd_start(args: argparse.Namespace) -> None:
    """Start the agent runtime (sandbox)."""
    workspace = AGENTS_DIR / args.name
    if not workspace.exists():
        print(
            f"Agent '{args.name}' not found. Run: cclaw create {args.name}",
            file=sys.stderr,
        )
        sys.exit(1)

    from .config import AgentConfig

    config = AgentConfig.from_dir(workspace)

    # Start sandbox
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


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat with an agent."""
    workspace = AGENTS_DIR / args.name
    if not workspace.exists():
        print(f"Agent '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    from .config import AgentConfig
    from .runtime import (
        create_knowledge,
        create_policy,
        create_provider,
        create_sandbox,
        create_tools_sync,
    )

    # Build the runner
    config = AgentConfig.from_dir(workspace)

    # Create provider
    provider = create_provider(config)

    # Create sandbox
    sandbox = create_sandbox(config)
    tools = create_tools_sync(config)

    # Create knowledge bridge if configured
    knowledge = create_knowledge(config)

    # Create policy engine
    policy = create_policy(config)

    # Create runner
    from .runner import AgentRunner

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

    print(f"Chatting with '{args.name}'. Type 'exit' or Ctrl+C to quit.\n")

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
                    elif event.type == "tool_result":
                        pass  # Don't print raw tool results
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


async def _cli_tool_approver(tool_call: Any) -> bool:
    """Ask the operator to approve a tool call."""
    args_preview = json.dumps(tool_call.arguments, ensure_ascii=True)
    prompt = f"Approve tool '{tool_call.name}' with arguments {args_preview}? [y/N]: "
    answer = await asyncio.to_thread(input, prompt)
    return answer.strip().lower() in {"y", "yes"}


def cmd_status(args: argparse.Namespace) -> None:
    """Check agent status."""
    workspace = _require_agent_workspace(args.name)

    from .config import AgentConfig

    config = AgentConfig.from_dir(workspace)
    desired_state = read_catalog_state(workspace)
    sync_required = catalog_sync_required(workspace)
    installed_connectors, installed_skills, missing_env = _status_from_lock(workspace)
    generated_mcp = generated_mcp_path(workspace)
    generated_policy = generated_policy_path(workspace)

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
        f"Catalog State: {catalog_state_path(workspace) if catalog_state_path(workspace).exists() else 'none'}"
    )
    print(
        f"Catalog Lock: {catalog_lock_path(workspace) if catalog_lock_path(workspace).exists() else 'none'}"
    )
    if catalog_state_path(workspace).exists():
        print(f"Catalog Sync: {'required' if sync_required else 'up to date'}")
    else:
        print("Catalog Sync: not initialized")
    print(f"Missing Env: {_format_mapping(missing_env)}")
    print(
        f"Missing Connector Deps: {_format_mapping(missing_connector_dependencies_from_lock(workspace))}"
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
    workspace = _require_agent_workspace(args.name)
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
    workspace = _require_agent_workspace(args.name)
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
    workspace = _require_agent_workspace(args.name)
    sync_agent_catalog(workspace)
    _print_sync_summary(args.name, workspace)


def cmd_skills_list(args: argparse.Namespace) -> None:
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
    workspace = _require_agent_workspace(args.name)
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
    workspace = _require_agent_workspace(args.name)
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
    workspace = _require_agent_workspace(args.name)
    sync_agent_catalog(workspace)
    _print_sync_summary(args.name, workspace)


def cmd_link(args: argparse.Namespace) -> None:
    """Link agent to ContextGraph."""
    workspace = AGENTS_DIR / args.name
    if not workspace.exists():
        print(f"Agent '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    config_path = workspace / "config.yaml"
    lines = config_path.read_text().splitlines() if config_path.exists() else []
    updates = {"cg_url": args.cg_url}

    if args.api_key.startswith("${") or args.api_key.startswith("env:"):
        updates["cg_api_key"] = args.api_key
    else:
        updates["cg_api_key"] = "${CONTEXTGRAPH_API_KEY}"
        print(
            f"WARNING: For security, set CONTEXTGRAPH_API_KEY in your environment "
            f"instead of storing the key in config.yaml.\n"
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


def main() -> None:
    from .logging_config import setup_logging

    parser = argparse.ArgumentParser(
        prog="cclaw", description="ContextClaw — Knowledge-aware agent orchestrator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p = sub.add_parser("create", help="Create a new agent workspace")
    p.add_argument("name")
    p.add_argument(
        "--template", default="default", choices=["default", "research", "coding"]
    )
    p.add_argument(
        "--provider", default="claude", choices=["claude", "openai", "ollama"]
    )
    p.set_defaults(func=cmd_create)

    # start
    p = sub.add_parser("start", help="Start the agent runtime")
    p.add_argument("name")
    p.set_defaults(func=cmd_start)

    # chat
    p = sub.add_parser("chat", help="Interactive chat with an agent")
    p.add_argument("name")
    p.set_defaults(func=cmd_chat)

    # status
    p = sub.add_parser("status", help="Check agent status")
    p.add_argument("name")
    p.set_defaults(func=cmd_status)

    # link
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

    # Global flags
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
