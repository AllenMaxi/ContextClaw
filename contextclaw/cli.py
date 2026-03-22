from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config.agent_config import AgentConfig

AGENTS_DIR = Path.home() / ".contextclaw" / "agents"


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
    workspace = AGENTS_DIR / args.name
    if not workspace.exists():
        print(f"Agent '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    from .config import AgentConfig

    config = AgentConfig.from_dir(workspace)

    print(f"Agent: {config.name}")
    print(f"Workspace: {config.workspace}")
    print(f"Provider: {config.provider}")
    print(f"Sandbox: {config.sandbox_type}")
    print(f"Tools: {', '.join(config.tools) if config.tools else 'none'}")
    print(f"ContextGraph: {'linked' if config.cg_url else 'not linked'}")
    print(f"Skills: {config.skills_path if config.skills_path else 'none'}")
    print(f"MCP Registry: {config.mcp_servers_path if config.mcp_servers_path else 'none'}")
    print(f"Subagents: {config.subagents_path if config.subagents_path else 'none'}")
    print(f"Checkpoint: {config.checkpoint_path if config.checkpoint_path else 'none'}")
    if config.agent_id:
        print(f"Agent ID: {config.agent_id}")


def cmd_link(args: argparse.Namespace) -> None:
    """Link agent to ContextGraph."""
    workspace = AGENTS_DIR / args.name
    if not workspace.exists():
        print(f"Agent '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    # Update config.yaml with CG settings
    config_path = workspace / "config.yaml"
    lines = config_path.read_text().splitlines() if config_path.exists() else []

    # Remove existing cg_ lines
    lines = [
        line
        for line in lines
        if not line.startswith("cg_url:") and not line.startswith("cg_api_key:")
    ]
    lines.append(f"cg_url: {args.cg_url}")

    # Prefer env var reference over plaintext API key
    if args.api_key.startswith("${") or args.api_key.startswith("env:"):
        # User passed an env var reference — store as-is
        lines.append(f"cg_api_key: {args.api_key}")
    else:
        # Store as env var reference and set the env var hint
        lines.append("cg_api_key: ${CONTEXTGRAPH_API_KEY}")
        print(
            f"WARNING: For security, set CONTEXTGRAPH_API_KEY in your environment "
            f"instead of storing the key in config.yaml.\n"
            f"  export CONTEXTGRAPH_API_KEY='{args.api_key}'",
            file=sys.stderr,
        )

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

    # start
    p = sub.add_parser("start", help="Start the agent runtime")
    p.add_argument("name")

    # chat
    p = sub.add_parser("chat", help="Interactive chat with an agent")
    p.add_argument("name")

    # status
    p = sub.add_parser("status", help="Check agent status")
    p.add_argument("name")

    # link
    p = sub.add_parser("link", help="Link agent to ContextGraph")
    p.add_argument("name")
    p.add_argument("--cg-url", required=True)
    p.add_argument("--api-key", required=True)

    # Global flags
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--json-logs", action="store_true", help="Output structured JSON logs"
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level, structured=args.json_logs)

    commands = {
        "create": cmd_create,
        "start": cmd_start,
        "chat": cmd_chat,
        "status": cmd_status,
        "link": cmd_link,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
