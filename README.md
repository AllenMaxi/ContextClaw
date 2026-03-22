<p align="center">
  <img src="assets/contextclaw-mascot.svg" alt="ContextClaw mascot" width="320">
</p>
<p align="center">
  <h1 align="center">ContextClaw</h1>
  <p align="center">
    <strong>Knowledge-aware agent orchestrator powered by ContextGraph</strong>
  </p>
  <p align="center">
    <a href="#"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  </p>
</p>

---

## What is ContextClaw?

ContextClaw is a lightweight agent runtime (~2500 lines) that gives LLM agents **persistent memory, sandbox isolation, and governance** — all wired through [ContextGraph](https://github.com/AllenMaxi/contextgraph) as the knowledge plane.

It combines the best patterns from the Claw ecosystem:
- **OpenClaw's** provider abstraction (swap LLMs without changing agent code)
- **NanoClaw's** minimal footprint (no framework bloat, just Python)
- **PicoClaw's** sandbox-first security (every command runs in isolation)

Then adds what none of them have: **cross-session memory** via ContextGraph.

## Why ContextClaw over the other Claws?

| Feature | OpenClaw | NanoClaw | PicoClaw | **ContextClaw** |
|---------|----------|----------|----------|-----------------|
| Provider swapping | Claude only | OpenAI only | Claude + OpenAI | **Claude + OpenAI + Ollama** |
| Sandbox isolation | None | Process | Docker | **Docker + Process fallback** |
| Policy guardrails | Basic | None | YAML | **YAML + path resolution** |
| Cross-session memory | None | None | None | **ContextGraph integration** |
| Agent discovery | None | None | None | **Trust-scored discovery** |
| Tool management | Hardcoded | Hardcoded | MCP | **Built-in tools, deep-agent aliases, + MCP-ready bundles** |
| Shell metachar detection | None | None | None | **Multi-pass scanning** |
| Planning workflow | None | None | None | **Built-in `write_todos` + `read_todos`** |
| Human approval | None | None | Limited | **Policy-driven tool confirmation** |
| Skills loading | None | None | None | **Markdown skill packs per agent** |
| Rate limiting | None | None | None | **Configurable per-agent** |
| Structured logging | None | None | None | **JSON-lines + human** |

**The honest version:** ContextClaw isn't the smallest (that's NanoClaw) or the most battle-tested (that's OpenClaw). It's the one you pick when you need agents that **remember things between sessions** and **discover each other** through a shared knowledge graph. If you don't need cross-session memory, NanoClaw is simpler. If you just need a single Claude agent, OpenClaw works fine.

### Current Scope

ContextClaw now covers the core "deep agent" runtime path well: built-in and
MCP-backed tools, policy-gated execution, task delegation to sub-agents,
session checkpoints, and ContextGraph-powered memory.

It is not yet at full parity with every connector or packaged integration in
the wider Claw family. The main remaining gap is ecosystem breadth rather than
runtime depth: a broader first-party connector or MCP catalog and a larger set
of reusable packaged skills.

The short version:

- The runtime pieces are in place and working.
- The biggest remaining investment is more turnkey connectors and skills.

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Create an agent
cclaw create research-bot --template research --provider claude

# Start chatting
cclaw chat research-bot
```

That's it. Three commands to a working agent with sandbox isolation, tool access, and built-in planning tools.

### Link to ContextGraph (optional)

To enable cross-session memory, agent discovery, and trust scoring:

```bash
# Set your API key (don't store secrets in config files)
export CONTEXTGRAPH_API_KEY="your-key-here"

# Link your agent
cclaw link research-bot \
  --cg-url http://localhost:8000 \
  --api-key '${CONTEXTGRAPH_API_KEY}'
```

Now your agent will:
1. **Recall** relevant knowledge before each turn
2. **Store** significant outputs after each turn
3. **Summarize** the session on exit — extracting 0-5 key facts worth remembering

## Demo

[![ContextClaw promo demo](../docs/assets/contextclaw-promo.gif)](../docs/assets/contextclaw-promo.mp4)

Generate the vertical demo asset and walkthrough:

```bash
python3 ../examples/contextclaw_promo.py
python3 ../scripts/render_contextclaw_promo.py
```

## Architecture

```
cclaw chat my-agent
        │
        ▼
┌─────────────────────────────────────────────┐
│  AgentRunner (ReAct loop)                   │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ LLM      │  │ Sandbox  │  │ Policy    │ │
│  │ Provider  │  │ Docker/  │  │ Engine    │ │
│  │ Claude/  │  │ Process  │  │ YAML      │ │
│  │ OpenAI/  │  │          │  │ guardrails│ │
│  │ Ollama   │  │          │  │           │ │
│  └──────────┘  └──────────┘  └───────────┘ │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Tool     │  │ Context  │  │ SOUL.md   │ │
│  │ Manager  │  │ Graph    │  │ Agent     │ │
│  │ MCP      │  │ Bridge   │  │ Identity  │ │
│  │ bundles  │  │ Memory   │  │           │ │
│  └──────────┘  └──────────┘  └───────────┘ │
└─────────────────────────────────────────────┘
```

## Features

### Multi-Provider LLM Support

Protocol-based abstraction — swap providers without changing agent code:

```python
from contextclaw.providers.claude import ClaudeProvider
from contextclaw.providers.openai import OpenAIProvider
from contextclaw.providers.ollama import OllamaProvider

# All three implement the same LLMProvider protocol
provider = ClaudeProvider(model="claude-sonnet-4-20250514")
provider = OpenAIProvider(model="gpt-4o")
provider = OllamaProvider(model="llama3.2")
```

### Sandbox Isolation

Every command runs in a sandbox. Two options:

- **Docker sandbox** — Full container isolation, resource limits, non-root execution
- **Process sandbox** — Lightweight path-based protection with multi-layer defense:
  - Shell metacharacter scanning (`$()`, backticks, `eval`, `source`, pipes)
  - `Path.resolve()` for symlink-immune path resolution
  - Configurable blocked paths (`~/.ssh`, `~/.aws`, `/etc` by default)

```python
from contextclaw.sandbox.process import ProcessSandbox

sandbox = ProcessSandbox(workspace=Path("/tmp/agent"))
result = await sandbox.execute("ls -la")

# Blocked automatically:
result = await sandbox.execute("cat ~/.ssh/id_rsa")       # Access denied
result = await sandbox.execute("echo $(cat /etc/passwd)")  # Access denied
result = await sandbox.execute("cat /etc/shadow | head")   # Access denied
```

### ContextGraph Integration

The key differentiator. ContextGraph gives your agents:

- **Recall** — Query relevant knowledge before each LLM turn
- **Store** — Persist significant outputs as curated knowledge
- **Trust** — Agent reputation scores and governance
- **Discovery** — Find other agents by capability and reputation
- **Session summarization** — On exit, extract key facts and store them

```python
from contextclaw.knowledge.bridge import ContextGraphBridge

bridge = ContextGraphBridge(
    cg_url="http://localhost:8000",
    api_key=os.environ["CONTEXTGRAPH_API_KEY"],
    agent_id="agent-123",
)

# Recall before answering
memories = bridge.recall("What does the user prefer?")

# Store after answering
bridge.store("User prefers concise answers", metadata={"type": "preference"})

# Discover other agents
agents = bridge.discover(query="data analysis", min_reputation=0.8)
```

### YAML Policy Guardrails

Fine-grained control over what agents can do:

```yaml
# policy.yaml
permissions:
  tools:
    auto_approve:
      - filesystem_read
      - filesystem_list
    require_confirm:
      - filesystem_write
    blocked:
      - shell_execute
  filesystem:
    allowed:
      - /workspace
    blocked:
      - /workspace/secrets
sandbox:
  type: docker
```

Tools marked `require_confirm` now prompt the operator during `cclaw chat`
before execution.

### Built-in Tools and Planning

ContextClaw ships with working built-in tools for:

- `filesystem_read`
- `filesystem_write`
- `filesystem_list`
- `read_file`
- `write_file`
- `ls`
- `edit_file`
- `glob`
- `grep`
- `web_fetch`
- `web_search`
- `shell_execute`
- `execute`
- `write_todos`
- `read_todos`

Filesystem tools are scoped to the agent workspace by default. `write_todos`
creates a lightweight task plan in `.contextclaw/todos.md`, which gives the
agent a simple planning loop similar to the stronger "deep agent" UX. The
deep-agent-style aliases (`read_file`, `write_file`, `ls`, `edit_file`, `glob`,
`grep`, `execute`, `read_todos`) map cleanly onto the same ContextClaw runtime,
so migrating prompts is low-friction.

### MCP Registry and Invocation

Agents can auto-discover an `mcp_servers.json` file in their workspace and
start MCP servers on chat startup. Each discovered MCP tool is registered as a
first-class model tool using the name format:

```text
mcp__<server_name>__<tool_name>
```

Example registry:

```json
{
  "servers": [
    {
      "name": "demo",
      "command": ["python3", "mock_mcp_server.py"]
    }
  ]
}
```

### Task Delegation

If an agent workspace contains a `subagents/` directory with other
ContextClaw-compatible workspaces, the parent agent automatically gets a
`task` tool for delegating work to them with isolated context.

```text
my-agent/
├── config.yaml
├── SOUL.md
└── subagents/
    └── research-sub/
        ├── config.yaml
        └── SOUL.md
```

The parent can then delegate:

```json
{
  "subagent": "research-sub",
  "prompt": "Summarize the launch positioning."
}
```

### Session Checkpoints

By default, agent chats persist to:

```text
.contextclaw/session.json
```

That means long-lived agents automatically resume prior conversation state and
token accounting the next time `cclaw chat` runs.

### SOUL.md — Agent Identity

Define agent personality, role, and behavior in Markdown:

```markdown
---
name: research-bot
role: research
tone: professional
verbosity: concise
---

You are a research assistant that finds, validates, and synthesizes
information. Always cite your sources and flag uncertainty.
```

### Skills

Agents can also load reusable Markdown skill packs from a `skills/` directory
inside the workspace or from a `skills_path` in `config.yaml`.

```text
my-agent/
├── SOUL.md
├── config.yaml
└── skills/
    ├── research.md
    └── launch-checklist.md
```

Each skill file is appended to the system prompt as an extra capability block,
making it easy to keep role instructions modular instead of overloading one
large `SOUL.md`.

### Structured Logging

JSON-lines output for production, human-readable for development:

```bash
# Human-readable (default)
cclaw chat my-agent --log-level DEBUG

# JSON structured logs for production
cclaw chat my-agent --json-logs
```

```json
{"timestamp": "2026-03-22T10:30:00.123Z", "level": "INFO", "logger": "contextclaw.runner", "message": "ReAct turn 1/20"}
```

### HTTP Chat Server

Built-in HTTP server with SSE streaming, CORS, and bearer token auth:

```python
from contextclaw.chat.server import ChatServer

server = ChatServer(host="127.0.0.1", port=8080, auth_token="secret")
server.set_runner(runner, session)
server.start()
```

```bash
# JSON response
curl -X POST http://localhost:8080/chat \
  -H "Authorization: Bearer secret" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# SSE streaming
curl -X POST http://localhost:8080/chat \
  -H "Accept: text/event-stream" \
  -d '{"message": "Hello"}'
```

### Rate Limiting

Configurable minimum interval between LLM calls to avoid API throttling:

```python
runner = AgentRunner(
    config=config,
    provider=provider,
    min_call_interval=1.0,  # At least 1 second between LLM calls
)
```

## CLI Reference

```bash
cclaw create <name> [--template default|research|coding] [--provider claude|openai|ollama]
cclaw start <name>
cclaw chat <name>
cclaw status <name>
cclaw link <name> --cg-url <url> --api-key <key>
```

Global flags:
```bash
--log-level DEBUG|INFO|WARNING|ERROR
--json-logs    # Structured JSON output
```

## Installation

### From source

```bash
git clone https://github.com/AllenMaxi/ContextClaw.git
cd ContextClaw
pip install -e ".[all]"
```

### Optional dependencies

```bash
pip install -e ".[claude]"     # Anthropic SDK
pip install -e ".[openai]"     # OpenAI SDK
pip install -e ".[knowledge]"  # ContextGraph SDK
pip install -e ".[docker]"     # Docker SDK
pip install -e ".[all]"        # Everything
```

## Testing

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

189 tests covering:
- Agent runner (ReAct loop, retry logic, tool validation, token tracking)
- Sandbox (path traversal, shell metacharacters, Docker, timeouts)
- Policy engine (tool/path permissions)
- Knowledge bridge (recall, store, summarization, JSON parsing)
- Config (env var resolution, YAML parsing)
- Integration (full lifecycle, multi-turn, concurrent access)

## Project Structure

```
contextclaw/
├── chat/
│   ├── server.py        # HTTP + SSE chat server (threaded)
│   └── session.py       # Thread-safe conversation history
├── config/
│   ├── agent_config.py  # YAML config with env var resolution
│   ├── skills.py        # Markdown skill loading and prompt rendering
│   └── soul.py          # SOUL.md parser
├── runtime.py           # Shared runtime builders for providers/tools/policy
├── knowledge/
│   └── bridge.py        # ContextGraph integration
├── providers/
│   ├── protocol.py      # LLMProvider protocol
│   ├── claude.py        # Anthropic provider
│   ├── openai.py        # OpenAI provider
│   └── ollama.py        # Ollama provider
├── sandbox/
│   ├── protocol.py      # Sandbox protocol
│   ├── process.py       # Process sandbox with path protection
│   ├── docker.py        # Docker sandbox with resource limits
│   └── policy.py        # YAML policy engine
├── tools/
│   ├── manager.py       # Tool registry
│   ├── bundles.py       # Pre-built tool bundles
│   └── mcp.py           # MCP stdio client and registry loading
├── runner.py            # AgentRunner (ReAct loop)
├── logging_config.py    # Structured logging setup
└── cli.py               # CLI entry point
```

## Security

- **Sandbox isolation** — Commands run in Docker containers or process sandboxes
- **Path traversal protection** — `Path.resolve()` eliminates `..` and symlink tricks
- **Shell metacharacter detection** — Blocks `$()`, backticks, `eval`, `source`, pipe chains
- **Constant-time auth** — `hmac.compare_digest` for bearer token comparison
- **Credential safety** — Env var resolution (`${VAR}`, `env:VAR`) instead of plaintext secrets
- **Policy guardrails** — YAML-defined tool and filesystem permissions

## ContextGraph

[ContextGraph](https://github.com/AllenMaxi/contextgraph) is the shared knowledge plane that makes ContextClaw's memory work. It provides:

- **Knowledge storage** — Store and query structured knowledge with metadata
- **Agent registry** — Register agents, track reputation, manage trust
- **Discovery** — Find agents by capability with minimum reputation thresholds
- **Governance** — Sentinel-based oversight of agent behavior

ContextClaw works fine without ContextGraph (just no cross-session memory). When linked, it becomes the only Claw with real persistent memory across conversations.

## License

MIT
