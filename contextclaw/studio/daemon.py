from __future__ import annotations

import asyncio
import json
import os
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import StudioService


class ProjectPayload(BaseModel):
    root: str | None = None
    entry_agent: str = "orchestrator"
    provider: str = "claude"


class WorkflowUpdatePayload(BaseModel):
    config: dict[str, Any] | None = None
    body: str | None = None


class AgentCreatePayload(BaseModel):
    name: str
    template: str = "default"
    provider: str = "claude"


class AgentUpdatePayload(BaseModel):
    soul_body: str | None = None
    config_updates: dict[str, str] = Field(default_factory=dict)


class RunPayload(BaseModel):
    prompt: str
    source: str = "desktop"
    agent_name: str | None = None


class ApprovalPayload(BaseModel):
    approved: bool


class MemoryPinPayload(BaseModel):
    pinned: bool = True


class MemoryCompactionPayload(BaseModel):
    proposal_ids: list[str]
    summary: str = ""


class ConnectorInstallPayload(BaseModel):
    agent_name: str
    connector_id: str


class SkillInstallPayload(BaseModel):
    agent_name: str
    skill_id: str
    no_deps: bool = False


class ContextGraphLinkPayload(BaseModel):
    cg_url: str
    api_key: str
    agent_name: str | None = None
    register_agent: bool = Field(default=False, alias="register")
    org_id: str = "default"
    capabilities: list[str] = Field(default_factory=list)


class CompactPayload(BaseModel):
    agent_name: str | None = None
    reason: str = "manual"


class MemoryFilePayload(BaseModel):
    scope: str
    agent_name: str | None = None
    filename: str = ""
    content: str = ""
    append: bool = False
    revision_id: str = ""


def _frontend_dist_dir() -> Path:
    """Resolve the built frontend directory.

    Checks two locations:
    1. Installed package data: contextclaw/studio/_frontend/ (inside wheel)
    2. Development repo: ../../studio-ui/dist/ (relative to this file)
    """
    # Installed location (pip install contextclaw[studio])
    installed = Path(__file__).resolve().parent / "_frontend"
    if installed.exists() and (installed / "index.html").exists():
        return installed
    # Development location (running from source)
    return Path(__file__).resolve().parents[2] / "studio-ui" / "dist"


def _serialize_studio_event(event: dict[str, Any]) -> str:
    payload = json.dumps(event, default=str)
    return f"id: {event.get('id', '')}\nevent: studio.event\ndata: {payload}\n\n"


def _heartbeat_frame() -> str:
    return 'event: studio.heartbeat\ndata: {"status":"ok"}\n\n'


def create_app(service: StudioService | None = None) -> FastAPI:
    studio = service or StudioService()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> Any:
        try:
            yield
        finally:
            studio.shutdown(timeout=5.0)

    app = FastAPI(title="ContextClaw Studio", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:4173",
            "http://localhost:4173",
            "tauri://localhost",
            "http://tauri.localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    frontend_dist = _frontend_dist_dir()
    frontend_assets = frontend_dist / "assets"
    frontend_ready = frontend_dist.exists() and (frontend_dist / "index.html").exists()
    if frontend_assets.exists():
        app.mount(
            "/studio/assets",
            StaticFiles(directory=frontend_assets),
            name="studio-assets",
        )

    def _coerce_root(raw: str | None) -> Path:
        return Path(raw or ".").resolve()

    def _wrap(callable_obj, *args, **kwargs):
        try:
            return callable_obj(*args, **kwargs)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/status")
    def status() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "contextclaw-studio",
            "frontend_ready": frontend_ready,
            "project_open": studio._layout is not None,
        }

    @app.get("/projects/current")
    def current_project() -> dict[str, Any]:
        return _wrap(studio.current_project)

    @app.post("/projects/open")
    def open_project(payload: ProjectPayload) -> dict[str, Any]:
        return _wrap(studio.open_project, _coerce_root(payload.root), initialize=True)

    @app.post("/projects/init")
    def init_project(payload: ProjectPayload) -> dict[str, Any]:
        return _wrap(
            studio.init_project,
            _coerce_root(payload.root),
            entry_agent=payload.entry_agent,
            provider=payload.provider,
        )

    @app.get("/workflow")
    def get_workflow() -> dict[str, Any]:
        return _wrap(studio.get_workflow)

    @app.put("/workflow")
    def update_workflow(payload: WorkflowUpdatePayload) -> dict[str, Any]:
        return _wrap(
            studio.update_workflow, config_data=payload.config, body=payload.body
        )

    @app.post("/workflow/validate")
    def validate_workflow() -> dict[str, Any]:
        return _wrap(studio.validate_current_workflow)

    @app.get("/agents")
    def list_agents() -> list[dict[str, Any]]:
        return _wrap(studio.list_agents)

    @app.post("/agents")
    def create_agent(payload: AgentCreatePayload) -> dict[str, Any]:
        return _wrap(
            studio.create_agent,
            name=payload.name,
            template=payload.template,
            provider=payload.provider,
        )

    @app.get("/agents/{agent_id}")
    def get_agent(agent_id: str) -> dict[str, Any]:
        return _wrap(studio.get_agent, agent_id)

    @app.put("/agents/{agent_id}")
    def update_agent(agent_id: str, payload: AgentUpdatePayload) -> dict[str, Any]:
        return _wrap(
            studio.update_agent,
            agent_id,
            soul_body=payload.soul_body,
            config_updates=payload.config_updates,
        )

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return _wrap(studio.list_runs)

    @app.post("/runs")
    def start_run(payload: RunPayload) -> dict[str, Any]:
        return _wrap(
            studio.start_run,
            payload.prompt,
            source=payload.source,
            agent_name=payload.agent_name,
        )

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any] | None:
        return _wrap(studio.cancel_run, run_id)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any] | None:
        return _wrap(studio.get_run, run_id)

    @app.get("/runs/{run_id}/events")
    def get_events(run_id: str) -> list[dict[str, Any]]:
        return _wrap(studio.list_run_events, run_id)

    @app.get("/events/stream")
    async def stream_events(
        request: Request,
        run_id: str | None = None,
        heartbeat: float = 20.0,
    ) -> StreamingResponse:
        subscription_id, event_queue = studio.subscribe_events(run_id=run_id)

        async def _stream() -> Any:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.to_thread(
                            event_queue.get,
                            True,
                            heartbeat,
                        )
                    except queue.Empty:
                        yield _heartbeat_frame()
                        continue
                    yield _serialize_studio_event(event)
            finally:
                studio.unsubscribe_events(subscription_id)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/approvals")
    def list_approvals() -> list[dict[str, Any]]:
        return _wrap(studio.list_approvals)

    @app.post("/approvals/{approval_id}/approve")
    def approve(approval_id: str) -> dict[str, Any] | None:
        return _wrap(studio.resolve_approval, approval_id, approved=True)

    @app.post("/approvals/{approval_id}/deny")
    def deny(approval_id: str) -> dict[str, Any] | None:
        return _wrap(studio.resolve_approval, approval_id, approved=False)

    @app.get("/memory")
    def list_memory() -> list[dict[str, Any]]:
        return _wrap(studio.list_memory_proposals)

    @app.get("/memory-files")
    def list_memory_files(agent_name: str | None = None) -> list[dict[str, Any]]:
        return _wrap(studio.list_memory_files, agent_name)

    @app.get("/memory-files/content")
    def get_memory_file(
        scope: str,
        agent_name: str | None = None,
        filename: str = "",
    ) -> dict[str, Any]:
        return _wrap(
            studio.read_memory_file,
            scope=scope,
            agent_name=agent_name,
            filename=filename,
        )

    @app.put("/memory-files/content")
    def put_memory_file(payload: MemoryFilePayload) -> dict[str, Any]:
        return _wrap(
            studio.write_memory_file,
            scope=payload.scope,
            content=payload.content,
            agent_name=payload.agent_name,
            filename=payload.filename,
            append=payload.append,
        )

    @app.get("/memory-files/revisions")
    def get_memory_file_revisions(
        scope: str,
        agent_name: str | None = None,
        filename: str = "",
    ) -> dict[str, Any]:
        return _wrap(
            studio.list_memory_file_revisions,
            scope=scope,
            agent_name=agent_name,
            filename=filename,
        )

    @app.get("/memory-files/revision")
    def get_memory_file_revision(
        scope: str,
        revision_id: str,
        agent_name: str | None = None,
        filename: str = "",
    ) -> dict[str, Any]:
        return _wrap(
            studio.read_memory_file_revision,
            scope=scope,
            revision_id=revision_id,
            agent_name=agent_name,
            filename=filename,
        )

    @app.post("/memory-files/restore")
    def restore_memory_file(payload: MemoryFilePayload) -> dict[str, Any]:
        return _wrap(
            studio.restore_memory_file,
            scope=payload.scope,
            revision_id=payload.revision_id,
            agent_name=payload.agent_name,
            filename=payload.filename,
        )

    @app.post("/memory-files/sync")
    def sync_memory_file(payload: MemoryFilePayload) -> dict[str, Any]:
        return _wrap(
            studio.sync_memory_file,
            scope=payload.scope,
            agent_name=payload.agent_name,
            filename=payload.filename,
            revision_id=payload.revision_id,
        )

    @app.get("/context")
    def get_context(agent_name: str | None = None) -> dict[str, Any]:
        return _wrap(studio.get_context, agent_name)

    @app.post("/compact/preview")
    def preview_compact(payload: CompactPayload) -> dict[str, Any]:
        return _wrap(
            studio.preview_compact,
            payload.agent_name,
            reason=payload.reason,
        )

    @app.post("/compact/apply")
    def apply_compact(payload: CompactPayload) -> dict[str, Any]:
        return _wrap(
            studio.apply_compact,
            payload.agent_name,
            reason=payload.reason,
        )

    @app.post("/compact/reject")
    def reject_compact(payload: CompactPayload) -> dict[str, Any]:
        return _wrap(studio.reject_compact, payload.agent_name)

    @app.post("/memory/{proposal_id}/pin")
    def pin_memory(
        proposal_id: str, payload: MemoryPinPayload
    ) -> dict[str, Any] | None:
        return _wrap(studio.pin_memory, proposal_id, pinned=payload.pinned)

    @app.post("/memory/{proposal_id}/sync")
    def sync_memory(proposal_id: str) -> dict[str, Any] | None:
        return _wrap(studio.sync_memory_proposal, proposal_id)

    @app.post("/memory/{proposal_id}/reject")
    def reject_memory(proposal_id: str) -> dict[str, Any] | None:
        return _wrap(studio.discard_memory, proposal_id)

    @app.post("/memory/compactions")
    def compact_memory(payload: MemoryCompactionPayload) -> dict[str, Any]:
        return _wrap(
            studio.compact_memory,
            payload.proposal_ids,
            summary=payload.summary,
        )

    @app.get("/docs/proposals")
    def list_docs() -> list[dict[str, Any]]:
        return _wrap(studio.list_docs_proposals)

    @app.post("/docs/proposals/{proposal_id}/apply")
    def apply_doc(proposal_id: str) -> dict[str, Any] | None:
        return _wrap(studio.apply_docs_proposal, proposal_id)

    @app.post("/docs/proposals/{proposal_id}/reject")
    def reject_doc(proposal_id: str) -> dict[str, Any] | None:
        return _wrap(studio.reject_docs_proposal, proposal_id)

    @app.get("/connectors")
    def connectors_status() -> dict[str, Any]:
        return _wrap(studio.connectors_status)

    @app.post("/connectors/install")
    def install_connector(payload: ConnectorInstallPayload) -> dict[str, Any]:
        return _wrap(studio.install_connector, payload.agent_name, payload.connector_id)

    @app.post("/skills/install")
    def install_skill(payload: SkillInstallPayload) -> dict[str, Any]:
        return _wrap(
            studio.install_skill,
            payload.agent_name,
            payload.skill_id,
            no_deps=payload.no_deps,
        )

    @app.get("/contextgraph/status")
    def contextgraph_status(agent_name: str | None = None) -> dict[str, Any]:
        return _wrap(studio.contextgraph_status, agent_name=agent_name)

    @app.post("/contextgraph/link")
    def link_contextgraph(payload: ContextGraphLinkPayload) -> dict[str, Any]:
        return _wrap(
            studio.link_contextgraph,
            cg_url=payload.cg_url,
            api_key=payload.api_key,
            agent_name=payload.agent_name,
            register=payload.register_agent,
            org_id=payload.org_id,
            capabilities=payload.capabilities,
        )

    @app.post("/contextgraph/sync")
    def sync_contextgraph() -> dict[str, Any]:
        return _wrap(studio.sync_contextgraph)

    @app.get("/studio", response_class=HTMLResponse)
    def dashboard_root() -> Any:
        if frontend_ready:
            return RedirectResponse(url="/studio/", status_code=307)
        return _render_dashboard()

    @app.get("/studio/", response_class=HTMLResponse)
    def dashboard() -> Any:
        if frontend_ready:
            return FileResponse(frontend_dist / "index.html")
        return _render_dashboard()

    return app


def _render_dashboard() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ContextClaw Studio</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #0d1117; color: #e6edf3; }
    header { padding: 20px 24px; border-bottom: 1px solid #30363d; background: #161b22; }
    main { padding: 24px; display: grid; gap: 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; }
    h1, h2 { margin: 0 0 12px; }
    button { background: #238636; color: white; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
    input, textarea { width: 100%; background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 8px; padding: 10px; margin-bottom: 12px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0d1117; padding: 12px; border-radius: 8px; }
    ul { margin: 0; padding-left: 18px; }
  </style>
</head>
<body>
  <header>
    <h1>ContextClaw Studio</h1>
    <div id="project"></div>
  </header>
  <main>
    <div class="grid">
      <section class="card">
        <h2>Run Prompt</h2>
        <input id="target-agent" placeholder="Optional target agent (leave empty for workflow routing)">
        <textarea id="prompt" rows="6" placeholder="Describe the task for the orchestrator or routed agent"></textarea>
        <button onclick="startRun()">Start Run</button>
        <pre id="run-result"></pre>
      </section>
      <section class="card">
        <h2>Agents</h2>
        <input id="new-agent-name" placeholder="New agent name">
        <input id="new-agent-provider" value="claude" placeholder="Provider">
        <input id="new-agent-template" value="default" placeholder="Template">
        <button onclick="createAgent()">Create Agent</button>
        <div id="agents"></div>
      </section>
      <section class="card">
        <h2>Runs</h2>
        <div id="runs"></div>
      </section>
      <section class="card">
        <h2>Catalog</h2>
        <input id="catalog-agent" placeholder="Agent name">
        <input id="connector-id" placeholder="Connector id">
        <button onclick="installConnector()">Install Connector</button>
        <input id="skill-id" placeholder="Skill id">
        <button onclick="installSkill()">Install Skill</button>
        <pre id="catalog-result"></pre>
      </section>
      <section class="card">
        <h2>Approvals</h2>
        <div id="approvals"></div>
      </section>
      <section class="card">
        <h2>Memory Queue</h2>
        <div id="memory"></div>
      </section>
      <section class="card">
        <h2>Memory Files</h2>
        <input id="memory-file-agent" placeholder="Optional memory-file agent">
        <input id="memory-file-scope" value="agent" placeholder="Scope: user, project, agent">
        <input id="memory-file-name" placeholder="Optional filename (AGENTS.md or MEMORY.md)">
        <input id="memory-file-revision" placeholder="Optional revision id">
        <textarea id="memory-file-content" rows="8" placeholder="Memory file content"></textarea>
        <button onclick="loadMemoryFile()">Load</button>
        <button onclick="loadMemoryRevision()">Load Revision</button>
        <button onclick="loadMemoryHistory()">History</button>
        <button onclick="saveMemoryFile()">Save</button>
        <button onclick="restoreMemoryFile()">Restore Revision</button>
        <button onclick="syncMemoryFile()">Sync To ContextGraph</button>
        <div id="memory-files"></div>
        <pre id="memory-file-history"></pre>
      </section>
      <section class="card">
        <h2>Context</h2>
        <input id="context-agent" placeholder="Optional context agent">
        <button onclick="refreshContext()">Inspect Context</button>
        <button onclick="previewCompact()">Preview Compact</button>
        <button onclick="applyCompact()">Apply Compact</button>
        <button onclick="rejectCompact()">Reject Pending</button>
        <pre id="context"></pre>
      </section>
      <section class="card">
        <h2>Docs Proposals</h2>
        <div id="docs"></div>
      </section>
      <section class="card">
        <h2>Workflow</h2>
        <button onclick="validateWorkflow()">Validate Workflow</button>
        <pre id="workflow"></pre>
      </section>
      <section class="card">
        <h2>ContextGraph</h2>
        <div id="contextgraph"></div>
      </section>
    </div>
  </main>
  <script>
    async function fetchJson(path, options) {
      const response = await fetch(path, options);
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || JSON.stringify(body));
      return body;
    }
    function renderList(targetId, items, renderItem) {
      const root = document.getElementById(targetId);
      if (!items.length) {
        root.innerHTML = '<p>None</p>';
        return;
      }
      root.innerHTML = '<ul>' + items.map(renderItem).join('') + '</ul>';
    }
    async function refresh() {
      try {
        const project = await fetchJson('/projects/current');
        document.getElementById('project').textContent = project.root + ' | branch: ' + (project.git.branch || 'n/a');
      } catch (error) {
        document.getElementById('project').textContent = error.message;
      }
      const agents = await fetchJson('/agents');
      renderList('agents', agents, item => `<li><strong>${item.name}</strong> (${item.provider})</li>`);
      const runs = await fetchJson('/runs');
      renderList('runs', runs.slice(0, 8), item => `<li><strong>${item.selected_agent}</strong> ${item.status}<br>${item.prompt}</li>`);
      const approvals = await fetchJson('/approvals');
      renderList('approvals', approvals, item => `<li>${item.tool_name} <button onclick="resolveApproval('${item.id}', true)">Approve</button> <button onclick="resolveApproval('${item.id}', false)">Deny</button></li>`);
      const memory = await fetchJson('/memory');
      renderList('memory', memory, item => `<li>${item.content}<br><button onclick="pinMemory('${item.id}', ${item.pinned ? 'false' : 'true'})">${item.pinned ? 'Unpin' : 'Pin'}</button> <button onclick="syncMemory('${item.id}')">Sync</button> <button onclick="rejectMemory('${item.id}')">Reject</button></li>`);
      const fileAgent = selectedMemoryFileAgent();
      const fileSuffix = fileAgent ? `?agent_name=${encodeURIComponent(fileAgent)}` : '';
      const memoryFiles = await fetchJson('/memory-files' + fileSuffix);
      renderList('memory-files', memoryFiles, item => `<li>${item.scope}: ${item.filename} (${item.token_estimate} tokens, ${item.revision_count || 0} revisions)<br>current=${item.current_revision_id || 'n/a'} synced=${item.last_synced_revision_id || 'n/a'}<br><button onclick="loadMemoryFile('${item.scope}', '${item.filename}')">Load</button> <button onclick="loadMemoryHistory('${item.scope}', '${item.filename}')">History</button> <button onclick="syncMemoryFile('${item.scope}', '${item.filename}')">Sync</button></li>`);
      const docs = await fetchJson('/docs/proposals');
      renderList('docs', docs, item => `<li>${item.path} <button onclick="applyDoc('${item.id}')">Apply</button> <button onclick="rejectDoc('${item.id}')">Reject</button></li>`);
      const workflow = await fetchJson('/workflow');
      document.getElementById('workflow').textContent = JSON.stringify(workflow.config, null, 2);
      const contextgraph = await fetchJson('/contextgraph/status');
      document.getElementById('contextgraph').innerHTML = '<pre>' + JSON.stringify(contextgraph, null, 2) + '</pre>';
      await refreshContext(false);
    }
    function selectedContextAgent() {
      const explicit = document.getElementById('context-agent').value.trim();
      const target = document.getElementById('target-agent').value.trim();
      return explicit || target || null;
    }
    function selectedMemoryFileAgent() {
      const explicit = document.getElementById('memory-file-agent').value.trim();
      const contextAgent = selectedContextAgent();
      return explicit || contextAgent || null;
    }
    async function refreshContext(raiseOnError = true) {
      try {
        const agent = selectedContextAgent();
        const suffix = agent ? `?agent_name=${encodeURIComponent(agent)}` : '';
        const context = await fetchJson('/context' + suffix);
        document.getElementById('context').textContent = JSON.stringify(context, null, 2);
        return context;
      } catch (error) {
        document.getElementById('context').textContent = error.message;
        if (raiseOnError) throw error;
        return null;
      }
    }
    async function previewCompact() {
      const result = await fetchJson('/compact/preview', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({agent_name: selectedContextAgent(), reason: 'dashboard_preview'})
      });
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function applyCompact() {
      const result = await fetchJson('/compact/apply', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({agent_name: selectedContextAgent(), reason: 'dashboard_apply'})
      });
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function rejectCompact() {
      const result = await fetchJson('/compact/reject', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({agent_name: selectedContextAgent()})
      });
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function startRun() {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) return;
      const agentName = document.getElementById('target-agent').value.trim();
      const run = await fetchJson('/runs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt, source: 'dashboard', agent_name: agentName || null})
      });
      document.getElementById('run-result').textContent = JSON.stringify(run, null, 2);
      setTimeout(refresh, 500);
    }
    async function createAgent() {
      const name = document.getElementById('new-agent-name').value.trim();
      const provider = document.getElementById('new-agent-provider').value.trim() || 'claude';
      const template = document.getElementById('new-agent-template').value.trim() || 'default';
      if (!name) return;
      await fetchJson('/agents', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name, provider, template})
      });
      await refresh();
    }
    async function installConnector() {
      const agentName = document.getElementById('catalog-agent').value.trim();
      const connectorId = document.getElementById('connector-id').value.trim();
      if (!agentName || !connectorId) return;
      const result = await fetchJson('/connectors/install', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({agent_name: agentName, connector_id: connectorId})
      });
      document.getElementById('catalog-result').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function installSkill() {
      const agentName = document.getElementById('catalog-agent').value.trim();
      const skillId = document.getElementById('skill-id').value.trim();
      if (!agentName || !skillId) return;
      const result = await fetchJson('/skills/install', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({agent_name: agentName, skill_id: skillId, no_deps: false})
      });
      document.getElementById('catalog-result').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function validateWorkflow() {
      const result = await fetchJson('/workflow/validate', {method: 'POST'});
      document.getElementById('workflow').textContent = JSON.stringify(result, null, 2);
    }
    async function resolveApproval(id, approved) {
      await fetchJson(`/approvals/${id}/${approved ? 'approve' : 'deny'}`, {method: 'POST'});
      await refresh();
    }
    async function pinMemory(id, pinned) {
      await fetchJson(`/memory/${id}/pin`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pinned})
      });
      await refresh();
    }
    async function syncMemory(id) {
      await fetchJson(`/memory/${id}/sync`, {method: 'POST'});
      await refresh();
    }
    async function rejectMemory(id) {
      await fetchJson(`/memory/${id}/reject`, {method: 'POST'});
      await refresh();
    }
    async function loadMemoryFile(scopeOverride = null, filenameOverride = null) {
      const scope = scopeOverride || document.getElementById('memory-file-scope').value.trim() || 'agent';
      const filename = filenameOverride !== null ? filenameOverride : document.getElementById('memory-file-name').value.trim();
      const agentName = selectedMemoryFileAgent();
      const params = new URLSearchParams({scope});
      if (agentName) params.set('agent_name', agentName);
      if (filename) params.set('filename', filename);
      const result = await fetchJson('/memory-files/content?' + params.toString());
      document.getElementById('memory-file-scope').value = result.scope;
      document.getElementById('memory-file-name').value = result.filename;
      document.getElementById('memory-file-revision').value = result.current_revision_id || '';
      document.getElementById('memory-file-content').value = result.content || '';
      document.getElementById('memory-file-history').textContent = '';
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function loadMemoryRevision() {
      const scope = document.getElementById('memory-file-scope').value.trim() || 'agent';
      const filename = document.getElementById('memory-file-name').value.trim();
      const revisionId = document.getElementById('memory-file-revision').value.trim();
      const agentName = selectedMemoryFileAgent();
      if (!revisionId) return;
      const params = new URLSearchParams({scope, revision_id: revisionId});
      if (agentName) params.set('agent_name', agentName);
      if (filename) params.set('filename', filename);
      const result = await fetchJson('/memory-files/revision?' + params.toString());
      document.getElementById('memory-file-scope').value = result.scope;
      document.getElementById('memory-file-name').value = result.filename;
      document.getElementById('memory-file-content').value = result.content || '';
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
    }
    async function loadMemoryHistory(scopeOverride = null, filenameOverride = null) {
      const scope = scopeOverride || document.getElementById('memory-file-scope').value.trim() || 'agent';
      const filename = filenameOverride !== null ? filenameOverride : document.getElementById('memory-file-name').value.trim();
      const agentName = selectedMemoryFileAgent();
      const params = new URLSearchParams({scope});
      if (agentName) params.set('agent_name', agentName);
      if (filename) params.set('filename', filename);
      const result = await fetchJson('/memory-files/revisions?' + params.toString());
      if (result.revisions && result.revisions.length) {
        const current = result.revisions.find(item => item.current) || result.revisions[0];
        document.getElementById('memory-file-revision').value = current.id || '';
      }
      const lines = (result.revisions || []).map(item => {
        const flags = [];
        if (item.current) flags.push('current');
        if (item.synced_memory_id) flags.push(`synced:${item.synced_memory_id}`);
        if (item.source_revision_id) flags.push(`from:${item.source_revision_id}`);
        return `${item.id} | ${item.action} | ${item.token_estimate || 0} tokens${flags.length ? ' | ' + flags.join(', ') : ''}`;
      });
      document.getElementById('memory-file-history').textContent = lines.length ? lines.join('\n') : 'No revisions recorded.';
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
    }
    async function saveMemoryFile() {
      const scope = document.getElementById('memory-file-scope').value.trim() || 'agent';
      const filename = document.getElementById('memory-file-name').value.trim();
      const content = document.getElementById('memory-file-content').value;
      const agentName = selectedMemoryFileAgent();
      const result = await fetchJson('/memory-files/content', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({scope, filename, content, agent_name: agentName, append: false})
      });
      document.getElementById('memory-file-name').value = result.filename;
      document.getElementById('memory-file-revision').value = result.current_revision_id || '';
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function restoreMemoryFile() {
      const scope = document.getElementById('memory-file-scope').value.trim() || 'agent';
      const filename = document.getElementById('memory-file-name').value.trim();
      const revisionId = document.getElementById('memory-file-revision').value.trim();
      const agentName = selectedMemoryFileAgent();
      if (!revisionId) return;
      const result = await fetchJson('/memory-files/restore', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({scope, filename, revision_id: revisionId, agent_name: agentName})
      });
      document.getElementById('memory-file-content').value = result.content || '';
      document.getElementById('memory-file-revision').value = result.current_revision_id || '';
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function syncMemoryFile(scopeOverride = null, filenameOverride = null) {
      const scope = scopeOverride || document.getElementById('memory-file-scope').value.trim() || 'agent';
      const filename = filenameOverride !== null ? filenameOverride : document.getElementById('memory-file-name').value.trim();
      const revisionId = document.getElementById('memory-file-revision').value.trim();
      const agentName = selectedMemoryFileAgent();
      const result = await fetchJson('/memory-files/sync', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({scope, filename, revision_id: revisionId, agent_name: agentName})
      });
      document.getElementById('context').textContent = JSON.stringify(result, null, 2);
      await refresh();
    }
    async function applyDoc(id) {
      await fetchJson(`/docs/proposals/${id}/apply`, {method: 'POST'});
      await refresh();
    }
    async function rejectDoc(id) {
      await fetchJson(`/docs/proposals/${id}/reject`, {method: 'POST'});
      await refresh();
    }
    let refreshTimer = null;
    function scheduleRefresh() {
      if (refreshTimer !== null) return;
      refreshTimer = window.setTimeout(async () => {
        refreshTimer = null;
        await refresh();
      }, 180);
    }
    function connectLiveFeed() {
      if (typeof EventSource !== 'function') return;
      const source = new EventSource('/events/stream');
      source.addEventListener('studio.event', () => scheduleRefresh());
      source.addEventListener('studio.heartbeat', () => {});
      window.addEventListener('beforeunload', () => source.close(), {once: true});
    }
    refresh();
    connectLiveFeed();
  </script>
</body>
</html>"""


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is required to run ContextClaw Studio. Install the `studio` extra."
        ) from exc

    host = os.environ.get("CONTEXTCLAW_STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("CONTEXTCLAW_STUDIO_PORT", "8765"))
    uvicorn.run(create_app(), host=host, port=port)
