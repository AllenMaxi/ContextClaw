from __future__ import annotations

import asyncio
import json
import logging
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..catalog_engine import (
    connector_bundles_from_lock,
    generated_mcp_path,
    load_connector_specs,
    load_skill_specs,
    read_catalog_state,
    sync_agent_catalog,
    validate_connector_prerequisites,
    write_catalog_state,
)
from ..config.agent_config import AgentConfig, _resolve_env
from ..context_engine import ContextController
from ..chat.session import ChatSession
from ..knowledge import ContextGraphBridge
from ..memory_files import (
    list_memory_file_revisions as list_memory_file_revisions_entry,
    list_tracked_memory_files,
    mark_memory_file_revision_synced,
    read_memory_file as read_memory_file_entry,
    read_memory_file_revision as read_memory_file_revision_entry,
    restore_memory_file_revision as restore_memory_file_revision_entry,
    render_memory_files_prompt,
    write_memory_file as write_memory_file_entry,
)
from ..providers.protocol import ToolCall
from ..runner import AgentRunner
from ..runtime import (
    create_policy,
    create_provider,
    create_sandbox,
    create_tools,
)
from ..tools.manager import ToolDefinition, ToolManager
from ..workflow import (
    WorkflowConfig,
    load_workflow,
    route_prompt,
    validate_workflow,
    workflow_from_dict,
    write_workflow,
)
from ..project import (
    ProjectLayout,
    ensure_project,
    get_project_layout,
    resolve_agent_workspace,
    scaffold_agent_workspace,
)
from .journal import StudioJournal
from .knowledge import StudioKnowledgeBridge

logger = logging.getLogger(__name__)


def _path_inside_any_root(target: Path, roots: list[Path]) -> bool:
    """Return True if *target* is equal to or inside at least one *root*."""
    for root in roots:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


class RunCancelledError(RuntimeError):
    """Raised when an operator cancels a run."""


@dataclass
class _PendingApproval:
    run_id: str
    decision: bool | None = None
    event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _RunHandle:
    run_id: str
    task: asyncio.Task[None] | None = None
    cancel_requested: asyncio.Event | None = None


@dataclass
class _EventSubscriber:
    run_id: str | None
    events: queue.Queue[dict[str, Any]]


class StudioService:
    """Studio control plane — owns a single shared event loop for all runs."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        provider_factory: Any | None = None,
        initialize_project: bool = False,
    ) -> None:
        self._layout: ProjectLayout | None = None
        self._journal: StudioJournal | None = None
        self._provider_factory = provider_factory
        self._handles: dict[str, _RunHandle] = {}
        self._pending_approvals: dict[str, _PendingApproval] = {}
        self._event_subscribers: dict[str, _EventSubscriber] = {}
        self._lock = threading.Lock()
        # Shared event loop — started lazily on first run
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        if project_root is not None:
            self.open_project(project_root, initialize=initialize_project)

    # ------------------------------------------------------------------
    # Shared event loop lifecycle
    # ------------------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Start the background event-loop thread if not already running."""
        # Capture into local to avoid TOCTOU race with _loop_main's finally
        loop = self._loop
        if loop is not None and loop.is_running():
            return loop
        with self._lock:
            loop = self._loop
            if loop is not None and loop.is_running():
                return loop
            self._loop_ready.clear()
            thread = threading.Thread(
                target=self._loop_main, name="studio-event-loop", daemon=True
            )
            thread.start()
            self._loop_ready.wait(timeout=10)
            loop = self._loop
            if loop is None or not loop.is_running():
                raise RuntimeError("Failed to start studio event loop")
            self._loop_thread = thread
        return loop

    def _loop_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            # Drain remaining tasks with a bounded timeout
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.wait(pending, timeout=5.0))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the shared event loop and join the thread."""
        loop = self._loop
        thread = self._loop_thread
        if loop is None or thread is None:
            return
        # Cancel every active run task
        with self._lock:
            handles = list(self._handles.values())
        for handle in handles:
            if handle.task is not None and not handle.task.done():
                loop.call_soon_threadsafe(handle.task.cancel)
        # Ask the loop to stop
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "Studio event-loop thread did not stop within %.1fs", timeout
            )
        self._loop_thread = None
        self._loop_ready.clear()

    def _fanout_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._event_subscribers.values())
        for subscriber in subscribers:
            if subscriber.run_id and subscriber.run_id != event["run_id"]:
                continue
            try:
                subscriber.events.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.events.get_nowait()
                except queue.Empty:
                    continue
                try:
                    subscriber.events.put_nowait(event)
                except queue.Full:
                    continue

    def subscribe_events(
        self,
        *,
        run_id: str | None = None,
        maxsize: int = 256,
    ) -> tuple[str, queue.Queue[dict[str, Any]]]:
        subscription_id = f"sub_{uuid.uuid4().hex[:12]}"
        event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._event_subscribers[subscription_id] = _EventSubscriber(
                run_id=run_id,
                events=event_queue,
            )
        return subscription_id, event_queue

    def unsubscribe_events(self, subscription_id: str) -> None:
        with self._lock:
            self._event_subscribers.pop(subscription_id, None)

    # ------------------------------------------------------------------
    # Project and workflow
    # ------------------------------------------------------------------

    def _ensure_project(self) -> tuple[ProjectLayout, StudioJournal]:
        if self._layout is None or self._journal is None:
            auto_layout = get_project_layout()
            if auto_layout is None:
                raise ValueError("No project is currently open")
            self.open_project(auto_layout.root, initialize=False)
        return self._layout, self._journal

    def open_project(self, root: Path, *, initialize: bool = False) -> dict[str, Any]:
        root = root.resolve()
        if initialize:
            layout = ensure_project(root)
        else:
            layout = get_project_layout(root)
            if layout is None:
                raise ValueError("Project has no Workflow.md or agents directory")
        journal = StudioJournal(
            layout.runtime_dir / "studio.db",
            on_event=self._fanout_event,
        )
        journal.set_current_project(layout.root)
        self._layout = layout
        self._journal = journal
        if layout.workflow_path.exists():
            config, _ = load_workflow(layout.workflow_path)
            issues = validate_workflow(config, project_root=layout.root)
            if issues:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Workflow validation issues on open: %s",
                    "; ".join(issues),
                )
        return self.current_project()

    def init_project(
        self,
        root: Path,
        *,
        entry_agent: str = "orchestrator",
        provider: str = "claude",
    ) -> dict[str, Any]:
        layout = ensure_project(root, entry_agent=entry_agent)
        scaffold_agent_workspace(
            layout,
            name=entry_agent,
            template="default",
            provider=provider,
        )
        self.open_project(layout.root, initialize=False)
        return self.current_project()

    def current_project(self) -> dict[str, Any]:
        layout, journal = self._ensure_project()
        return {
            "root": str(layout.root),
            "workflow_path": str(layout.workflow_path),
            "agents_dir": str(layout.agents_dir),
            "runtime_dir": str(layout.runtime_dir),
            "git": self._git_status(layout.root),
            "current_project": journal.get_current_project(),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        _, journal = self._ensure_project()
        return journal.list_runs()

    def _git_status(self, root: Path) -> dict[str, Any]:
        branch = ""
        dirty: list[str] = []
        try:
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty_output = subprocess.run(
                ["git", "status", "--short"],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            ).stdout
            dirty = [line for line in dirty_output.splitlines() if line.strip()]
        except OSError:
            pass
        return {"branch": branch, "dirty_files": dirty, "dirty": bool(dirty)}

    def get_workflow(self) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        config, body = load_workflow(layout.workflow_path)
        return {"config": config.to_frontmatter(), "body": body}

    def update_workflow(
        self,
        *,
        config_data: dict[str, Any] | None = None,
        body: str | None = None,
    ) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        current_config, current_body = load_workflow(layout.workflow_path)
        config = workflow_from_dict(config_data or current_config.to_frontmatter())
        write_workflow(layout.workflow_path, config, body=body or current_body)
        return self.get_workflow()

    def validate_current_workflow(self) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        config, _ = load_workflow(layout.workflow_path)
        return {"issues": validate_workflow(config, project_root=layout.root)}

    def _default_agent_name(self) -> str:
        layout, _ = self._ensure_project()
        workflow_config, _ = load_workflow(layout.workflow_path)
        return workflow_config.entry_agent

    def _context_target(
        self,
        agent_name: str | None = None,
    ) -> tuple[ProjectLayout, WorkflowConfig, AgentConfig, ContextController, str]:
        layout, _ = self._ensure_project()
        workflow_config, _ = load_workflow(layout.workflow_path)
        selected_agent = (agent_name or workflow_config.entry_agent).strip()
        if not selected_agent:
            raise ValueError("agent_name must not be empty")
        workspace = resolve_agent_workspace(
            selected_agent,
            project_layout=layout,
            legacy_agents_dir=Path.home() / ".contextclaw" / "agents",
        )
        if not workspace.exists():
            raise FileNotFoundError(f"Agent '{selected_agent}' was not found")
        config = AgentConfig.from_dir(workspace)
        controller = ContextController(
            config.workspace,
            memory_policy=workflow_config.memory_policy,
        )
        system_prompt = self._build_system_prompt(config)
        return layout, workflow_config, config, controller, system_prompt

    def _build_system_prompt(self, config: AgentConfig) -> str:
        system_prompt = ""
        if config.soul_path and config.soul_path.exists():
            from ..config.soul import load_soul

            system_prompt = load_soul(config.soul_path).body

        from ..config.skills import render_skills_prompt

        skills_prompt = render_skills_prompt(config.skills_path)
        if skills_prompt:
            if system_prompt:
                system_prompt += "\n\n"
            system_prompt += skills_prompt
        memory_prompt, _ = render_memory_files_prompt(config.workspace)
        sections = [
            item.strip() for item in (system_prompt, memory_prompt) if item.strip()
        ]
        return "\n\n".join(sections)

    def _context_tools(
        self, config: AgentConfig
    ) -> tuple[list[dict[str, Any]], list[str]]:
        tools = ToolManager()
        bundle_names: list[str] = []
        for bundle_name in config.tools + connector_bundles_from_lock(config.workspace):
            if bundle_name not in bundle_names:
                bundle_names.append(bundle_name)
        for bundle_name in bundle_names:
            tools.register_bundle(bundle_name)
        self._register_studio_tools(tools)
        dynamic_registries: list[str] = []
        if config.mcp_servers_path and config.mcp_servers_path.exists():
            dynamic_registries.append(str(config.mcp_servers_path))
        generated_mcp = generated_mcp_path(config.workspace)
        if generated_mcp.exists():
            dynamic_registries.append(str(generated_mcp))
        return tools.list_tools(), dynamic_registries

    def _load_context_payload(
        self,
        config: AgentConfig,
        *,
        system_prompt: str,
    ) -> dict[str, Any]:
        checkpoint_path = config.checkpoint_path
        if checkpoint_path and checkpoint_path.exists():
            try:
                payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, ValueError):
                payload = {}
            if isinstance(payload, dict):
                return payload
        session = ChatSession(system_prompt=system_prompt, max_history=0)
        return {
            "session": session.to_dict(),
            "total_usage": {},
            "saved_at": time.time(),
        }

    def _write_context_payload(
        self,
        config: AgentConfig,
        payload: dict[str, Any],
    ) -> None:
        checkpoint_path = config.checkpoint_path
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def get_context(self, agent_name: str | None = None) -> dict[str, Any]:
        _, _, config, controller, system_prompt = self._context_target(agent_name)
        payload = self._load_context_payload(config, system_prompt=system_prompt)
        tools, dynamic_registries = self._context_tools(config)
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            "memory_policy": controller.policy,
            "budget": controller.inspect_state(
                payload,
                system_prompt=system_prompt,
                tools=tools,
            ),
            "working_memory": controller.load_working_memory(),
            "pending_compact": controller.load_pending_compact(),
            "memory_files": list_tracked_memory_files(config.workspace),
            "tool_estimate_mode": "static_bundle_snapshot",
            "dynamic_tool_registries": dynamic_registries,
        }

    def list_memory_files(self, agent_name: str | None = None) -> list[dict[str, Any]]:
        _, _, config, _, _ = self._context_target(agent_name)
        return list_tracked_memory_files(config.workspace)

    def read_memory_file(
        self,
        *,
        scope: str,
        agent_name: str | None = None,
        filename: str = "",
    ) -> dict[str, Any]:
        _, _, config, _, _ = self._context_target(agent_name)
        result = read_memory_file_entry(
            config.workspace,
            scope=scope,
            filename=filename,
        )
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            **result,
        }

    def read_memory_file_revision(
        self,
        *,
        scope: str,
        revision_id: str,
        agent_name: str | None = None,
        filename: str = "",
    ) -> dict[str, Any]:
        _, _, config, _, _ = self._context_target(agent_name)
        result = read_memory_file_revision_entry(
            config.workspace,
            scope=scope,
            revision_id=revision_id,
            filename=filename,
        )
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            **result,
        }

    def list_memory_file_revisions(
        self,
        *,
        scope: str,
        agent_name: str | None = None,
        filename: str = "",
    ) -> dict[str, Any]:
        _, _, config, _, _ = self._context_target(agent_name)
        result = list_memory_file_revisions_entry(
            config.workspace,
            scope=scope,
            filename=filename,
        )
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            **result,
        }

    def write_memory_file(
        self,
        *,
        scope: str,
        content: str,
        agent_name: str | None = None,
        filename: str = "",
        append: bool = False,
    ) -> dict[str, Any]:
        _, _, config, _, _ = self._context_target(agent_name)
        result = write_memory_file_entry(
            config.workspace,
            scope=scope,
            content=content,
            filename=filename,
            append=append,
        )
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            **result,
        }

    def sync_memory_file(
        self,
        *,
        scope: str,
        agent_name: str | None = None,
        filename: str = "",
        revision_id: str = "",
    ) -> dict[str, Any]:
        _, _, config, _, _ = self._context_target(agent_name)
        if not config.cg_url or not config.agent_id:
            raise ValueError("Agent is not linked to ContextGraph")
        if revision_id:
            payload = read_memory_file_revision_entry(
                config.workspace,
                scope=scope,
                revision_id=revision_id,
                filename=filename,
            )
            selected_revision = dict(payload["revision"])
            content = str(payload.get("content", "")).strip()
        else:
            payload = read_memory_file_entry(
                config.workspace,
                scope=scope,
                filename=filename,
            )
            content = str(payload.get("content", "")).strip()
            selected_revision = {}
            current_revision_id = str(payload.get("current_revision_id", "")).strip()
            if current_revision_id:
                listing = list_memory_file_revisions_entry(
                    config.workspace,
                    scope=scope,
                    filename=payload["filename"],
                )
                for item in listing["revisions"]:
                    if item.get("id") == current_revision_id:
                        selected_revision = dict(item)
                        break
        if not content:
            raise ValueError("Memory file is empty")
        chosen_revision_id = str(selected_revision.get("id", "")).strip()
        existing_memory_id = str(selected_revision.get("synced_memory_id", "")).strip()
        if chosen_revision_id and existing_memory_id:
            return {
                "agent": config.name,
                "workspace": str(config.workspace),
                **payload,
                "synced_memory_id": existing_memory_id,
                "revision": selected_revision,
                "already_synced": True,
            }
        bridge = ContextGraphBridge(
            cg_url=config.cg_url,
            api_key=config.cg_api_key,
            agent_id=config.agent_id,
        )
        parent_memory_id = str(payload.get("last_synced_memory_id", "")).strip()
        derived_from_memory_ids = [parent_memory_id] if parent_memory_id else []
        result = bridge.store(
            content,
            metadata={
                "source": "memory_file",
                "agent": config.name,
                "scope": str(payload["scope"]),
                "filename": str(payload["filename"]),
                "path": str(payload["path"]),
                "revision_id": chosen_revision_id,
                "source_revision_id": str(
                    selected_revision.get("source_revision_id", "")
                ),
                "content_hash": str(
                    selected_revision.get(
                        "content_hash", payload.get("content_hash", "")
                    )
                ),
            },
            citations=[str(payload["path"])],
            evidence=[str(payload["path"])],
            memory_kind="artifact",
            summary=f"{payload['scope']} memory file {payload['filename']}",
            tags=["memory-file", str(payload["scope"]), str(payload["filename"])],
            section_schema={
                "scope": str(payload["scope"]),
                "filename": str(payload["filename"]),
                "revision_id": chosen_revision_id,
                "source_revision_id": str(
                    selected_revision.get("source_revision_id", "")
                ),
                "content_hash": str(
                    selected_revision.get(
                        "content_hash", payload.get("content_hash", "")
                    )
                ),
            },
            token_estimate=int(payload.get("token_estimate", 0)) or None,
            importance_score=0.8
            if str(payload["scope"]) in {"project", "agent"}
            else 0.6,
            pinned=str(payload["scope"]) in {"project", "agent"},
            parent_memory_id=parent_memory_id,
            derived_from_memory_ids=derived_from_memory_ids,
        )
        memory_id = ""
        if isinstance(result, dict):
            memory_id = str(result.get("id", "")) or str(
                (result.get("memory") or {}).get("memory_id", "")
            )
        updated_payload = payload
        updated_revision = selected_revision
        if chosen_revision_id and memory_id:
            sync_result = mark_memory_file_revision_synced(
                config.workspace,
                scope=scope,
                revision_id=chosen_revision_id,
                memory_id=memory_id,
                filename=str(payload["filename"]),
            )
            updated_payload = {
                key: value
                for key, value in sync_result.items()
                if key != "synced_revision"
            }
            updated_revision = dict(
                sync_result.get("synced_revision", selected_revision)
            )
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            **updated_payload,
            "synced_memory_id": memory_id,
            "revision": updated_revision,
            "already_synced": False,
        }

    def restore_memory_file(
        self,
        *,
        scope: str,
        revision_id: str,
        agent_name: str | None = None,
        filename: str = "",
    ) -> dict[str, Any]:
        _, _, config, _, _ = self._context_target(agent_name)
        result = restore_memory_file_revision_entry(
            config.workspace,
            scope=scope,
            revision_id=revision_id,
            filename=filename,
        )
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            **result,
        }

    def preview_compact(
        self,
        agent_name: str | None = None,
        *,
        reason: str = "manual",
    ) -> dict[str, Any]:
        _, _, config, controller, system_prompt = self._context_target(agent_name)
        payload = self._load_context_payload(config, system_prompt=system_prompt)
        tools, _ = self._context_tools(config)
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            "preview": controller.preview_compact(
                payload,
                system_prompt=system_prompt,
                tools=tools,
                reason=reason,
            ),
        }

    def apply_compact(
        self,
        agent_name: str | None = None,
        *,
        reason: str = "manual",
    ) -> dict[str, Any]:
        _, _, config, controller, system_prompt = self._context_target(agent_name)
        payload = self._load_context_payload(config, system_prompt=system_prompt)
        tools, _ = self._context_tools(config)
        session = ChatSession.from_dict(
            payload.get("session", {}),
            system_prompt=system_prompt,
            max_history=0,
        )
        updated_session, result = controller.apply_compact_to_chat_session(
            session,
            system_prompt=system_prompt,
            tools=tools,
            total_usage=payload.get("total_usage", {}),
            reason=reason,
        )
        updated_payload = {
            **payload,
            "session": updated_session.to_dict(),
            "saved_at": time.time(),
        }
        self._write_context_payload(config, updated_payload)
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            "result": result,
        }

    def reject_compact(self, agent_name: str | None = None) -> dict[str, Any]:
        _, _, config, controller, _ = self._context_target(agent_name)
        result = controller.reject_pending_compact()
        return {
            "agent": config.name,
            "workspace": str(config.workspace),
            **result,
        }

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def list_agents(self) -> list[dict[str, Any]]:
        layout, _ = self._ensure_project()
        agents: list[dict[str, Any]] = []
        for child in (
            sorted(layout.agents_dir.iterdir()) if layout.agents_dir.exists() else []
        ):
            if not child.is_dir():
                continue
            config_path = child / "config.yaml"
            soul_path = child / "SOUL.md"
            if not config_path.exists() and not soul_path.exists():
                continue
            config = AgentConfig.from_dir(child)
            agents.append(
                {
                    "name": config.name,
                    "workspace": str(child),
                    "provider": config.provider,
                    "cg_url": config.cg_url,
                    "agent_id": config.agent_id,
                    "skills_path": str(config.skills_path)
                    if config.skills_path
                    else "",
                }
            )
        return agents

    def create_agent(
        self,
        *,
        name: str,
        template: str = "default",
        provider: str = "claude",
    ) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        workspace = scaffold_agent_workspace(
            layout, name=name, template=template, provider=provider
        )
        config = AgentConfig.from_dir(workspace)
        return {
            "name": config.name,
            "workspace": str(workspace),
            "provider": config.provider,
        }

    def get_agent(self, name: str) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        workspace = resolve_agent_workspace(
            name,
            project_layout=layout,
            legacy_agents_dir=Path.home() / ".contextclaw" / "agents",
        )
        config = AgentConfig.from_dir(workspace)
        soul_text = (
            config.soul_path.read_text(encoding="utf-8")
            if config.soul_path and config.soul_path.exists()
            else ""
        )
        return {
            "name": config.name,
            "workspace": str(workspace),
            "provider": config.provider,
            "model": config.model,
            "cg_url": config.cg_url,
            "agent_id": config.agent_id,
            "soul": soul_text,
        }

    def update_agent(
        self,
        name: str,
        *,
        soul_body: str | None = None,
        config_updates: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        workspace = resolve_agent_workspace(
            name,
            project_layout=layout,
            legacy_agents_dir=Path.home() / ".contextclaw" / "agents",
        )
        if soul_body is not None:
            (workspace / "SOUL.md").write_text(soul_body, encoding="utf-8")
        if config_updates:
            self._rewrite_config(workspace / "config.yaml", config_updates)
        return self.get_agent(name)

    def _rewrite_config(self, config_path: Path, updates: dict[str, str]) -> None:
        lines = (
            config_path.read_text(encoding="utf-8").splitlines()
            if config_path.exists()
            else []
        )
        managed_keys = set(updates)
        filtered = [
            line
            for line in lines
            if not any(line.startswith(f"{key}:") for key in managed_keys)
        ]
        for key, value in updates.items():
            filtered.append(f"{key}: {value}")
        config_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Runs and event journal
    # ------------------------------------------------------------------

    def start_run(
        self,
        prompt: str,
        *,
        source: str = "desktop",
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        layout, journal = self._ensure_project()
        workflow_config, _ = load_workflow(layout.workflow_path)
        matched_rule: dict[str, Any] | None = None
        if agent_name:
            selected_agent = agent_name.strip()
            if not selected_agent:
                raise ValueError("agent_name must not be empty")
            matched_rule = {"mode": "direct", "agent": selected_agent}
        else:
            selected_agent, matched_rule = route_prompt(workflow_config, prompt)
        workspace = layout.agents_dir / selected_agent
        if not workspace.exists():
            raise FileNotFoundError(f"Agent '{selected_agent}' was not found")
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run = journal.create_run(
            run_id=run_id,
            project_root=layout.root,
            agent_name=workflow_config.entry_agent,
            selected_agent=selected_agent,
            source=source,
            prompt=prompt,
            workflow_rule=matched_rule,
        )
        # asyncio.Event must be created on the event loop thread (Python 3.10+
        # does not bind at creation, but we schedule the coroutine on the shared
        # loop so the event will be used from that loop's context).
        cancel_event = asyncio.Event()
        handle = _RunHandle(run_id=run_id, cancel_requested=cancel_event)
        with self._lock:
            self._handles[run_id] = handle
        loop = self._ensure_loop()
        # Schedule the run on the shared loop.  The coroutine itself binds
        # handle.task from within asyncio.current_task(), eliminating any
        # external task-scan race.
        asyncio.run_coroutine_threadsafe(
            self._run_async(
                layout,
                workflow_config,
                run_id,
                prompt,
                selected_agent,
                source,
                cancel_event,
            ),
            loop,
        )
        return run

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            handle = self._handles.get(run_id)
        if handle is None:
            return self.get_run(run_id)
        loop = self._loop
        if loop is not None:
            # Signal the cooperative cancel event (checked between LLM turns)
            if handle.cancel_requested is not None:
                loop.call_soon_threadsafe(handle.cancel_requested.set)
            # Also cancel the asyncio task for hard cancellation
            if handle.task is not None and not handle.task.done():
                loop.call_soon_threadsafe(handle.task.cancel)
        _, journal = self._ensure_project()
        journal.record_event(
            run_id,
            source="studio",
            event_type="run.cancel_requested",
            payload={"run_id": run_id},
        )
        return journal.update_run(run_id, status="cancelling")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        _, journal = self._ensure_project()
        return journal.get_run(run_id)

    def list_run_events(self, run_id: str) -> list[dict[str, Any]]:
        _, journal = self._ensure_project()
        return journal.list_events(run_id)

    def _provider_for(self, config: AgentConfig) -> Any:
        if self._provider_factory is not None:
            return self._provider_factory(config)
        return create_provider(config)

    def _register_studio_tools(self, tools: Any) -> None:
        for name, description, parameters in (
            (
                "memory_propose",
                "Propose a durable memory for review and later ContextGraph sync.",
                {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["content"],
                },
            ),
            (
                "docs_propose",
                "Propose a Markdown document or docs update for operator review.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            ),
        ):
            if tools.get_tool(name) is None:
                tools.register(
                    ToolDefinition(
                        name=name,
                        description=description,
                        parameters=parameters,
                    )
                )

    def _normalize_runner_event(self, event_type: str) -> str:
        mapping = {
            "text": "message.assistant",
            "tool_call": "tool.call",
            "tool_result": "tool.result",
            "knowledge_recalled": "memory.recalled",
            "context_budget": "context.budget",
            "compact_preview": "context.compact.preview",
            "context_compacted": "context.compacted",
            "done": "run.output",
            "error": "run.error",
        }
        return mapping.get(event_type, event_type)

    async def _run_async(
        self,
        layout: ProjectLayout,
        workflow_config: WorkflowConfig,
        run_id: str,
        prompt: str,
        selected_agent: str,
        source: str,
        cancel_event: asyncio.Event,
    ) -> None:
        # Bind the asyncio.Task to the handle so cancel_run can cancel it
        current = asyncio.current_task()
        if current is not None:
            current.set_name(f"run-{run_id}")
            with self._lock:
                handle = self._handles.get(run_id)
                if handle is not None:
                    handle.task = current

        _, journal = self._ensure_project()
        workspace = layout.agents_dir / selected_agent
        config = AgentConfig.from_dir(workspace)
        final_text = ""
        tool_calls = 0

        journal.update_run(run_id, status="running")
        journal.record_event(
            run_id,
            source="studio",
            event_type="run.started",
            payload={"prompt": prompt, "agent": selected_agent},
        )
        journal.record_event(
            run_id,
            source=source,
            event_type="message.user",
            payload={"content": prompt},
        )

        sandbox = None
        tools = None
        provider = self._provider_for(config)
        try:
            sandbox = create_sandbox(config)
            tools = await create_tools(config)
            self._register_studio_tools(tools)
            policy = create_policy(config)
            knowledge = StudioKnowledgeBridge(
                cg_url=config.cg_url,
                api_key=config.cg_api_key,
                agent_id=config.agent_id,
                memory_sink=self._make_memory_sink(layout, run_id, selected_agent),
                auto_store=False,
            )

            async def _approval(tool_call: ToolCall) -> bool:
                return await asyncio.to_thread(
                    self._wait_for_approval,
                    layout,
                    run_id,
                    selected_agent,
                    tool_call,
                )

            runner = AgentRunner(
                config=config,
                provider=provider,
                sandbox=sandbox,
                tools=tools,
                knowledge=knowledge,
                policy=policy,
                tool_approver=_approval,
                docs_proposer=self._make_docs_sink(
                    layout, workflow_config, run_id, selected_agent
                ),
                memory_proposer=self._make_memory_sink(layout, run_id, selected_agent),
                provider_factory=self._provider_factory,
            )
            if sandbox:
                await sandbox.start()
            async for event in runner.run(prompt):
                payload = dict(event.data)
                if event.type in {"text", "done", "tool_call", "tool_result"}:
                    payload.setdefault("agent", selected_agent)
                journal.record_event(
                    run_id,
                    source="runner",
                    event_type=self._normalize_runner_event(event.type),
                    payload=payload,
                )
                if event.type == "text":
                    final_text = event.data.get("content", final_text)
                elif event.type == "done":
                    final_text = event.data.get("content", final_text)
                elif event.type == "tool_call":
                    tool_calls += 1
                # Cooperative cancellation check between events
                if cancel_event.is_set():
                    raise RunCancelledError("Run cancelled by operator")
            stored = await runner.close_session()
            journal.record_event(
                run_id,
                source="studio",
                event_type="run.finished",
                payload={
                    "result": final_text,
                    "tool_calls": tool_calls,
                    "memory_proposals": len(stored),
                },
            )
            journal.update_run(run_id, status="completed", result=final_text)
        except (RunCancelledError, asyncio.CancelledError) as exc:
            journal.record_event(
                run_id,
                source="studio",
                event_type="run.cancelled",
                payload={"error": str(exc), "reason": "cancelled"},
            )
            journal.update_run(run_id, status="cancelled", result=str(exc))
        except (ConnectionError, TimeoutError, OSError) as exc:
            journal.record_event(
                run_id,
                source="studio",
                event_type="run.failed",
                payload={"error": str(exc), "reason": "transient"},
            )
            journal.update_run(run_id, status="failed", result=str(exc))
        except (ValueError, PermissionError, FileNotFoundError) as exc:
            journal.record_event(
                run_id,
                source="studio",
                event_type="run.failed",
                payload={"error": str(exc), "reason": "config"},
            )
            journal.update_run(run_id, status="failed", result=str(exc))
        except Exception as exc:  # noqa: BLE001 — last-resort; log at error level
            logger.error("Unexpected error in run %s: %s", run_id, exc, exc_info=True)
            journal.record_event(
                run_id,
                source="studio",
                event_type="run.failed",
                payload={"error": str(exc), "reason": "unexpected"},
            )
            journal.update_run(run_id, status="failed", result=str(exc))
        finally:
            if sandbox:
                await sandbox.stop()
            if tools is not None:
                await tools.stop_all()
            with self._lock:
                self._handles.pop(run_id, None)

    # ------------------------------------------------------------------
    # Approvals and proposal sinks
    # ------------------------------------------------------------------

    def _wait_for_approval(
        self,
        layout: ProjectLayout,
        run_id: str,
        agent_name: str,
        tool_call: ToolCall,
        *,
        timeout: int = 300,
    ) -> bool:
        _, journal = self._ensure_project()
        approval_id = f"apr_{uuid.uuid4().hex[:12]}"
        approval = journal.create_approval(
            approval_id=approval_id,
            run_id=run_id,
            project_root=layout.root,
            agent_name=agent_name,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
        )
        journal.record_event(
            run_id,
            source="studio",
            event_type="approval.requested",
            payload=approval,
        )
        pending = _PendingApproval(run_id=run_id)
        with self._lock:
            self._pending_approvals[approval_id] = pending
        resolved = pending.event.wait(timeout=timeout)
        if not resolved:
            result = journal.resolve_approval(
                approval_id,
                approved=False,
                resolution_reason="timeout",
            )
            journal.record_event(
                run_id,
                source="studio",
                event_type="approval.resolved",
                payload={
                    **(result or {"id": approval_id, "status": "denied"}),
                    "resolution_reason": "timeout",
                    "timeout_seconds": timeout,
                },
            )
            with self._lock:
                self._pending_approvals.pop(approval_id, None)
            return False
        journal.record_event(
            run_id,
            source="studio",
            event_type="approval.resolved",
            payload=journal.get_approval(approval_id) or {"id": approval_id},
        )
        with self._lock:
            self._pending_approvals.pop(approval_id, None)
        return bool(pending.decision)

    def list_approvals(self, *, pending_only: bool = False) -> list[dict[str, Any]]:
        _, journal = self._ensure_project()
        return journal.list_approvals(pending_only=pending_only)

    def resolve_approval(
        self, approval_id: str, *, approved: bool
    ) -> dict[str, Any] | None:
        _, journal = self._ensure_project()
        updated = journal.resolve_approval(approval_id, approved=approved)
        with self._lock:
            pending = self._pending_approvals.get(approval_id)
        if pending is not None:
            pending.decision = approved
            pending.event.set()
        return updated

    def _make_memory_sink(
        self, layout: ProjectLayout, run_id: str, agent_name: str
    ) -> Any:
        def _sink(content: str, metadata: dict[str, Any]) -> dict[str, Any]:
            _, journal = self._ensure_project()
            proposal = journal.create_memory_proposal(
                proposal_id=f"mem_{uuid.uuid4().hex[:12]}",
                run_id=run_id,
                project_root=layout.root,
                agent_name=agent_name,
                content=content,
                metadata=metadata,
            )
            journal.record_event(
                run_id,
                source="studio",
                event_type="memory.proposed",
                payload=proposal,
            )
            return proposal

        return _sink

    def _make_docs_sink(
        self,
        layout: ProjectLayout,
        workflow_config: WorkflowConfig,
        run_id: str,
        agent_name: str,
    ) -> Any:
        allowed_roots = [
            (layout.root / str(item)).resolve()
            for item in list(workflow_config.docs_policy.get("roots", []))
        ]

        def _sink(path_arg: str, content: str, summary: str) -> dict[str, Any]:
            raw = Path(path_arg)
            if raw.is_absolute():
                raise PermissionError("docs proposals must use project-relative paths")
            target = (layout.root / raw).resolve()
            if not _path_inside_any_root(target, allowed_roots):
                raise PermissionError(
                    "docs proposal path must stay inside configured docs roots"
                )
            _, journal = self._ensure_project()
            proposal = journal.create_docs_proposal(
                proposal_id=f"doc_{uuid.uuid4().hex[:12]}",
                run_id=run_id,
                project_root=layout.root,
                agent_name=agent_name,
                path=str(target.relative_to(layout.root)),
                summary=summary,
                content=content,
            )
            journal.record_event(
                run_id,
                source="studio",
                event_type="docs.proposed",
                payload=proposal,
            )
            return proposal

        return _sink

    # ------------------------------------------------------------------
    # Memory and docs review
    # ------------------------------------------------------------------

    def list_memory_proposals(self) -> list[dict[str, Any]]:
        _, journal = self._ensure_project()
        return journal.list_memory_proposals()

    def pin_memory(
        self, proposal_id: str, *, pinned: bool = True
    ) -> dict[str, Any] | None:
        _, journal = self._ensure_project()
        return journal.update_memory_proposal(proposal_id, pinned=pinned)

    def discard_memory(self, proposal_id: str) -> dict[str, Any] | None:
        _, journal = self._ensure_project()
        return journal.update_memory_proposal(proposal_id, status="rejected")

    def compact_memory(
        self,
        proposal_ids: list[str],
        *,
        summary: str = "",
    ) -> dict[str, Any]:
        layout, journal = self._ensure_project()
        selected = [
            proposal
            for proposal_id in proposal_ids
            if (proposal := journal.get_memory_proposal(proposal_id)) is not None
        ]
        if not selected:
            raise ValueError("No memory proposals matched the requested ids")
        merged = "\n".join(f"- {item['content']}" for item in selected)
        for item in selected:
            journal.update_memory_proposal(item["id"], status="compacted")
        merged_proposal = journal.create_memory_proposal(
            proposal_id=f"mem_{uuid.uuid4().hex[:12]}",
            run_id=selected[0]["run_id"],
            project_root=layout.root,
            agent_name=selected[0]["agent_name"],
            content=merged,
            metadata={
                "type": "compaction",
                "memory_kind": "compact",
                "summary": summary,
                "tags": ["compact", "review-queue"],
                "section_schema": {"compacted_from": [item["id"] for item in selected]},
                "compacted_from": [item["id"] for item in selected],
            },
        )
        return merged_proposal

    def sync_memory_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        layout, journal = self._ensure_project()
        proposal = journal.get_memory_proposal(proposal_id)
        if proposal is None:
            return None
        config = AgentConfig.from_dir(layout.agents_dir / proposal["agent_name"])
        if not config.cg_url or not config.agent_id:
            raise ValueError("Agent is not linked to ContextGraph")
        bridge = ContextGraphBridge(
            cg_url=config.cg_url,
            api_key=config.cg_api_key,
            agent_id=config.agent_id,
        )
        raw_metadata = dict(proposal["metadata"])
        metadata = (
            dict(raw_metadata.pop("metadata"))
            if isinstance(raw_metadata.get("metadata"), dict)
            else dict(raw_metadata)
        )
        evidence = raw_metadata.pop("evidence", None)
        citations = raw_metadata.pop("citations", None)
        extra_fields: dict[str, Any] = {}
        for key in (
            "memory_kind",
            "summary",
            "tags",
            "parent_memory_id",
            "derived_from_memory_ids",
            "token_estimate",
            "section_schema",
            "importance_score",
            "pinned",
        ):
            value = metadata.pop(key, None)
            if value not in (None, "", [], {}):
                extra_fields[key] = value
        result = bridge.store(
            proposal["content"],
            metadata=metadata,
            evidence=list(evidence) if isinstance(evidence, list) else None,
            citations=list(citations) if isinstance(citations, list) else None,
            **extra_fields,
        )
        memory_id = ""
        if isinstance(result, dict):
            memory_id = str(result.get("id", "")) or str(
                (result.get("memory") or {}).get("memory_id", "")
            )
        updated = journal.update_memory_proposal(
            proposal_id,
            status="synced",
            synced_memory_id=memory_id,
        )
        if updated is not None:
            journal.record_event(
                proposal["run_id"],
                source="studio",
                event_type="memory.synced",
                payload=updated,
            )
        return updated

    def sync_contextgraph(self) -> dict[str, Any]:
        synced: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        for proposal in self.list_memory_proposals():
            if proposal["status"] != "pending_review":
                continue
            try:
                result = self.sync_memory_proposal(proposal["id"])
            except (ConnectionError, TimeoutError, OSError) as exc:
                logger.warning(
                    "Transient sync failure for proposal %s: %s",
                    proposal["id"],
                    exc,
                )
                failed.append({"id": proposal["id"], "error": str(exc)})
                continue
            except (ValueError, PermissionError) as exc:
                logger.warning("Sync rejected for proposal %s: %s", proposal["id"], exc)
                failed.append({"id": proposal["id"], "error": str(exc)})
                continue
            if result is not None:
                synced.append(result)
        return {"synced": synced, "failed": failed}

    def list_docs_proposals(self) -> list[dict[str, Any]]:
        _, journal = self._ensure_project()
        return journal.list_docs_proposals()

    def apply_docs_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        layout, journal = self._ensure_project()
        proposal = journal.get_docs_proposal(proposal_id)
        if proposal is None:
            return None
        target = (layout.root / proposal["path"]).resolve()
        workflow, _ = load_workflow(layout.workflow_path)
        allowed_roots = [
            (layout.root / str(item)).resolve()
            for item in list(workflow.docs_policy.get("roots", []))
        ]
        if not _path_inside_any_root(target, allowed_roots):
            raise PermissionError("docs proposal path is outside configured docs roots")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(proposal["content"], encoding="utf-8")
        updated = journal.update_docs_proposal(proposal_id, status="applied")
        if updated is not None:
            journal.record_event(
                proposal["run_id"],
                source="studio",
                event_type="docs.applied",
                payload=updated,
            )
        return updated

    def reject_docs_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        _, journal = self._ensure_project()
        return journal.update_docs_proposal(proposal_id, status="rejected")

    # ------------------------------------------------------------------
    # Connectors, skills, ContextGraph status
    # ------------------------------------------------------------------

    def _serialize_connector_spec(self, spec: Any) -> dict[str, Any]:
        return {
            "id": spec.id,
            "version": spec.version,
            "display_name": spec.display_name,
            "description": spec.description,
            "stability": spec.stability,
            "tags": list(spec.tags),
            "type": spec.type,
            "bundles": list(spec.bundles),
            "tools_exposed": list(spec.tools_exposed),
            "required_env": dict(spec.required_env),
            "prerequisites": dict(spec.prerequisites),
            "docs_url": spec.runtime.docs_url if spec.runtime else "",
        }

    def connectors_status(self) -> dict[str, Any]:
        connector_specs = load_connector_specs()
        return {
            "available": [
                self._serialize_connector_spec(spec)
                for spec in connector_specs.values()
            ],
            "agents": self.list_agents(),
        }

    def install_connector(self, agent_name: str, connector_id: str) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        workspace = layout.agents_dir / agent_name
        connector_specs = load_connector_specs()
        spec = connector_specs.get(connector_id)
        if spec is None:
            raise ValueError(f"Unknown connector '{connector_id}'")
        state = read_catalog_state(workspace)
        if spec.id not in state.connectors:
            missing = validate_connector_prerequisites([spec.id], connector_specs)
            if missing.get(spec.id):
                raise ValueError(
                    f"Missing prerequisites: {', '.join(missing[spec.id])}"
                )
            state.connectors.append(spec.id)
            write_catalog_state(workspace, state)
        result = sync_agent_catalog(workspace)
        return {
            "agent": agent_name,
            "connector": connector_id,
            "missing_env": result.missing_env.get(connector_id, []),
        }

    def install_skill(
        self,
        agent_name: str,
        skill_id: str,
        *,
        no_deps: bool = False,
    ) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        workspace = layout.agents_dir / agent_name
        connector_specs = load_connector_specs()
        skill_specs = load_skill_specs()
        spec = skill_specs.get(skill_id)
        if spec is None:
            raise ValueError(f"Unknown skill '{skill_id}'")
        state = read_catalog_state(workspace)
        auto_installed: list[str] = []
        if not no_deps:
            for connector_id in spec.requires_connectors:
                if connector_id not in state.connectors:
                    state.connectors.append(connector_id)
                    auto_installed.append(connector_id)
        if spec.id not in state.skills:
            state.skills.append(spec.id)
        missing = validate_connector_prerequisites(state.connectors, connector_specs)
        if any(missing.values()):
            raise ValueError(f"Missing prerequisites: {missing}")
        write_catalog_state(workspace, state)
        result = sync_agent_catalog(workspace)
        return {
            "agent": agent_name,
            "skill": skill_id,
            "auto_installed": auto_installed,
            "missing_env": result.missing_env,
        }

    def contextgraph_status(self, agent_name: str | None = None) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        if agent_name is None:
            config, _ = load_workflow(layout.workflow_path)
            agent_name = config.entry_agent
        config = AgentConfig.from_dir(layout.agents_dir / agent_name)
        pending = [
            item
            for item in self.list_memory_proposals()
            if item["agent_name"] == agent_name and item["status"] == "pending_review"
        ]
        return {
            "agent": agent_name,
            "linked": bool(config.cg_url and config.agent_id),
            "cg_url": config.cg_url,
            "agent_id": config.agent_id,
            "pending_sync": len(pending),
        }

    def link_contextgraph(
        self,
        *,
        cg_url: str,
        api_key: str,
        agent_name: str | None = None,
        register: bool = False,
        org_id: str = "default",
        capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        layout, _ = self._ensure_project()
        if agent_name is None:
            workflow, _ = load_workflow(layout.workflow_path)
            agent_name = workflow.entry_agent
        workspace = layout.agents_dir / agent_name
        config_path = workspace / "config.yaml"
        updates = {"cg_url": cg_url, "cg_api_key": api_key}
        if register:
            resolved_api_key = _resolve_env(
                api_key, env_fallback="CONTEXTGRAPH_API_KEY"
            )
            if not resolved_api_key:
                raise ValueError(
                    "ContextGraph registration requires a resolvable API key"
                )
            bridge = ContextGraphBridge(cg_url=cg_url, api_key=resolved_api_key)
            agent_id = bridge.register(agent_name, org_id, capabilities=capabilities)
            updates["agent_id"] = agent_id
            if bridge.api_key:
                updates["cg_api_key"] = "${CONTEXTGRAPH_AGENT_KEY}"
        self._rewrite_config(config_path, updates)
        return self.contextgraph_status(agent_name=agent_name)
