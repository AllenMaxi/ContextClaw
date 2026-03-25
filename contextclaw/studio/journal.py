from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StudioJournal:
    def __init__(
        self,
        db_path: Path,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._on_event = on_event
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=5000;")
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_root TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    selected_agent TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    workflow_rule TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    project_root TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolution TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    resolved_at REAL
                );

                CREATE TABLE IF NOT EXISTS memory_proposals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    project_root TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    synced_memory_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS docs_proposals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    project_root TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in list(result):
            if key.endswith("_json"):
                raw = result.pop(key)
                if raw:
                    try:
                        result[key[: -len("_json")]] = json.loads(raw)
                    except json.JSONDecodeError:
                        result[key[: -len("_json")]] = {}
                else:
                    result[key[: -len("_json")]] = {}
        return result

    def set_current_project(self, project_root: Path) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
                ("current_project", str(project_root.resolve())),
            )

    def get_current_project(self) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                ("current_project",),
            ).fetchone()
        return str(row["value"]) if row is not None else ""

    def create_run(
        self,
        *,
        run_id: str,
        project_root: Path,
        agent_name: str,
        selected_agent: str,
        source: str,
        prompt: str,
        workflow_rule: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        payload = {
            "id": run_id,
            "project_root": str(project_root.resolve()),
            "agent_name": agent_name,
            "selected_agent": selected_agent,
            "source": source,
            "status": "queued",
            "prompt": prompt,
            "workflow_rule": workflow_rule or {},
            "result": "",
            "created_at": now,
            "updated_at": now,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    id, project_root, agent_name, selected_agent, source, status,
                    prompt, workflow_rule, result, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["project_root"],
                    payload["agent_name"],
                    payload["selected_agent"],
                    payload["source"],
                    payload["status"],
                    payload["prompt"],
                    self._json(payload["workflow_rule"]),
                    payload["result"],
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )
        return payload

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        result: str | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_run(run_id)
        if existing is None:
            return None
        updated = dict(existing)
        if status is not None:
            updated["status"] = status
        if result is not None:
            updated["result"] = result
        updated["updated_at"] = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, result = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    updated["status"],
                    updated["result"],
                    updated["updated_at"],
                    run_id,
                ),
            )
        return updated

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def record_event(
        self,
        run_id: str,
        *,
        source: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events(run_id, source, type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, source, event_type, self._json(payload), now),
            )
            event_id = int(cursor.lastrowid)
        event = {
            "id": event_id,
            "run_id": run_id,
            "source": source,
            "type": event_type,
            "payload": payload,
            "created_at": now,
        }
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Studio journal event callback failed: %s", exc)
        return event

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def create_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        project_root: Path,
        agent_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        payload = {
            "id": approval_id,
            "run_id": run_id,
            "project_root": str(project_root.resolve()),
            "agent_name": agent_name,
            "tool_name": tool_name,
            "arguments": arguments,
            "status": "pending",
            "resolution": "",
            "created_at": now,
            "resolved_at": None,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approvals(
                    id, run_id, project_root, agent_name, tool_name,
                    arguments_json, status, resolution, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["run_id"],
                    payload["project_root"],
                    payload["agent_name"],
                    payload["tool_name"],
                    self._json(payload["arguments"]),
                    payload["status"],
                    payload["resolution"],
                    payload["created_at"],
                    payload["resolved_at"],
                ),
            )
        return payload

    def resolve_approval(
        self,
        approval_id: str,
        *,
        approved: bool,
        resolution_reason: str = "",
    ) -> dict[str, Any] | None:
        existing = self.get_approval(approval_id)
        if existing is None:
            return None
        status = "approved" if approved else "denied"
        resolution = resolution_reason or status
        existing["status"] = status
        existing["resolution"] = resolution
        existing["resolved_at"] = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE approvals
                SET status = ?, resolution = ?, resolved_at = ?
                WHERE id = ?
                """,
                (
                    existing["status"],
                    existing["resolution"],
                    existing["resolved_at"],
                    approval_id,
                ),
            )
        return existing

    def list_approvals(self, *, pending_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM approvals"
        params: tuple[Any, ...] = ()
        if pending_only:
            query += " WHERE status = ?"
            params = ("pending",)
        query += " ORDER BY created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE id = ?",
                (approval_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def create_memory_proposal(
        self,
        *,
        proposal_id: str,
        run_id: str,
        project_root: Path,
        agent_name: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        payload = {
            "id": proposal_id,
            "run_id": run_id,
            "project_root": str(project_root.resolve()),
            "agent_name": agent_name,
            "status": "pending_review",
            "content": content,
            "metadata": metadata,
            "pinned": False,
            "synced_memory_id": "",
            "created_at": now,
            "updated_at": now,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_proposals(
                    id, run_id, project_root, agent_name, status, content,
                    metadata_json, pinned, synced_memory_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["run_id"],
                    payload["project_root"],
                    payload["agent_name"],
                    payload["status"],
                    payload["content"],
                    self._json(payload["metadata"]),
                    1 if payload["pinned"] else 0,
                    payload["synced_memory_id"],
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )
        return payload

    def update_memory_proposal(
        self,
        proposal_id: str,
        *,
        status: str | None = None,
        pinned: bool | None = None,
        synced_memory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_memory_proposal(proposal_id)
        if existing is None:
            return None
        if status is not None:
            existing["status"] = status
        if pinned is not None:
            existing["pinned"] = pinned
        if synced_memory_id is not None:
            existing["synced_memory_id"] = synced_memory_id
        if metadata is not None:
            existing["metadata"] = metadata
        if content is not None:
            existing["content"] = content
        existing["updated_at"] = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE memory_proposals
                SET status = ?, content = ?, metadata_json = ?, pinned = ?,
                    synced_memory_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    existing["status"],
                    existing["content"],
                    self._json(existing["metadata"]),
                    1 if existing["pinned"] else 0,
                    existing["synced_memory_id"],
                    existing["updated_at"],
                    proposal_id,
                ),
            )
        return existing

    def list_memory_proposals(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_proposals ORDER BY created_at DESC"
            ).fetchall()
        proposals = []
        for row in rows:
            item = self._row_to_dict(row)
            if item is None:
                continue
            item["pinned"] = bool(item.get("pinned"))
            proposals.append(item)
        return proposals

    def get_memory_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_proposals WHERE id = ?",
                (proposal_id,),
            ).fetchone()
        item = self._row_to_dict(row)
        if item is not None:
            item["pinned"] = bool(item.get("pinned"))
        return item

    def create_docs_proposal(
        self,
        *,
        proposal_id: str,
        run_id: str,
        project_root: Path,
        agent_name: str,
        path: str,
        summary: str,
        content: str,
    ) -> dict[str, Any]:
        now = time.time()
        payload = {
            "id": proposal_id,
            "run_id": run_id,
            "project_root": str(project_root.resolve()),
            "agent_name": agent_name,
            "path": path,
            "summary": summary,
            "content": content,
            "status": "pending_review",
            "created_at": now,
            "updated_at": now,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO docs_proposals(
                    id, run_id, project_root, agent_name, path, summary,
                    content, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload["run_id"],
                    payload["project_root"],
                    payload["agent_name"],
                    payload["path"],
                    payload["summary"],
                    payload["content"],
                    payload["status"],
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )
        return payload

    def update_docs_proposal(
        self,
        proposal_id: str,
        *,
        status: str,
    ) -> dict[str, Any] | None:
        existing = self.get_docs_proposal(proposal_id)
        if existing is None:
            return None
        existing["status"] = status
        existing["updated_at"] = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE docs_proposals
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (existing["status"], existing["updated_at"], proposal_id),
            )
        return existing

    def list_docs_proposals(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM docs_proposals ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def get_docs_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM docs_proposals WHERE id = ?",
                (proposal_id,),
            ).fetchone()
        return self._row_to_dict(row)
