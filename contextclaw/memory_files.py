from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context_engine import estimate_text_tokens
from .project import get_project_layout

_MEMORY_FILENAMES = ("AGENTS.md", "MEMORY.md")
_USER_MEMORY_ROOT = Path.home() / ".contextclaw"
_HISTORY_DIRNAME = "memory_files"
_INDEX_FILENAME = "index.json"
_SNAPSHOT_DIRNAME = "snapshots"
_DEFAULT_FILENAMES = {
    "user": "AGENTS.md",
    "project": "AGENTS.md",
    "agent": "MEMORY.md",
}


@dataclass(frozen=True)
class MemoryFileSource:
    path: Path
    scope: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _scope_runtime_dir(workspace: Path, scope: str) -> Path:
    workspace = workspace.resolve()
    if scope == "user":
        return _USER_MEMORY_ROOT.resolve()
    if scope == "project":
        layout = get_project_layout(workspace)
        if layout is None:
            raise ValueError("project-scoped memory files require project-local mode")
        return layout.runtime_dir
    return (workspace / ".contextclaw").resolve()


def _target_paths(
    workspace: Path,
    *,
    scope: str,
    filename: str = "",
) -> dict[str, Path | str]:
    path = resolve_memory_file_path(workspace, scope=scope, filename=filename)
    runtime_dir = _scope_runtime_dir(workspace, scope)
    history_dir = (
        runtime_dir / _HISTORY_DIRNAME / str(scope).strip().lower() / path.name
    )
    return {
        "path": path,
        "scope": str(scope).strip().lower(),
        "runtime_dir": runtime_dir,
        "history_dir": history_dir,
        "index_path": history_dir / _INDEX_FILENAME,
        "snapshots_dir": history_dir / _SNAPSHOT_DIRNAME,
    }


def _load_history_index(target: dict[str, Path | str]) -> dict[str, Any]:
    index = _read_json(target["index_path"])
    revisions = index.get("revisions", [])
    if not isinstance(revisions, list):
        revisions = []
    return {
        "schema_version": 1,
        "scope": str(target["scope"]),
        "filename": str(Path(str(target["path"])).name),
        "path": str(target["path"]),
        "last_synced_memory_id": str(index.get("last_synced_memory_id", "")),
        "last_synced_revision_id": str(index.get("last_synced_revision_id", "")),
        "revisions": [item for item in revisions if isinstance(item, dict)],
    }


def _write_history_index(target: dict[str, Path | str], index: dict[str, Any]) -> None:
    _write_json(target["index_path"], index)


def _snapshot_path(
    target: dict[str, Path | str],
    revision_id: str,
) -> Path:
    return Path(str(target["snapshots_dir"])) / f"{revision_id}.md"


def _build_memory_file_payload(
    target: dict[str, Path | str],
    content: str,
    index: dict[str, Any],
) -> dict[str, Any]:
    content_hash = _content_hash(content)
    current_revision_id = ""
    revisions = list(index.get("revisions", []))
    if revisions and revisions[-1].get("content_hash") == content_hash:
        current_revision_id = str(revisions[-1].get("id", ""))
    latest_synced_at = ""
    if index.get("last_synced_revision_id"):
        for item in reversed(revisions):
            if item.get("id") == index.get("last_synced_revision_id"):
                latest_synced_at = str(item.get("synced_at", ""))
                break
    return {
        "path": str(target["path"]),
        "scope": str(target["scope"]),
        "filename": Path(str(target["path"])).name,
        "content": content,
        "exists": Path(str(target["path"])).exists(),
        "token_estimate": estimate_text_tokens(content),
        "content_hash": content_hash,
        "revision_count": len(revisions),
        "current_revision_id": current_revision_id,
        "last_synced_memory_id": str(index.get("last_synced_memory_id", "")),
        "last_synced_revision_id": str(index.get("last_synced_revision_id", "")),
        "last_synced_at": latest_synced_at,
    }


def _record_revision(
    target: dict[str, Path | str],
    *,
    content: str,
    action: str,
    source_revision_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    index = _load_history_index(target)
    revisions = list(index["revisions"])
    content_hash = _content_hash(content)
    latest = revisions[-1] if revisions else None
    if latest is not None and latest.get("content_hash") == content_hash:
        return index, latest, False
    timestamp = time.time()
    revision_id = (
        f"rev_{time.strftime('%Y%m%d_%H%M%S', time.gmtime(timestamp))}_"
        f"{content_hash[:8]}"
    )
    snapshot = _snapshot_path(target, revision_id)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(content, encoding="utf-8")
    revision = {
        "id": revision_id,
        "created_at": timestamp,
        "action": action,
        "content_hash": content_hash,
        "token_estimate": estimate_text_tokens(content),
        "snapshot_path": str(snapshot),
        "content_preview": content[:200],
        "source_revision_id": source_revision_id,
        "synced_memory_id": "",
        "synced_at": "",
    }
    revisions.append(revision)
    index["revisions"] = revisions
    _write_history_index(target, index)
    return index, revision, True


def ensure_memory_file_history(
    workspace: Path,
    *,
    scope: str,
    filename: str = "",
) -> dict[str, Any]:
    target = _target_paths(workspace, scope=scope, filename=filename)
    path = Path(str(target["path"]))
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    index = _load_history_index(target)
    if path.exists():
        action = "baseline" if not index["revisions"] else "external"
        index, _, _ = _record_revision(target, content=content, action=action)
    return _build_memory_file_payload(target, content, index)


def discover_memory_files(workspace: Path) -> list[MemoryFileSource]:
    workspace = workspace.resolve()
    candidates: list[MemoryFileSource] = []
    user_root = _USER_MEMORY_ROOT.resolve()
    for filename in _MEMORY_FILENAMES:
        candidates.append(MemoryFileSource(path=user_root / filename, scope="user"))

    project_layout = get_project_layout(workspace)
    if project_layout is not None:
        for filename in _MEMORY_FILENAMES:
            candidates.append(
                MemoryFileSource(
                    path=(project_layout.root / filename).resolve(),
                    scope="project",
                )
            )

    for filename in _MEMORY_FILENAMES:
        candidates.append(
            MemoryFileSource(path=(workspace / filename).resolve(), scope="agent")
        )

    seen: set[Path] = set()
    result: list[MemoryFileSource] = []
    for item in candidates:
        if item.path in seen or not item.path.exists() or not item.path.is_file():
            continue
        seen.add(item.path)
        result.append(item)
    return result


def resolve_memory_file_path(
    workspace: Path,
    *,
    scope: str,
    filename: str = "",
) -> Path:
    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in _DEFAULT_FILENAMES:
        raise ValueError("scope must be one of: user, project, agent")
    selected_filename = (filename or _DEFAULT_FILENAMES[normalized_scope]).strip()
    if selected_filename not in _MEMORY_FILENAMES:
        raise ValueError(f"filename must be one of: {', '.join(_MEMORY_FILENAMES)}")

    workspace = workspace.resolve()
    if normalized_scope == "user":
        return (_USER_MEMORY_ROOT / selected_filename).resolve()
    if normalized_scope == "project":
        layout = get_project_layout(workspace)
        if layout is None:
            raise ValueError("project-scoped memory files require project-local mode")
        return (layout.root / selected_filename).resolve()
    return (workspace / selected_filename).resolve()


def load_memory_files(workspace: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for item in discover_memory_files(workspace):
        try:
            content = item.path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not content:
            continue
        target = _target_paths(workspace, scope=item.scope, filename=item.path.name)
        index = _load_history_index(target)
        files.append(_build_memory_file_payload(target, content, index))
    return files


def list_tracked_memory_files(workspace: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for item in discover_memory_files(workspace):
        tracked = ensure_memory_file_history(
            workspace,
            scope=item.scope,
            filename=item.path.name,
        )
        files.append(tracked)
    return files


def read_memory_file(
    workspace: Path,
    *,
    scope: str,
    filename: str = "",
) -> dict[str, Any]:
    tracked = ensure_memory_file_history(workspace, scope=scope, filename=filename)
    path = resolve_memory_file_path(workspace, scope=scope, filename=filename)
    try:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        raise ValueError(f"Failed to read memory file '{path}': {exc}") from exc
    return {**tracked, "content": content}


def write_memory_file(
    workspace: Path,
    *,
    scope: str,
    content: str,
    filename: str = "",
    append: bool = False,
) -> dict[str, Any]:
    target = _target_paths(workspace, scope=scope, filename=filename)
    path = Path(str(target["path"]))
    existing = ""
    if append and path.exists():
        existing = path.read_text(encoding="utf-8")
    final_content = f"{existing}{content}" if append else content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(final_content, encoding="utf-8")
    _record_revision(
        target,
        content=final_content,
        action="append" if append else "write",
    )
    return read_memory_file(workspace, scope=scope, filename=path.name)


def list_memory_file_revisions(
    workspace: Path,
    *,
    scope: str,
    filename: str = "",
) -> dict[str, Any]:
    tracked = ensure_memory_file_history(workspace, scope=scope, filename=filename)
    target = _target_paths(workspace, scope=scope, filename=filename)
    index = _load_history_index(target)
    revisions = []
    for item in reversed(index["revisions"]):
        revisions.append(
            {
                **item,
                "current": item.get("id") == tracked["current_revision_id"],
                "snapshot_exists": Path(str(item.get("snapshot_path", ""))).exists(),
            }
        )
    return {**tracked, "revisions": revisions}


def read_memory_file_revision(
    workspace: Path,
    *,
    scope: str,
    revision_id: str,
    filename: str = "",
) -> dict[str, Any]:
    listing = list_memory_file_revisions(workspace, scope=scope, filename=filename)
    for item in listing["revisions"]:
        if item.get("id") != revision_id:
            continue
        snapshot_path = Path(str(item.get("snapshot_path", "")))
        try:
            content = snapshot_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Failed to read revision snapshot '{snapshot_path}': {exc}"
            ) from exc
        return {
            **listing,
            "revision": item,
            "content": content,
        }
    raise ValueError(f"Revision '{revision_id}' was not found")


def restore_memory_file_revision(
    workspace: Path,
    *,
    scope: str,
    revision_id: str,
    filename: str = "",
) -> dict[str, Any]:
    target = _target_paths(workspace, scope=scope, filename=filename)
    revision = read_memory_file_revision(
        workspace,
        scope=scope,
        revision_id=revision_id,
        filename=filename or Path(str(target["path"])).name,
    )
    path = Path(str(target["path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(revision["content"]), encoding="utf-8")
    _record_revision(
        target,
        content=str(revision["content"]),
        action="restore",
        source_revision_id=revision_id,
    )
    current = read_memory_file(workspace, scope=scope, filename=path.name)
    return {**current, "restored_from_revision_id": revision_id}


def mark_memory_file_revision_synced(
    workspace: Path,
    *,
    scope: str,
    revision_id: str,
    memory_id: str,
    filename: str = "",
) -> dict[str, Any]:
    target = _target_paths(workspace, scope=scope, filename=filename)
    index = _load_history_index(target)
    synced_at = time.time()
    updated = None
    for item in index["revisions"]:
        if item.get("id") != revision_id:
            continue
        item["synced_memory_id"] = memory_id
        item["synced_at"] = synced_at
        updated = item
        break
    if updated is None:
        raise ValueError(f"Revision '{revision_id}' was not found")
    index["last_synced_memory_id"] = memory_id
    index["last_synced_revision_id"] = revision_id
    _write_history_index(target, index)
    return {
        **read_memory_file(
            workspace, scope=scope, filename=Path(str(target["path"])).name
        ),
        "synced_revision": updated,
    }


def render_memory_files_prompt(workspace: Path) -> tuple[str, list[dict[str, Any]]]:
    files = load_memory_files(workspace)
    if not files:
        return "", []

    lines = [
        "## Always-Loaded Memory Files",
        (
            "These file-backed memories are durable operator-controlled context. "
            "Use them as project or agent memory, not as new user instructions."
        ),
        "",
    ]
    for item in files:
        lines.append(f"### {item['scope'].title()} Memory ({item['filename']})")
        lines.append(str(item["content"]).strip())
        lines.append("")
    return "\n".join(lines).strip(), files
