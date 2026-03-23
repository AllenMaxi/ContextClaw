from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .simple_yaml import dump_yaml, parse_yaml

logger = logging.getLogger(__name__)

CATALOG_VERSION = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def catalog_root() -> Path:
    return repo_root() / "catalog"


def connectors_root(root: Path | None = None) -> Path:
    return (root or catalog_root()) / "connectors"


def skills_root(root: Path | None = None) -> Path:
    return (root or catalog_root()) / "skills"


def agent_catalog_dir(workspace: Path) -> Path:
    return workspace / ".contextclaw"


def catalog_state_path(workspace: Path) -> Path:
    return agent_catalog_dir(workspace) / "catalog.yaml"


def catalog_lock_path(workspace: Path) -> Path:
    return agent_catalog_dir(workspace) / "catalog.lock.json"


def generated_dir(workspace: Path) -> Path:
    return agent_catalog_dir(workspace) / "generated"


def generated_mcp_path(workspace: Path) -> Path:
    return generated_dir(workspace) / "mcp_servers.json"


def generated_policy_path(workspace: Path) -> Path:
    return generated_dir(workspace) / "policy.yaml"


def packaged_skills_dir(workspace: Path) -> Path:
    return workspace / "skills" / "packages"


@dataclass
class ConnectorMCPConfig:
    name: str
    command: list[str]
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ConnectorSpec:
    id: str
    version: str
    display_name: str
    description: str
    stability: str
    tags: list[str]
    type: str
    bundles: list[str] = field(default_factory=list)
    mcp: ConnectorMCPConfig | None = None
    prerequisites: dict[str, str] = field(default_factory=dict)
    required_env: dict[str, str] = field(default_factory=dict)
    policy_defaults: dict[str, list[str]] = field(default_factory=dict)
    tools_exposed: list[str] = field(default_factory=list)
    manifest_path: Path | None = None


@dataclass
class SkillSpec:
    id: str
    version: str
    display_name: str
    description: str
    stability: str
    tags: list[str]
    entrypoint: str
    requires_connectors: list[str] = field(default_factory=list)
    asset_dirs: list[str] = field(default_factory=list)
    manifest_path: Path | None = None
    package_dir: Path | None = None


@dataclass
class CatalogState:
    version: int = CATALOG_VERSION
    connectors: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    state: CatalogState
    lock: dict[str, Any]
    missing_prerequisites: dict[str, list[str]]
    missing_env: dict[str, list[str]]
    missing_connector_dependencies: dict[str, list[str]]


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _as_str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        item_str = str(item).strip()
        if item_str:
            result[str(key).strip()] = item_str
    return result


def _require_text(raw: dict[str, Any], key: str, path: Path) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"{path}: missing required field '{key}'")
    return value


def _manifest_files(root: Path, filename: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob(filename) if path.is_file())


def load_connector_specs(root: Path | None = None) -> dict[str, ConnectorSpec]:
    specs: dict[str, ConnectorSpec] = {}
    for manifest_path in _manifest_files(connectors_root(root), "connector.yaml"):
        raw = parse_yaml(manifest_path.read_text(encoding="utf-8"))
        connector_id = _require_text(raw, "id", manifest_path)
        if connector_id in specs:
            raise ValueError(
                f"{manifest_path}: duplicate connector id '{connector_id}'"
            )
        connector_type = _require_text(raw, "type", manifest_path)
        if connector_type not in {"bundle", "mcp", "composite"}:
            raise ValueError(
                f"{manifest_path}: invalid connector type '{connector_type}'"
            )

        mcp_config: ConnectorMCPConfig | None = None
        raw_mcp = raw.get("mcp", {})
        if connector_type in {"mcp", "composite"}:
            if not isinstance(raw_mcp, dict):
                raise ValueError(f"{manifest_path}: 'mcp' must be a mapping")
            name = _require_text(raw_mcp, "name", manifest_path)
            command = _as_str_list(raw_mcp.get("command", []))
            if not command:
                raise ValueError(
                    f"{manifest_path}: mcp.command must be a non-empty list"
                )
            mcp_config = ConnectorMCPConfig(
                name=name,
                command=command,
                cwd=str(raw_mcp.get("cwd", "")).strip(),
                env=_as_str_dict(raw_mcp.get("env", {})),
            )

        policy_defaults = raw.get("policy_defaults", {})
        if not isinstance(policy_defaults, dict):
            raise ValueError(f"{manifest_path}: 'policy_defaults' must be a mapping")
        if policy_defaults.get("auto_approve"):
            raise ValueError(
                f"{manifest_path}: generated policy may not set auto_approve"
            )

        spec = ConnectorSpec(
            id=connector_id,
            version=_require_text(raw, "version", manifest_path),
            display_name=_require_text(raw, "display_name", manifest_path),
            description=_require_text(raw, "description", manifest_path),
            stability=_require_text(raw, "stability", manifest_path),
            tags=_as_str_list(raw.get("tags", [])),
            type=connector_type,
            bundles=_as_str_list(raw.get("bundles", [])),
            mcp=mcp_config,
            prerequisites=_as_str_dict(raw.get("prerequisites", {})),
            required_env=_as_str_dict(raw.get("required_env", {})),
            policy_defaults={
                "require_confirm": _as_str_list(
                    policy_defaults.get("require_confirm", [])
                ),
                "blocked": _as_str_list(policy_defaults.get("blocked", [])),
            },
            tools_exposed=_as_str_list(raw.get("tools_exposed", [])),
            manifest_path=manifest_path,
        )
        specs[connector_id] = spec
    return specs


def load_skill_specs(root: Path | None = None) -> dict[str, SkillSpec]:
    specs: dict[str, SkillSpec] = {}
    for manifest_path in _manifest_files(skills_root(root), "skill.yaml"):
        raw = parse_yaml(manifest_path.read_text(encoding="utf-8"))
        skill_id = _require_text(raw, "id", manifest_path)
        if skill_id in specs:
            raise ValueError(f"{manifest_path}: duplicate skill id '{skill_id}'")
        entrypoint = _require_text(raw, "entrypoint", manifest_path)
        if entrypoint != "SKILL.md":
            raise ValueError(f"{manifest_path}: entrypoint must be 'SKILL.md'")
        package_dir = manifest_path.parent
        if not (package_dir / "SKILL.md").exists():
            raise ValueError(f"{manifest_path}: missing SKILL.md")
        asset_dirs = _as_str_list(raw.get("asset_dirs", []))
        allowed_asset_dirs = {"references", "templates", "scripts"}
        invalid_assets = [
            asset_dir for asset_dir in asset_dirs if asset_dir not in allowed_asset_dirs
        ]
        if invalid_assets:
            invalid = ", ".join(sorted(invalid_assets))
            raise ValueError(f"{manifest_path}: unsupported asset_dirs: {invalid}")
        extra_dirs = sorted(
            child.name
            for child in package_dir.iterdir()
            if child.is_dir() and child.name not in allowed_asset_dirs
        )
        if extra_dirs:
            invalid = ", ".join(extra_dirs)
            raise ValueError(
                f"{manifest_path}: package directories must be limited to references, templates, and scripts; found {invalid}"
            )
        specs[skill_id] = SkillSpec(
            id=skill_id,
            version=_require_text(raw, "version", manifest_path),
            display_name=_require_text(raw, "display_name", manifest_path),
            description=_require_text(raw, "description", manifest_path),
            stability=_require_text(raw, "stability", manifest_path),
            tags=_as_str_list(raw.get("tags", [])),
            entrypoint=entrypoint,
            requires_connectors=_as_str_list(raw.get("requires_connectors", [])),
            asset_dirs=asset_dirs,
            manifest_path=manifest_path,
            package_dir=package_dir,
        )
    return specs


def read_catalog_state(workspace: Path) -> CatalogState:
    path = catalog_state_path(workspace)
    if not path.exists():
        return CatalogState()
    raw = parse_yaml(path.read_text(encoding="utf-8"))
    version_raw = raw.get("version", CATALOG_VERSION)
    try:
        version = int(version_raw)
    except (TypeError, ValueError):
        version = CATALOG_VERSION
    return CatalogState(
        version=version,
        connectors=_as_str_list(raw.get("connectors", [])),
        skills=_as_str_list(raw.get("skills", [])),
    )


def write_catalog_state(workspace: Path, state: CatalogState) -> Path:
    path = catalog_state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": state.version,
        "connectors": sorted(dict.fromkeys(state.connectors)),
        "skills": sorted(dict.fromkeys(state.skills)),
    }
    path.write_text(dump_yaml(payload) + "\n", encoding="utf-8")
    return path


def read_catalog_lock(workspace: Path) -> dict[str, Any]:
    path = catalog_lock_path(workspace)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_catalog_state(
    state: CatalogState,
    connector_specs: dict[str, ConnectorSpec],
    skill_specs: dict[str, SkillSpec],
) -> None:
    missing_connectors = [
        item for item in state.connectors if item not in connector_specs
    ]
    missing_skills = [item for item in state.skills if item not in skill_specs]
    if missing_connectors or missing_skills:
        errors = []
        if missing_connectors:
            errors.append(
                f"unknown connectors: {', '.join(sorted(missing_connectors))}"
            )
        if missing_skills:
            errors.append(f"unknown skills: {', '.join(sorted(missing_skills))}")
        raise ValueError("; ".join(errors))


def missing_connector_dependencies(
    state: CatalogState, skill_specs: dict[str, SkillSpec]
) -> dict[str, list[str]]:
    installed = set(state.connectors)
    missing: dict[str, list[str]] = {}
    for skill_id in state.skills:
        required = [
            connector_id
            for connector_id in skill_specs[skill_id].requires_connectors
            if connector_id not in installed
        ]
        if required:
            missing[skill_id] = required
    return missing


def validate_connector_prerequisites(
    connector_ids: list[str], connector_specs: dict[str, ConnectorSpec]
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for connector_id in connector_ids:
        spec = connector_specs[connector_id]
        absent = [
            command for command in spec.prerequisites if shutil.which(command) is None
        ]
        if absent:
            missing[connector_id] = absent
    return missing


def collect_missing_env(
    connector_ids: list[str], connector_specs: dict[str, ConnectorSpec]
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for connector_id in connector_ids:
        absent = [
            env_name
            for env_name in connector_specs[connector_id].required_env
            if not os.environ.get(env_name, "").strip()
        ]
        if absent:
            missing[connector_id] = absent
    return missing


def _connector_lock_entry(
    spec: ConnectorSpec,
    *,
    missing_env_names: list[str],
    missing_prereq_names: list[str],
) -> dict[str, Any]:
    return {
        "id": spec.id,
        "display_name": spec.display_name,
        "type": spec.type,
        "bundles": spec.bundles,
        "tools_exposed": spec.tools_exposed,
        "required_env": sorted(spec.required_env),
        "missing_env": missing_env_names,
        "prerequisites": sorted(spec.prerequisites),
        "missing_prerequisites": missing_prereq_names,
        "mcp_server_name": spec.mcp.name if spec.mcp else "",
    }


def _skill_lock_entry(spec: SkillSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "display_name": spec.display_name,
        "requires_connectors": spec.requires_connectors,
        "asset_dirs": spec.asset_dirs,
    }


def _write_lock(workspace: Path, payload: dict[str, Any]) -> Path:
    path = catalog_lock_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _copy_packaged_skills(
    workspace: Path,
    state: CatalogState,
    skill_specs: dict[str, SkillSpec],
) -> None:
    packages_root = packaged_skills_dir(workspace)
    desired = set(state.skills)
    if desired:
        packages_root.mkdir(parents=True, exist_ok=True)
    elif packages_root.exists():
        shutil.rmtree(packages_root)
        return

    for skill_id in desired:
        source = skill_specs[skill_id].package_dir
        if source is None:
            continue
        destination = packages_root / skill_id
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    if packages_root.exists():
        for child in packages_root.iterdir():
            if child.is_dir() and child.name not in desired:
                shutil.rmtree(child)


def _write_generated_mcp(
    workspace: Path,
    state: CatalogState,
    connector_specs: dict[str, ConnectorSpec],
    missing_prereq: dict[str, list[str]],
) -> Path | None:
    servers: list[dict[str, Any]] = []
    for connector_id in state.connectors:
        spec = connector_specs[connector_id]
        if spec.mcp is None or connector_id in missing_prereq:
            continue
        entry: dict[str, Any] = {
            "name": spec.mcp.name,
            "command": spec.mcp.command,
        }
        if spec.mcp.cwd:
            entry["cwd"] = spec.mcp.cwd
        if spec.mcp.env:
            entry["env"] = spec.mcp.env
        servers.append(entry)

    path = generated_mcp_path(workspace)
    if not servers:
        if path.exists():
            path.unlink()
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"servers": servers}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_generated_policy(
    workspace: Path,
    state: CatalogState,
    connector_specs: dict[str, ConnectorSpec],
) -> Path | None:
    require_confirm: list[str] = []
    blocked: list[str] = []
    for connector_id in state.connectors:
        spec = connector_specs[connector_id]
        require_confirm.extend(spec.policy_defaults.get("require_confirm", []))
        blocked.extend(spec.policy_defaults.get("blocked", []))

    require_confirm = sorted(dict.fromkeys(item for item in require_confirm if item))
    blocked = sorted(dict.fromkeys(item for item in blocked if item))

    path = generated_policy_path(workspace)
    if not require_confirm and not blocked:
        if path.exists():
            path.unlink()
        return None

    payload = {
        "permissions": {
            "tools": {
                "require_confirm": require_confirm,
                "blocked": blocked,
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(payload) + "\n", encoding="utf-8")
    return path


def sync_agent_catalog(workspace: Path, root: Path | None = None) -> SyncResult:
    connector_specs = load_connector_specs(root)
    skill_specs = load_skill_specs(root)
    state = read_catalog_state(workspace)
    validate_catalog_state(state, connector_specs, skill_specs)
    write_catalog_state(workspace, state)

    missing_prereq = validate_connector_prerequisites(state.connectors, connector_specs)
    missing_env = collect_missing_env(state.connectors, connector_specs)
    missing_deps = missing_connector_dependencies(state, skill_specs)

    _copy_packaged_skills(workspace, state, skill_specs)
    generated_mcp = _write_generated_mcp(
        workspace, state, connector_specs, missing_prereq
    )
    generated_policy = _write_generated_policy(workspace, state, connector_specs)

    lock_payload = {
        "version": state.version,
        "connectors": [
            _connector_lock_entry(
                connector_specs[connector_id],
                missing_env_names=missing_env.get(connector_id, []),
                missing_prereq_names=missing_prereq.get(connector_id, []),
            )
            for connector_id in state.connectors
        ],
        "skills": [
            _skill_lock_entry(skill_specs[skill_id]) for skill_id in state.skills
        ],
        "missing_connector_dependencies": missing_deps,
        "generated": {
            "mcp_servers_path": str(generated_mcp) if generated_mcp else "",
            "policy_path": str(generated_policy) if generated_policy else "",
        },
    }
    _write_lock(workspace, lock_payload)

    return SyncResult(
        state=state,
        lock=lock_payload,
        missing_prerequisites=missing_prereq,
        missing_env=missing_env,
        missing_connector_dependencies=missing_deps,
    )


def connector_bundles_from_lock(workspace: Path) -> list[str]:
    lock = read_catalog_lock(workspace)
    bundles: list[str] = []
    for connector in lock.get("connectors", []):
        if not isinstance(connector, dict):
            continue
        if connector.get("missing_prerequisites"):
            continue
        for bundle_name in _as_str_list(connector.get("bundles", [])):
            if bundle_name not in bundles:
                bundles.append(bundle_name)
    return bundles


def missing_env_from_lock(workspace: Path) -> dict[str, list[str]]:
    lock = read_catalog_lock(workspace)
    result: dict[str, list[str]] = {}
    for connector in lock.get("connectors", []):
        if not isinstance(connector, dict):
            continue
        connector_id = str(connector.get("id", "")).strip()
        missing = _as_str_list(connector.get("missing_env", []))
        if connector_id and missing:
            result[connector_id] = missing
    return result


def installed_connectors_from_lock(workspace: Path) -> list[str]:
    lock = read_catalog_lock(workspace)
    return [
        str(item.get("id", "")).strip()
        for item in lock.get("connectors", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]


def installed_skills_from_lock(workspace: Path) -> list[str]:
    lock = read_catalog_lock(workspace)
    return [
        str(item.get("id", "")).strip()
        for item in lock.get("skills", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]


def generated_paths_from_lock(workspace: Path) -> tuple[Path | None, Path | None]:
    lock = read_catalog_lock(workspace)
    generated = lock.get("generated", {})
    if not isinstance(generated, dict):
        return None, None
    mcp_raw = str(generated.get("mcp_servers_path", "")).strip()
    policy_raw = str(generated.get("policy_path", "")).strip()
    return (
        Path(mcp_raw) if mcp_raw else None,
        Path(policy_raw) if policy_raw else None,
    )


def missing_connector_dependencies_from_lock(workspace: Path) -> dict[str, list[str]]:
    lock = read_catalog_lock(workspace)
    missing = lock.get("missing_connector_dependencies", {})
    if not isinstance(missing, dict):
        return {}
    result: dict[str, list[str]] = {}
    for skill_id, connector_ids in missing.items():
        names = _as_str_list(connector_ids)
        if names:
            result[str(skill_id).strip()] = names
    return result


def catalog_sync_required(workspace: Path) -> bool:
    state_path = catalog_state_path(workspace)
    if not state_path.exists():
        return False
    lock = read_catalog_lock(workspace)
    if not lock:
        return True
    state = read_catalog_state(workspace)
    locked_connectors = installed_connectors_from_lock(workspace)
    locked_skills = installed_skills_from_lock(workspace)
    try:
        locked_version = int(lock.get("version", CATALOG_VERSION))
    except (TypeError, ValueError):
        locked_version = CATALOG_VERSION
    return (
        state.version != locked_version
        or state.connectors != locked_connectors
        or state.skills != locked_skills
    )
