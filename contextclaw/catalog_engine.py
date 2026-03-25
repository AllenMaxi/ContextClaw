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


def generated_connectors_path(workspace: Path) -> Path:
    return generated_dir(workspace) / "connectors.json"


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
class ConnectorRuntimeConfig:
    driver: str
    transport: str
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    headers_env: dict[str, str] = field(default_factory=dict)
    auth: str = "none"
    capabilities: list[str] = field(default_factory=lambda: ["tools"])
    tool_allowlist: list[str] = field(default_factory=list)
    tool_prefix: str = ""
    default_policy: dict[str, list[str]] = field(default_factory=dict)
    timeouts: dict[str, float] = field(default_factory=dict)
    output_limit_tokens: int = 0
    doctor_checks: list[str] = field(default_factory=list)
    docs_url: str = ""
    adapter: str = ""

    @property
    def command_parts(self) -> list[str]:
        parts: list[str] = []
        if self.command:
            parts.append(self.command)
        parts.extend(self.args)
        return parts


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
    runtime: ConnectorRuntimeConfig | None = None
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


def _as_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key).strip()] = float(item)
        except (TypeError, ValueError):
            continue
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


def _parse_runtime_config(
    raw: dict[str, Any],
    *,
    connector_id: str,
    path: Path,
    required_env: dict[str, str],
) -> tuple[ConnectorRuntimeConfig | None, ConnectorMCPConfig | None]:
    raw_runtime = raw.get("runtime", {})
    raw_mcp = raw.get("mcp", {})

    if raw_runtime and raw_mcp:
        raise ValueError(f"{path}: use either 'runtime' or legacy 'mcp', not both")

    legacy_mcp: ConnectorMCPConfig | None = None
    runtime: ConnectorRuntimeConfig | None = None

    if raw_runtime:
        if not isinstance(raw_runtime, dict):
            raise ValueError(f"{path}: 'runtime' must be a mapping")
        driver = _require_text(raw_runtime, "driver", path)
        if driver not in {"managed_mcp", "python_adapter"}:
            raise ValueError(f"{path}: invalid runtime.driver '{driver}'")
        transport = _require_text(raw_runtime, "transport", path)
        if transport not in {"stdio", "http", "sse"}:
            raise ValueError(f"{path}: invalid runtime.transport '{transport}'")

        command = str(raw_runtime.get("command", "")).strip()
        args = _as_str_list(raw_runtime.get("args", []))
        url = str(raw_runtime.get("url", "")).strip()
        if driver == "managed_mcp":
            if transport == "stdio" and not command:
                raise ValueError(
                    f"{path}: managed_mcp stdio connectors require runtime.command"
                )
            if transport in {"http", "sse"} and not url:
                raise ValueError(
                    f"{path}: managed_mcp {transport} connectors require runtime.url"
                )
        if driver == "python_adapter":
            adapter = _require_text(raw_runtime, "adapter", path)
        else:
            adapter = ""

        runtime = ConnectorRuntimeConfig(
            driver=driver,
            transport=transport,
            name=str(raw_runtime.get("name", connector_id)).strip() or connector_id,
            command=command,
            args=args,
            url=url,
            cwd=str(raw_runtime.get("cwd", "")).strip(),
            env=_as_str_dict(raw_runtime.get("env", {})),
            headers_env=_as_str_dict(raw_runtime.get("headers_env", {})),
            auth=str(
                raw_runtime.get(
                    "auth",
                    "env"
                    if required_env
                    else ("none" if driver == "python_adapter" else "none"),
                )
            ).strip()
            or "none",
            capabilities=_as_str_list(raw_runtime.get("capabilities", ["tools"]))
            or ["tools"],
            tool_allowlist=_as_str_list(raw_runtime.get("tool_allowlist", [])),
            tool_prefix=str(raw_runtime.get("tool_prefix", "")).strip()
            or f"mcp__{connector_id}",
            default_policy={
                "require_confirm": _as_str_list(
                    raw_runtime.get("default_policy", {}).get("require_confirm", [])
                    if isinstance(raw_runtime.get("default_policy", {}), dict)
                    else []
                ),
                "blocked": _as_str_list(
                    raw_runtime.get("default_policy", {}).get("blocked", [])
                    if isinstance(raw_runtime.get("default_policy", {}), dict)
                    else []
                ),
            },
            timeouts=_as_float_dict(raw_runtime.get("timeouts", {})),
            output_limit_tokens=int(raw_runtime.get("output_limit_tokens", 0) or 0),
            doctor_checks=_as_str_list(
                raw_runtime.get(
                    "doctor_checks", ["env", "prerequisites", "connectivity", "tools"]
                )
            ),
            docs_url=str(raw_runtime.get("docs_url", "")).strip(),
            adapter=adapter,
        )
        if (
            runtime.driver == "managed_mcp"
            and runtime.transport == "stdio"
            and runtime.command
        ):
            legacy_mcp = ConnectorMCPConfig(
                name=runtime.name,
                command=runtime.command_parts,
                cwd=runtime.cwd,
                env=runtime.env,
            )
        return runtime, legacy_mcp

    if raw_mcp:
        if not isinstance(raw_mcp, dict):
            raise ValueError(f"{path}: 'mcp' must be a mapping")
        name = _require_text(raw_mcp, "name", path)
        command_parts = _as_str_list(raw_mcp.get("command", []))
        if not command_parts:
            raise ValueError(f"{path}: mcp.command must be a non-empty list")
        legacy_mcp = ConnectorMCPConfig(
            name=name,
            command=command_parts,
            cwd=str(raw_mcp.get("cwd", "")).strip(),
            env=_as_str_dict(raw_mcp.get("env", {})),
        )
        runtime = ConnectorRuntimeConfig(
            driver="managed_mcp",
            transport="stdio",
            name=name,
            command=command_parts[0],
            args=command_parts[1:],
            cwd=legacy_mcp.cwd,
            env=legacy_mcp.env,
            auth="env" if required_env else "none",
            capabilities=["tools"],
            tool_prefix=f"mcp__{name}",
            doctor_checks=["env", "prerequisites", "connectivity", "tools"],
            docs_url=str(raw.get("docs_url", "")).strip(),
        )
        return runtime, legacy_mcp

    return None, None


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
        required_env = _as_str_dict(raw.get("required_env", {}))
        runtime_config: ConnectorRuntimeConfig | None = None
        mcp_config: ConnectorMCPConfig | None = None
        if connector_type in {"mcp", "composite"}:
            runtime_config, mcp_config = _parse_runtime_config(
                raw,
                connector_id=connector_id,
                path=manifest_path,
                required_env=required_env,
            )
            if runtime_config is None:
                raise ValueError(
                    f"{manifest_path}: non-bundle connectors require either 'runtime' or 'mcp'"
                )

        policy_defaults = raw.get("policy_defaults", raw.get("default_policy", {}))
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
            runtime=runtime_config,
            prerequisites=_as_str_dict(raw.get("prerequisites", {})),
            required_env=required_env,
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
    runtime: dict[str, Any] = {}
    if spec.runtime is not None:
        runtime = {
            "driver": spec.runtime.driver,
            "transport": spec.runtime.transport,
            "name": spec.runtime.name,
            "command": spec.runtime.command,
            "args": spec.runtime.args,
            "url": spec.runtime.url,
            "cwd": spec.runtime.cwd,
            "env": spec.runtime.env,
            "headers_env": spec.runtime.headers_env,
            "auth": spec.runtime.auth,
            "capabilities": spec.runtime.capabilities,
            "tool_allowlist": spec.runtime.tool_allowlist,
            "tool_prefix": spec.runtime.tool_prefix,
            "default_policy": spec.runtime.default_policy,
            "timeouts": spec.runtime.timeouts,
            "output_limit_tokens": spec.runtime.output_limit_tokens,
            "doctor_checks": spec.runtime.doctor_checks,
            "docs_url": spec.runtime.docs_url,
            "adapter": spec.runtime.adapter,
        }
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
        "runtime": runtime,
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


def _write_generated_connectors(
    workspace: Path,
    state: CatalogState,
    connector_specs: dict[str, ConnectorSpec],
    missing_prereq: dict[str, list[str]],
    missing_env: dict[str, list[str]],
) -> Path | None:
    connectors: list[dict[str, Any]] = []
    for connector_id in state.connectors:
        spec = connector_specs[connector_id]
        if spec.runtime is None or connector_id in missing_prereq:
            continue
        connectors.append(
            {
                "id": spec.id,
                "display_name": spec.display_name,
                "description": spec.description,
                "stability": spec.stability,
                "tags": spec.tags,
                "required_env": sorted(spec.required_env),
                "missing_env": missing_env.get(connector_id, []),
                "prerequisites": sorted(spec.prerequisites),
                "missing_prerequisites": missing_prereq.get(connector_id, []),
                "tools_exposed": spec.tools_exposed,
                "runtime": {
                    "driver": spec.runtime.driver,
                    "transport": spec.runtime.transport,
                    "name": spec.runtime.name,
                    "command": spec.runtime.command,
                    "args": spec.runtime.args,
                    "url": spec.runtime.url,
                    "cwd": spec.runtime.cwd,
                    "env": spec.runtime.env,
                    "headers_env": spec.runtime.headers_env,
                    "auth": spec.runtime.auth,
                    "capabilities": spec.runtime.capabilities,
                    "tool_allowlist": spec.runtime.tool_allowlist,
                    "tool_prefix": spec.runtime.tool_prefix,
                    "default_policy": spec.runtime.default_policy,
                    "timeouts": spec.runtime.timeouts,
                    "output_limit_tokens": spec.runtime.output_limit_tokens,
                    "doctor_checks": spec.runtime.doctor_checks,
                    "docs_url": spec.runtime.docs_url,
                    "adapter": spec.runtime.adapter,
                },
            }
        )

    path = generated_connectors_path(workspace)
    if not connectors:
        if path.exists():
            path.unlink()
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": CATALOG_VERSION, "connectors": connectors}, indent=2)
        + "\n",
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
    generated_connectors = _write_generated_connectors(
        workspace,
        state,
        connector_specs,
        missing_prereq,
        missing_env,
    )
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
            "connectors_path": str(generated_connectors)
            if generated_connectors
            else "",
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
