from __future__ import annotations

from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Minimal YAML parser
# ---------------------------------------------------------------------------
# Supports the policy schema only: nested keys via indentation, list items
# as "- value" lines, and scalar key: value pairs.  No PyYAML dependency.

def _parse_inline_list(value: str) -> list[str] | None:
    """Parse a ``[item1, item2]`` inline YAML list. Returns None if not a list."""
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return None
    inner = value[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    for item in inner.split(","):
        item = item.strip().strip("'\"")
        if item:
            items.append(item)
    return items


def _parse_policy_yaml(text: str) -> dict:
    """Parse a policy YAML file into a nested Python dict/list structure.

    Handles:
    - Scalar key: value pairs
    - Inline list syntax: ``key: [a, b, c]``
    - Nested sections via 2-space indentation
    - List items prefixed with "- "
    - Inline comments (#)
    - Blank lines
    """
    lines: list[tuple[int, str]] = []  # (indent_level, raw_content)
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(stripped) - len(stripped.lstrip())
        content = stripped.lstrip()
        # Strip inline comments (safe for our simple schema)
        if " #" in content:
            content = content[: content.index(" #")].rstrip()
        lines.append((indent, content))

    def parse_block(start: int, base_indent: int) -> tuple[dict | list, int]:
        """Recursively parse lines[start:] at *base_indent* into a dict or list."""
        # Peek ahead to decide: list block or dict block
        if start < len(lines) and lines[start][1].startswith("- "):
            return parse_list(start, base_indent)
        return parse_dict(start, base_indent)

    def parse_list(start: int, base_indent: int) -> tuple[list, int]:
        result: list = []
        i = start
        while i < len(lines):
            indent, content = lines[i]
            if indent < base_indent:
                break
            if indent == base_indent and content.startswith("- "):
                result.append(content[2:].strip())
                i += 1
            else:
                break
        return result, i

    def parse_dict(start: int, base_indent: int) -> tuple[dict, int]:
        result: dict = {}
        i = start
        while i < len(lines):
            indent, content = lines[i]
            if indent < base_indent:
                break
            if indent != base_indent:
                # Unexpected deeper indent — skip (robustness)
                i += 1
                continue
            if ":" in content:
                key, _, value = content.partition(":")
                key = key.strip()
                value = value.strip()
                if value:
                    # Check for inline list syntax: key: [a, b, c]
                    inline = _parse_inline_list(value)
                    result[key] = inline if inline is not None else value
                    i += 1
                else:
                    # Nested block starts on next line(s)
                    i += 1
                    if i < len(lines) and lines[i][0] > base_indent:
                        child_indent = lines[i][0]
                        child, i = parse_block(i, child_indent)
                        result[key] = child
                    else:
                        result[key] = {}
            else:
                i += 1
        return result, i

    if not lines:
        return {}

    root_indent = lines[0][0]
    result, _ = parse_dict(0, root_indent)
    return result


def _as_list(value: object) -> list[str]:
    """Coerce a parsed YAML value to a list of strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value else []
    return []


def _as_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return default


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Evaluate agent actions against a YAML policy file.

    Example usage::

        engine = PolicyEngine.from_file(Path("policy.yaml"))
        verdict = engine.check_tool("filesystem_write")   # "allow"|"confirm"|"block"
        allowed = engine.check_path("/workspace/data.txt")
        cfg     = engine.get_sandbox_config()
    """

    def __init__(self, policy: dict) -> None:
        self._policy = policy
        self._permissions: dict = policy.get("permissions", {})
        self._sandbox_cfg: dict = policy.get("sandbox", {})
        self._tools_cfg: dict = self._permissions.get("tools", {})
        self._fs_cfg: dict = self._permissions.get("filesystem", {})

        # Tool lists
        self._auto_approve: set[str] = set(_as_list(self._tools_cfg.get("auto_approve", [])))
        self._require_confirm: set[str] = set(_as_list(self._tools_cfg.get("require_confirm", [])))
        self._blocked_tools: set[str] = set(_as_list(self._tools_cfg.get("blocked", [])))

        # Filesystem lists
        self._allowed_paths: list[str] = _as_list(self._fs_cfg.get("allowed", []))
        self._blocked_paths: list[str] = [
            str(Path(p).expanduser())
            for p in _as_list(self._fs_cfg.get("blocked", []))
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: Path) -> PolicyEngine:
        """Load and parse a policy YAML file."""
        raw = _parse_policy_yaml(path.read_text(encoding="utf-8"))
        return cls(raw)

    @classmethod
    def from_text(cls, text: str) -> PolicyEngine:
        """Parse a policy from a YAML string (useful for testing)."""
        return cls(_parse_policy_yaml(text))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_tool(self, name: str) -> Literal["allow", "confirm", "block"]:
        """Return the policy verdict for tool *name*.

        Precedence: blocked > auto_approve > require_confirm > confirm (default).
        """
        if name in self._blocked_tools:
            return "block"
        if name in self._auto_approve:
            return "allow"
        if name in self._require_confirm:
            return "confirm"
        # Unknown tools require confirmation by default (safe default)
        return "confirm"

    def check_path(self, path: str | Path) -> bool:
        """Return True if *path* is accessible under the current policy.

        A path is allowed when it starts with an entry in *allowed* AND does
        not start with any entry in *blocked*.  If no allowed paths are
        configured every path passes the allow check (permissive by default).
        """
        resolved = str(Path(path).expanduser().resolve())

        # Check blocked first
        for blocked in self._blocked_paths:
            if resolved.startswith(blocked):
                return False

        # Check allowed (if configured)
        if self._allowed_paths:
            for allowed in self._allowed_paths:
                allowed_resolved = str(Path(allowed).expanduser().resolve())
                if resolved.startswith(allowed_resolved):
                    return True
            return False  # Not in any allowed path

        return True  # No allow-list configured → permissive

    def get_sandbox_config(self) -> dict:
        """Return the sandbox configuration block from the policy."""
        cfg = dict(self._sandbox_cfg)

        # Normalise resource_limits sub-dict if present
        limits_raw = cfg.get("resource_limits", {})
        if isinstance(limits_raw, dict):
            limits: dict = {}
            for k, v in limits_raw.items():
                try:
                    limits[k] = int(v) if isinstance(v, str) else v
                except (ValueError, TypeError):
                    limits[k] = v
            cfg["resource_limits"] = limits

        return cfg
