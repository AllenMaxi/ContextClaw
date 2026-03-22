from __future__ import annotations

import json
from pathlib import Path

from .manager import ToolDefinition


def load_bundle(name: str, bundles_path: Path | None = None) -> list[ToolDefinition]:
    """Load tool definitions from a named bundle in the bundle JSON file.

    Args:
        name: The bundle name (e.g. "filesystem", "web", "shell").
        bundles_path: Path to the JSON file containing bundle definitions.
                      Defaults to bundles.json next to this module.

    Returns:
        List of ToolDefinition instances for the requested bundle.

    Raises:
        FileNotFoundError: If the bundles file does not exist.
        KeyError: If the named bundle is not found in the file.
    """
    if bundles_path is None:
        bundles_path = Path(__file__).parent / "bundles.json"

    if not bundles_path.exists():
        raise FileNotFoundError(f"Bundles file not found: {bundles_path}")

    with bundles_path.open("r", encoding="utf-8") as fh:
        data: dict[str, list[dict]] = json.load(fh)

    if name not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Bundle '{name}' not found. Available bundles: {available}")

    return [
        ToolDefinition(
            name=entry["name"],
            description=entry.get("description", ""),
            parameters=entry.get("parameters", {}),
        )
        for entry in data[name]
    ]
