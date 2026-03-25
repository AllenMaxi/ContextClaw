from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any

from .catalog_engine import load_connector_specs


def _status_payload(connector_id: str) -> dict[str, Any]:
    specs = load_connector_specs()
    if connector_id not in specs:
        raise ValueError(f"Unknown connector '{connector_id}'")
    spec = specs[connector_id]
    missing_env = [
        env_name
        for env_name in spec.required_env
        if not os.environ.get(env_name, "").strip()
    ]
    missing_prereq = [
        command for command in spec.prerequisites if not shutil.which(command)
    ]
    return {
        "connector": spec.id,
        "display_name": spec.display_name,
        "description": spec.description,
        "type": spec.type,
        "tools_exposed": spec.tools_exposed,
        "missing_env": missing_env,
        "missing_prerequisites": missing_prereq,
    }


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if not args:
        raise SystemExit(
            "Usage: python -m contextclaw.catalog_mcp_server <connector-id>"
        )
    connector_id = args[0]
    specs = load_connector_specs()
    if connector_id not in specs:
        raise SystemExit(f"Unknown connector '{connector_id}'")

    spec = specs[connector_id]
    tool_name = "status"

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue

        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            _emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": spec.id, "version": spec.version},
                        "capabilities": {"tools": {}},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": tool_name,
                                "description": f"Report configuration status for the {spec.display_name} connector",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                },
                            }
                        ]
                    },
                }
            )
        elif method == "tools/call":
            text = json.dumps(_status_payload(connector_id), ensure_ascii=True)
            _emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": text}]},
                }
            )
        else:
            _emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported method {method}",
                    },
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
