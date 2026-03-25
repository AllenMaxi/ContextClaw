from __future__ import annotations

import asyncio
import base64
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _truncate_output(text: str, output_limit_tokens: int) -> str:
    if output_limit_tokens <= 0:
        return text
    limit = max(output_limit_tokens * 4, 512)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


@dataclass
class GitHubAPIConfig:
    token: str
    base_url: str = "https://api.github.com"
    request_timeout: float = 20.0
    output_limit_tokens: int = 0


class GitHubConnectorBackend:
    def __init__(self, config: GitHubAPIConfig) -> None:
        self.config = config

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        if not self.config.token:
            raise RuntimeError("Missing GITHUB_TOKEN")

        url = self.config.base_url.rstrip("/") + path
        if query:
            encoded = urllib.parse.urlencode(
                {
                    key: value
                    for key, value in query.items()
                    if value not in ("", None, [])
                },
                doseq=True,
            )
            if encoded:
                url = f"{url}?{encoded}"

        payload: bytes | None = None
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ContextClaw/0.1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if body is not None:
            payload = json.dumps(body, ensure_ascii=True).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout,
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub API {exc.code} {exc.reason}: {_truncate_output(payload, self.config.output_limit_tokens)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GitHub API unreachable: {exc.reason}") from exc

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


class GitHubConnectorClient:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        request_timeout: float = 20.0,
        output_limit_tokens: int = 0,
        backend: GitHubConnectorBackend | None = None,
    ) -> None:
        self.token = token
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.output_limit_tokens = output_limit_tokens
        self.backend = backend or GitHubConnectorBackend(
            GitHubAPIConfig(
                token=token,
                base_url=base_url,
                request_timeout=request_timeout,
                output_limit_tokens=output_limit_tokens,
            )
        )

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "repo_get_file",
                "description": "Read a file from a GitHub repository at an optional ref.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "ref": {"type": "string"},
                    },
                    "required": ["owner", "repo", "path"],
                },
            },
            {
                "name": "repo_put_file",
                "description": "Create or update a file in a GitHub repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "message": {"type": "string"},
                        "branch": {"type": "string"},
                        "sha": {"type": "string"},
                    },
                    "required": ["owner", "repo", "path", "content", "message"],
                },
            },
            {
                "name": "issues_list",
                "description": "List issues for a repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "state": {"type": "string"},
                        "per_page": {"type": "integer"},
                    },
                    "required": ["owner", "repo"],
                },
            },
            {
                "name": "issue_create",
                "description": "Create an issue in a repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["owner", "repo", "title"],
                },
            },
            {
                "name": "pull_requests_list",
                "description": "List pull requests for a repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "state": {"type": "string"},
                        "base": {"type": "string"},
                        "head": {"type": "string"},
                        "per_page": {"type": "integer"},
                    },
                    "required": ["owner", "repo"],
                },
            },
            {
                "name": "pull_request_create",
                "description": "Create a pull request in a repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                        "head": {"type": "string"},
                        "base": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["owner", "repo", "title", "head", "base"],
                },
            },
            {
                "name": "branch_create",
                "description": "Create a branch from a starting SHA or the default branch.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "branch": {"type": "string"},
                        "from_sha": {"type": "string"},
                    },
                    "required": ["owner", "repo", "branch"],
                },
            },
            {
                "name": "workflows_list",
                "description": "List GitHub Actions workflows for a repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                    },
                    "required": ["owner", "repo"],
                },
            },
            {
                "name": "workflow_dispatch",
                "description": "Dispatch a GitHub Actions workflow.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "workflow_id": {"type": "string"},
                        "ref": {"type": "string"},
                        "inputs": {"type": "object"},
                    },
                    "required": ["owner", "repo", "workflow_id", "ref"],
                },
            },
            {
                "name": "status",
                "description": "Return GitHub connector readiness and token status.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    async def health(self) -> dict[str, Any]:
        if not self.token:
            return {
                "authenticated": False,
                "healthy": False,
                "message": "Missing GITHUB_TOKEN",
            }
        try:
            data = await asyncio.to_thread(self.backend.request, "GET", "/rate_limit")
        except Exception as exc:  # noqa: BLE001
            return {
                "authenticated": True,
                "healthy": False,
                "message": str(exc),
            }
        return {
            "authenticated": True,
            "healthy": True,
            "remaining": data.get("rate", {}).get("remaining"),
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "status":
            return json.dumps(await self.health(), ensure_ascii=True)

        if not self.token:
            return "Error: Missing GITHUB_TOKEN"

        if name == "repo_get_file":
            owner = str(arguments.get("owner", "")).strip()
            repo = str(arguments.get("repo", "")).strip()
            path = str(arguments.get("path", "")).strip()
            ref = str(arguments.get("ref", "")).strip()
            data = await asyncio.to_thread(
                self.backend.request,
                "GET",
                f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}",
                query={"ref": ref} if ref else None,
            )
            content = str(data.get("content", ""))
            if data.get("encoding") == "base64":
                decoded = base64.b64decode(content.encode("utf-8")).decode(
                    "utf-8", errors="replace"
                )
            else:
                decoded = content
            return json.dumps(
                {
                    "path": data.get("path", path),
                    "sha": data.get("sha", ""),
                    "size": data.get("size"),
                    "content": _truncate_output(decoded, self.output_limit_tokens),
                },
                ensure_ascii=True,
            )

        if name == "repo_put_file":
            owner = str(arguments.get("owner", "")).strip()
            repo = str(arguments.get("repo", "")).strip()
            path = str(arguments.get("path", "")).strip()
            body = {
                "message": str(arguments.get("message", "")),
                "content": base64.b64encode(
                    str(arguments.get("content", "")).encode("utf-8")
                ).decode("utf-8"),
            }
            branch = str(arguments.get("branch", "")).strip()
            sha = str(arguments.get("sha", "")).strip()
            if branch:
                body["branch"] = branch
            if sha:
                body["sha"] = sha
            data = await asyncio.to_thread(
                self.backend.request,
                "PUT",
                f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}",
                body=body,
            )
            return json.dumps(
                {
                    "content": data.get("content", {}),
                    "commit": data.get("commit", {}),
                },
                ensure_ascii=True,
            )

        if name == "issues_list":
            data = await asyncio.to_thread(
                self.backend.request,
                "GET",
                f"/repos/{arguments['owner']}/{arguments['repo']}/issues",
                query={
                    "state": arguments.get("state", "open"),
                    "per_page": int(arguments.get("per_page", 20) or 20),
                },
            )
            return json.dumps(data, ensure_ascii=True)

        if name == "issue_create":
            data = await asyncio.to_thread(
                self.backend.request,
                "POST",
                f"/repos/{arguments['owner']}/{arguments['repo']}/issues",
                body={
                    "title": str(arguments.get("title", "")),
                    "body": str(arguments.get("body", "")),
                    "labels": arguments.get("labels", []),
                },
            )
            return json.dumps(data, ensure_ascii=True)

        if name == "pull_requests_list":
            data = await asyncio.to_thread(
                self.backend.request,
                "GET",
                f"/repos/{arguments['owner']}/{arguments['repo']}/pulls",
                query={
                    "state": arguments.get("state", "open"),
                    "base": arguments.get("base", ""),
                    "head": arguments.get("head", ""),
                    "per_page": int(arguments.get("per_page", 20) or 20),
                },
            )
            return json.dumps(data, ensure_ascii=True)

        if name == "pull_request_create":
            data = await asyncio.to_thread(
                self.backend.request,
                "POST",
                f"/repos/{arguments['owner']}/{arguments['repo']}/pulls",
                body={
                    "title": str(arguments.get("title", "")),
                    "head": str(arguments.get("head", "")),
                    "base": str(arguments.get("base", "")),
                    "body": str(arguments.get("body", "")),
                },
            )
            return json.dumps(data, ensure_ascii=True)

        if name == "branch_create":
            owner = str(arguments.get("owner", "")).strip()
            repo = str(arguments.get("repo", "")).strip()
            branch = str(arguments.get("branch", "")).strip()
            from_sha = str(arguments.get("from_sha", "")).strip()
            if not from_sha:
                repo_data = await asyncio.to_thread(
                    self.backend.request,
                    "GET",
                    f"/repos/{owner}/{repo}",
                )
                default_branch = str(repo_data.get("default_branch", "")).strip()
                ref_data = await asyncio.to_thread(
                    self.backend.request,
                    "GET",
                    f"/repos/{owner}/{repo}/git/ref/heads/{urllib.parse.quote(default_branch)}",
                )
                from_sha = str(ref_data.get("object", {}).get("sha", "")).strip()
            data = await asyncio.to_thread(
                self.backend.request,
                "POST",
                f"/repos/{owner}/{repo}/git/refs",
                body={"ref": f"refs/heads/{branch}", "sha": from_sha},
            )
            return json.dumps(data, ensure_ascii=True)

        if name == "workflows_list":
            data = await asyncio.to_thread(
                self.backend.request,
                "GET",
                f"/repos/{arguments['owner']}/{arguments['repo']}/actions/workflows",
            )
            return json.dumps(data, ensure_ascii=True)

        if name == "workflow_dispatch":
            await asyncio.to_thread(
                self.backend.request,
                "POST",
                f"/repos/{arguments['owner']}/{arguments['repo']}/actions/workflows/{arguments['workflow_id']}/dispatches",
                body={
                    "ref": str(arguments.get("ref", "")),
                    "inputs": arguments.get("inputs", {}),
                },
            )
            return json.dumps({"ok": True}, ensure_ascii=True)

        raise KeyError(f"Unknown GitHub tool '{name}'")
