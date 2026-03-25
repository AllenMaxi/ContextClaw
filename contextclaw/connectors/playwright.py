from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PlaywrightBackendError(RuntimeError):
    pass


class DefaultPlaywrightBackend:
    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._page = None
        self.console_messages: list[str] = []
        self.network_events: list[str] = []
        self.current_url = ""

    async def launch(self, *, headless: bool = True) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise PlaywrightBackendError(
                "Playwright is not installed. Run `pip install playwright` and `playwright install chromium`."
            ) from exc

        if self._page is not None:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        page = await self._browser.new_page()
        page.on("console", lambda msg: self.console_messages.append(msg.text))
        page.on(
            "response",
            lambda response: self.network_events.append(
                f"{response.status} {response.url}"
            ),
        )
        self._page = page

    async def navigate(self, url: str) -> dict[str, Any]:
        page = self._require_page()
        response = await page.goto(url, wait_until="networkidle")
        self.current_url = page.url
        title = await page.title()
        return {
            "url": page.url,
            "status": response.status if response else None,
            "title": title,
        }

    async def snapshot_text(self, selector: str) -> str:
        page = self._require_page()
        if selector == "body":
            return await page.locator("body").inner_text()
        return await page.locator(selector).inner_text()

    async def click(self, selector: str) -> None:
        page = self._require_page()
        await page.locator(selector).click()

    async def type(self, selector: str, text: str, *, clear: bool = True) -> None:
        page = self._require_page()
        locator = page.locator(selector)
        if clear:
            await locator.fill("")
        await locator.fill(text)

    async def screenshot(self, path: Path, *, full_page: bool = True) -> str:
        page = self._require_page()
        path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(path), full_page=full_page)
        return str(path)

    async def close(self) -> None:
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    def _require_page(self):
        if self._page is None:
            raise PlaywrightBackendError(
                "Browser is not launched. Run browser_launch first."
            )
        return self._page


class PlaywrightConnectorClient:
    backend_factory = DefaultPlaywrightBackend

    def __init__(
        self,
        *,
        workspace: Path,
        output_limit_tokens: int = 0,
    ) -> None:
        self.workspace = workspace
        self.output_limit_tokens = output_limit_tokens
        self.backend = self.backend_factory()

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "browser_launch",
                "description": "Launch a Playwright browser session.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"headless": {"type": "boolean"}},
                },
            },
            {
                "name": "navigate",
                "description": "Navigate the active browser page to a URL.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            {
                "name": "snapshot_text",
                "description": "Capture text content from the page or a selector.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"selector": {"type": "string"}},
                },
            },
            {
                "name": "click",
                "description": "Click a selector on the active page.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"selector": {"type": "string"}},
                    "required": ["selector"],
                },
            },
            {
                "name": "type",
                "description": "Type text into a selector on the active page.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string"},
                        "text": {"type": "string"},
                        "clear": {"type": "boolean"},
                    },
                    "required": ["selector", "text"],
                },
            },
            {
                "name": "screenshot",
                "description": "Save a screenshot under the agent workspace.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "full_page": {"type": "boolean"},
                    },
                },
            },
            {
                "name": "console_logs",
                "description": "Return collected console messages from the active page.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "network_log",
                "description": "Return collected network responses from the active page.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "close",
                "description": "Close the active Playwright browser session.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "status",
                "description": "Return Playwright connector readiness and current browser URL.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    async def health(self) -> dict[str, Any]:
        try:
            await self.backend.launch(headless=True)
        except Exception as exc:  # noqa: BLE001
            return {"authenticated": True, "healthy": False, "message": str(exc)}
        return {
            "authenticated": True,
            "healthy": True,
            "url": getattr(self.backend, "current_url", ""),
        }

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "browser_launch":
            await self.backend.launch(headless=bool(arguments.get("headless", True)))
            return json.dumps({"ok": True}, ensure_ascii=True)
        if name == "navigate":
            result = await self.backend.navigate(str(arguments.get("url", "")))
            return json.dumps(result, ensure_ascii=True)
        if name == "snapshot_text":
            text = await self.backend.snapshot_text(
                str(arguments.get("selector", "body") or "body")
            )
            return json.dumps({"text": self._truncate(text)}, ensure_ascii=True)
        if name == "click":
            await self.backend.click(str(arguments.get("selector", "")))
            return json.dumps({"ok": True}, ensure_ascii=True)
        if name == "type":
            await self.backend.type(
                str(arguments.get("selector", "")),
                str(arguments.get("text", "")),
                clear=bool(arguments.get("clear", True)),
            )
            return json.dumps({"ok": True}, ensure_ascii=True)
        if name == "screenshot":
            rel_path = str(arguments.get("path", ".contextclaw/playwright.png"))
            path = (self.workspace / rel_path).resolve()
            try:
                path.relative_to(self.workspace.resolve())
            except ValueError as exc:
                raise PlaywrightBackendError(
                    "Screenshots must stay inside the agent workspace."
                ) from exc
            saved = await self.backend.screenshot(
                path,
                full_page=bool(arguments.get("full_page", True)),
            )
            return json.dumps({"path": saved}, ensure_ascii=True)
        if name == "console_logs":
            return json.dumps(
                {"messages": list(getattr(self.backend, "console_messages", []))},
                ensure_ascii=True,
            )
        if name == "network_log":
            return json.dumps(
                {"events": list(getattr(self.backend, "network_events", []))},
                ensure_ascii=True,
            )
        if name == "close":
            await self.backend.close()
            return json.dumps({"ok": True}, ensure_ascii=True)
        if name == "status":
            return json.dumps(await self.health(), ensure_ascii=True)
        raise KeyError(f"Unknown Playwright tool '{name}'")

    def _truncate(self, text: str) -> str:
        if self.output_limit_tokens <= 0:
            return text
        limit = max(self.output_limit_tokens * 4, 512)
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"
