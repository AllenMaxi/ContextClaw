from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

logger = logging.getLogger(__name__)

# Maximum request body size: 1 MiB (prevents OOM from malicious Content-Length)
_MAX_BODY_BYTES = 1 * 1024 * 1024


class ChatHandler(BaseHTTPRequestHandler):
    # Instance-level attributes set by ChatServer before serving.
    # Each ChatServer instance sets these on the *class* it passes to
    # HTTPServer, but since we only support one runner per server this
    # is acceptable.  A threading.Lock serialises runner access so two
    # concurrent requests cannot interleave conversation state.
    runner: Any = None
    session: Any = None
    auth_token: str = ""
    cors_origin: str = ""
    _runner_lock: threading.Lock = threading.Lock()
    _loop: asyncio.AbstractEventLoop | None = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug(format, *args)

    # ------------------------------------------------------------------
    # Auth + CORS
    # ------------------------------------------------------------------

    def _check_auth(self) -> bool:
        if not self.auth_token:
            return True
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.auth_token}"
        return hmac.compare_digest(header.encode("utf-8"), expected.encode("utf-8"))

    def _add_cors_headers(self) -> None:
        origin = self.cors_origin or "*"
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization, Accept"
        )

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._add_cors_headers()
        self.end_headers()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        if not self._check_auth():
            self._send_json({"error": "unauthorized"}, status=401)
            return
        if self.path != "/chat":
            self._send_json({"error": "not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            self._send_json({"error": "invalid Content-Length"}, status=400)
            return

        if length > _MAX_BODY_BYTES:
            self._send_json(
                {"error": f"request body too large (max {_MAX_BODY_BYTES} bytes)"},
                status=413,
            )
            return

        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        if not isinstance(data, dict):
            self._send_json({"error": "request body must be a JSON object"}, status=400)
            return

        message = data.get("message", "").strip()
        if not message:
            self._send_json({"error": "message required"}, status=400)
            return

        wants_sse = "text/event-stream" in self.headers.get("Accept", "")

        if wants_sse:
            self._handle_sse(message)
        else:
            self._handle_json_chat(message)

    def do_GET(self) -> None:
        if not self._check_auth():
            self._send_json({"error": "unauthorized"}, status=401)
            return
        if self.path == "/status":
            self._send_json({"status": "ok", "runner": self.runner is not None})
        elif self.path == "/history":
            history = self.session.get_messages() if self.session is not None else []
            self._send_json({"history": history})
        else:
            self._send_json({"error": "not found"}, status=404)

    # ------------------------------------------------------------------
    # Chat handlers
    # ------------------------------------------------------------------

    def _handle_json_chat(self, message: str) -> None:
        """Run the async runner and return a plain JSON response."""
        if self.runner is None:
            self._send_json({"error": "no runner attached"}, status=503)
            return

        try:
            events = self._run_async(message)
        except ConnectionError as exc:
            logger.warning("Provider connection error during chat: %s", exc)
            self._send_json({"error": "LLM provider unavailable"}, status=502)
            return
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during chat: %s", exc)
            self._send_json({"error": "internal server error"}, status=500)
            return

        reply_parts: list[str] = []
        for event in events:
            if event.type == "text":
                reply_parts.append(event.data.get("content", ""))

        self._send_json({"reply": "".join(reply_parts)})

    def _handle_sse(self, message: str) -> None:
        """Stream events via Server-Sent Events."""
        if self.runner is None:
            self._send_json({"error": "no runner attached"}, status=503)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._add_cors_headers()
        self.end_headers()

        try:
            events = self._run_async(message)
            for event in events:
                self._sse_send({"type": event.type, **event.data})
        except ConnectionError as exc:
            logger.warning("Provider connection error during SSE: %s", exc)
            self._sse_send({"error": "LLM provider unavailable"})
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during SSE: %s", exc)
            self._sse_send({"error": "internal server error"})

        self._sse_done()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_async(self, message: str) -> list:
        """Collect all events from the async runner.

        Uses a lock to serialise concurrent requests — only one conversation
        turn runs at a time, preventing interleaved session state.
        Uses a shared event loop to avoid asyncio.run() conflicts in threads.
        """

        async def _collect() -> list:
            events = []
            async for event in self.runner.run(message):
                events.append(event)
            return events

        with self._runner_lock:
            loop = ChatHandler._loop
            if loop is None or loop.is_closed():
                loop = asyncio.new_event_loop()
                ChatHandler._loop = loop
            return loop.run_until_complete(_collect())

    def _sse_send(self, payload: dict) -> None:
        data = json.dumps(payload)
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def _sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._add_cors_headers()
        self.end_headers()
        self.wfile.write(body)


class ThreadingChatServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a separate thread."""

    daemon_threads = True


class ChatServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        auth_token: str = "",
        cors_origin: str = "",
    ) -> None:
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.cors_origin = cors_origin
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def set_runner(self, runner: Any, session: Any) -> None:
        ChatHandler.runner = runner
        ChatHandler.session = session

    def start(self) -> None:
        ChatHandler.auth_token = self.auth_token
        ChatHandler.cors_origin = self.cors_origin
        self._server = ThreadingChatServer((self.host, self.port), ChatHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("ChatServer started on %s:%d", self.host, self.port)

    def stop(self) -> None:
        # Stop accepting new requests first
        if self._server is not None:
            server = self._server
            server.shutdown()
            server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

        # Acquire the runner lock to ensure no in-flight request is running,
        # then summarise the session safely.
        with ChatHandler._runner_lock:
            if ChatHandler.runner is not None:
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(ChatHandler.runner.close_session())
                    finally:
                        loop.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Session summarization on shutdown failed: %s", exc)

            # Close the shared event loop
            if ChatHandler._loop is not None and not ChatHandler._loop.is_closed():
                ChatHandler._loop.close()
                ChatHandler._loop = None

        logger.info("ChatServer stopped")
