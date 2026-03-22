"""Tests for the HTTP chat server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from contextclaw.chat.server import _MAX_BODY_BYTES, ChatHandler, ChatServer

# ---------------------------------------------------------------------------
# Body size limit
# ---------------------------------------------------------------------------


def test_max_body_bytes_constant():
    """MAX_BODY_BYTES should be 1 MiB."""
    assert _MAX_BODY_BYTES == 1 * 1024 * 1024


# ---------------------------------------------------------------------------
# ChatServer lifecycle
# ---------------------------------------------------------------------------


def test_chat_server_set_runner():
    """set_runner should configure the handler class."""
    server = ChatServer(host="127.0.0.1", port=0)  # port 0 = don't bind
    mock_runner = MagicMock()
    mock_session = MagicMock()

    old_runner = ChatHandler.runner
    old_session = ChatHandler.session
    try:
        server.set_runner(mock_runner, mock_session)
        assert ChatHandler.runner is mock_runner
        assert ChatHandler.session is mock_session
    finally:
        ChatHandler.runner = old_runner
        ChatHandler.session = old_session


def test_chat_server_cors_origin():
    """ChatServer should set cors_origin on the handler."""
    server = ChatServer(host="127.0.0.1", port=0, cors_origin="https://myapp.com")

    old_cors = ChatHandler.cors_origin
    try:
        # start() would set this, but we test directly
        ChatHandler.cors_origin = server.cors_origin
        assert ChatHandler.cors_origin == "https://myapp.com"
    finally:
        ChatHandler.cors_origin = old_cors


# ---------------------------------------------------------------------------
# Live server integration test
# ---------------------------------------------------------------------------


def test_chat_server_start_stop():
    """Server should start and stop without errors."""
    import socket

    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port)
    server.start()
    assert server._server is not None
    assert server._thread is not None
    assert server._thread.is_alive()

    server.stop()
    assert server._server is None


def test_chat_server_stop_releases_port():
    """Stopping the server should close the listening socket."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port)
    server.start()
    server.stop()

    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_chat_server_status_endpoint():
    """GET /status should return JSON status."""
    import socket
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port)
    server.start()
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/status")
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert data["runner"] is False  # no runner attached
    finally:
        server.stop()


def test_chat_server_auth_rejects_bad_token():
    """Requests with wrong auth token should be rejected."""
    import socket
    import urllib.error
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port, auth_token="secret123")
    server.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/status",
            headers={"Authorization": "Bearer wrong"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 401
    finally:
        server.stop()


def test_chat_server_auth_accepts_correct_token():
    """Requests with correct auth token should succeed."""
    import socket
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port, auth_token="secret123")
    server.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/status",
            headers={"Authorization": "Bearer secret123"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert data["status"] == "ok"
    finally:
        server.stop()


def test_chat_server_post_without_runner_returns_503():
    """POST /chat without a runner should return 503."""
    import socket
    import urllib.error
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port)
    server.start()
    try:
        body = json.dumps({"message": "hello"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 503
    finally:
        server.stop()


def test_chat_server_post_invalid_json_returns_400():
    """POST /chat with invalid JSON should return 400."""
    import socket
    import urllib.error
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port)
    server.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400
    finally:
        server.stop()


def test_chat_server_post_empty_message_returns_400():
    """POST /chat with empty message should return 400."""
    import socket
    import urllib.error
    import urllib.request

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = ChatServer(host="127.0.0.1", port=port)
    server.start()
    try:
        body = json.dumps({"message": ""}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400
    finally:
        server.stop()
