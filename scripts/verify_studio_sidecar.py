from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TOKEN_HEADER = "X-ContextClaw-Token"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a built Studio sidecar")
    parser.add_argument(
        "--sidecar", required=True, help="Path to the built sidecar binary"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout", type=float, default=12.0)
    return parser.parse_args()


def _reserve_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _request_json(method: str, url: str, token: str) -> dict[str, object]:
    request = Request(url, method=method)
    request.add_header(TOKEN_HEADER, token)
    with urlopen(request, timeout=1.0) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _wait_for_status(
    host: str, port: int, token: str, timeout: float
) -> dict[str, object]:
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/status"
    while time.time() < deadline:
        try:
            return _request_json("GET", url, token)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _shutdown(host: str, port: int, token: str) -> None:
    url = f"http://{host}:{port}/shutdown"
    try:
        _request_json("POST", url, token)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return


def main() -> None:
    args = _parse_args()
    sidecar_path = Path(args.sidecar).resolve()
    if not sidecar_path.is_file():
        raise SystemExit(f"Sidecar binary not found: {sidecar_path}")

    token = f"verify-{int(time.time() * 1000)}"
    port = _reserve_port(args.host)
    process = subprocess.Popen(
        [
            str(sidecar_path),
        ],
        env={
            **dict(os.environ),
            "CONTEXTCLAW_STUDIO_HOST": args.host,
            "CONTEXTCLAW_STUDIO_PORT": str(port),
            "CONTEXTCLAW_STUDIO_TOKEN": token,
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        payload = _wait_for_status(args.host, port, token, args.timeout)
        if payload.get("status") != "ok":
            raise SystemExit(f"Unexpected sidecar status payload: {payload}")
        _shutdown(args.host, port, token)
        deadline = time.time() + args.timeout
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.2)
        process.kill()
        process.wait(timeout=5)
        raise SystemExit("Sidecar did not stop after graceful shutdown request")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
