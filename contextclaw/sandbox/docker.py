from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .protocol import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    cpu: float = 1.0  # CPU core count (fractional allowed)
    memory_mb: int = 512  # Memory limit in MiB
    timeout_seconds: int = 300


@dataclass
class DockerSandbox:
    """Container-per-agent sandbox using the Docker Python SDK.

    Mounts *workspace* at /workspace inside the container and runs all
    commands as a non-root user (uid 1000).  The container is created fresh
    on start() and removed on stop().
    """

    workspace: Path
    image: str = "python:3.12-slim"
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    _container: Any = field(default=None, init=False, repr=False)
    _docker: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Pull image if needed, create and start the container."""
        try:
            import docker  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("docker SDK is required for DockerSandbox: pip install docker") from exc

        client = await asyncio.to_thread(docker.from_env)
        self._docker = client

        container_name = f"contextclaw-{uuid.uuid4().hex[:12]}"
        nano_cpus = int(self.limits.cpu * 1_000_000_000)
        mem_limit = f"{self.limits.memory_mb}m"

        logger.info("Starting Docker sandbox: %s (image=%s)", container_name, self.image)

        self._container = await asyncio.to_thread(
            client.containers.run,
            self.image,
            name=container_name,
            command="sleep infinity",
            detach=True,
            remove=False,
            volumes={
                str(self.workspace.resolve()): {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            },
            working_dir="/workspace",
            user="1000:1000",
            nano_cpus=nano_cpus,
            mem_limit=mem_limit,
            network_mode="bridge",
            security_opt=["no-new-privileges:true"],
        )

    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Run *command* inside the container via exec_run with timeout."""
        if self._container is None:
            raise RuntimeError("DockerSandbox.start() must be called before execute()")

        if timeout <= 0:
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr="Invalid timeout: must be positive",
            )

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._container.exec_run,
                    cmd=["sh", "-c", command],
                    workdir="/workspace",
                    user="1000:1000",
                    demux=True,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("Docker command timed out after %ds: %s", timeout, command[:200])
            return ExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                timed_out=True,
            )
        except OSError as exc:
            logger.error("Docker OS error executing command: %s", exc)
            return ExecutionResult(exit_code=1, stdout="", stderr=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("Docker unexpected error executing command: %s", exc)
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
            )

        exit_code: int = result.exit_code or 0
        output = result.output or (b"", b"")
        try:
            stdout_bytes, stderr_bytes = output
        except (TypeError, ValueError):
            stdout_bytes, stderr_bytes = b"", b""
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")

        return ExecutionResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def stop(self) -> None:
        """Stop and remove the container."""
        if self._container is None:
            return
        container_name = getattr(self._container, "name", "unknown")
        try:
            await asyncio.to_thread(self._container.stop, timeout=5)
            await asyncio.to_thread(self._container.remove, force=True)
            logger.info("Docker sandbox stopped: %s", container_name)
        except OSError as exc:
            logger.warning("Error stopping Docker sandbox %s: %s", container_name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error stopping Docker sandbox %s: %s", container_name, exc)
        finally:
            self._container = None
            self._docker = None
