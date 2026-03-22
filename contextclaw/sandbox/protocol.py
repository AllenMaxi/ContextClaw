from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Sandbox(Protocol):
    async def start(self) -> None: ...
    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult: ...
    async def stop(self) -> None: ...
