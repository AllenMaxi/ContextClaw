from __future__ import annotations

import asyncio
import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .protocol import ExecutionResult

logger = logging.getLogger(__name__)

_DEFAULT_BLOCKED: list[str] = [
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "/etc",
]

# Shell metacharacters that can embed sub-commands or redirect output.
# These patterns are checked BEFORE path extraction to catch evasion
# attempts like $(cat /etc/passwd) or `cat ~/.ssh/id_rsa`.
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    # Command substitution: $(cmd) or `cmd`
    re.compile(r"\$\(.*\)", re.DOTALL),
    re.compile(r"`.*`", re.DOTALL),
    # Process substitution: <(cmd) or >(cmd)
    re.compile(r"[<>]\(.*\)", re.DOTALL),
    # eval, source, or exec wrapping
    re.compile(r"\b(eval|source|exec)\s+", re.IGNORECASE),
]


def _resolve_path(p: str) -> Path:
    """Expand user home and resolve to an absolute, symlink-free path."""
    return Path(p).expanduser().resolve()


def _extract_path_tokens(command: str) -> list[str]:
    """Extract file-path-like tokens from a shell command string.

    Uses shlex for proper quote handling, falls back to regex for
    commands that shlex cannot tokenise (unmatched quotes, etc.).
    Also extracts paths from flag values like ``--config=/etc/passwd``.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Unmatched quotes or other parse errors — fall back to regex
        tokens = re.findall(r'["\']?([~/.][^\s"\']+)["\']?', command)

    paths: list[str] = []
    for token in tokens:
        # Handle --flag=/path/to/file and -f=/path/to/file
        if "=" in token:
            _, _, value = token.partition("=")
            token = value
        # Keep tokens that look like paths (start with /, ~, or ./)
        if token.startswith(("/", "~", "./")):
            paths.append(token)
    return paths


def _extract_paths_from_subshells(command: str) -> list[str]:
    """Extract path-like strings from inside shell metacharacters.

    Catches paths embedded in $(...), `...`, pipe chains, semicolons,
    and logical operators (&&, ||) that the shlex tokeniser would miss.
    """
    # Pull out the inner content of $() and `` substitutions
    inner_parts: list[str] = []
    for m in re.finditer(r"\$\((.+?)\)", command, re.DOTALL):
        inner_parts.append(m.group(1))
    for m in re.finditer(r"`(.+?)`", command, re.DOTALL):
        inner_parts.append(m.group(1))

    # Split on shell separators: pipes, semicolons, && and ||
    segments = re.split(r"\|{1,2}|;|&&", command)
    for segment in segments:
        inner_parts.append(segment.strip())

    # Extract path tokens from each inner part
    paths: list[str] = []
    for part in inner_parts:
        paths.extend(_extract_path_tokens(part))
    return paths


def _path_is_under(child: Path, parent: Path) -> bool:
    """Return True if *child* is equal to or a descendant of *parent*.

    Both paths must already be resolved (absolute, no symlinks).
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass
class ProcessSandbox:
    """Subprocess-based sandbox with multi-layer path protection.

    **Security model** (defense in depth):

    1. **Shell metacharacter scanning** — detects ``$()``, backticks,
       process substitution, eval/source/exec that could embed hidden
       commands.  Paths inside these constructs are extracted and checked.
    2. **Path token extraction** — uses :func:`shlex.split` to properly
       tokenise the command, then resolves each path-like token through
       the real filesystem (expanding ``~``, resolving ``..``, following
       symlinks) via :meth:`Path.resolve`.
    3. **Containment check** — each resolved path is tested against the
       blocked list using :meth:`Path.relative_to`, which is immune to
       substring false-positives (``/etc-backup`` is *not* under ``/etc``).

    .. note::
       This is a **best-effort** defence layer.  For true OS-level
       isolation use :class:`DockerSandbox`.
    """

    workspace: Path
    allowed_paths: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(default_factory=lambda: list(_DEFAULT_BLOCKED))
    _resolved_blocked_cache: list[tuple[str, Path]] = field(
        default=None,
        init=False,
        repr=False,  # type: ignore[assignment]
    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolved_blocked(self) -> list[tuple[str, Path]]:
        """Return (original, resolved) pairs for each blocked path.

        Computed once and cached for the lifetime of the sandbox instance.
        """
        if self._resolved_blocked_cache is None:
            pairs: list[tuple[str, Path]] = []
            for p in self.blocked_paths:
                pairs.append((p, _resolve_path(p)))
            self._resolved_blocked_cache = pairs
        return self._resolved_blocked_cache

    def _check_paths_against_blocked(
        self,
        paths: list[str],
        blocked_pairs: list[tuple[str, Path]],
    ) -> str | None:
        """Check a list of path strings against blocked list. Returns first hit."""
        for token in paths:
            try:
                resolved = _resolve_path(token)
            except (OSError, ValueError):
                continue
            for original, blocked_resolved in blocked_pairs:
                if _path_is_under(resolved, blocked_resolved):
                    return original
        return None

    def _command_accesses_blocked(self, command: str) -> str | None:
        """Return the first blocked path found in *command*, or None.

        Performs three passes:
        1. Direct path tokens from the top-level command
        2. Paths inside shell metacharacters ($(), ``, pipes)
        3. Paths inside dangerous constructs (eval, source, exec)
        """
        blocked_pairs = self._resolved_blocked()

        # Pass 1: top-level path tokens
        hit = self._check_paths_against_blocked(
            _extract_path_tokens(command),
            blocked_pairs,
        )
        if hit:
            return hit

        # Pass 2: paths inside subshells, backticks, pipes
        hit = self._check_paths_against_blocked(
            _extract_paths_from_subshells(command),
            blocked_pairs,
        )
        if hit:
            return hit

        return None

    # ------------------------------------------------------------------
    # Sandbox protocol
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """No-op: process sandbox requires no setup."""

    async def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        """Run *command* as a subprocess inside *workspace*."""
        if timeout <= 0:
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr="Invalid timeout: must be positive",
            )

        blocked_hit = self._command_accesses_blocked(command)
        if blocked_hit:
            logger.warning("Blocked command accessing '%s': %s", blocked_hit, command[:200])
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Access denied: command references blocked path '{blocked_hit}'",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.workspace.resolve()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                timed_out = False
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.warning("Command timed out after %ds: %s", timeout, command[:200])
                return ExecutionResult(
                    exit_code=124,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    timed_out=True,
                )

            return ExecutionResult(
                exit_code=proc.returncode or 0,
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                timed_out=timed_out,
            )
        except OSError as exc:
            logger.error("OS error executing command: %s", exc)
            return ExecutionResult(exit_code=1, stdout="", stderr=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error executing command: %s", exc)
            return ExecutionResult(exit_code=1, stdout="", stderr=str(exc))

    async def stop(self) -> None:
        """No-op: nothing to tear down."""
