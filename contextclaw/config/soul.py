from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SoulConfig:
    name: str = ""
    role: str = ""
    tone: str = ""
    verbosity: str = ""
    extra: dict[str, str] = field(default_factory=dict)  # other frontmatter fields
    body: str = ""                                        # markdown body (system prompt)


_KNOWN_FIELDS = {"name", "role", "tone", "verbosity"}


def load_soul(path: Path) -> SoulConfig:
    """Parse a SOUL.md file with YAML frontmatter.

    Format::

        ---
        name: Research Assistant
        role: researcher
        tone: professional
        verbosity: concise
        ---

        You are a research assistant...

    Lines outside ``---`` delimiters are collected as the markdown body.
    Frontmatter is parsed as simple ``key: value`` pairs (no nesting).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    frontmatter: dict[str, str] = {}
    body_lines: list[str] = []

    if lines and lines[0].strip() == "---":
        # Find closing ---
        close_idx = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                close_idx = i
                break

        if close_idx is not None:
            for line in lines[1:close_idx]:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, value = line.partition(":")
                    frontmatter[key.strip()] = value.strip()
            body_lines = lines[close_idx + 1 :]
        else:
            # No closing --- found; treat entire file as body
            body_lines = lines
    else:
        body_lines = lines

    extra = {k: v for k, v in frontmatter.items() if k not in _KNOWN_FIELDS}

    return SoulConfig(
        name=frontmatter.get("name", ""),
        role=frontmatter.get("role", ""),
        tone=frontmatter.get("tone", ""),
        verbosity=frontmatter.get("verbosity", ""),
        extra=extra,
        body="\n".join(body_lines).strip(),
    )
