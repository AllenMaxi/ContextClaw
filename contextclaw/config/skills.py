from __future__ import annotations

from pathlib import Path


def _skill_label(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def load_skills(path: Path) -> list[tuple[str, str]]:
    """Load markdown skills from a file or directory.

    Files are returned as ``(label, content)`` pairs. Directories are scanned
    recursively for ``*.md`` files in lexical order.
    """
    if not path.exists():
        return []

    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        return [(path.name, text)] if text else []

    root = path.resolve()
    skills: list[tuple[str, str]] = []
    for skill_file in sorted(root.rglob("*.md")):
        text = skill_file.read_text(encoding="utf-8").strip()
        if text:
            skills.append((_skill_label(skill_file, root), text))
    return skills


def render_skills_prompt(path: Path | None) -> str:
    """Render skills into a compact prompt appendix."""
    if path is None:
        return ""

    skills = load_skills(path)
    if not skills:
        return ""

    parts = [
        "Additional skills are available. Use them when a task clearly matches one.",
    ]
    for label, content in skills:
        parts.append(f"[Skill: {label}]")
        parts.append(content)
    return "\n\n".join(parts).strip()
