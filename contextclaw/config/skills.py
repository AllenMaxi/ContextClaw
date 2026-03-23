from __future__ import annotations

from pathlib import Path


def _skill_label(path: Path, root: Path | None = None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def _package_skill_files(root: Path) -> list[tuple[str, str]]:
    skills: list[tuple[str, str]] = []
    for manifest in sorted(root.rglob("skill.yaml")):
        package_dir = manifest.parent.resolve()
        skill_file = package_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        text = skill_file.read_text(encoding="utf-8").strip()
        if text:
            skills.append((_skill_label(skill_file, root), text))
    return skills


def _is_under_package(path: Path, package_dirs: set[Path]) -> bool:
    return any(
        package_dir == path or package_dir in path.parents
        for package_dir in package_dirs
    )


def load_skills(path: Path) -> list[tuple[str, str]]:
    """Load markdown skills from a file or directory.

    Files are returned as ``(label, content)`` pairs. Directories are scanned
    recursively for ``*.md`` files in lexical order.

    Packaged skills are detected by the presence of ``skill.yaml``. For those
    packages, only ``SKILL.md`` is injected into the prompt; markdown under
    ``references`` or ``templates`` stays available on disk without being
    auto-loaded.
    """
    if not path.exists():
        return []

    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        return [(path.name, text)] if text else []

    root = path.resolve()
    skills: list[tuple[str, str]] = []
    packaged_files = _package_skill_files(root)
    skills.extend(packaged_files)
    package_dirs = {
        manifest.parent.resolve() for manifest in sorted(root.rglob("skill.yaml"))
    }
    for skill_file in sorted(root.rglob("*.md")):
        resolved = skill_file.resolve()
        if _is_under_package(resolved, package_dirs):
            continue
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
