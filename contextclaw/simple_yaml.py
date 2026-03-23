from __future__ import annotations

from typing import Any


def _strip_inline_comment(line: str) -> str:
    if " #" not in line:
        return line
    return line[: line.index(" #")].rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        try:
            return int(value)
        except ValueError:
            pass
    return value


def _parse_inline_list(value: str) -> list[Any] | None:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return None
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar(item.strip()) for item in inner.split(",") if item.strip()]


def parse_yaml(text: str) -> dict[str, Any]:
    """Parse a minimal YAML subset into a nested dict/list structure."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(stripped) - len(stripped.lstrip())
        content = _strip_inline_comment(stripped.lstrip())
        if content:
            lines.append((indent, content))

    def parse_block(start: int, base_indent: int) -> tuple[Any, int]:
        if start < len(lines) and lines[start][1].startswith("- "):
            return parse_list(start, base_indent)
        return parse_dict(start, base_indent)

    def parse_list(start: int, base_indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        index = start
        while index < len(lines):
            indent, content = lines[index]
            if indent < base_indent:
                break
            if indent != base_indent or not content.startswith("- "):
                break
            item = content[2:].strip()
            if item:
                inline = _parse_inline_list(item)
                if inline is not None:
                    result.append(inline)
                else:
                    result.append(_parse_scalar(item))
                index += 1
                continue

            index += 1
            if index < len(lines) and lines[index][0] > base_indent:
                child_indent = lines[index][0]
                child, index = parse_block(index, child_indent)
                result.append(child)
            else:
                result.append("")
        return result, index

    def parse_dict(start: int, base_indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        index = start
        while index < len(lines):
            indent, content = lines[index]
            if indent < base_indent:
                break
            if indent != base_indent:
                index += 1
                continue
            if ":" not in content:
                index += 1
                continue
            key, _, raw_value = content.partition(":")
            key = key.strip()
            value = raw_value.strip()
            if value:
                inline = _parse_inline_list(value)
                result[key] = inline if inline is not None else _parse_scalar(value)
                index += 1
                continue

            index += 1
            if index < len(lines) and lines[index][0] > base_indent:
                child_indent = lines[index][0]
                child, index = parse_block(index, child_indent)
                result[key] = child
            else:
                result[key] = {}
        return result, index

    if not lines:
        return {}
    root_indent = lines[0][0]
    parsed, _ = parse_dict(0, root_indent)
    return parsed


def _format_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def dump_yaml(data: Any, indent: int = 0) -> str:
    """Serialize a nested dict/list/scalar structure to a small YAML subset."""
    prefix = " " * indent

    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                if value:
                    lines.append(f"{prefix}{key}:")
                    lines.append(dump_yaml(value, indent + 2))
                else:
                    lines.append(
                        f"{prefix}{key}: []"
                        if isinstance(value, list)
                        else f"{prefix}{key}: {{}}"
                    )
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(value)}")
        return "\n".join(lines)

    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines)

    return f"{prefix}{_format_scalar(data)}"
