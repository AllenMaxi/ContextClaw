from __future__ import annotations

import platform
from pathlib import Path

DEFAULT_SIDECAR_BASE_NAME = "contextclaw-studio-daemon"

_ARCH_ALIASES = {
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "amd64": "x86_64",
    "x86_64": "x86_64",
}


def detect_target_triple(
    system_name: str | None = None,
    machine_name: str | None = None,
) -> str:
    system_value = (system_name or platform.system()).lower()
    machine_value = _ARCH_ALIASES.get((machine_name or platform.machine()).lower())
    if machine_value is None:
        raise ValueError(
            f"Unsupported architecture: {machine_name or platform.machine()}"
        )
    if system_value in {"darwin", "mac", "macos"}:
        return f"{machine_value}-apple-darwin"
    if system_value == "linux":
        return f"{machine_value}-unknown-linux-gnu"
    if system_value in {"windows", "win32"}:
        return f"{machine_value}-pc-windows-msvc"
    raise ValueError(f"Unsupported platform: {system_name or platform.system()}")


def target_executable_suffix(target_triple: str) -> str:
    return ".exe" if "windows" in target_triple else ""


def external_bin_name(
    base_name: str = DEFAULT_SIDECAR_BASE_NAME,
    target_triple: str | None = None,
) -> str:
    triple = target_triple or detect_target_triple()
    return f"{base_name}-{triple}{target_executable_suffix(triple)}"


def sidecar_binary_path(
    src_tauri_root: Path,
    *,
    base_name: str = DEFAULT_SIDECAR_BASE_NAME,
    target_triple: str | None = None,
) -> Path:
    return src_tauri_root / "binaries" / external_bin_name(base_name, target_triple)


def pyinstaller_data_separator(target_triple: str) -> str:
    return ";" if target_executable_suffix(target_triple) == ".exe" else ":"
