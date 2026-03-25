from __future__ import annotations

from pathlib import Path

import pytest

from contextclaw.studio.sidecar import (
    DEFAULT_SIDECAR_BASE_NAME,
    detect_target_triple,
    external_bin_name,
    pyinstaller_data_separator,
    sidecar_binary_path,
)


@pytest.mark.parametrize(
    ("system_name", "machine_name", "expected"),
    [
        ("Darwin", "arm64", "aarch64-apple-darwin"),
        ("Darwin", "x86_64", "x86_64-apple-darwin"),
        ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
        ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
    ],
)
def test_detect_target_triple_maps_supported_hosts(
    system_name: str,
    machine_name: str,
    expected: str,
):
    assert detect_target_triple(system_name, machine_name) == expected


def test_detect_target_triple_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        detect_target_triple("Darwin", "sparc")


def test_sidecar_binary_path_uses_tauri_binaries_dir(tmp_path: Path):
    binary_path = sidecar_binary_path(
        tmp_path,
        base_name=DEFAULT_SIDECAR_BASE_NAME,
        target_triple="aarch64-apple-darwin",
    )

    assert binary_path == tmp_path / "binaries" / external_bin_name(
        DEFAULT_SIDECAR_BASE_NAME,
        "aarch64-apple-darwin",
    )
    assert pyinstaller_data_separator("aarch64-apple-darwin") == ":"
    assert pyinstaller_data_separator("x86_64-pc-windows-msvc") == ";"
