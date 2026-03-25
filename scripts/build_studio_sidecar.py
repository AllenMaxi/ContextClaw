from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_TAURI_ROOT = REPO_ROOT / "studio-ui" / "src-tauri"
ENTRYPOINT = Path(__file__).resolve().parent / "studio_sidecar_entry.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ContextClaw Studio sidecar")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--target-triple", default="")
    parser.add_argument("--src-tauri-root", default=str(SRC_TAURI_ROOT))
    return parser.parse_args()


def _require_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit(
            "PyInstaller is required to build the Studio sidecar. "
            "Install `contextclaw[desktop-build]` or `pyinstaller` first."
        )


def build_sidecar(
    *,
    python_bin: str,
    target_triple: str,
    src_tauri_root: Path,
) -> Path:
    from contextclaw.studio.sidecar import (
        DEFAULT_SIDECAR_BASE_NAME,
        pyinstaller_data_separator,
        sidecar_binary_path,
    )

    target_path = sidecar_binary_path(
        src_tauri_root,
        base_name=DEFAULT_SIDECAR_BASE_NAME,
        target_triple=target_triple,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    pyinstaller_dir = src_tauri_root / "target" / "pyinstaller"
    build_dir = pyinstaller_dir / "build"
    spec_dir = pyinstaller_dir / "spec"
    config_dir = pyinstaller_dir / "config"
    build_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    add_data_separator = pyinstaller_data_separator(target_triple)
    add_data_args = [
        "--add-data",
        f"{REPO_ROOT / 'catalog'}{add_data_separator}catalog",
        "--add-data",
        f"{REPO_ROOT / 'contextclaw' / 'tools' / 'bundles.json'}"
        f"{add_data_separator}contextclaw/tools",
    ]
    command = [
        python_bin,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        target_path.name,
        "--distpath",
        str(target_path.parent),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(spec_dir),
        "--collect-data",
        "contextclaw",
        "--collect-submodules",
        "contextclaw",
        "--collect-submodules",
        "fastapi",
        "--collect-submodules",
        "uvicorn",
        *add_data_args,
        str(ENTRYPOINT),
    ]
    env = dict(os.environ)
    env["PYINSTALLER_CONFIG_DIR"] = str(config_dir)
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)
    return target_path


def main() -> None:
    from contextclaw.studio.sidecar import detect_target_triple

    args = _parse_args()
    _require_pyinstaller()
    target_triple = args.target_triple or detect_target_triple()
    output_path = build_sidecar(
        python_bin=args.python_bin,
        target_triple=target_triple,
        src_tauri_root=Path(args.src_tauri_root).resolve(),
    )
    print(output_path)


if __name__ == "__main__":
    main()
