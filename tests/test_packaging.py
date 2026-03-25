from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

HAS_BUILD = importlib.util.find_spec("build") is not None


@pytest.mark.skipif(
    not HAS_BUILD,
    reason="python-build tooling is not installed",
)
def test_wheel_contains_built_studio_frontend(tmp_path: Path):
    package_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(tmp_path.glob("contextclaw-*.whl"))
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        assert "contextclaw/studio/_frontend/index.html" in names
        assert any(
            name.startswith("contextclaw/studio/_frontend/assets/") for name in names
        )

        index_html = archive.read("contextclaw/studio/_frontend/index.html").decode(
            "utf-8"
        )
        assert "<html" in index_html.lower()
        assert 'id="root"' in index_html or "id='root'" in index_html
