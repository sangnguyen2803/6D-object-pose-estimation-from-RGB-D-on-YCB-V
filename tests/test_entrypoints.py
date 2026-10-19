"""Check entry points and root paths after moving scripts into subfolders."""

import subprocess
import sys
from pathlib import Path

import pytest

from clutterpick.utils.common import PROJECT, load_config, resolve


def test_project_paths_still_resolve_from_repository_root():
    root = Path(__file__).resolve().parents[1]
    assert PROJECT == root
    assert resolve("configs/default.yaml") == root / "configs/default.yaml"
    assert load_config()["data"]["name"] == "ycbv"


@pytest.mark.parametrize(
    "script",
    [
        "main.py",
        "scripts/data/prepare_ycbv.py",
        "scripts/inference/infer_rgbd.py",
        "scripts/validation/validate_refinement_synthetic.py",
    ],
)
def test_help_works_from_an_unrelated_directory(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(PROJECT / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
