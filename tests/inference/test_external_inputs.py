"""External RGB-D input mistakes must fail before loading model checkpoints."""

import subprocess
import sys

import pytest

from clutterpick.utils.common import PROJECT


@pytest.mark.parametrize(
    "case, expected",
    [
        ("nan", "Depth scale must be finite and positive"),
        ("inf", "Depth scale must be finite and positive"),
        ("0", "Depth scale must be finite and positive"),
        ("missing_rgb", "--rgb must point to an existing file"),
        ("missing_depth", "--depth must point to an existing file"),
        ("invalid_k", "Expected finite upper-triangular K"),
    ],
)
def test_external_input_errors(case, expected, tmp_path):
    rgb, depth = tmp_path / "rgb.png", tmp_path / "depth.png"
    if case != "missing_rgb":
        rgb.touch()
    if case != "missing_depth":
        depth.touch()
    intrinsics = ["500", "0", "320", "0", "500", "240", "0", "0", "1"]
    if case == "invalid_k":
        intrinsics[3] = "1"
    scale = case if case in {"nan", "inf", "0"} else "1"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/inference/infer_rgbd.py"),
            "--rgb", str(rgb),
            "--depth", str(depth),
            "--intrinsics", *intrinsics,
            "--depth-scale-mm", scale,
            "--config", str(tmp_path / "missing_config.yaml"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert expected in result.stderr
    assert "Traceback" not in result.stderr
