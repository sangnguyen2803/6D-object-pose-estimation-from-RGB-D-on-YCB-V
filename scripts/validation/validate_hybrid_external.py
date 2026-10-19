"""Exercise the external RGB-D CLI and verify all learned stages work without GT."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from validate_ycbv import validate

from clutterpick.data.dataset import load_index
from clutterpick.utils.common import load_config, read_json, resolve, write_json


def main():
    cfg = load_config("configs/hybrid.yaml")
    index, _ = load_index(cfg)
    frame = next(
        frame for frame in index["frames"] if frame["key"] == "ycbv_test/000048/1074"
    )
    output = resolve("artifacts/ycbv/research/external_demo")
    args = [
        sys.executable,
        str(resolve("scripts/inference/infer_rgbd.py")),
        "--config",
        "configs/hybrid.yaml",
        "--rgb",
        frame["rgb"],
        "--depth",
        frame["depth"],
        "--depth-scale-mm",
        str(frame["depth_scale_mm"]),
        "--output",
        str(output),
        "--intrinsics",
        *[str(value) for value in frame["K"]],
    ]
    subprocess.run(args, check=True, timeout=240)
    payload = read_json(output / "external_000000_0" / "poses.json")
    reference = read_json(
        resolve("artifacts/ycbv/research/cli_demo/ycbv_test_000048_1074/poses.json")
    )
    assert payload["ground_truth_pose_used_only_for_evaluation"] is False
    assert payload["learned_point_refinement_enabled"]
    assert len(payload["instances"]) == len(reference["instances"])
    deviations = []
    for i, (predicted, expected) in enumerate(
        zip(payload["instances"], reference["instances"])
    ):
        assert predicted["obj_id"] == expected["obj_id"]
        first = np.asarray(predicted["T_camera_object"])
        second = np.asarray(expected["T_camera_object"])
        deviations.append(
            {
                "obj_id": predicted["obj_id"],
                "translation_mm": float(
                    np.linalg.norm(first[:3, 3] - second[:3, 3]) * 1000
                ),
                "rotation_degrees": float(
                    np.rad2deg(
                        Rotation.from_matrix(
                            first[:3, :3] @ second[:3, :3].T
                        ).magnitude()
                    )
                ),
            }
        )
        a = output / "external_000000_0"
        b = resolve("artifacts/ycbv/research/cli_demo/ycbv_test_000048_1074")
        np.testing.assert_array_equal(
            np.asarray(Image.open(a / "masks" / f"instance_{i:03d}.png")),
            np.asarray(Image.open(b / "masks" / f"instance_{i:03d}.png")),
        )
        np.testing.assert_array_equal(
            trimesh.load(
                a / "pointclouds" / f"instance_{i:03d}.ply", process=False
            ).vertices,
            trimesh.load(
                b / "pointclouds" / f"instance_{i:03d}.ply", process=False
            ).vertices,
        )
    rows = read_json(output / "run_summary.json")
    assert rows[0]["gt_instances"] == 0 and not rows[0]["pose_diagnostics"]
    checks = validate(output, 1)
    result = {
        "passed": True,
        "checks": checks,
        "gt_instances": 0,
        "masks_and_filtered_clouds_equal_without_GT": True,
        "pose_cross_process_deviations": deviations,
        "exact_pose_equivalence_verified": False,
        "scope": "hybrid external CLI; same registered RGB-D/K/CAD and checkpoints, no labels supplied; "
        "cross-process pose values vary, exact equivalence is not asserted",
    }
    report = resolve("reports/ycbv/research/external_input_validation.json")
    if report.exists():
        previous = read_json(report)
        if previous.get("initial_strict_comparison"):
            result["initial_strict_comparison"] = previous["initial_strict_comparison"]
    write_json(report, result)
    print(result)


if __name__ == "__main__":
    main()
