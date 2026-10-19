"""Compare an unlabeled RGB-D smoke run with its dataset-interface reference."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.utils.common import read_json, resolve, write_json


def main():
    external = read_json(
        resolve("artifacts/ycbv/external_smoke/external_000000_0/poses.json")
    )
    reference = read_json(
        resolve("artifacts/ycbv/demo_rgbd/ycbv_test_000048_1074/poses.json")
    )
    assert not external["uses_ground_truth_pose_for_inference"]
    assert not external["uses_ground_truth_masks"]
    assert not external["ground_truth_pose_used_only_for_evaluation"]
    assert len(external["instances"]) == len(reference["instances"])
    differences = []
    for first, second in zip(external["instances"], reference["instances"]):
        assert first["obj_id"] == second["obj_id"]
        a, b = (
            np.asarray(first["T_camera_object"]),
            np.asarray(second["T_camera_object"]),
        )
        translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000)
        rotation = float(
            np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())
        )
        differences.append(
            {
                "obj_id": first["obj_id"],
                "translation_difference_mm": translation,
                "rotation_difference_degrees": rotation,
            }
        )
        # Compare physical pose, allowing small numerical variation across GPU runs.
        assert translation < 0.1 and rotation < 0.1, (
            "Unlabeled-input inference differs materially"
        )
    result = {
        "passed": True,
        "estimated_poses": len(differences),
        "gt_labels_supplied": False,
        "max_translation_difference_mm": max(
            item["translation_difference_mm"] for item in differences
        ),
        "max_rotation_difference_degrees": max(
            item["rotation_difference_degrees"] for item in differences
        ),
        "comparison_tolerance": {"translation_mm": 0.1, "rotation_degrees": 0.1},
        "provided_inputs": ["RGB", "uint16 depth", "K", "depth scale"],
        "pose_differences": differences,
        "scope": "same YCB-V image through unlabeled interface; not a new camera benchmark",
    }
    write_json(resolve("reports/ycbv/external_input_validation.json"), result)
    print({key: value for key, value in result.items() if key != "pose_differences"})


if __name__ == "__main__":
    main()
