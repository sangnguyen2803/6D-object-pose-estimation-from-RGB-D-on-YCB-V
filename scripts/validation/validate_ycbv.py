"""Validate transforms, provenance, ablation cohorts and BOP unit export."""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.utils.common import read_json, resolve, write_json


def validate(directory, expected_frames=None, oracle=False):
    directory = resolve(directory)
    rows = read_json(directory / "run_summary.json")
    assert len({row["frame_key"] for row in rows}) == len(rows), "Duplicate frames"
    if expected_frames is not None:
        assert len(rows) == expected_frames, "Incomplete benchmark"
    expected = []
    for row in rows:
        payload = read_json(Path(row["output"]) / "poses.json")
        assert payload["uses_ground_truth_pose_for_inference"] is False
        assert payload["uses_ground_truth_masks"] is oracle
        assert payload["units"] == "m" and not payload["grasp_planning_performed"]
        assert (
            row["gt_instances"]
            == read_json(Path(row["output"]) / "pose_diagnostics.json")[
                "gt_instance_count"
            ]
        )
        for pose in payload["instances"]:
            if "T_camera_object" not in pose:
                continue
            matrix = np.asarray(pose["T_camera_object"])
            assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
            assert np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            assert np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)
            assert matrix[2, 3] > 0 and 1 <= pose["obj_id"] <= 21
            expected.append(
                (int(row["scene_id"]), row["image_id"], pose["obj_id"], matrix)
            )
    with (directory / "bop_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(expected)
    for row, (scene, image, obj, matrix) in zip(csv_rows, expected):
        assert (int(row["scene_id"]), int(row["im_id"]), int(row["obj_id"])) == (
            scene,
            image,
            obj,
        )
        assert np.allclose(
            np.fromstring(row["R"], sep=" ").reshape(3, 3), matrix[:3, :3], atol=1e-8
        )
        assert np.allclose(
            np.fromstring(row["t"], sep=" "), matrix[:3, 3] * 1000, atol=1e-5
        )
    return {
        "frames": len(rows),
        "valid_exported_poses": len(expected),
        "oracle_masks": oracle,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-frames", type=int, default=900)
    args = parser.parse_args()
    results = {
        "benchmark": validate("artifacts/ycbv/benchmark_full", args.expected_frames)
    }
    audit = read_json(resolve("reports/ycbv/data_audit.json"))
    assert not audit["errors"]
    assert audit["official_test_targets_match"]
    rows = read_json(resolve("artifacts/ycbv/benchmark_full/run_summary.json"))
    keys = {row["frame_key"] for row in rows}
    index = read_json(resolve("data/ycbv/index.json"))
    assert sum(row["gt_instances"] for row in rows) == sum(
        len(frame["instances"]) for frame in index["frames"] if frame["key"] in keys
    )
    if args.expected_frames == 900:
        splits = read_json(resolve("data/ycbv/splits.json"))
        assert keys == {index["frames"][i]["key"] for i in splits["test"]}
    manifest = read_json(resolve("reports/ycbv/download_manifest.json"))
    for record in manifest["pretrained_models"]:
        assert (
            hashlib.sha256(resolve(record["path"]).read_bytes()).hexdigest()
            == record["sha256"]
        ), "Checkpoint hash changed"
    experiments = read_json(resolve("artifacts/ycbv/experiments/summary.json"))
    cohorts = [value["frame_keys"] for value in experiments.values()]
    assert cohorts and all(keys == cohorts[0] for keys in cohorts), (
        "Ablation cohorts differ"
    )
    for name, value in experiments.items():
        results[name] = validate(
            "artifacts/ycbv/experiments/" + name, value["frames"], value["oracle_masks"]
        )
    results["demo"] = validate("artifacts/ycbv/demo_rgbd", 1)
    write_json(
        resolve("reports/ycbv/delivery_validation.json"),
        {"passed": True, "checks": results},
    )
    files = []
    for folder in ["src/clutterpick", "scripts", "configs", "tests", "docs"]:
        files.extend(
            path
            for path in resolve(folder).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    files.extend(
        resolve(name)
        for name in [
            "main.py",
            "README.md",
            "PROJECT_SUMMARY_VI.md",
            "pyproject.toml",
            "reports/ycbv/measured_summary.json",
            "reports/ycbv/delivery_validation.json",
            "reports/ycbv/download_manifest.json",
            "third_party/cosypose_source.json",
        ]
    )
    delivered = [
        "reports/ycbv/RESULTS_VI.md",
        "reports/ycbv/data_audit.json",
        "reports/ycbv/data_split.json",
        "reports/ycbv/pose_recall.png",
        "reports/ycbv/mask_ap_per_object.png",
        "reports/ycbv/external_input_validation.json",
        "reports/ycbv/failure_examples.json",
        "artifacts/ycbv/benchmark_full/run_summary.json",
        "artifacts/ycbv/benchmark_full/bop_predictions.csv",
        "artifacts/ycbv/experiments/summary.json",
        "artifacts/ycbv/demo_rgbd/run_summary.json",
    ]
    delivered.extend(
        "artifacts/ycbv/demo_rgbd/ycbv_test_000048_1074/" + name
        for name in [
            "poses.json",
            "final_visualization.png",
            "instances_3d.png",
            "instances_3d.html",
        ]
    )
    files.extend(resolve(name) for name in delivered if resolve(name).exists())
    write_json(
        resolve("reports/ycbv/artifact_manifest.json"),
        {
            str(path.relative_to(resolve("."))): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(files)
        },
    )
    print(results)


if __name__ == "__main__":
    main()
