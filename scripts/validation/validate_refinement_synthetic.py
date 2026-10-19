"""Independent synthetic CAD corruptions; no real test pose is read."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.evaluation.metrics import pose_errors
from clutterpick.geometry.icp import trimmed_icp
from clutterpick.geometry.point_cloud import transform_points
from clutterpick.geometry.pose_estimation import GeometricPoseEstimator
from clutterpick.geometry.refinement import bounded, fit_metrics, refine
from clutterpick.utils.common import load_config, resolve, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument(
        "--output", default="reports/ycbv/extension/synthetic_validation.json"
    )
    args = parser.parse_args()
    cfg = load_config("configs/enhanced.yaml")
    geometry = GeometricPoseEstimator(cfg)
    rng = np.random.default_rng(args.seed)
    rows = []
    for obj in cfg["data"]["object_ids"]:
        mesh, model, normals = geometry.mesh(obj)
        for fraction, outliers in [(0.8, 0.0), (0.5, 0.25), (0.25, 0.5)]:
            truth = np.eye(4)
            truth[:3, :3] = Rotation.random(random_state=rng).as_matrix()
            truth[:3, 3] = [0.02, -0.01, 0.8]
            world = transform_points(model, truth)
            world_normals = normals @ truth[:3, :3].T
            visible = np.sum(world_normals * (-world), axis=1) > 0
            crop = world[:, 0] < np.quantile(world[visible, 0], fraction)
            indices = np.flatnonzero(visible & crop)
            selected = rng.choice(indices, min(1200, len(indices)), replace=False)
            observed = world[selected] + rng.normal(0, 0.001, (len(selected), 3))
            background = observed[
                rng.choice(len(observed), int(len(observed) * outliers))
            ] + rng.uniform(
                [-0.03, -0.03, 0.03],
                [0.03, 0.03, 0.10],
                (int(len(observed) * outliers), 3),
            )
            observed = np.concatenate([observed, background])
            initial = truth.copy()
            initial[:3, 3] += rng.normal(0, 0.01, 3)
            initial[:3, :3] = (
                Rotation.from_rotvec(rng.normal(0, np.deg2rad(2), 3)).as_matrix()
                @ initial[:3, :3]
            )
            pose_cfg = cfg["pose"]
            subset = observed[
                np.random.default_rng(cfg["seed"] + obj).choice(
                    len(observed),
                    min(len(observed), pose_cfg["observed_points"]),
                    replace=False,
                )
            ]
            legacy, after, _ = trimmed_icp(
                model,
                subset,
                initial,
                pose_cfg["icp_iterations"],
                pose_cfg["correspondence_limit_m"],
                pose_cfg["trim_fraction"],
            )
            before = fit_metrics(model, subset, initial)["rmse_m"]
            if not (
                bounded(legacy, initial, pose_cfg)[0]
                and after
                < before * (1 - pose_cfg["depth_min_relative_fit_improvement"])
            ):
                legacy = initial
            enhanced, diagnostic = refine(
                model, normals, observed, initial, pose_cfg, cfg["seed"] + obj
            )
            rows.append(
                {
                    "obj_id": obj,
                    "visible_crop_quantile": fraction,
                    "background_ratio_relative_to_foreground": outliers,
                    "baseline": pose_errors(mesh, legacy, truth),
                    "enhanced": pose_errors(mesh, enhanced, truth),
                    "selected_candidate": diagnostic["selected_candidate"],
                    "quality": diagnostic["quality"]["state"],
                }
            )
        print("Synthetic object", obj, "/21", flush=True)
    summary = {
        name: {
            "mean_ADD_mm": float(np.mean([row[name]["ADD_m"] * 1000 for row in rows])),
            "median_translation_mm": float(
                np.median([row[name]["translation_error_mm"] for row in rows])
            ),
        }
        for name in ["baseline", "enhanced"]
    }
    write_json(
        resolve(args.output),
        {
            "seed": args.seed,
            "cases": len(rows),
            "rows": rows,
            "summary": summary,
            "config": cfg,
            "uses_real_test_gt": False,
            "scope": "synthetic CAD corruption validation, not equivalent to real RGB-D validation",
            "parameter_selection": "fixed engineering configuration; no test-GT search",
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
