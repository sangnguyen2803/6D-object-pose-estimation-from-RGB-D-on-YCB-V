"""Fixed-test-cohort ablations, depth corruption stress tests, and learned 3D cleanup."""

import copy
import gc
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.data.dataset import load_index, select_frames
from clutterpick.evaluation.metrics import evaluate_detector
from clutterpick.evaluation.reporting import per_object, summarize
from clutterpick.inference.pipeline import run_frame
from clutterpick.models.detector import load_detector
from clutterpick.models.fusion import FusionDetector
from clutterpick.utils.common import load_config, read_json, resolve, write_json
from clutterpick.utils.outputs import export_bop


class CorruptedDetector:
    requires_depth = True

    def __init__(self, base, mode):
        self.base = base
        self.mode = mode
        self.device, self.config, self.checkpoint = (
            base.device,
            base.config,
            base.checkpoint,
        )
        self.rng = np.random.default_rng(2026)

    def predict(self, rgb, score_threshold=None, depth=None, k=None):
        depth = depth.copy()
        if self.mode == "missing_all":
            depth[:] = 0
        elif self.mode.startswith("dropout"):
            depth[self.rng.random(depth.shape) < float(self.mode.split("_")[1])] = 0
        elif self.mode == "noise_5mm":
            valid = depth > 0
            depth[valid] = np.maximum(
                depth[valid] + self.rng.normal(0, 0.005, valid.sum()), 0
            )
        return self.base.predict(rgb, score_threshold=score_threshold, depth=depth, k=k)


def point_diagnostics(rows):
    values = [
        item["point_contamination_fraction"]
        for row in rows
        for item in read_json(Path(row["output"]) / "geometry_diagnostics.json")[
            "instances"
        ]
        if item["point_contamination_fraction"] is not None
    ]
    return float(np.mean(values)) if values else None


def class_diagnostics(rows, index):
    metrics = per_object(rows, index)
    lookup = {
        obj["instance_id"]: str(obj["obj_id"])
        for frame in index["frames"]
        for obj in frame["instances"]
    }
    samples = {key: [] for key in metrics}
    for row in rows:
        for item in read_json(Path(row["output"]) / "geometry_diagnostics.json")[
            "instances"
        ]:
            value = item["point_contamination_fraction"]
            if value is not None:
                samples[lookup[item["reference_instance_id"]]].append(value)
    for key, values in samples.items():
        metrics[key]["mean_contamination"] = float(np.mean(values)) if values else None
    return metrics


def paired_pose(before, after, index):
    keys = {row["frame_key"] for row in before}
    references = {
        obj["instance_id"]
        for frame in index["frames"]
        if frame["key"] in keys
        for obj in frame["instances"]
    }

    def successes(rows):
        return {
            item["reference_instance_id"]
            for row in rows
            for item in row["pose_diagnostics"]
            if item["ADD_or_S_success_0_1d"]
        }

    first, second = successes(before), successes(after)
    return {
        "gt_instances": len(references),
        "fail_to_pass": len(second - first),
        "pass_to_fail": len(first - second),
        "both_pass": len(first & second),
        "both_fail": len(references - (first | second)),
        "threshold": "0.1 CAD diameter; ADD(-S); does not imply every numerical error improves",
    }


def main():
    torch.set_num_threads(4)
    cfg = load_config("configs/research.yaml")
    index, splits = load_index(cfg)
    test = select_frames(index, splits["test"], 96, cfg["seed"])
    pose_frames = select_frames(index, splits["test"], 24, cfg["seed"])
    holdout = select_frames(index, splits["holdout"], 60, cfg["seed"])
    root = resolve("artifacts/ycbv/research/evaluation")
    report_path = resolve("reports/ycbv/research/ablation.json")
    reports = read_json(report_path) if report_path.exists() else {}
    runtime = {}
    try:
        for variant in cfg["research"]["variants"]:
            checkpoint = resolve(f"artifacts/ycbv/research/fusion/{variant}/best.pt")
            checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            previous = reports.get(variant, {})
            if (
                previous.get("checkpoint_sha256") == checksum
                and previous.get("evaluation_config") == cfg
                and previous["test"]["frame_keys"] == [frame["key"] for frame in test]
                and previous.get("pose_frame_keys")
                == [frame["key"] for frame in pose_frames]
            ):
                print("Evaluation already complete", variant, flush=True)
                continue
            detector = FusionDetector(checkpoint, cfg)
            result = {
                "test": evaluate_detector(
                    detector, test, root / variant / "test.json", boundary=True
                ),
                "pbr_holdout": evaluate_detector(
                    detector, holdout, root / variant / "holdout.json", boundary=True
                ),
            }
            result["robustness"] = {}
            if variant in ["rgb", "early", "late", "gated"]:
                for mode in ["missing_all", "dropout_0.3", "dropout_0.8", "noise_5mm"]:
                    result["robustness"][mode] = evaluate_detector(
                        CorruptedDetector(detector, mode),
                        test,
                        root / variant / (mode + ".json"),
                        boundary=True,
                    )
            rows = []
            directory = root / variant / "pose"
            for frame in pose_frames:
                rows.append(
                    run_frame(
                        frame,
                        cfg,
                        detector,
                        directory / frame["key"].replace("/", "_"),
                        runtime=runtime,
                        plots=False,
                        export_clouds=False,
                    )
                )
            write_json(directory / "run_summary.json", rows)
            export_bop(rows, directory)
            result["pose"] = {
                **summarize(rows),
                "mean_contamination": point_diagnostics(rows),
                "per_object": class_diagnostics(rows, index),
            }
            result.update(
                checkpoint_sha256=checksum,
                evaluation_config=cfg,
                pose_frame_keys=[frame["key"] for frame in pose_frames],
            )
            reports[variant] = result
            write_json(report_path, reports)
            print(
                "Evaluation complete",
                variant,
                result["test"]["metrics"]["segm"]["AP_50_95"],
                flush=True,
            )
            del detector
            gc.collect()
            torch.cuda.empty_cache()
        baseline = load_detector(resolve(cfg["detector"]["checkpoint"]), cfg)
        write_json(
            resolve("reports/ycbv/research/published_detector.json"),
            {
                "test": evaluate_detector(
                    baseline, test, root / "published" / "test.json", boundary=True
                ),
                "pbr_holdout": evaluate_detector(
                    baseline,
                    holdout,
                    root / "published" / "holdout.json",
                    boundary=True,
                ),
            },
        )
        with_cleaner = copy.deepcopy(cfg)
        with_cleaner["point_refiner"] = {
            "enabled": True,
            "checkpoint": "artifacts/ycbv/research/point_network/best.pt",
        }
        point_rows = []
        reference = []
        directory = root / "point_network" / "pose"
        for frame in pose_frames:
            reference.append(
                run_frame(
                    frame,
                    cfg,
                    baseline,
                    root / "point_baseline" / frame["key"].replace("/", "_"),
                    runtime=runtime,
                    plots=False,
                    export_clouds=False,
                )
            )
            point_rows.append(
                run_frame(
                    frame,
                    with_cleaner,
                    baseline,
                    directory / frame["key"].replace("/", "_"),
                    runtime=runtime,
                    plots=False,
                    export_clouds=False,
                )
            )
        write_json(directory / "run_summary.json", point_rows)
        export_bop(point_rows, directory)
        learned = {
            **summarize(point_rows),
            "mean_contamination": point_diagnostics(point_rows),
        }
        write_json(root / "point_baseline" / "run_summary.json", reference)
        export_bop(reference, root / "point_baseline")
        assert [row["frame_key"] for row in point_rows] == [
            row["frame_key"] for row in reference
        ]
        before_cloud = {
            item["reference_instance_id"]: item
            for row in reference
            for item in read_json(Path(row["output"]) / "geometry_diagnostics.json")[
                "instances"
            ]
        }
        after_cloud = {
            item["reference_instance_id"]: item
            for row in point_rows
            for item in read_json(Path(row["output"]) / "geometry_diagnostics.json")[
                "instances"
            ]
        }
        retention = []
        for key, before in before_cloud.items():
            foreground = (
                before["retained_points"] * (1 - before["point_contamination_fraction"])
                if before["point_contamination_fraction"] is not None
                else 0
            )
            if not foreground:
                continue
            after = after_cloud.get(key)
            kept = (
                after["retained_points"] * (1 - after["point_contamination_fraction"])
                if after and after["point_contamination_fraction"] is not None
                else 0
            )
            retention.append(kept / foreground)
        learned["mean_foreground_retention"] = (
            float(np.mean(retention)) if retention else None
        )
        write_json(
            resolve("reports/ycbv/research/point_real_comparison.json"),
            {
                "learned_3d": learned,
                "original_2d_mask": {
                    **summarize(reference),
                    "mean_contamination": point_diagnostics(reference),
                },
                "source_masks": "same published RGB detector/robust ICP configuration on fixed 24 frame cohort; threshold selected only on PBR validation",
                "paired": paired_pose(reference, point_rows, index),
                "gt_used_for_inference": False,
            },
        )
        # End-to-end demo of all three: gated fusion + query head + learned point cleanup.
        detector = FusionDetector(
            resolve("artifacts/ycbv/research/fusion/gated_query/best.pt"), cfg
        )
        directory = resolve("artifacts/ycbv/research/demo")
        row = run_frame(
            pose_frames[0],
            with_cleaner,
            detector,
            directory / pose_frames[0]["key"].replace("/", "_"),
            runtime=runtime,
            plots=True,
            export_clouds=True,
        )
        write_json(directory / "run_summary.json", [row])
        export_bop([row], directory)
        combined = [row]
        directory = root / "combined" / "pose"
        for frame in pose_frames[1:]:
            combined.append(
                run_frame(
                    frame,
                    with_cleaner,
                    detector,
                    directory / frame["key"].replace("/", "_"),
                    runtime=runtime,
                    plots=False,
                    export_clouds=False,
                )
            )
        write_json(directory / "run_summary.json", combined)
        export_bop(combined, directory)
        write_json(
            resolve("reports/ycbv/research/combined_pose.json"),
            {
                **summarize(combined),
                "mean_contamination": point_diagnostics(combined),
                "paired_vs_gated_query_without_point_filter": paired_pose(
                    read_json(root / "gated_query" / "pose" / "run_summary.json"),
                    combined,
                    index,
                ),
                "per_object": class_diagnostics(combined, index),
                "frame_keys": [row["frame_key"] for row in combined],
                "scope": "gated_query + validation-selected learned point filter + robust ICP; fixed 24 test frames",
            },
        )
        print("Research ablations complete", flush=True)
    finally:
        for estimator in runtime.values():
            if hasattr(estimator, "close"):
                estimator.close()


if __name__ == "__main__":
    main()
