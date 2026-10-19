"""Paired architecture ablation with warm-up, repeated timings and unchanged weights."""

import argparse
import copy
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.data.dataset import load_index, select_frames
from clutterpick.evaluation.reporting import per_object, summarize
from clutterpick.inference.pipeline import get_estimator, run_frame
from clutterpick.models.detector import load_detector
from clutterpick.utils.common import load_config, resolve, write_json
from clutterpick.utils.outputs import export_bop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "batch_only", "cache_only", "optimized"],
    )
    parser.add_argument("--output", default="artifacts/ycbv/architecture/pilot")
    parser.add_argument("--report", default="reports/ycbv/architecture/pilot.json")
    parser.add_argument(
        "--reuse-baseline",
        help="Reuse completed sequential baseline measurements; preserves timing provenance.",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = load_config("configs/enhanced.yaml")
    optimized = copy.deepcopy(base)
    optimized["runtime"] = {"pose_batch_size": 4, "lazy_scene_geometry": True}
    optimized["pose"]["cache_model_trees"] = True
    optimized["pose"]["resident_renderer"] = True
    optimized["pose"]["quality_retry"] = True
    resolve("configs/optimized.yaml").write_text(
        yaml.safe_dump(optimized, sort_keys=False), encoding="utf-8"
    )
    configs = {"baseline": base, "optimized": optimized}
    configs["optimized_fast"] = copy.deepcopy(optimized)
    configs["optimized_fast"]["pose"]["quality_retry"] = False
    configs["batch_only"] = copy.deepcopy(base)
    configs["batch_only"]["runtime"] = {"pose_batch_size": 4}
    configs["cache_only"] = copy.deepcopy(base)
    configs["cache_only"]["pose"]["cache_model_trees"] = True
    configs["resident_only"] = copy.deepcopy(base)
    configs["resident_only"]["pose"]["resident_renderer"] = True
    index, splits = load_index(base)
    frames = select_frames(index, splits["test"], args.limit, base["seed"])
    detector = load_detector(resolve(base["detector"]["checkpoint"]), base)
    root = resolve(args.output)
    reports = {}
    if args.reuse_baseline:
        from clutterpick.utils.common import read_json

        previous = read_json(resolve(args.reuse_baseline))
        assert (
            previous["cohort_size"] == len(frames)
            and previous["repetitions"] == args.repeats
        )
        assert previous["variants"]["baseline"]["frames"] == [
            frame["key"] for frame in frames
        ]
        reports["baseline"] = previous["variants"]["baseline"]
    for name in args.variants:
        cfg = configs[name]
        runtime = {}
        try:
            estimator = get_estimator(
                cfg, cfg["pose"]["backend"], runtime, str(detector.device)
            )
            for obj in cfg["data"]["object_ids"]:
                estimator.geometry.mesh(obj)
            # Warm the exact cohort before timing; includes renderer/model/texture initialization.
            for frame in frames:
                run_frame(
                    frame,
                    cfg,
                    detector,
                    root / name / "warmup" / frame["key"].replace("/", "_"),
                    runtime=runtime,
                    plots=False,
                    export_clouds=False,
                )
            print("Warmup complete", name, flush=True)
            renderer_before = estimator.renderer.cache_stats.copy()
            rows = []
            peaks = []
            workspace = []
            for repeat in range(args.repeats):
                repeat_rows = []
                for number, frame in enumerate(frames):
                    if detector.device.type == "cuda":
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                    row = run_frame(
                        frame,
                        cfg,
                        detector,
                        root
                        / name
                        / f"repeat_{repeat}"
                        / frame["key"].replace("/", "_"),
                        runtime=runtime,
                        plots=False,
                        export_clouds=False,
                    )
                    if detector.device.type == "cuda":
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated()
                        peaks.append(peak / 2**20)
                        workspace.append((peak - before) / 2**20)
                    repeat_rows.append(row)
                    if (number + 1) % 12 == 0:
                        print(
                            "Benchmark",
                            name,
                            "repeat",
                            repeat,
                            "frame",
                            number + 1,
                            "/",
                            len(frames),
                            flush=True,
                        )
                directory = root / name / f"repeat_{repeat}"
                write_json(directory / "run_summary.json", repeat_rows)
                export_bop(repeat_rows, directory)
                rows.append(repeat_rows)
            times = np.array(
                [
                    row["timing"]["inference_s_before_reference_evaluation"]
                    for part in rows
                    for row in part
                ]
            )
            components = {
                key: float(
                    np.median([row["timing"][key] for part in rows for row in part])
                )
                for key in [
                    "detection_s",
                    "scene_geometry_s",
                    "instance_geometry_s",
                    "pose_s",
                ]
            }
            reports[name] = {
                "config": cfg,
                "repeats": [summarize(part) for part in rows],
                "per_object": per_object(rows[0], index),
                "frames": [frame["key"] for frame in frames],
                "timing": {
                    "p50_s": float(np.median(times)),
                    "p95_s": float(np.quantile(times, 0.95)),
                    "frames_per_s": float(1 / times.mean()),
                    "components_median_s": components,
                },
                "memory": {
                    "peak_torch_allocated_MiB": max(peaks) if peaks else None,
                    "peak_incremental_workspace_MiB": max(workspace)
                    if workspace
                    else None,
                    "scope": "PyTorch allocations; excludes renderer/driver memory",
                },
                "warmup_frames": len(frames),
                "output": str(root / name),
            }
            reports[name]["renderer"] = {
                key: estimator.renderer.cache_stats[key] - renderer_before[key]
                for key in renderer_before
            }
            write_json(
                resolve(args.report),
                {
                    "variants": reports,
                    "hardware": torch.cuda.get_device_name()
                    if torch.cuda.is_available()
                    else "CPU",
                    "cohort_size": len(frames),
                    "repetitions": args.repeats,
                    "uses_gt_for_inference": False,
                    "reused_baseline_report": args.reuse_baseline,
                    "scope": "paired same frame cohort, detector/pose checkpoints and robust ICP; exploratory test evaluation; "
                    "warm timings before reference scoring; excludes model initialization and report visualization",
                },
            )
            print("Completed", name, reports[name]["timing"], flush=True)
        finally:
            for value in runtime.values():
                if hasattr(value, "close"):
                    value.close()
            value = None  # Do not keep the previous estimator alive through the loop variable.
            del estimator, runtime
            gc.collect()
            if detector.device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
