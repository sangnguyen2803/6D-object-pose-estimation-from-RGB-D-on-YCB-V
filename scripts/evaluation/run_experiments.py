"""Fixed-cohort YCB-V ablations; oracle masks explicitly identified."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.data.dataset import load_index, select_frames
from clutterpick.evaluation.reporting import summarize
from clutterpick.inference.pipeline import run_frame
from clutterpick.models.detector import load_detector
from clutterpick.utils.common import load_config, read_json, resolve, write_json
from clutterpick.utils.outputs import export_bop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output", default="artifacts/ycbv/experiments")
    args = parser.parse_args()
    import numpy as np
    import torch

    torch.set_num_threads(4)
    config = load_config(args.config)
    index, splits = load_index(config)
    frames = select_frames(index, splits["test"], args.limit, config["seed"])
    detector = load_detector(resolve(config["detector"]["checkpoint"]), config)
    modes = [
        ("mask_rgbd", False, "mask", "cosypose_rgbd"),
        ("bbox_rgbd", False, "bbox", "cosypose_rgbd"),
        ("reference_mask_rgbd", True, "mask", "cosypose_rgbd"),
        ("mask_rgb", False, "mask", "cosypose"),
    ]
    runtime, results = {}, {}
    try:
        for name, oracle, extraction, backend in modes:
            rows = []
            directory = resolve(args.output) / name
            for frame in frames:
                rows.append(
                    run_frame(
                        frame,
                        config,
                        detector,
                        directory / frame["key"].replace("/", "_"),
                        oracle,
                        backend,
                        plots=False,
                        depth_extraction=extraction,
                        runtime=runtime,
                        export_clouds=False,
                    )
                )
                write_json(directory / "run_summary.json", rows)
                print(name, len(rows), len(frames), flush=True)
            export_bop(rows, directory)
            contamination = [
                item["point_contamination_fraction"]
                for row in rows
                for item in read_json(
                    Path(row["output"]) / "geometry_diagnostics.json"
                )["instances"]
                if item["point_contamination_fraction"] is not None
            ]
            results[name] = {
                **summarize(rows),
                "oracle_masks": oracle,
                "mean_point_contamination": float(np.mean(contamination))
                if contamination
                else None,
                "frame_keys": [row["frame_key"] for row in rows],
            }
            write_json(resolve(args.output) / "summary.json", results)
    finally:
        for estimator in runtime.values():
            if hasattr(estimator, "close"):
                estimator.close()


if __name__ == "__main__":
    main()
