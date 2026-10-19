"""Run the trained YCB-V object pipeline on registered RGB-D without GT labels."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.inference.pipeline import run_frame
from clutterpick.models.detector import load_detector
from clutterpick.utils.common import load_config, resolve, write_json
from clutterpick.utils.outputs import export_bop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=float, nargs=9, required=True, metavar="K")
    parser.add_argument("--depth-scale-mm", type=float, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="artifacts/ycbv/external_rgbd")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.depth_scale_mm <= 0:
        parser.error("Depth scale must be positive")
    import numpy as np
    import torch

    k = np.asarray(args.intrinsics).reshape(3, 3)
    if (
        not np.isfinite(k).all()
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1])
    ):
        parser.error("Expected finite calibrated K with positive focal lengths")
    torch.set_num_threads(4)
    config = load_config(args.config)
    config["data"]["name"] = "external_registered_rgbd"
    config["data"]["revision"] = "user_supplied"
    frame = {
        "key": "external/000000/0",
        "scene_id": "000000",
        "image_id": 0,
        "rgb": str(args.rgb.resolve()),
        "depth": str(args.depth.resolve()),
        "K": args.intrinsics,
        "depth_scale_mm": args.depth_scale_mm,
        "instances": [],
    }
    detector = load_detector(
        resolve(config["detector"]["checkpoint"]), config, args.device
    )
    directory = resolve(args.output)
    row = run_frame(frame, config, detector, directory / "external_000000_0")
    write_json(directory / "run_summary.json", [row])
    export_bop([row], directory)
    print({key: value for key, value in row.items() if key != "pose_diagnostics"})


if __name__ == "__main__":
    main()
