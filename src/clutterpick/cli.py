import argparse
import os

from clutterpick import __version__
from clutterpick.utils.common import load_config, resolve, write_json
from clutterpick.utils.outputs import export_bop


def positive_int(value):
    """Reject invalid counts before loading datasets or models."""
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected a positive integer") from None
    if count <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return count


def build_parser():
    """Define the commands and options shown by --help."""
    parser = argparse.ArgumentParser(
        description="YCBPose: instance segmentation -> point clouds -> 6D pose"
    )
    parser.add_argument("--version", action="version", version=f"YCBPose {__version__}")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="YAML configuration path (default: configs/default.yaml); place before the command",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="Build the dataset index and scene-based splits")
    audit = sub.add_parser("audit", help="Check indexed dataset assets and metadata")
    audit.add_argument("--output", default="reports/ycbv/data_audit.json")
    evaluate = sub.add_parser("evaluate", help="Measure detector bbox and mask AP")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument(
        "--architecture", choices=["cosypose", "maskrcnn", "yolo", "fusion"]
    )
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--limit", type=positive_int)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument(
        "--output", default="artifacts/ycbv/evaluation/detection.json"
    )
    train = sub.add_parser("train", help="Fine-tune the detector on train/validation scenes")
    train.add_argument("--epochs", type=positive_int)
    train.add_argument("--limit", type=positive_int)
    train.add_argument("--val-limit", type=positive_int, default=200)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume")
    for command in ["infer", "benchmark"]:
        infer = sub.add_parser(
            command,
            help="Export predictions and visualizations for selected frames"
            if command == "infer"
            else "Run inference on a frame cohort and export timings and poses",
        )
        infer.add_argument("--checkpoint")
        infer.add_argument(
            "--architecture", choices=["cosypose", "maskrcnn", "yolo", "fusion"]
        )
        infer.add_argument("--frame-key")
        infer.add_argument("--split", choices=["val", "test"], default="test")
        infer.add_argument(
            "--limit", type=positive_int, default=1 if command == "infer" else 100
        )
        infer.add_argument("--device", default="auto")
        infer.add_argument(
            "--pose-backend", choices=["cosypose", "cosypose_rgbd", "geometric"]
        )
        infer.add_argument(
            "--oracle-masks",
            action="store_true",
            help="Explicit reference-mask diagnostic",
        )
        infer.add_argument(
            "--depth-extraction", choices=["mask", "bbox"], default="mask"
        )
        infer.add_argument(
            "--output",
            default="artifacts/ycbv/" + ("demo" if command == "infer" else "benchmark"),
        )
        infer.add_argument(
            "--no-plots", action="store_true", help="Skip the scene 3D PNG plot"
        )
        infer.add_argument(
            "--skip-cloud-export", action="store_true", help="Skip PLY and scene instance-label exports"
        )
    return parser


def run_inference(frames, config, detector, args):
    """Process frames, save summaries and release cached models afterwards."""
    from clutterpick.inference.pipeline import run_frame

    rows, runtime = [], {}
    try:
        for frame in frames:
            directory = resolve(args.output) / frame["key"].replace("/", "_")
            row = run_frame(
                frame,
                config,
                detector,
                directory,
                args.oracle_masks,
                args.pose_backend,
                plots=args.command == "infer" and not args.no_plots,
                depth_extraction=args.depth_extraction,
                runtime=runtime,
                export_clouds=args.command == "infer" and not args.skip_cloud_export,
            )
            rows.append(row)
            write_json(resolve(args.output) / "run_summary.json", rows)
            print(
                {key: value for key, value in row.items() if key != "pose_diagnostics"},
                flush=True,
            )
        export_bop(rows, args.output)
    finally:
        for estimator in runtime.values():
            if hasattr(estimator, "close"):
                estimator.close()


def main():
    os.environ.setdefault("TORCH_HOME", str(resolve("artifacts/cache/torch")))
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    from clutterpick.data.dataset import build_index, load_index, select_frames

    if args.command == "index":
        _, splits = build_index(config)
        print({key: len(value) for key, value in splits.items()}, flush=True)
        return
    if args.command == "audit":
        from clutterpick.data.audit import audit_dataset

        print(audit_dataset(config, resolve(args.output)), flush=True)
        return
    import torch

    torch.set_num_threads(4)
    index, splits = load_index(config)
    if args.command == "train":
        if not splits["train"] or not splits["val"]:
            parser.error(
                "Training requires separate train/validation scenes. Run prepare_ycbv.py --groups base models test train-pbr, then index. Test cannot be used for training."
            )
        from clutterpick.training.train_detector import train_maskrcnn

        print(
            train_maskrcnn(
                config,
                args.epochs,
                args.limit,
                args.val_limit,
                args.resume,
                args.device,
            ),
            flush=True,
        )
        return
    checkpoint = resolve(args.checkpoint or config["detector"]["checkpoint"])
    if not checkpoint.exists() and not getattr(args, "oracle_masks", False):
        parser.error("Checkpoint unavailable; run scripts/data/prepare_ycbv.py")
    if getattr(args, "frame_key", None):
        frames = [frame for frame in index["frames"] if frame["key"] == args.frame_key]
    else:
        frames = select_frames(index, splits[args.split], args.limit, config["seed"])
    if not frames:
        parser.error(
            "Requested frame/split unavailable; prepare data and rebuild index"
        )
    from clutterpick.models.detector import load_detector

    detector = (
        None
        if getattr(args, "oracle_masks", False)
        else load_detector(checkpoint, config, args.device, args.architecture)
    )
    if args.command == "evaluate":
        from clutterpick.evaluation.metrics import evaluate_detector

        report = evaluate_detector(detector, frames, resolve(args.output))
        print(report["metrics"], flush=True)
        return
    run_inference(frames, config, detector, args)


if __name__ == "__main__":
    main()
