"""Cache predicted-mask ROI point labels from PBR train/val/holdout only."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.data.dataset import load_index, load_rgbd
from clutterpick.evaluation.metrics import match_references
from clutterpick.geometry.point_cloud import backproject, voxel_downsample
from clutterpick.models.detector import load_detector
from clutterpick.models.point_network import point_features
from clutterpick.utils.common import load_config, resolve, write_json


def main():
    import torch

    torch.set_num_threads(4)
    config = load_config("configs/research.yaml")
    index, splits = load_index(config)
    detector = load_detector(resolve(config["detector"]["checkpoint"]), config)
    records = []
    root = resolve("data/ycbv/point_training")
    for split in ["train", "val", "holdout"]:
        for number, frame_id in enumerate(splits[split]):
            frame = index["frames"][frame_id]
            if frame["group"] != "pbr_ycbv":
                raise ValueError("Point training cache must not use real test frames")
            rgb, depth, k = load_rgbd(frame)
            prediction = detector.predict(rgb)
            items = [
                {"obj_id": int(label), "mask": mask}
                for label, mask in zip(prediction["labels"], prediction["masks"])
            ]
            matches = match_references(items, frame, minimum_iou=0.3)
            for position, (reference, iou) in matches.items():
                mask = items[position]["mask"]
                selected = binary_dilation(mask, iterations=5)
                points, pixels = backproject(depth, k, selected, 0.1, 3.0)
                if len(points) < 40:
                    continue
                points, ids = voxel_downsample(points, 0.003, np.arange(len(points)))
                pixels = pixels[ids]
                colors = rgb[pixels[:, 0], pixels[:, 1]]
                truth = np.asarray(Image.open(reference["mask_visib"])) > 0
                membership = mask[pixels[:, 0], pixels[:, 1]].astype(np.float32)
                features = point_features(points, colors, membership)
                labels = truth[pixels[:, 0], pixels[:, 1]].astype(np.float32)
                rng = np.random.default_rng(config["seed"] + frame_id * 100 + position)
                selected_ids = rng.choice(len(points), 512, replace=len(points) < 512)
                target = (
                    root
                    / split
                    / (frame["key"].replace("/", "_") + f"_{position:03d}.npz")
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target,
                    features=features[selected_ids],
                    labels=labels[selected_ids],
                    obj_id=items[position]["obj_id"],
                )
                records.append(
                    {
                        "path": str(target),
                        "split": split,
                        "frame_key": frame["key"],
                        "obj_id": items[position]["obj_id"],
                        "source_mask_iou": iou,
                        "foreground_fraction": float(labels.mean()),
                        "uses_gt_only_for_training_labels": True,
                    }
                )
            if (number + 1) % 25 == 0:
                print(
                    "Point cache",
                    split,
                    number + 1,
                    "/",
                    len(splits[split]),
                    flush=True,
                )
    write_json(
        root / "index.json",
        {
            "records": records,
            "split_policy": "PBR scene-disjoint; no real test GT used",
            "features": [
                "local_X",
                "local_Y",
                "local_Z",
                "R",
                "G",
                "B",
                "pred_mask",
                "camera_depth",
                "neighbor_distance",
            ],
        },
    )
    print(
        {
            split: sum(row["split"] == split for row in records)
            for split in ["train", "val", "holdout"]
        }
    )


if __name__ == "__main__":
    main()
