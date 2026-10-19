"""Pixel-level YCB-V/BOP integrity, units and split checks."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from clutterpick.data.dataset import load_index, load_rgbd
from clutterpick.utils.common import object_ids, read_json, resolve, write_json


def check_frame(frame):
    rgb, depth, k = load_rgbd(frame)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid camera calibration")
    for obj in frame["instances"]:
        visible = np.asarray(Image.open(obj["mask_visib"])) > 0
        full = np.asarray(Image.open(obj["mask"])) > 0
        if (
            visible.shape != depth.shape
            or full.shape != depth.shape
            or np.any(visible & ~full)
        ):
            raise ValueError("Invalid visible/full mask dimensions or containment")
    return {
        "frame_key": frame["key"],
        "rgb_sha256": hashlib.sha256(Path(frame["rgb"]).read_bytes()).hexdigest(),
        "invalid_depth_fraction": float((depth <= 0).mean()),
        "instances": len(frame["instances"]),
    }


def audit_dataset(config, output):
    index, splits = load_index(config)
    errors, records = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            (frame["key"], pool.submit(check_frame, frame)) for frame in index["frames"]
        ]
        for key, job in jobs:
            try:
                records.append(job.result())
            except Exception as error:
                errors.append({"frame_key": key, "error": str(error)})
    groups = {}
    for record in records:
        groups.setdefault(record["rgb_sha256"], []).append(record["frame_key"])
    duplicates = [keys for keys in groups.values() if len(keys) > 1]
    from itertools import combinations

    names = [name for name in ["train", "val", "holdout", "test"] if name in splits]
    split_sets = {
        name: {index["frames"][i]["key"] for i in splits[name]} for name in names
    }
    overlap = {
        f"{a}/{b}": sorted(split_sets[a] & split_sets[b])
        for a, b in combinations(names, 2)
    }
    scene_sets = {
        name: {
            (index["frames"][i]["group"], index["frames"][i]["scene_id"])
            for i in splits[name]
        }
        for name in names
    }
    scene_overlap = {
        f"{a}/{b}": sorted(scene_sets[a] & scene_sets[b])
        for a, b in combinations(names, 2)
    }
    targets_path = resolve(config["data"]["root"]) / "test_targets_bop19.json"
    target_keys = (
        {(entry["scene_id"], entry["im_id"]) for entry in read_json(targets_path)}
        if targets_path.exists()
        else set()
    )
    actual_keys = {
        (int(index["frames"][i]["scene_id"]), index["frames"][i]["image_id"])
        for i in splits["test"]
    }
    result = {
        "dataset": config["data"]["name"],
        "dataset_revision": index["revision"],
        "decoded_frames": len(records),
        "decoded_instances": sum(record["instances"] for record in records),
        "errors": errors,
        "duplicate_rgb_groups": duplicates,
        "cross_split_frame_overlap": overlap,
        "official_test_targets_match": target_keys == actual_keys
        if target_keys
        else None,
        "cross_split_scene_overlap": scene_overlap,
        "object_ids": object_ids(config),
        "depth_scales_mm": sorted(
            {frame["depth_scale_mm"] for frame in index["frames"]}
        ),
        "mean_invalid_depth_fraction": float(
            np.mean([record["invalid_depth_fraction"] for record in records])
        ),
        "train_val_available": bool(splits["train"] and splits["val"]),
        "scope": "all local RGB-D/masks decoded; exact duplicates only; pretrained test-only mode allows empty train/val",
    }
    write_json(output, result)
    if (
        errors
        or any(overlap.values())
        or any(scene_overlap.values())
        or result["official_test_targets_match"] is False
    ):
        raise RuntimeError(f"Dataset audit failed; see {output}")
    return result
