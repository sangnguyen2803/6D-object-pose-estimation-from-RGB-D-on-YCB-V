"""Train point foreground classifier; choose threshold/checkpoint on validation."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.models.point_network import PointForegroundNetwork
from clutterpick.utils.common import read_json, resolve, set_seed, write_json


class Dataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.rows[index]["path"]) as data:
            return (
                torch.from_numpy(data["features"]),
                torch.from_numpy(data["labels"]),
                int(data["obj_id"]),
            )


def measure(model, loader, device):
    scores = []
    truths = []
    with torch.inference_mode():
        for features, labels, objects in loader:
            scores.append(
                model(features.to(device), objects.to(device))
                .sigmoid()
                .cpu()
                .numpy()
                .ravel()
            )
            truths.append(labels.numpy().ravel())
    scores = np.concatenate(scores)
    truth = np.concatenate(truths) > 0
    candidates = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & truth))
        fp = int(np.sum(predicted & ~truth))
        fn = int(np.sum(~predicted & truth))
        candidates.append(
            {
                "threshold": threshold,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "F1": 2 * tp / max(2 * tp + fp + fn, 1),
                "contamination": fp / max(tp + fp, 1),
            }
        )
    return max(candidates, key=lambda row: row["F1"]), candidates


def main():
    torch.set_num_threads(4)
    set_seed(42)
    records = read_json(resolve("data/ycbv/point_training/index.json"))["records"]
    groups = {
        split: [row for row in records if row["split"] == split]
        for split in ["train", "val", "holdout"]
    }
    assert all(groups.values())
    assert not (
        {row["frame_key"] for row in groups["train"]}
        & {row["frame_key"] for row in groups["val"]}
    )
    loaders = {
        split: torch.utils.data.DataLoader(
            Dataset(rows), batch_size=16, shuffle=split == "train", num_workers=0
        )
        for split, rows in groups.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointForegroundNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    directory = resolve("artifacts/ycbv/research/point_network")
    directory.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(20):
        model.train()
        total = 0.0
        started = time.perf_counter()
        for features, labels, objects in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            features, labels, objects = (
                features.to(device),
                labels.to(device),
                objects.to(device),
            )
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(features, objects)
                weights = torch.where(labels > 0, 1.0, 2.0)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, weight=weights
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        model.eval()
        validation, _ = measure(model, loaders["val"], device)
        record = {
            "epoch": epoch + 1,
            "loss": total / len(loaders["train"]),
            "seconds": time.perf_counter() - started,
            **validation,
        }
        history.append(record)
        write_json(directory / "history.json", history)
        if validation["F1"] > best:
            best = validation["F1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "threshold": validation["threshold"],
                    "sample_points": 512,
                    "train_frame_keys": sorted(
                        {row["frame_key"] for row in groups["train"]}
                    ),
                    "validation_frame_keys": sorted(
                        {row["frame_key"] for row in groups["val"]}
                    ),
                    "trained_on_test": False,
                    "history": history.copy(),
                },
                directory / "best.pt",
            )
        print("Point epoch", record, flush=True)
    payload = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    selected, curve = measure(model, loaders["holdout"], device)
    fixed = next(row for row in curve if row["threshold"] == payload["threshold"])
    write_json(
        resolve("reports/ycbv/research/point_holdout.json"),
        {
            "fixed_validation_threshold": payload["threshold"],
            "metrics": fixed,
            "thresholds_diagnostic_only": curve,
            "cases": len(groups["holdout"]),
            "scope": "independent PBR scenes for the new point network; not real test accuracy",
        },
    )
    print("Point heldout", fixed, flush=True)


if __name__ == "__main__":
    main()
