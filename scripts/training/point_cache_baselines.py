"""Label-based scoring only: show what mask membership alone achieves on held-out ROIs."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.utils.common import read_json, resolve, write_json


def main():
    rows = read_json(resolve("data/ycbv/point_training/index.json"))["records"]
    counts = {
        name: np.zeros(3, dtype=np.int64)
        for name in ["all_dilated_roi_points", "predicted_mask_membership"]
    }
    cases = 0
    for row in rows:
        if row["split"] != "holdout":
            continue
        cases += 1
        with np.load(row["path"]) as data:
            truth = data["labels"] > 0
            predictions = {
                "all_dilated_roi_points": np.ones(len(truth), dtype=bool),
                "predicted_mask_membership": data["features"][:, 6] > 0.5,
            }
            for name, predicted in predictions.items():
                counts[name] += np.array(
                    [
                        (predicted & truth).sum(),
                        (predicted & ~truth).sum(),
                        (~predicted & truth).sum(),
                    ]
                )
    metrics = {}
    for name, (tp, fp, fn) in counts.items():
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        metrics[name] = {
            "precision": precision,
            "recall": recall,
            "F1": 2 * precision * recall / max(precision + recall, 1e-9),
            "contamination": 1 - precision,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
        }
    result = {
        "cases": cases,
        "metrics": metrics,
        "scope": "same fixed 512 sampled held-out PBR ROI points; no fitting or threshold selection",
    }
    write_json(resolve("reports/ycbv/research/point_holdout_baselines.json"), result)
    print(result)


if __name__ == "__main__":
    main()
