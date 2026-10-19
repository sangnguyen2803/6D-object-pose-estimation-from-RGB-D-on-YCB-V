"""Aggregate measured poses; missing detections remain failures in GT denominator."""

from collections import Counter

import numpy as np

METRICS = ("ADD_success_0_1d", "ADD_S_success_0_1d", "ADD_or_S_success_0_1d")


def summarize(rows):
    gt = sum(row["gt_instances"] for row in rows)
    matched = [item for row in rows for item in row["pose_diagnostics"]]
    times = [row["timing"]["inference_s_before_reference_evaluation"] for row in rows]
    counts = {key: sum(bool(item[key]) for item in matched) for key in METRICS}
    return {
        "frames": len(rows),
        "gt_instances": gt,
        "matched_pose_instances": len(matched),
        "detections": sum(row["detections"] for row in rows),
        "success_counts": counts,
        "recall": {key: value / gt if gt else None for key, value in counts.items()},
        "median_inference_s_excluding_first": float(np.median(times[1:] or times))
        if times
        else None,
        "scope": "diagnostic ADD/ADD-S/ADD(-S), 0.1 diameter, all GT instances; not official BOP AR",
    }


def per_object(rows, index):
    lookup = {frame["key"]: frame for frame in index["frames"]}
    totals = Counter(
        obj["obj_id"] for row in rows for obj in lookup[row["frame_key"]]["instances"]
    )
    matches = [item for row in rows for item in row["pose_diagnostics"]]
    return {
        str(obj): {
            "gt": count,
            **{
                key: sum(item["obj_id"] == obj and item[key] for item in matches)
                / count
                for key in METRICS
            },
        }
        for obj, count in sorted(totals.items())
    }
