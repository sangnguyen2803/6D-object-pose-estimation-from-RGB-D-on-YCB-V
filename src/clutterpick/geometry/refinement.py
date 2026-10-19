"""Select between ICP results and the initial RGB pose, then report fit quality.

No GT pose, mask or class correction enters this module. Support is a geometric
diagnostic, never a calibrated probability of correct 6D pose.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from clutterpick.geometry.icp import point_plane, trimmed_icp


def fit_metrics(model, observed, pose, trim=0.8, tree=None):
    """Measure trimmed RMSE and the fraction of observed points near the CAD."""
    if not len(observed):
        return {"rmse_m": None, "coverage_10mm": 0.0, "support_40mm": 0.0, "points": 0}
    tree = tree if tree is not None else cKDTree(model)
    distances, _ = tree.query((observed - pose[:3, 3]) @ pose[:3, :3])
    selected = np.sort(distances)[: max(1, int(len(distances) * trim))]
    return {
        "rmse_m": float(np.sqrt(np.mean(selected**2))),
        "coverage_10mm": float(np.mean(distances < 0.01)),
        "support_40mm": float(np.mean(distances < 0.04)),
        "points": len(observed),
    }


def bounded(candidate, initial, cfg):
    """Check positive depth and the allowed translation/rotation update."""
    movement = float(np.linalg.norm(candidate[:3, 3] - initial[:3, 3]))
    angle = float(
        np.degrees(
            Rotation.from_matrix(candidate[:3, :3] @ initial[:3, :3].T).magnitude()
        )
    )
    valid = (
        np.isfinite(candidate).all()
        and candidate[2, 3] > 0
        and movement < cfg.get("depth_max_translation_update_m", 0.05)
        and angle < cfg.get("depth_max_rotation_update_degrees", 10)
    )
    return valid, movement, angle


def annotate_observability(diagnostic):
    """Conservative review flag from the robust candidate's measured constraints."""
    candidates = [
        candidate
        for candidate in diagnostic["candidates"]
        if candidate["method"] == "huber_point_to_plane"
    ]
    rank = candidates[0].get("observable_directions") if candidates else None
    quality = diagnostic["quality"]
    quality["robust_candidate_constraint_rank"] = rank
    quality["review_reasons"] = []
    if quality["state"] != "supported_geometry":
        quality["review_reasons"].append(quality["state"])
    if rank is not None and rank < 6:
        quality["review_reasons"].append(
            "robust_candidate_has_underconstrained_directions"
        )
    quality["requires_review"] = bool(quality["review_reasons"])
    return diagnostic


def evaluate_candidate(name, candidate, initial, before, model, observed, cfg, tree):
    """Check fit improvement and the maximum allowed pose change."""
    metrics = fit_metrics(model, observed, candidate, cfg["trim_fraction"], tree)
    valid, translation, angle = bounded(candidate, initial, cfg)
    improvement = (before["rmse_m"] - metrics["rmse_m"]) / max(before["rmse_m"], 1e-12)
    eligible = valid and improvement > cfg.get(
        "depth_min_relative_fit_improvement", 0.1
    )
    if name == "huber_point_to_plane":
        eligible = (
            eligible
            and metrics["support_40mm"] >= 0.5
            and metrics["coverage_10mm"] >= before["coverage_10mm"] - 0.02
        )
    return {
        "method": name,
        "eligible": bool(eligible),
        "relative_rmse_improvement": float(improvement),
        "translation_update_m": translation,
        "rotation_update_degrees": angle,
        **metrics,
    }


def refine(model, normals, points, initial, cfg, seed=42, min_points=40, tree=None):
    """Try both ICP methods and keep the best eligible pose, or the RGB pose."""
    # 1. Bỏ depth không hợp lệ và lấy mẫu có seed cố định.
    observed = np.asarray(points)
    observed = observed[np.isfinite(observed).all(axis=1)]
    rng = np.random.default_rng(seed)
    sample_size = min(len(observed), cfg["observed_points"])
    observed = observed[rng.choice(len(observed), sample_size, replace=False)]
    if tree is None:
        tree = cKDTree(model)
    before = fit_metrics(model, observed, initial, cfg["trim_fraction"], tree)

    # 2. Chạy hai thuật toán trên cùng tập điểm và pose ban đầu.
    candidates = []
    selected = initial.copy()
    selected_name = "rgb"
    if len(observed) >= min_points:
        point_pose, _, _ = trimmed_icp(
            model,
            observed,
            initial,
            cfg["icp_iterations"],
            cfg["correspondence_limit_m"],
            cfg["trim_fraction"],
            tree=tree,
        )
        plane_pose, plane_details = point_plane(
            model, normals, observed, initial, cfg, tree=tree
        )
        options = [
            ("trimmed_point_to_point", point_pose, {}),
            ("huber_point_to_plane", plane_pose, plane_details),
        ]

        # 3. Chỉ chọn pose đạt điều kiện và có RMSE nhỏ nhất.
        best_rmse = None
        for name, pose, details in options:
            candidate = evaluate_candidate(
                name, pose, initial, before, model, observed, cfg, tree
            )
            candidate.update(details)
            candidates.append(candidate)
            if candidate["eligible"]:
                if best_rmse is None or candidate["rmse_m"] < best_rmse:
                    best_rmse = candidate["rmse_m"]
                    selected = pose
                    selected_name = name

    # 4. Ghi mức hỗ trợ hình học và các lý do cần xem lại.
    quality = fit_metrics(model, observed, selected, cfg["trim_fraction"], tree)
    supported = (
        quality["points"] >= min_points
        and quality["coverage_10mm"] >= cfg.get("min_coverage", 0.65)
        and quality["rmse_m"] <= cfg.get("max_observed_rmse_m", 0.01)
    )
    if supported:
        state = "supported_geometry"
    elif len(observed) < min_points:
        state = "insufficient_depth"
    else:
        state = "weak_geometry"

    diagnostic = {
        "method": "robust_candidate_selection",
        "selected_candidate": selected_name,
        "accepted": selected_name != "rgb",
        "before": before,
        "after": quality,
        "candidates": candidates,
        "quality": {
            "state": state,
            "requires_review": not supported,
            **quality,
            "calibrated_probability": False,
            "limitation": "symmetry and wrong identity may fit geometry despite incorrect pose",
        },
    }
    return selected, annotate_observability(diagnostic)
