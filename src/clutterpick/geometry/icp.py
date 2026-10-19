"""Point-to-point and point-to-plane ICP, implemented with NumPy/SciPy.

All coordinates and distance thresholds use meters. These functions only fit
CAD points to observed depth; they do not use reference poses or masks.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from clutterpick.geometry.point_cloud import transform_points


def proper_rotation(matrix):
    """Project a matrix onto a rotation, correcting reflections after SVD."""
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def rigid_fit(source, target):
    """Find the rotation and translation between paired 3D points using SVD."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    rotation = proper_rotation(covariance).T
    translation = target_center - rotation @ source_center

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def trimmed_icp(
    model, observed, initial, iterations, max_distance, trim_fraction, tree=None
):
    """Point-to-point ICP: nearest neighbors, distance trimming, then SVD fit."""
    if tree is None:
        tree = cKDTree(model)
    pose = initial.copy()

    for _ in range(iterations):
        # 1. Đưa điểm quan sát về hệ CAD để tìm điểm gần nhất.
        local_points = (observed - pose[:3, 3]) @ pose[:3, :3]
        distances, indices = tree.query(local_points)

        # 2. Bỏ các cặp quá xa hoặc nằm ngoài phần dữ liệu giữ lại.
        cutoff = min(max_distance, np.quantile(distances, trim_fraction))
        keep = distances <= cutoff
        if keep.sum() < 12:
            break

        # 3. Tìm pose mới từ các cặp điểm còn lại.
        updated = rigid_fit(model[indices[keep]], observed[keep])
        translation_change = np.linalg.norm(updated[:3, 3] - pose[:3, 3])
        rotation_change = np.linalg.norm(updated[:3, :3] - pose[:3, :3])
        change = translation_change + rotation_change * 0.01
        pose = updated
        if change < 1e-5:
            break

    distances, _ = tree.query((observed - pose[:3, 3]) @ pose[:3, :3])
    trimmed = np.sort(distances)[: max(1, int(len(distances) * trim_fraction))]
    rmse = float(np.sqrt(np.mean(trimmed**2)))
    coverage = float(np.mean(distances < 0.01))
    return pose, rmse, coverage


def point_plane(model, normals, observed, initial, cfg, tree=None):
    """Point-to-plane ICP with Huber weights and damped least squares.

    The residual measures distance along the CAD surface normal. Each update
    rotates around the object origin and is limited to 2 degrees and 10 mm.
    """
    if tree is None:
        tree = cKDTree(model)
    pose = initial.copy()
    radius = max(
        float(np.sqrt(np.mean(np.sum((model - model.mean(0)) ** 2, axis=1)))),
        0.02,
    )
    eigenvalues = np.zeros(6)
    iterations = 0

    for iteration in range(cfg.get("robust_iterations", 20)):
        # 1. Ghép điểm quan sát với bề mặt CAD gần nhất.
        local_points = (observed - pose[:3, 3]) @ pose[:3, :3]
        distances, indices = tree.query(local_points)
        cutoff = min(cfg["correspondence_limit_m"], float(np.quantile(distances, 0.85)))
        keep = distances <= cutoff
        if keep.sum() < 30:
            break

        targets = observed[keep]
        source = transform_points(model[indices[keep]], pose)
        normal = normals[indices[keep]] @ pose[:3, :3].T
        normal_lengths = np.linalg.norm(normal, axis=1)
        valid = normal_lengths > 1e-8
        targets = targets[valid]
        source = source[valid]
        normal = normal[valid] / normal_lengths[valid, None]
        if len(source) < 30:
            break

        # 2. Tính sai số theo normal; Huber giảm ảnh hưởng của điểm nhiễu.
        residual = np.sum(normal * (targets - source), axis=1)
        delta = cfg.get("huber_delta_m", 0.004)
        weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12))
        jacobian = np.column_stack(
            (np.cross(source - pose[:3, 3], normal) / radius, normal)
        )

        # 3. Giải hệ 6 ẩn: ba thành phần xoay và ba thành phần tịnh tiến.
        information = (jacobian.T * weights) @ jacobian / max(weights.sum(), 1)
        eigenvalues = np.linalg.eigvalsh(information)
        rhs = jacobian.T @ (weights * residual) / max(weights.sum(), 1)
        damping = np.eye(6) * cfg.get("damping", 0.01)
        step = np.linalg.solve(information + damping, rhs)

        # 4. Giới hạn bước cập nhật rồi áp dụng vào pose hiện tại.
        angle = step[:3] / radius
        angle *= min(1.0, np.deg2rad(2) / max(np.linalg.norm(angle), 1e-12))
        translation = step[3:]
        translation *= min(1.0, 0.01 / max(np.linalg.norm(translation), 1e-12))
        pose[:3, :3] = Rotation.from_rotvec(angle).as_matrix() @ pose[:3, :3]
        pose[:3, 3] += translation
        iterations = iteration + 1
        if np.linalg.norm(translation) + radius * np.linalg.norm(angle) < 1e-5:
            break

    diagnostics = {
        "iterations": iterations,
        "information_eigenvalues": eigenvalues.tolist(),
        "observable_directions": int(np.sum(eigenvalues > 0.01)),
    }
    return pose, diagnostics
