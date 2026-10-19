"""YCB-V instance segmentation, visible point clouds and known-object 6D pose.

Reference masks/poses appear only in explicit oracle diagnostics or evaluation
after predictions. Grasp planning is outside the current project scope.
"""

import time

import numpy as np
from PIL import Image

from clutterpick.data.dataset import load_rgbd
from clutterpick.evaluation.metrics import evaluate_frame
from clutterpick.geometry.point_cloud import (
    backproject,
    instance_cloud,
    project_pixels,
    voxel_downsample,
    write_ply,
)
from clutterpick.geometry.pose_estimation import GeometricPoseEstimator
from clutterpick.utils.common import resolve
from clutterpick.utils.outputs import (
    save_frame_results,
    save_pose_clouds,
    save_scene_instances,
)
from clutterpick.visualization.draw import instance_color, overlay, plot_scene


def reference_detections(frame):
    masks, boxes, labels = [], [], []
    for obj in frame.get("instances", []):
        mask = np.asarray(Image.open(obj["mask_visib"])) > 0
        y, x = np.nonzero(mask)
        if len(x):
            masks.append(mask)
            boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
            labels.append(obj["obj_id"])
    h, w = np.asarray(Image.open(frame["rgb"])).shape[:2]
    return {
        "boxes": np.asarray(boxes).reshape(-1, 4),
        "labels": np.asarray(labels),
        "scores": np.ones(len(masks)),
        "masks": np.stack(masks) if masks else np.zeros((0, h, w), dtype=bool),
    }


def get_estimator(config, backend, runtime, device):
    if backend in {"cosypose", "cosypose_rgbd"}:
        if "cosypose" not in runtime:
            from clutterpick.models.cosypose import CosyPoseEstimator

            runtime["cosypose"] = CosyPoseEstimator(config, device)
        return runtime["cosypose"]
    if backend == "geometric":
        if "geometric" not in runtime:
            runtime["geometric"] = GeometricPoseEstimator(config)
        return runtime["geometric"]
    raise ValueError(f"Unsupported YCB-V pose backend: {backend}")


def depth_mask(mask, box, mode):
    """Choose the predicted mask or its bounding box for depth extraction."""
    if mode == "mask":
        return mask
    if mode != "bbox":
        raise ValueError("Depth extraction must be mask or bbox")
    selected = np.zeros_like(mask)
    x1, y1, x2, y2 = np.asarray(box).astype(int)
    height, width = mask.shape
    selected[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return selected


def clean_instance_points(
    points, colors, mask, k, label, cleaner, min_points, diagnostics
):
    """Apply the optional learned filter and update the cloud diagnostics."""
    pixels = project_pixels(points, k, mask.shape)
    membership = mask[pixels[:, 1], pixels[:, 0]]
    keep, cleaning_diagnostics = cleaner.filter(
        points,
        colors,
        label,
        membership=membership,
        min_points=min_points,
    )
    diagnostics["retained_points_before_learned"] = len(points)
    points, colors = points[keep], colors[keep]
    diagnostics["retained_points"] = len(points)
    diagnostics["learned_3d_refinement"] = cleaning_diagnostics
    return points, colors


def run_frame(
    frame,
    config,
    detector=None,
    output="artifacts/ycbv/inference",
    oracle_masks=False,
    pose_backend=None,
    plots=True,
    depth_extraction="mask",
    pose_checkpoint=None,
    runtime=None,
    export_clouds=True,
):
    """Run RGB-D loading, detection, geometry, pose estimation and evaluation."""
    if not oracle_masks and detector is None:
        raise ValueError("Prediction mode requires a detector")
    directory = resolve(output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "masks").mkdir(exist_ok=True)

    # 1. Đọc ảnh RGB-D và chạy detector.
    started = time.perf_counter()
    rgb, depth, k = load_rgbd(frame)
    timing = {"preprocessing_s": time.perf_counter() - started}
    begun = time.perf_counter()
    if oracle_masks:
        prediction = reference_detections(frame)
    elif getattr(detector, "requires_depth", False):
        prediction = detector.predict(rgb, depth=depth, k=k)
    else:
        prediction = detector.predict(rgb)
    timing["detection_s"] = time.perf_counter() - begun

    # 2. Dựng point cloud toàn cảnh khi cần xuất file hoặc vẽ hình.
    begun = time.perf_counter()
    need_scene = (
        export_clouds
        or plots
        or not config.get("runtime", {}).get("lazy_scene_geometry", False)
    )
    if need_scene:
        scene, pixels = backproject(
            depth,
            k,
            None,
            config["geometry"]["depth_min_m"],
            config["geometry"]["depth_max_m"],
        )
        scene, colors = voxel_downsample(
            scene, config["geometry"]["voxel_size_m"], rgb[pixels[:, 0], pixels[:, 1]]
        )
    else:
        scene, colors = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    timing["scene_geometry_s"] = time.perf_counter() - begun
    if export_clouds:
        write_ply(directory / "pointcloud.ply", scene, colors)

    # 3. Lấy pose estimator và point filter từ cache dùng chung giữa các frame.
    backend = pose_backend or config["pose"]["backend"]
    cache = runtime if runtime is not None else {}
    device = str(detector.device) if detector is not None else "auto"
    estimator = get_estimator(config, backend, cache, device)
    cleaning = config.get("point_refiner", {})
    cleaner = None
    if cleaning.get("enabled", False):
        if "point_cleaner" not in cache:
            from clutterpick.models.point_network import PointCleaner

            cache["point_cleaner"] = PointCleaner(cleaning["checkpoint"], device)
        cleaner = cache["point_cleaner"]
    geometry = estimator.geometry if hasattr(estimator, "geometry") else estimator
    items, poses = [], []
    batch_size = config.get("runtime", {}).get("pose_batch_size", 1)
    batched = (
        backend.startswith("cosypose")
        and batch_size > 1
        and hasattr(estimator, "estimate_rgbd_batch")
    )
    batch_masks, batch_colors = [], []
    timing.update(instance_geometry_s=0.0, pose_s=0.0)

    # 4. Tạo point cloud từng vật thể; dự đoán pose ngay nếu không chạy batch.
    for i, (box, label, score, mask) in enumerate(
        zip(
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
            prediction["masks"],
        )
    ):
        instance_id = f"{frame['key']}/pred_{i:03d}"
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            directory / "masks" / f"instance_{i:03d}.png"
        )
        selected = depth_mask(mask, box, depth_extraction)
        begun = time.perf_counter()
        points, point_colors, diagnostics = instance_cloud(
            depth, k, selected, rgb, config
        )
        if cleaner is not None:
            points, point_colors = clean_instance_points(
                points,
                point_colors,
                mask,
                k,
                int(label),
                cleaner,
                config["geometry"]["min_points"],
                diagnostics,
            )
        timing["instance_geometry_s"] += time.perf_counter() - begun
        cloud_path = f"pointclouds/instance_{i:03d}.ply"
        if export_clouds:
            write_ply(directory / cloud_path, points, point_colors)
        item = {
            "instance_id": instance_id,
            "obj_id": int(label),
            "score": float(score),
            "box": box,
            "mask": mask,
            "_points": points,
            "geometry": diagnostics,
            "mask_path": f"masks/instance_{i:03d}.png",
            "pointcloud_path": cloud_path if export_clouds else None,
            "instance_color": instance_color(i),
        }
        items.append(item)
        if batched:
            batch_masks.append(selected)
            batch_colors.append(point_colors)
            continue
        begun = time.perf_counter()
        if backend.startswith("cosypose"):
            pose = estimator.estimate_rgbd(
                rgb,
                depth,
                k,
                box,
                selected,
                int(label),
                points,
                depth_refinement=backend == "cosypose_rgbd",
            )
        else:
            pose = estimator.estimate(points, int(label))
        timing["pose_s"] += time.perf_counter() - begun
        poses.append(
            {
                "instance_id": instance_id,
                "obj_id": int(label),
                "score": float(score),
                **pose,
            }
        )
        if export_clouds:
            save_pose_clouds(
                directory, i, points, point_colors, pose, geometry, int(label)
            )

    # 5. Chạy pose theo batch nếu cấu hình đã bật.
    if batched:
        begun = time.perf_counter()
        results = estimator.estimate_rgbd_batch(
            rgb,
            depth,
            k,
            [item["box"] for item in items],
            batch_masks,
            [item["obj_id"] for item in items],
            [item["_points"] for item in items],
            depth_refinement=backend == "cosypose_rgbd",
            batch_size=batch_size,
        )
        if len(results) != len(items):
            raise RuntimeError("Pose batch output lost instances")
        timing["pose_s"] += time.perf_counter() - begun
        for i, (item, pose) in enumerate(zip(items, results)):
            poses.append(
                {
                    "instance_id": item["instance_id"],
                    "obj_id": item["obj_id"],
                    "score": item["score"],
                    **pose,
                }
            )
            if export_clouds:
                save_pose_clouds(
                    directory,
                    i,
                    item["_points"],
                    batch_colors[i],
                    pose,
                    geometry,
                    item["obj_id"],
                )
    if export_clouds:
        save_scene_instances(directory, scene, k, depth.shape, items)
    timing["inference_s_before_reference_evaluation"] = time.perf_counter() - started

    # 6. Chỉ đọc GT để đánh giá sau khi đã dự đoán xong mọi pose.
    pose_diagnostics, cloud_diagnostics = evaluate_frame(
        items, poses, frame, geometry, k, config, depth.shape
    )

    # 7. Lưu kết quả và ảnh minh họa.
    provenance = {
        "dataset": config["data"].get("name"),
        "dataset_revision": config["data"]["revision"],
        "frame_key": frame["key"],
        "units": "m",
        "coordinate_frame": "camera_BOP_x_right_y_down_z_forward",
        "uses_ground_truth_masks": oracle_masks,
        "uses_ground_truth_pose_for_inference": False,
        "ground_truth_pose_used_only_for_evaluation": bool(frame.get("instances")),
        "detector_checkpoint": None if oracle_masks else detector.checkpoint,
        "pose_backend": backend,
        "pose_checkpoints": getattr(estimator, "checkpoints", None),
        "point_clouds_exported": export_clouds,
        "point_instance_labels_source": ("reference" if oracle_masks else "predicted")
        + " visible 2D masks backprojected through measured depth",
        "learned_point_refinement_enabled": cleaner is not None,
        "learned_point_checkpoint": getattr(cleaner, "checkpoint", None),
        "scene_instance_labels_before_learned_filter": True,
        "grasp_planning_performed": False,
        "config": config,
    }
    save_frame_results(
        directory,
        provenance,
        items,
        poses,
        pose_diagnostics,
        cloud_diagnostics,
        timing,
        len(frame.get("instances", [])),
    )
    overlay(rgb, items, k, directory / "final_visualization.png", oracle_masks, poses)
    if plots:
        plot_scene(scene, colors, directory / "scene_3d.png", poses, oracle_masks)
    if runtime is None and hasattr(estimator, "close"):
        estimator.close()
    return {
        "frame_key": frame["key"],
        "scene_id": frame["scene_id"],
        "image_id": frame["image_id"],
        "detections": len(items),
        "poses_ok": sum(pose["status"] == "ok" for pose in poses),
        "gt_instances": len(frame.get("instances", [])),
        "timing": timing,
        "output": str(directory),
        "oracle_masks": oracle_masks,
        "pose_backend": backend,
        "pose_diagnostics": pose_diagnostics,
    }
