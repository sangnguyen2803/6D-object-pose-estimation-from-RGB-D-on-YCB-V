"""Write point clouds, prediction JSON files and BOP CSV results."""

import csv

import numpy as np

from clutterpick.geometry.point_cloud import project_pixels, transform_points, write_ply
from clutterpick.utils.common import read_json, resolve, write_json
from clutterpick.visualization.draw import instance_color


def export_bop(rows, output):
    """Export estimated object-to-camera poses in BOP CSV millimeter units."""
    path = resolve(output) / "bop_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
        for row in rows:
            payload = read_json(resolve(row["output"]) / "poses.json")
            for pose in payload["instances"]:
                if "T_camera_object" not in pose:
                    continue
                matrix = np.asarray(pose["T_camera_object"])
                writer.writerow(
                    [
                        int(row["scene_id"]),
                        row["image_id"],
                        pose["obj_id"],
                        pose["score"],
                        " ".join(f"{value:.9f}" for value in matrix[:3, :3].ravel()),
                        " ".join(f"{value * 1000:.6f}" for value in matrix[:3, 3]),
                        row["timing"]["inference_s_before_reference_evaluation"],
                    ]
                )


def save_pose_clouds(directory, index, points, colors, pose, geometry, obj_id):
    """Save observed points in object coordinates and the CAD in camera coordinates."""
    if "T_camera_object" not in pose:
        return
    transform = np.asarray(pose["T_camera_object"])
    local_points = (points - transform[:3, 3]) @ transform[:3, :3]
    name = f"instance_{index:03d}.ply"
    write_ply(directory / "object_pointclouds" / name, local_points, colors)
    _, cad, _ = geometry.mesh(obj_id)
    write_ply(directory / "aligned_cad" / name, transform_points(cad, transform))


def save_scene_instances(directory, scene, k, image_shape, items):
    """Color scene points by predicted instance; earlier instances take priority."""
    pixels = project_pixels(scene, k, image_shape)
    labels = np.zeros(len(scene), dtype=np.int32)
    colors = np.full((len(scene), 3), 120, dtype=np.uint8)
    for index, item in reversed(list(enumerate(items))):
        keep = item["mask"][pixels[:, 1], pixels[:, 0]]
        labels[keep] = index + 1
        colors[keep] = instance_color(index)
    write_ply(directory / "instances_pointcloud.ply", scene, colors)
    np.save(directory / "point_instance_ids.npy", labels)


def save_frame_results(
    directory,
    provenance,
    items,
    poses,
    pose_diagnostics,
    cloud_diagnostics,
    timing,
    gt_count,
):
    """Write the JSON files shared by inference, the viewer and reports."""
    detections = []
    for item in items:
        detections.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"mask", "_points"}
            }
        )
    write_json(directory / "detection.json", {**provenance, "instances": detections})
    write_json(directory / "poses.json", {**provenance, "instances": poses})

    quality = [
        {
            "instance_id": pose["instance_id"],
            "obj_id": pose["obj_id"],
            "diagnostic": pose.get("pose_quality"),
        }
        for pose in poses
    ]
    write_json(
        directory / "pose_quality.json",
        {
            "instances": quality,
            "uses_gt": False,
            "scope": "geometric support diagnostic; not calibrated correctness or robot safety",
        },
    )
    write_json(
        directory / "pose_diagnostics.json",
        {
            "matched_instances": pose_diagnostics,
            "gt_instance_count": gt_count,
            "scope": "diagnostic ADD/ADD-S/ADD(-S) at 0.1 diameter; not official BOP AR",
        },
    )
    write_json(
        directory / "geometry_diagnostics.json",
        {
            "instances": cloud_diagnostics,
            "scope": "reference evaluation after inference",
        },
    )
    write_json(directory / "timing.json", timing)
