"""Instance overlays and visible RGB-D scenes with estimated pose axes."""

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def instance_color(index):
    return tuple(
        int(channel * 255)
        for channel in colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.72, 0.95)
    )


def project(points, k):
    points = np.asarray(points)
    projected = points @ k.T
    valid = points[:, 2] > 0.01
    return projected[:, :2] / np.maximum(projected[:, 2:3], 0.01), valid


def overlay(rgb, items, k, path, reference=False, poses=None):
    array = rgb.copy()
    for item in reversed(items):
        color = np.asarray(item["instance_color"])
        array[item["mask"]] = (array[item["mask"]] * 0.5 + color * 0.5).astype(np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    pose_lookup = {pose["instance_id"]: pose for pose in poses or []}
    for item in items:
        box = item["box"]
        draw.rectangle(tuple(box), outline=tuple(item["instance_color"]), width=2)
        draw.text(
            (box[0], max(23, box[1] - 12)),
            f"obj {item['obj_id']} {item['score']:.2f}",
            fill="white",
        )
        quality = pose_lookup.get(item["instance_id"], {}).get("pose_quality")
        if quality is not None:
            label = (
                "DEPTH: REVIEW" if quality["requires_review"] else "DEPTH: SUPPORTED"
            )
            draw.text(
                (box[0] + 3, box[1] + 3),
                label,
                fill="yellow" if quality["requires_review"] else "white",
            )
    for pose in poses or []:
        if "T_camera_object" not in pose:
            continue
        matrix = np.asarray(pose["T_camera_object"])
        axes = np.vstack([matrix[:3, 3], matrix[:3, 3] + matrix[:3, :3].T * 0.04])
        uv, valid = project(axes, k)
        for i, color in enumerate([(255, 60, 60), (60, 255, 60), (60, 120, 255)]):
            if valid[0] and valid[i + 1] and np.isfinite(uv[[0, i + 1]]).all():
                draw.line([tuple(uv[0]), tuple(uv[i + 1])], fill=color, width=3)
    draw.rectangle((0, 0, image.width, 22), fill=(20, 20, 20))
    draw.text(
        (6, 5),
        "REFERENCE MASK DIAGNOSTIC"
        if reference
        else "INSTANCE SEGMENTATION / ESTIMATED 6D POSES",
        fill="white",
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def plot_scene(points, colors, path, poses=None, reference=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(projection="3d")
    stride = max(1, len(points) // 12000)
    if len(points):
        cloud = points[::stride]
        axes.scatter(
            cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors[::stride] / 255, s=0.5
        )
    for pose in poses or []:
        if "T_camera_object" in pose:
            matrix = np.asarray(pose["T_camera_object"])
            for i, color in enumerate(["red", "green", "blue"]):
                axes.quiver(*matrix[:3, 3], *(matrix[:3, i] * 0.04), color=color)
    axes.set(
        xlabel="camera X (m)",
        ylabel="camera Y (m)",
        zlabel="camera Z (m)",
        title="REFERENCE MASK DIAGNOSTIC"
        if reference
        else "Visible RGB-D cloud and estimated 6D poses",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
