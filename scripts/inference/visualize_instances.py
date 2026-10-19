"""Export measured point clouds as PNG and a self-contained interactive HTML."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    payload = json.loads((directory / "detection.json").read_text(encoding="utf-8"))
    poses = json.loads((directory / "poses.json").read_text(encoding="utf-8"))[
        "instances"
    ]
    scene = trimesh.load(directory / "pointcloud.ply", process=False)
    background = np.asarray(scene.vertices)[:: max(1, len(scene.vertices) // 8000)]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=background[:, 0],
            y=background[:, 1],
            z=background[:, 2],
            mode="markers",
            marker={"size": 1, "color": "#aaaaaa", "opacity": 0.25},
            name="Scene",
            visible="legendonly",
        )
    )
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*background.T, c="lightgray", s=0.3, alpha=0.3)
    observed = []
    for i, item in enumerate(payload["instances"]):
        cloud = trimesh.load(directory / item["pointcloud_path"], process=False)
        points = np.asarray(cloud.vertices)
        points = points[:: max(1, len(points) // 4000)]
        color = np.asarray(item["instance_color"]) / 255
        if len(points):
            observed.append(points)
            label = f"Instance {i}: object {item['obj_id']}"
            ax.scatter(*points.T, s=2, color=color, label=label)
            hexcolor = "#" + "".join(f"{int(v):02x}" for v in item["instance_color"])
            figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker={"size": 2, "color": hexcolor},
                    name=label,
                )
            )
        pose = poses[i]
        if "T_camera_object" in pose:
            matrix = np.asarray(pose["T_camera_object"])
            origin = matrix[:3, 3]
            for axis, c in enumerate(["red", "green", "blue"]):
                end = origin + 0.04 * matrix[:3, axis]
                xyz = np.stack([origin, end])
                figure.add_trace(
                    go.Scatter3d(
                        x=xyz[:, 0],
                        y=xyz[:, 1],
                        z=xyz[:, 2],
                        mode="lines",
                        line={"color": c, "width": 5},
                        legendgroup=f"pose{i}",
                        name=f"Pose {i} axis {axis}",
                        showlegend=axis == 0,
                    )
                )
            cad = trimesh.load(
                directory / "aligned_cad" / f"instance_{i:03d}.ply", process=False
            )
            vertices = np.asarray(cad.vertices)[:: max(1, len(cad.vertices) // 3000)]
            figure.add_trace(
                go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode="markers",
                    marker={"size": 1, "color": "orange"},
                    name=f"Aligned CAD {i}",
                    visible="legendonly",
                )
            )
    ax.set(
        xlabel="Camera X (m)",
        ylabel="Camera Y (m)",
        zlabel="Camera Z (m)",
        title="Predicted visible instance point clouds",
    )
    if observed:
        low, high = np.quantile(np.concatenate(observed), [0.01, 0.99], axis=0)
        padding = np.maximum((high - low) * 0.15, 0.02)
        ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
        ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
        ax.set_zlim(low[2] - padding[2], high[2] + padding[2])
        ax.set_title("Visible instance clouds (view focused on 1st–99th percentiles)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "instances_3d.png", dpi=160)
    plt.close(fig)
    figure.update_layout(
        title="YCBPose: visible depth and estimated object frames",
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        legend={"title": "Click to toggle instances / CAD"},
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
    )
    figure.write_html(
        directory / "instances_3d.html", include_plotlyjs=True, full_html=True
    )
    print(directory / "instances_3d.html")


if __name__ == "__main__":
    main()
