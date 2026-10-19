"""Check metric scale against CAD using reference pose; diagnostic only."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2] / "audit"


def read_vertices(path):
    with path.open("rb") as f:
        lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        count = int(
            next(l for l in lines if l.startswith("element vertex")).split()[-1]
        )
        if "format binary_little_endian 1.0" in lines:
            # All supplied binary CAD meshes have six float32 vertex properties.
            start = lines.index(f"element vertex {count}")
            properties = []
            for line in lines[start + 1 :]:
                if line.startswith("element"):
                    break
                if line.startswith("property"):
                    _, dtype, name = line.split()
                    if dtype != "float":
                        raise ValueError("Unsupported CAD property")
                    properties.append(name)
            values = np.frombuffer(
                f.read(count * len(properties) * 4), dtype="<f4"
            ).reshape(count, len(properties))
            return values[:, [properties.index(x) for x in ["x", "y", "z"]]]
        rows = [f.readline().decode("ascii") for _ in range(count)]
        return np.array([[float(x) for x in l.split()[:3]] for l in rows])


def main():
    output = []
    for sample_path in sorted((ROOT / "samples").glob("*/sample_audit.json")):
        sample = json.loads(sample_path.read_text())
        gt = sample["first_instance_gt"]
        verts = read_vertices(
            ROOT / "cad/IndustryShapes_cad_models" / f"obj_{gt['obj_id']:06d}.ply"
        )
        r = np.array(gt["cam_R_m2c"]).reshape(3, 3)
        t = np.array(gt["cam_t_m2c"])
        transformed = (verts @ r.T + t) / 1000
        with (sample_path.parent / "visible_points_gt_mask.ply").open() as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            points = np.loadtxt(f)[:, :3]
        if len(points) != sample["valid_depth_visible_pixels"]:
            raise ValueError("Point cloud count differs from sample audit")
        distances, _ = cKDTree(transformed).query(points)
        check = {
            "sample": sample["tag"],
            "object_id": gt["obj_id"],
            "cad_vertex_count": len(verts),
            "valid_points": len(points),
            "point_to_nearest_cad_vertex_mm_p50_p90_p95": (
                np.percentile(distances, [50, 90, 95]) * 1000
            ).tolist(),
            "diagnostic_uses_gt_pose_and_gt_mask": True,
            "note": "One-sided distance to vertices, not mesh-surface distance or pose prediction error. Includes sensor/label error.",
        }
        output.append(check)
    (ROOT / "geometry_smoke_checks.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
