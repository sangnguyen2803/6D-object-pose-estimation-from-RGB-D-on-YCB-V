import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from clutterpick.geometry.pose_estimation import GeometricPoseEstimator
from clutterpick.geometry.refinement import refine
from clutterpick.models.cosypose import CosyPoseEstimator
from clutterpick.utils.common import load_config


def test_pose_batch_chunks_keep_order_duplicate_classes_and_image_calibration():
    estimator = object.__new__(CosyPoseEstimator)
    estimator.device = torch.device("cpu")
    estimator.config = load_config("configs/enhanced.yaml")

    class Meshes:
        def select(self, labels):
            from types import SimpleNamespace

            return SimpleNamespace(
                sample_points=lambda count, deterministic: torch.zeros(
                    len(labels), count, 3
                )
            )

    estimator.mesh_db = Meshes()
    calls = []

    def initialize(boxes, points, k):
        poses = torch.eye(4)[None].repeat(len(boxes), 1, 1)
        poses[:, 0, 3] = boxes[:, 0] / 10000
        poses[:, 2, 3] = 1
        return poses

    estimator.initialize = initialize

    def network(images, k, labels, poses, n_iterations):
        assert images.shape == (len(labels), 3, 12, 16) and k.shape == (
            len(labels),
            3,
            3,
        )
        assert torch.all(k[:, 0, 0] == 100) and images.max() <= 1
        calls.append(list(labels))
        return {f"iteration={n_iterations}": {"TCO_output": poses.clone()}}

    estimator.networks = {"coarse": network, "refiner": network}
    estimator.finalize_rgbd = lambda pose, points, obj_id, used: {
        "obj_id": obj_id,
        "T_camera_object": pose,
    }
    labels = [1, 19, 19, 14, 1]
    boxes = np.array([[i, 0, 10, 10] for i in range(5)])
    results = estimator.estimate_rgbd_batch(
        np.full((12, 16, 3), 255, dtype=np.uint8),
        np.ones((12, 16)),
        np.diag([100, 100, 1]),
        boxes,
        [np.ones((12, 16), dtype=bool)] * 5,
        labels,
        [np.ones((40, 3))] * 5,
        batch_size=2,
    )
    assert [result["obj_id"] for result in results] == labels
    assert [result["neural_pose_batch_size"] for result in results] == [2, 2, 2, 2, 1]
    np.testing.assert_allclose(
        [result["T_camera_object"][0, 3] for result in results], np.arange(5) / 10000
    )
    assert len(calls) == 6 and calls[2] == ["obj_000019", "obj_000014"]
    assert (
        estimator.estimate_rgbd_batch(None, None, None, [], [], [], [], batch_size=2)
        == []
    )
    with pytest.raises(ValueError, match="matching"):
        estimator.estimate_rgbd_batch(None, None, None, [], [], [1], [], batch_size=2)
    estimator.config["pose"]["quality_retry"] = True
    estimator.finalize_rgbd = lambda pose, points, obj_id, used: {
        "obj_id": obj_id,
        "T_camera_object": pose,
        "pose_quality": {
            "state": "weak_geometry",
            "rmse_m": 0.02,
            "coverage_10mm": 0.2,
        },
    }
    estimator.estimate_rgbd = lambda *args, **kwargs: {
        "status": "ok",
        "T_camera_object": np.eye(4),
        "pose_quality": {
            "state": "supported_geometry",
            "rmse_m": 0.002,
            "coverage_10mm": 0.9,
        },
    }
    recovered = estimator.estimate_rgbd_batch(
        np.zeros((12, 16, 3), dtype=np.uint8),
        np.ones((12, 16)),
        np.diag([100, 100, 1]),
        boxes[:1],
        [np.ones((12, 16))],
        [1],
        [np.ones((40, 3))],
        batch_size=2,
    )
    assert (
        recovered[0]["quality_recovery"]["selected"]
        and not recovered[0]["quality_recovery"]["uses_gt"]
    )


def test_cached_cad_tree_has_identical_refinement_and_isolated_object_indices():
    rng = np.random.default_rng(4)
    model = rng.normal(size=(2000, 3))
    model /= np.linalg.norm(model, axis=1)[:, None]
    scales = np.array([0.08, 0.05, 0.03])
    model *= scales
    normals = model / scales**2
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    observed = model[model[:, 2] < 0] + [0, 0, 0.8]
    initial = np.eye(4)
    initial[2, 3] = 0.81
    cfg = load_config("configs/enhanced.yaml")
    first, diag1 = refine(model, normals, observed, initial, cfg["pose"])
    second, diag2 = refine(
        model, normals, observed, initial, cfg["pose"], tree=cKDTree(model)
    )
    np.testing.assert_array_equal(first, second)
    assert diag1 == diag2
    geometry = GeometricPoseEstimator(cfg)
    geometry.cache = {1: (None, model, normals), 2: (None, model * 2, normals)}
    assert geometry.registration_tree(1) is geometry.registration_tree(1)
    assert geometry.registration_tree(1) is not geometry.registration_tree(2)


def test_batch_pipeline_retains_two_same_class_instances_and_skips_unrequested_scene(
    tmp_path, monkeypatch
):
    import trimesh
    from PIL import Image

    from clutterpick.inference import pipeline
    from clutterpick.utils.common import read_json, write_json

    cfg = load_config("configs/enhanced.yaml")
    cfg["data"]["models"] = str(tmp_path)
    cfg["runtime"] = {"pose_batch_size": 2, "lazy_scene_geometry": True}
    write_json(tmp_path / "models_info.json", {"21": {"diameter": 100}})
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    Image.fromarray(np.full((24, 24), 5000, dtype=np.uint16)).save(
        tmp_path / "depth.png"
    )
    masks = np.zeros((2, 24, 24), dtype=bool)
    masks[0, 6:18, 3:10] = True
    masks[1, 6:18, 14:21] = True

    class Geometry:
        def mesh(self, obj):
            mesh = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
            return mesh, mesh.vertices, None

    class Estimator:
        geometry = Geometry()

        def estimate_rgbd(self, *args, **kwargs):
            raise AssertionError("Unexpected sequential path")

        def estimate_rgbd_batch(
            self,
            rgb,
            depth,
            k,
            boxes,
            masks,
            obj_ids,
            points,
            depth_refinement,
            batch_size,
        ):
            assert obj_ids == [21, 21] and batch_size == 2
            assert points[0][:, 0].mean() < points[1][:, 0].mean()
            result = []
            for cloud in points:
                pose = np.eye(4)
                pose[:3, 3] = cloud.mean(0)
                result.append({"status": "ok", "T_camera_object": pose})
            return result

    class Detector:
        device = "cpu"
        checkpoint = "fixture"

        def predict(self, rgb):
            return {
                "boxes": np.array([[3, 6, 10, 18], [14, 6, 21, 18]]),
                "labels": np.array([21, 21]),
                "scores": np.array([0.9, 0.8]),
                "masks": masks,
            }

    def unexpected_scene(*args, **kwargs):
        raise AssertionError("Unexpected full-scene construction")

    monkeypatch.setattr(pipeline, "backproject", unexpected_scene)
    monkeypatch.setattr(pipeline, "get_estimator", lambda *args: Estimator())
    monkeypatch.setattr(pipeline, "overlay", lambda *args: None)
    frame = {
        "key": "fixture/000048/0",
        "scene_id": "000048",
        "image_id": 0,
        "rgb": str(tmp_path / "rgb.png"),
        "depth": str(tmp_path / "depth.png"),
        "depth_scale_mm": 0.1,
        "K": [100, 0, 12, 0, 100, 12, 0, 0, 1],
        "instances": [],
    }
    row = pipeline.run_frame(
        frame, cfg, Detector(), tmp_path / "output", plots=False, export_clouds=False
    )
    poses = read_json(tmp_path / "output" / "poses.json")["instances"]
    assert (
        row["detections"] == row["poses_ok"] == 2
        and poses[0]["instance_id"] != poses[1]["instance_id"]
    )
    assert poses[0]["T_camera_object"][0][3] < poses[1]["T_camera_object"][0][3]
