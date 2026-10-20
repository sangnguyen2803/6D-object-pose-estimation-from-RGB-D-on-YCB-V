import copy
import csv

import numpy as np
import pytest
import trimesh
import yaml
from PIL import Image

from clutterpick.evaluation.reporting import summarize
from clutterpick.models.pretrained import read_published_config
from clutterpick.utils.common import load_config, read_json, write_json
from clutterpick.utils.outputs import export_bop


def test_bop_export_converts_meters_to_millimeters(tmp_path):
    matrix = np.eye(4)
    matrix[:3, 3] = [0.1, -0.2, 0.75]
    write_json(
        tmp_path / "poses.json",
        {"instances": [{"obj_id": 21, "score": 0.9, "T_camera_object": matrix}]},
    )
    export_bop(
        [
            {
                "output": str(tmp_path),
                "scene_id": "000048",
                "image_id": 5,
                "timing": {"inference_s_before_reference_evaluation": 1.2},
            }
        ],
        tmp_path / "export" / "nested",
    )
    with (tmp_path / "export" / "nested" / "bop_predictions.csv").open() as stream:
        row = list(csv.DictReader(stream))[0]
    np.testing.assert_allclose(np.fromstring(row["t"], sep=" "), [100, -200, 750])
    assert int(row["obj_id"]) == 21


def test_bop_export_empty_run_creates_header(tmp_path):
    output = tmp_path / "empty_run"
    export_bop([], output)
    with (output / "bop_predictions.csv").open() as stream:
        rows = list(csv.reader(stream))
    assert rows == [["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]]


def test_published_config_loader_rejects_arbitrary_constructors(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("!!python/object/apply:os.system ['echo unsafe']", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        read_published_config(path)


def test_missing_instances_remain_in_pose_denominator():
    row = {
        "gt_instances": 2,
        "detections": 1,
        "timing": {"inference_s_before_reference_evaluation": 1},
        "pose_diagnostics": [
            {
                "ADD_success_0_1d": True,
                "ADD_S_success_0_1d": True,
                "ADD_or_S_success_0_1d": True,
            }
        ],
    }
    assert summarize([row])["recall"]["ADD_or_S_success_0_1d"] == 0.5


def test_ycbv_index_keeps_test_out_of_scene_holdout(tmp_path):
    from clutterpick.data.dataset import build_index

    config = load_config()
    config["data"].update(
        root=str(tmp_path / "raw"),
        index=str(tmp_path / "index.json"),
        splits=str(tmp_path / "splits.json"),
        index_report=str(tmp_path / "report.json"),
        object_ids=[21],
        validation_scenes=["000001"],
    )
    for split, scene in [
        ("train_real", "000000"),
        ("train_real", "000001"),
        ("test", "000001"),
    ]:
        folder = tmp_path / "raw" / split / scene
        for name in ["rgb", "depth", "mask", "mask_visib"]:
            (folder / name).mkdir(parents=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            folder / "rgb/000000.png"
        )
        Image.fromarray(np.ones((4, 4), dtype=np.uint16)).save(
            folder / "depth/000000.png"
        )
        for name in ["mask", "mask_visib"]:
            Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(
                folder / name / "000000_000000.png"
            )
        write_json(
            folder / "scene_gt.json",
            {
                "0": [
                    {
                        "obj_id": 21,
                        "cam_R_m2c": np.eye(3).ravel(),
                        "cam_t_m2c": [0, 0, 500],
                    }
                ]
            },
        )
        write_json(folder / "scene_gt_info.json", {"0": [{"visib_fract": 1.0}]})
        write_json(
            folder / "scene_camera.json",
            {"0": {"cam_K": np.eye(3).ravel(), "depth_scale": 0.1}},
        )
    index, splits = build_index(config)
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert index["frames"][splits["test"][0]]["group"] == "ycbv_test"
    assert not (set(splits["test"]) & (set(splits["train"]) | set(splits["val"])))


def test_changing_gt_pose_changes_evaluation_but_not_prediction(tmp_path, monkeypatch):
    from clutterpick.inference import pipeline

    config = load_config()
    config["data"]["models"] = str(tmp_path)
    write_json(tmp_path / "models_info.json", {"21": {"diameter": 100}})
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    depth = np.full((24, 24), 5000, dtype=np.uint16)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[6:18, 6:18] = 255
    for name, array in [("rgb", rgb), ("depth", depth), ("mask", mask)]:
        Image.fromarray(array).save(tmp_path / f"{name}.png")
    transform = np.eye(4)
    transform[2, 3] = 0.5
    mesh = trimesh.creation.box(extents=[0.05, 0.05, 0.05])

    class Geometry:
        def mesh(self, obj):
            return mesh, mesh.vertices, {}

    class Estimator:
        geometry = Geometry()

        def estimate_rgbd(
            self, rgb, depth, k, box, mask, obj, points, depth_refinement
        ):
            assert obj == 21
            return {"status": "ok", "T_camera_object": transform.copy()}

    class Detector:
        device = "cpu"
        checkpoint = "fixture"

        def predict(self, rgb):
            return {
                "boxes": np.array([[6, 6, 18, 18]]),
                "labels": np.array([21]),
                "scores": np.array([0.9]),
                "masks": (mask > 0)[None],
            }

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
        "instances": [
            {
                "obj_id": 21,
                "instance_id": "fixture/0/0",
                "mask_visib": str(tmp_path / "mask.png"),
                "R_m2c": np.eye(3).ravel().tolist(),
                "t_m2c_mm": [0, 0, 500],
            }
        ],
    }
    first = pipeline.run_frame(
        frame, config, Detector(), tmp_path / "first", plots=False, export_clouds=False
    )
    changed = copy.deepcopy(frame)
    changed["instances"][0]["t_m2c_mm"] = [0, 0, 900]
    second = pipeline.run_frame(
        changed,
        config,
        Detector(),
        tmp_path / "second",
        plots=False,
        export_clouds=False,
    )
    p1 = read_json(tmp_path / "first/poses.json")
    p2 = read_json(tmp_path / "second/poses.json")
    assert p1["instances"] == p2["instances"]
    assert p1["uses_ground_truth_pose_for_inference"] is False
    assert first["pose_diagnostics"][0]["ADD_success_0_1d"]
    assert not second["pose_diagnostics"][0]["ADD_success_0_1d"]
