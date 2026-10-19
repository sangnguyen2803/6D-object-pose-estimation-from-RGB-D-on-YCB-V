import numpy as np
from PIL import Image

from clutterpick.evaluation.metrics import evaluate_detector


class Detector:
    device = "cpu"
    checkpoint = "synthetic_evaluation_fixture"

    def __init__(self, mask, empty=False):
        self.mask, self.empty = mask, empty

    def predict(self, rgb, score_threshold=None):
        if self.empty:
            return {
                "boxes": np.zeros((0, 4)),
                "scores": np.zeros(0),
                "labels": np.zeros(0, dtype=int),
                "masks": np.zeros((0, *rgb.shape[:2]), dtype=bool),
            }
        return {
            "boxes": np.array([[2, 2, 8, 8]]),
            "scores": np.array([0.9]),
            "labels": np.array([3]),
            "masks": self.mask[None],
        }


def fixture_frame(tmp_path):
    rgb, mask_path = tmp_path / "rgb.png", tmp_path / "mask.png"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(rgb)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    return {
        "id": 0,
        "key": "fixture/scene/0",
        "rgb": str(rgb),
        "instances": [{"obj_id": 3, "mask_visib": str(mask_path), "visib_fract": 0.4}],
    }, mask


def test_perfect_masks_score_one_and_misses_remain_in_visibility_denominator(tmp_path):
    frame, mask = fixture_frame(tmp_path)
    perfect = evaluate_detector(Detector(mask), [frame])
    assert perfect["metrics"]["segm"]["AP_50_95"] > 0.999
    missed = evaluate_detector(Detector(mask, empty=True), [frame])
    assert missed["metrics"]["segm"]["AP_50_95"] == 0
    assert missed["visibility_mask_recall_iou50_score50"]["25_50"] == {
        "gt": 1,
        "matched": 0,
        "recall": 0.0,
    }


def test_boundary_ap_perfect_and_empty_predictions(tmp_path):
    frame, mask = fixture_frame(tmp_path)
    perfect = evaluate_detector(Detector(mask), [frame], boundary=True)
    missed = evaluate_detector(Detector(mask, empty=True), [frame], boundary=True)
    assert perfect["metrics"]["boundary"]["AP_50_95"] > 0.999
    assert missed["metrics"]["boundary"]["AP_50_95"] == 0
