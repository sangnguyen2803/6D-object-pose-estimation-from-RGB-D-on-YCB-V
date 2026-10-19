import numpy as np

from clutterpick.geometry.icp import point_plane
from clutterpick.geometry.point_cloud import transform_points
from clutterpick.geometry.refinement import annotate_observability, refine
from clutterpick.utils.common import load_config


def ellipsoid():
    rng = np.random.default_rng(12)
    directions = rng.normal(size=(6000, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    scales = np.array([0.09, 0.055, 0.035])
    model = directions * scales
    normals = model / scales**2
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return model, normals


def test_robust_refinement_recovers_depth_with_partial_visibility_and_background():
    model, normals = ellipsoid()
    truth = np.eye(4)
    truth[2, 3] = 0.8
    observed = transform_points(model[model[:, 2] < -0.005][::2], truth)
    rng = np.random.default_rng(6)
    observed += rng.normal(0, 0.0005, observed.shape)
    background = rng.uniform([-0.1, -0.1, 0.95], [0.1, 0.1, 1.1], size=(250, 3))
    initial = truth.copy()
    initial[2, 3] += 0.015
    cfg = load_config("configs/enhanced.yaml")["pose"]
    result, diagnostic = refine(
        model, normals, np.concatenate([observed, background]), initial, cfg
    )
    assert np.linalg.norm(result[:3, 3] - truth[:3, 3]) < 0.005
    assert diagnostic["accepted"]
    assert diagnostic["after"]["rmse_m"] < diagnostic["before"]["rmse_m"]
    assert diagnostic["quality"]["calibrated_probability"] is False


def test_missing_depth_keeps_rgb_pose_and_requests_review():
    model, normals = ellipsoid()
    initial = np.eye(4)
    initial[2, 3] = 0.8
    result, diagnostic = refine(
        model,
        normals,
        np.zeros((0, 3)),
        initial,
        load_config("configs/enhanced.yaml")["pose"],
    )
    np.testing.assert_array_equal(result, initial)
    assert not diagnostic["accepted"]
    assert diagnostic["quality"]["state"] == "insufficient_depth"


def test_unrelated_background_is_not_accepted_as_depth_refinement():
    model, normals = ellipsoid()
    initial = np.eye(4)
    initial[2, 3] = 0.8
    observed = model + [0, 0, 1.2]
    result, diagnostic = refine(
        model, normals, observed, initial, load_config("configs/enhanced.yaml")["pose"]
    )
    np.testing.assert_array_equal(result, initial)
    assert diagnostic["quality"]["requires_review"]


def test_planar_support_reports_unobservable_directions():
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 30), np.linspace(-0.1, 0.1, 30))
    model = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    normals = np.tile([0, 0, -1.0], (len(model), 1))
    initial = np.eye(4)
    initial[2, 3] = 0.82
    result, diagnostic = point_plane(
        model,
        normals,
        model + [0, 0, 0.8],
        initial,
        load_config("configs/enhanced.yaml")["pose"],
    )
    assert abs(result[2, 3] - 0.8) < 0.001
    assert diagnostic["observable_directions"] < 6
    quality = annotate_observability(
        {
            "candidates": [{"method": "huber_point_to_plane", **diagnostic}],
            "quality": {"state": "supported_geometry", "requires_review": False},
        }
    )["quality"]
    assert quality["requires_review"]
    assert quality["review_reasons"] == [
        "robust_candidate_has_underconstrained_directions"
    ]
