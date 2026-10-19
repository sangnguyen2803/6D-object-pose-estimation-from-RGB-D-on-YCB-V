import numpy as np
import torch

from clutterpick.models.point_network import PointForegroundNetwork, point_features


def test_point_features_are_finite_for_flat_cloud_and_use_predicted_membership():
    points = np.column_stack([np.arange(40) * 0.001, np.zeros(40), np.ones(40)])
    features = point_features(
        points, np.full((40, 3), 120, dtype=np.uint8), np.ones(40)
    )
    assert features.shape == (40, 9) and np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, 6], 1)


def test_local_point_attention_is_permutation_equivariant_and_trainable():
    torch.manual_seed(5)
    model = PointForegroundNetwork()
    features = torch.randn(1, 32, 9)
    obj = torch.tensor([14])
    order = torch.randperm(32)
    original = model(features, obj)
    reordered = model(features[:, order], obj)
    torch.testing.assert_close(reordered, original[:, order], rtol=1e-4, atol=1e-5)
    torch.nn.functional.binary_cross_entropy_with_logits(
        original, torch.ones_like(original)
    ).backward()
    assert model.classifier[-1].weight.grad.abs().sum() > 0
