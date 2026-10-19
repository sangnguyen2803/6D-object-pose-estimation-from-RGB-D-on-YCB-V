from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from clutterpick.evaluation.boundary import boundary_mask
from clutterpick.models.fusion import RGBDFPN, QueryMaskPredictor, depth_channels


def test_metric_planar_depth_normals_and_missing_depth():
    depth = np.ones((8, 8))
    depth[0, 0] = 0
    scalar, normals = depth_channels(
        depth, np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]])
    )
    np.testing.assert_allclose(normals[3, 3], [0, 0, -1])
    assert scalar[0, 0] == 0 and not normals[0, 0].any()
    assert np.isfinite(normals).all()


class Body(nn.Module):
    def forward(self, x):
        value = x.mean(1, keepdim=True)
        return OrderedDict(
            (str(i), value.repeat(1, c, 1, 1))
            for i, c in enumerate([256, 512, 1024, 2048])
        )


class Backbone(nn.Module):
    out_channels = 256

    def __init__(self):
        super().__init__()
        self.body = Body()

    def fpn(self, maps):
        return OrderedDict((key, value[:, :256]) for key, value in maps.items())

    def forward(self, x):
        return self.fpn(self.body(x))


def test_gated_fusion_falls_back_exactly_to_rgb_when_depth_missing():
    original = Backbone()
    model = RGBDFPN(original, "gated")
    rgb = torch.rand(1, 3, 3, 4)
    actual = model(torch.cat([rgb, torch.zeros_like(rgb)], 1))
    for key, value in original(rgb).items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


def test_query_head_returns_56_masks_and_can_learn_residual():
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    predictor = QueryMaskPredictor(MaskRCNNPredictor(256, 256, 22), 22)
    features = torch.randn(1, 256, 28, 28)
    output = predictor(features)
    assert output.shape == (1, 22, 56, 56)
    output.mean().backward()
    assert predictor.output.weight.grad.abs().sum() > 0


def test_boundary_keeps_border_objects_and_is_sensitive_to_large_mask_shift():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:60, :60] = True
    edge = boundary_mask(mask)
    assert edge[0, 0] and not edge[30, 30]
    shifted = np.roll(mask, 3, axis=1)
    region_iou = (
        np.logical_and(mask, shifted).sum() / np.logical_or(mask, shifted).sum()
    )
    other = boundary_mask(shifted)
    boundary_iou = np.logical_and(edge, other).sum() / np.logical_or(edge, other).sum()
    assert boundary_iou < region_iou
