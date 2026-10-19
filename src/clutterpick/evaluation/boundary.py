"""Boundary AP using min(mask IoU, inner-boundary IoU), matching reference protocol."""

import cv2
import numpy as np
from pycocotools import mask as mask_api
from pycocotools.cocoeval import COCOeval

from clutterpick.evaluation.metrics import encode_mask


def boundary_mask(mask, ratio=0.02):
    mask = mask.astype(np.uint8)
    width = max(1, int(round(ratio * np.hypot(*mask.shape))))
    padded = np.pad(mask, 1)
    eroded = cv2.erode(padded, np.ones((3, 3), dtype=np.uint8), iterations=width)[
        1:-1, 1:-1
    ]
    return (mask - eroded) > 0


class BoundaryCOCOeval(COCOeval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundaries = {}

    def computeIoU(self, image, category):
        ordinary = super().computeIoU(image, category)
        if np.asarray(ordinary).size == 0:
            return ordinary
        gt = self._gts[image, category]
        dt = sorted(self._dts[image, category], key=lambda item: -item["score"])[
            : self.params.maxDets[-1]
        ]

        def encoded(item, kind):
            key = (kind, item["id"])
            if key not in self.boundaries:
                self.boundaries[key] = encode_mask(
                    boundary_mask(mask_api.decode(item["segmentation"]))
                )
            return self.boundaries[key]

        crowd = [int(item["iscrowd"]) for item in gt]
        boundary = mask_api.iou(
            [encoded(item, "dt") for item in dt],
            [encoded(item, "gt") for item in gt],
            crowd,
        )
        combined = np.minimum(ordinary, boundary)
        combined[:, np.asarray(crowd, dtype=bool)] = np.asarray(ordinary)[
            :, np.asarray(crowd, dtype=bool)
        ]
        return combined
