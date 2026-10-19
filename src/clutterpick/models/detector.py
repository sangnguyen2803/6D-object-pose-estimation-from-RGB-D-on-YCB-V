from pathlib import Path

import numpy as np


def choose_device(requested="auto"):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    return torch.device(requested)


def create_maskrcnn(config, pretrained=False):
    import torch
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops.misc import FrozenBatchNorm2d

    d = config["detector"]
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,
        min_size=d["min_size"],
        max_size=d["max_size"],
        box_detections_per_img=d["max_detections"],
        box_score_thresh=0.05,
    )

    # Torchvision otherwise changes normalization/freezing when weights=None.
    # Preserve the pretrained architecture for loading and resuming checkpoints.
    def freeze_batch_norm(module):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.BatchNorm2d):
                replacement = FrozenBatchNorm2d(child.num_features, eps=child.eps)
                replacement.load_state_dict(
                    {
                        k: v
                        for k, v in child.state_dict().items()
                        if k != "num_batches_tracked"
                    }
                )
                setattr(module, name, replacement)
            else:
                freeze_batch_norm(child)

    freeze_batch_norm(model.backbone.body)
    for name, param in model.backbone.body.named_parameters():
        param.requires_grad_(name.startswith(("layer2.", "layer3.", "layer4.")))
    from clutterpick.utils.common import object_ids

    classes = max(object_ids(config)) + 1
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, classes
    )
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, classes
    )
    return model


class MaskRCNNDetector:
    def __init__(self, checkpoint, config, device="auto"):
        import torch

        self.device = choose_device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = payload.get("config", config)
        from clutterpick.utils.common import object_ids

        if object_ids(self.config) != object_ids(config):
            raise ValueError(
                "Checkpoint object classes do not match the requested dataset"
            )
        self.model = create_maskrcnn(self.config)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()
        self.checkpoint = str(Path(checkpoint).resolve())

    def predict(self, rgb, score_threshold=None):
        import torch

        image = (
            torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(self.device) / 255
        )
        with torch.inference_mode():
            out = self.model([image])[0]
        threshold = (
            self.config["detector"]["score_threshold"]
            if score_threshold is None
            else score_threshold
        )
        keep = out["scores"] >= threshold
        result = {k: v[keep].detach().cpu().numpy() for k, v in out.items()}
        result["masks"] = (
            result["masks"][:, 0] >= self.config["detector"]["mask_threshold"]
        )
        return result


class YOLODetector:
    def __init__(self, checkpoint, config, device="auto"):
        from ultralytics import YOLO

        self.model, self.config = YOLO(str(checkpoint)), config
        from clutterpick.utils.common import object_ids

        expected = {i - 1: f"object_{i}" for i in object_ids(config)}
        if self.model.names != expected:
            raise ValueError(
                "YOLO checkpoint must use the exported YCB-V object_1..object_21 mapping"
            )
        self.device = str(choose_device(device))
        self.checkpoint = str(Path(checkpoint).resolve())

    def predict(self, rgb, score_threshold=None):
        from PIL import Image

        threshold = (
            self.config["detector"]["score_threshold"]
            if score_threshold is None
            else score_threshold
        )
        pred = self.model.predict(
            source=Image.fromarray(rgb),
            conf=threshold,
            device=self.device,
            imgsz=self.config["detector"]["max_size"],
            retina_masks=True,
            max_det=self.config["detector"]["max_detections"],
            verbose=False,
        )[0]
        h, w = rgb.shape[:2]
        if pred.masks is None:
            return {
                "boxes": np.zeros((0, 4)),
                "scores": np.zeros(0),
                "labels": np.zeros(0, dtype=int),
                "masks": np.zeros((0, h, w), dtype=bool),
            }
        return {
            "boxes": pred.boxes.xyxy.cpu().numpy(),
            "scores": pred.boxes.conf.cpu().numpy(),
            "labels": pred.boxes.cls.cpu().numpy().astype(int) + 1,
            "masks": pred.masks.data.cpu().numpy() > 0.5,
        }


def load_detector(checkpoint, config, device="auto", architecture=None):
    if (architecture or config["detector"]["architecture"]) == "fusion":
        from clutterpick.models.fusion import FusionDetector

        return FusionDetector(checkpoint, config, device)
    cls = {
        "cosypose": CosyPoseDetector,
        "yolo": YOLODetector,
        "maskrcnn": MaskRCNNDetector,
    }[architecture or config["detector"]["architecture"]]
    return cls(checkpoint, config, device)


class CosyPoseDetector(MaskRCNNDetector):
    """Published BOP20 Mask R-CNN; strict state loading and YCB-V class mapping."""

    def __init__(self, checkpoint, config, device="auto"):
        import copy

        import torch

        from clutterpick.models.pretrained import read_published_config

        self.device = choose_device(device)
        self.config = copy.deepcopy(config)
        published = read_published_config(Path(checkpoint).parent / "config.yaml")
        resize = published["input_resize"]
        self.config["detector"].update(min_size=min(resize), max_size=max(resize))
        mapping = {
            label: int(category)
            for label, category in published["label_to_category_id"].items()
            if str(label).startswith("obj_")
        }
        if sorted(mapping.values()) != list(range(1, 22)) or any(
            int(label.split("_")[1]) != category for label, category in mapping.items()
        ):
            raise ValueError("Unexpected published YCB-V category mapping")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = create_maskrcnn(self.config)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.checkpoint = str(Path(checkpoint).resolve())
        self.pretrained_source = "CosyPose BOP20 YCB-V SYNT+REAL"
