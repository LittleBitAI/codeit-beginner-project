"""Faster R-CNN model construction."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"


def build_model(num_classes: int, *, pretrained: bool = False) -> nn.Module:
    """Build the CPU-friendly MobileNetV3 Faster R-CNN baseline."""
    if num_classes < 2:
        raise ValueError("num_classes must include background and at least one object class")
    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        )
        input_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
        return model

    return fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )
