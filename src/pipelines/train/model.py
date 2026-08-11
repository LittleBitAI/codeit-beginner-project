"""설정으로 선택한 torchvision detection model을 만듭니다."""

from __future__ import annotations

from functools import partial

from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from .mmdetection_adapter import (
    DEFAULT_INPUT_SIZE,
    MMDETECTION_ARCHITECTURES,
    build_mmdetection_model,
)


ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
SUPPORTED_ARCHITECTURES = (
    ARCHITECTURE,
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    *MMDETECTION_ARCHITECTURES,
)


def _replace_faster_rcnn_head(model: nn.Module, num_classes: int) -> nn.Module:
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
    return model


def _replace_retinanet_head(model: nn.Module, num_classes: int) -> nn.Module:
    current = model.head.classification_head
    model.head.classification_head = RetinaNetClassificationHead(
        current.cls_logits.in_channels,
        current.num_anchors,
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    return model


def build_model(
    num_classes: int,
    *,
    architecture: str = ARCHITECTURE,
    pretrained: bool = False,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> nn.Module:
    """선택한 detection model을 class 수에 맞춰 만듭니다."""
    if num_classes < 2:
        raise ValueError("num_classes must include background and at least one object class")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported train architecture: {architecture}")

    if architecture in MMDETECTION_ARCHITECTURES:
        # MMDetection detector는 torchvision과 호출 방식이 달라 adapter가 감쌉니다.
        # `input_size`는 이쪽에서만 씁니다.
        return build_mmdetection_model(
            num_classes,
            architecture=architecture,
            pretrained=pretrained,
            input_size=input_size,
        )

    if architecture == ARCHITECTURE and pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        )
        return _replace_faster_rcnn_head(model, num_classes)

    if architecture == "fasterrcnn_resnet50_fpn_v2" and pretrained:
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        )
        return _replace_faster_rcnn_head(model, num_classes)

    if architecture == "retinanet_resnet50_fpn_v2" and pretrained:
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
        )
        return _replace_retinanet_head(model, num_classes)

    builders = {
        ARCHITECTURE: fasterrcnn_mobilenet_v3_large_320_fpn,
        "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
        "retinanet_resnet50_fpn_v2": retinanet_resnet50_fpn_v2,
    }
    return builders[architecture](weights=None, weights_backbone=None, num_classes=num_classes)
