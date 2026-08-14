"""MMDetection detector를 기존 Train model 계약에 맞춥니다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.common.train_contract import (
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_INPUT_SIZE,
    MMDETECTION_ARCHITECTURES,
)

from .errors import TrainError


# 고를 수 있는 이름과 기본값은 GUI와 함께 쓰는 계약(`src/common/train_contract.py`)에
# 있습니다. 여기서는 그 이름이 어떤 config와 checkpoint를 뜻하는지만 정합니다. 이름을
# 계약에서 풀어 받지 않는 것은 순서가 바뀌면 두 model이 조용히 뒤바뀌기 때문입니다.
DINO_ARCHITECTURE = "dino_r50_4scale"
CASCADE_ARCHITECTURE = "cascade_rcnn_swin_t_fpn"
PAD_MULTIPLE = 32
MODEL_CONFIG_SCHEMA_VERSION = 1
# mmdet 3.3.0이 거부하지만 **직접 확인해 본** mmcv 버전 하나입니다. 자세한 이유는
# _shimmed_mmcv_version에 적었습니다. 이 하나 말고는 손대지 않습니다.
MMCV_SHIM_EXACT = (2, 2, 0)
MMCV_SHIM_VERSION = "2.1.999"

DINO_CHECKPOINT = (
    "https://download.openmmlab.com/mmdetection/v3.0/dino/"
    "dino-4scale_r50_8xb2-12e_coco/"
    "dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"
)
CASCADE_CHECKPOINT = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.2/"
    "cascade_mask_rcnn_swin_tiny_patch4_window7.pth"
)


def _data_preprocessor() -> dict[str, Any]:
    # Adapter가 RGB [0, 1] tensor를 직접 정규화하고 padding합니다.
    return {
        "type": "DetDataPreprocessor",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "bgr_to_rgb": False,
        "pad_size_divisor": PAD_MULTIPLE,
    }


def _dino_config(num_classes: int) -> dict[str, Any]:
    return {
        "type": "DINO",
        "num_queries": 900,
        "with_box_refine": True,
        "as_two_stage": True,
        "data_preprocessor": _data_preprocessor(),
        "backbone": {
            "type": "ResNet",
            "depth": 50,
            "num_stages": 4,
            "out_indices": (1, 2, 3),
            "frozen_stages": 1,
            "norm_cfg": {"type": "BN", "requires_grad": False},
            "norm_eval": True,
            "style": "pytorch",
            "with_cp": True,
            "init_cfg": None,
        },
        "neck": {
            "type": "ChannelMapper",
            "in_channels": [512, 1024, 2048],
            "kernel_size": 1,
            "out_channels": 256,
            "act_cfg": None,
            "norm_cfg": {"type": "GN", "num_groups": 32},
            "num_outs": 4,
        },
        "encoder": {
            "num_layers": 6,
            "layer_cfg": {
                "self_attn_cfg": {
                    "embed_dims": 256,
                    "num_levels": 4,
                    "dropout": 0.0,
                },
                "ffn_cfg": {
                    "embed_dims": 256,
                    "feedforward_channels": 2048,
                    "ffn_drop": 0.0,
                },
            },
        },
        "decoder": {
            "num_layers": 6,
            "return_intermediate": True,
            "layer_cfg": {
                "self_attn_cfg": {
                    "embed_dims": 256,
                    "num_heads": 8,
                    "dropout": 0.0,
                },
                "cross_attn_cfg": {
                    "embed_dims": 256,
                    "num_levels": 4,
                    "dropout": 0.0,
                },
                "ffn_cfg": {
                    "embed_dims": 256,
                    "feedforward_channels": 2048,
                    "ffn_drop": 0.0,
                },
            },
            "post_norm_cfg": None,
        },
        "positional_encoding": {
            "num_feats": 128,
            "normalize": True,
            "offset": 0.0,
            "temperature": 20,
        },
        "bbox_head": {
            "type": "DINOHead",
            "num_classes": num_classes,
            "sync_cls_avg_factor": True,
            "loss_cls": {
                "type": "FocalLoss",
                "use_sigmoid": True,
                "gamma": 2.0,
                "alpha": 0.25,
                "loss_weight": 1.0,
            },
            "loss_bbox": {"type": "L1Loss", "loss_weight": 5.0},
            "loss_iou": {"type": "GIoULoss", "loss_weight": 2.0},
        },
        "dn_cfg": {
            "label_noise_scale": 0.5,
            "box_noise_scale": 1.0,
            "group_cfg": {
                "dynamic": True,
                "num_groups": None,
                "num_dn_queries": 100,
            },
        },
        "train_cfg": {
            "assigner": {
                "type": "HungarianAssigner",
                "match_costs": [
                    {"type": "FocalLossCost", "weight": 2.0},
                    {"type": "BBoxL1Cost", "weight": 5.0, "box_format": "xywh"},
                    {"type": "IoUCost", "iou_mode": "giou", "weight": 2.0},
                ],
            }
        },
        "test_cfg": {"max_per_img": 300},
    }


def _cascade_bbox_head(num_classes: int, target_stds: Sequence[float]) -> dict[str, Any]:
    return {
        "type": "ConvFCBBoxHead",
        "num_shared_convs": 4,
        "num_shared_fcs": 1,
        "in_channels": 256,
        "conv_out_channels": 256,
        "fc_out_channels": 1024,
        "roi_feat_size": 7,
        "num_classes": num_classes,
        "bbox_coder": {
            "type": "DeltaXYWHBBoxCoder",
            "target_means": [0.0, 0.0, 0.0, 0.0],
            "target_stds": list(target_stds),
        },
        "reg_class_agnostic": False,
        "reg_decoded_bbox": True,
        "norm_cfg": {"type": "SyncBN", "requires_grad": True},
        "loss_cls": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 1.0,
        },
        "loss_bbox": {"type": "GIoULoss", "loss_weight": 10.0},
    }


def _rcnn_stage(iou: float) -> dict[str, Any]:
    return {
        "assigner": {
            "type": "MaxIoUAssigner",
            "pos_iou_thr": iou,
            "neg_iou_thr": iou,
            "min_pos_iou": iou,
            "match_low_quality": False,
            "ignore_iof_thr": -1,
        },
        "sampler": {
            "type": "RandomSampler",
            "num": 512,
            "pos_fraction": 0.25,
            "neg_pos_ub": -1,
            "add_gt_as_proposals": True,
        },
        "pos_weight": -1,
        "debug": False,
    }


def _cascade_config(num_classes: int) -> dict[str, Any]:
    return {
        "type": "CascadeRCNN",
        "data_preprocessor": _data_preprocessor(),
        "backbone": {
            "type": "SwinTransformer",
            "embed_dims": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.2,
            "patch_norm": True,
            "out_indices": (0, 1, 2, 3),
            "with_cp": True,
            "convert_weights": True,
            "init_cfg": None,
        },
        "neck": {
            "type": "FPN",
            "in_channels": [96, 192, 384, 768],
            "out_channels": 256,
            "num_outs": 5,
        },
        "rpn_head": {
            "type": "RPNHead",
            "in_channels": 256,
            "feat_channels": 256,
            "anchor_generator": {
                "type": "AnchorGenerator",
                "scales": [8],
                "ratios": [0.5, 1.0, 2.0],
                "strides": [4, 8, 16, 32, 64],
            },
            "bbox_coder": {
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [1.0, 1.0, 1.0, 1.0],
            },
            "loss_cls": {
                "type": "CrossEntropyLoss",
                "use_sigmoid": True,
                "loss_weight": 1.0,
            },
            "loss_bbox": {
                "type": "SmoothL1Loss",
                "beta": 1.0 / 9.0,
                "loss_weight": 1.0,
            },
        },
        "roi_head": {
            "type": "CascadeRoIHead",
            "num_stages": 3,
            "stage_loss_weights": [1.0, 0.5, 0.25],
            "bbox_roi_extractor": {
                "type": "SingleRoIExtractor",
                "roi_layer": {
                    "type": "RoIAlign",
                    "output_size": 7,
                    "sampling_ratio": 0,
                },
                "out_channels": 256,
                "featmap_strides": [4, 8, 16, 32],
            },
            "bbox_head": [
                _cascade_bbox_head(num_classes, [0.1, 0.1, 0.2, 0.2]),
                _cascade_bbox_head(num_classes, [0.05, 0.05, 0.1, 0.1]),
                _cascade_bbox_head(num_classes, [0.033, 0.033, 0.067, 0.067]),
            ],
        },
        "train_cfg": {
            "rpn": {
                "assigner": {
                    "type": "MaxIoUAssigner",
                    "pos_iou_thr": 0.7,
                    "neg_iou_thr": 0.3,
                    "min_pos_iou": 0.3,
                    "match_low_quality": True,
                    "ignore_iof_thr": -1,
                },
                "sampler": {
                    "type": "RandomSampler",
                    "num": 256,
                    "pos_fraction": 0.5,
                    "neg_pos_ub": -1,
                    "add_gt_as_proposals": False,
                },
                "allowed_border": 0,
                "pos_weight": -1,
                "debug": False,
            },
            "rpn_proposal": {
                "nms_pre": 2000,
                "max_per_img": 2000,
                "nms": {"type": "nms", "iou_threshold": 0.7},
                "min_bbox_size": 0,
            },
            "rcnn": [_rcnn_stage(0.5), _rcnn_stage(0.6), _rcnn_stage(0.7)],
        },
        "test_cfg": {
            "rpn": {
                "nms_pre": 1000,
                "max_per_img": 1000,
                "nms": {"type": "nms", "iou_threshold": 0.7},
                "min_bbox_size": 0,
            },
            "rcnn": {
                "score_thr": 0.05,
                "nms": {"type": "nms", "iou_threshold": 0.5},
                "max_per_img": 100,
            },
        },
    }


def _shimmed_mmcv_version(version: str) -> str | None:
    """mmdet의 mmcv 상한만 통과시킬 가짜 버전을 정합니다. 필요 없으면 ``None``입니다.

    mmdet 3.3.0은 ``import mmdet`` 도중 ``assert mmcv < 2.2.0``을 겁니다. 그런데 지금
    torch에 맞는 mmcv 확장 wheel은 **2.2.0뿐이고**, mmdet은 2024-01 이후 새 release가
    없어 이 상한이 열릴 일이 없습니다. mmcv 2.2.0 release note에는 2.1.0 대비
    breaking change가 없고 NPU 연산자 추가와 버그 수정뿐입니다.

    **직접 확인한 그 버전에만** 적용합니다. 범위로 열어 두면 아직 나오지도 않은 2.2.1이나
    2.2.99까지 함께 통과해, 정말로 맞지 않는 조합이 설치 문제로 보고되는 대신 알 수 없는
    자리에서 깨집니다. 공식 호환표는 여전히 mmdet 3.3.0에 ``mmcv<2.2.0``을 적어 두고
    있으므로, 새 버전을 쓰려면 그때 다시 확인하고 이 값을 옮겨야 합니다.

    ``+a8073c7pt2.12.0cu126`` 같은 local 꼬리표는 같은 소스를 어느 torch에 맞춰
    빌드했는지만 나타내므로 떼고 봅니다.
    """

    parts = version.split("+", 1)[0].split(".")
    try:
        numbers = tuple(int(part) for part in parts)
    except ValueError:
        return None
    if numbers == MMCV_SHIM_EXACT:
        return MMCV_SHIM_VERSION
    return None


def build_mmdetection_config(
    architecture: str, *, foreground_classes: int
) -> dict[str, Any]:
    """승인된 MMDetection detector 설정을 class 수에 맞춰 만듭니다."""
    if architecture not in MMDETECTION_ARCHITECTURES:
        raise ValueError(f"unsupported MMDetection architecture: {architecture}")
    if foreground_classes < 1:
        raise ValueError("foreground_classes must be positive")
    if architecture == DINO_ARCHITECTURE:
        return _dino_config(foreground_classes)
    return _cascade_config(foreground_classes)


def model_config_metadata(input_size: int) -> dict[str, int | str]:
    return {
        "schema_version": MODEL_CONFIG_SCHEMA_VERSION,
        "input_size": input_size,
        "resize": "longest_edge",
        "pad_multiple": PAD_MULTIPLE,
    }


def prepare_mmdetection_batch(
    images: Sequence[torch.Tensor],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    input_size: int,
) -> tuple[torch.Tensor, tuple[dict[str, torch.Tensor], ...], list[dict[str, Any]]]:
    """RGB image와 1-based target을 resize·padding하고 0-based로 바꿉니다."""
    if input_size < 1:
        raise ValueError("input_size must be positive")
    padded_images: list[torch.Tensor] = []
    prepared_targets: list[dict[str, torch.Tensor]] = []
    metadata: list[dict[str, Any]] = []
    for image, target in zip(images, targets, strict=True):
        if image.ndim != 3 or image.shape[0] != 3:
            raise TrainError("MMDetection image must have shape [3, H, W]")
        _, height, width = image.shape
        scale = input_size / max(height, width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        mean = resized.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = resized.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        normalized = (resized - mean) / std
        padded_height = ((resized_height + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        padded_width = ((resized_width + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        padded_images.append(
            F.pad(
                normalized,
                (0, padded_width - resized_width, 0, padded_height - resized_height),
            )
        )

        labels = target["labels"]
        if labels.numel() and bool((labels < 1).any().item()):
            raise TrainError("MMDetection input labels must be contiguous from 1")
        converted = dict(target)
        converted["labels"] = labels - 1
        boxes = target["boxes"].clone()
        scale_x = resized_width / width
        scale_y = resized_height / height
        if boxes.numel():
            boxes[:, (0, 2)] *= scale_x
            boxes[:, (1, 3)] *= scale_y
        converted["boxes"] = boxes
        prepared_targets.append(converted)
        image_id = target.get("image_id")
        metadata.append(
            {
                "img_id": int(image_id.reshape(-1)[0].item()) if image_id is not None else -1,
                "ori_shape": (height, width),
                "img_shape": (resized_height, resized_width),
                "pad_shape": (padded_height, padded_width),
                "scale_factor": (scale_x, scale_y),
            }
        )
    # DeformableDETR.pre_transformer가 batch_data_samples[0].batch_input_shape를 읽습니다.
    # DetDataPreprocessor를 거치지 않으므로 batch 전체의 padding 크기를 여기서 남깁니다.
    batch_height = max(image.shape[1] for image in padded_images)
    batch_width = max(image.shape[2] for image in padded_images)
    prepared_images = [
        F.pad(
            image,
            (0, batch_width - image.shape[2], 0, batch_height - image.shape[1]),
        )
        for image in padded_images
    ]
    for entry in metadata:
        entry["batch_input_shape"] = (batch_height, batch_width)
    return torch.stack(prepared_images), tuple(prepared_targets), metadata


def _flatten_losses(losses: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """이름에 loss가 든 항목만 남깁니다.

    Cascade R-CNN은 ``s0.acc``처럼 정확도 지표를 loss와 같은 dict에 담아 돌려줍니다.
    학습 loop은 받은 값을 모두 더하므로, 걸러 내지 않으면 정확도가 오를수록 목적함수가
    커져 학습과 검증 점수가 모두 뒤집힙니다.
    """

    flattened: dict[str, torch.Tensor] = {}
    for name, value in losses.items():
        if "loss" not in name:
            continue
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if not values or any(not isinstance(item, torch.Tensor) for item in values):
            raise TrainError(f"MMDetection loss '{name}' must contain tensors")
        scalars = [item if item.numel() == 1 else item.mean() for item in values]
        flattened[name] = torch.stack(scalars).sum()
    return flattened


class MMDetectionAdapter(nn.Module):
    """MMDetection detector를 torchvision-style loss 호출로 감쌉니다."""

    def __init__(
        self,
        detector: nn.Module,
        *,
        input_size: int,
        data_sample_type: type[Any],
        instance_data_type: type[Any],
    ) -> None:
        super().__init__()
        self.detector = detector
        self.input_size = input_size
        self._data_sample_type = data_sample_type
        self._instance_data_type = instance_data_type

    def forward(
        self,
        images: Sequence[torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> Mapping[str, torch.Tensor]:
        batch, converted, metadata = prepare_mmdetection_batch(
            images, targets, input_size=self.input_size
        )
        samples = []
        for target, metainfo in zip(converted, metadata, strict=True):
            sample = self._data_sample_type()
            sample.set_metainfo(metainfo)
            sample.gt_instances = self._instance_data_type(
                bboxes=target["boxes"], labels=target["labels"]
            )
            samples.append(sample)
        return _flatten_losses(self.detector.loss(batch, samples))


def _allowed_pretrained_key(architecture: str, key: str) -> bool:
    if architecture == DINO_ARCHITECTURE:
        return "bbox_head.cls_branches" in key or "label_embedding" in key
    return (
        "roi_head.mask" in key
        or ("roi_head.bbox_head" in key and (".fc_cls." in key or ".fc_reg." in key))
    )


def _unfold_reduction_order(value: torch.Tensor) -> torch.Tensor:
    out_channel, in_channel = value.shape
    reshaped = value.reshape(out_channel, 4, in_channel // 4)
    return reshaped[:, (0, 2, 1, 3), :].transpose(1, 2).reshape(out_channel, in_channel)


def _unfold_norm_order(value: torch.Tensor) -> torch.Tensor:
    in_channel = value.shape[0]
    reshaped = value.reshape(4, in_channel // 4)
    return reshaped[(0, 2, 1, 3), :].transpose(0, 1).reshape(in_channel)


def _modern_backbone_entry(
    architecture: str, name: str, value: Any
) -> tuple[str, Any]:
    """옛 Swin checkpoint의 backbone 항목을 MMDetection 3 이름으로 바꿉니다.

    공개된 Swin 검출 checkpoint는 원래 Swin 저장소의 이름을 씁니다. 지금 backbone은
    ``stages``, ``patch_embed.projection``, ``attn.w_msa``, ``ffn.layers``를 쓰므로
    바꾸지 않으면 backbone key가 거의 하나도 맞지 않습니다. 맞지 않는 key는 조용히
    걸러지므로, ``pretrained=true``인데 사실상 scratch로 학습하게 됩니다.

    ``downsample``은 이름뿐 아니라 **값의 순서**도 다릅니다. MMDetection의 PatchMerging이
    원본과 다른 순서로 4칸을 펼치기 때문에, 이름만 바꾸면 가중치가 뒤섞인 채 실립니다.
    바꾸는 규칙은 ``mmdet.models.backbones.swin.swin_converter``와 같습니다.
    """

    if architecture != CASCADE_ARCHITECTURE or not name.startswith("backbone."):
        return name, value
    inner = name.removeprefix("backbone.")
    if inner.startswith("patch_embed"):
        return "backbone." + inner.replace("proj.", "projection."), value
    if not inner.startswith("layers"):
        return name, value
    if "attn." in inner:
        inner = inner.replace("attn.", "attn.w_msa.")
    elif "mlp.fc1." in inner:
        inner = inner.replace("mlp.fc1.", "ffn.layers.0.0.")
    elif "mlp.fc2." in inner:
        inner = inner.replace("mlp.fc2.", "ffn.layers.1.")
    elif isinstance(value, torch.Tensor) and "downsample.reduction." in inner:
        value = _unfold_reduction_order(value)
    elif isinstance(value, torch.Tensor) and "downsample.norm." in inner:
        value = _unfold_norm_order(value)
    return "backbone." + inner.replace("layers", "stages", 1), value


def _load_pretrained(detector: nn.Module, architecture: str, loader: Any) -> None:
    source = DINO_CHECKPOINT if architecture == DINO_ARCHITECTURE else CASCADE_CHECKPOINT
    document = loader.load_checkpoint(source, map_location="cpu")
    raw_state = document.get("state_dict", document) if isinstance(document, Mapping) else None
    if not isinstance(raw_state, Mapping):
        raise TrainError("MMDetection pretrained checkpoint has no state_dict")
    expected = detector.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    mismatched: list[str] = []
    for raw_name, raw_value in raw_state.items():
        name, value = _modern_backbone_entry(
            architecture, raw_name.removeprefix("module."), raw_value
        )
        if _allowed_pretrained_key(architecture, name):
            continue
        if name not in expected:
            continue
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            mismatched.append(name)
            continue
        filtered[name] = value
    if mismatched:
        raise TrainError(
            "MMDetection pretrained state shape mismatch: " + ", ".join(sorted(mismatched))
        )
    detector.load_state_dict(filtered, strict=False)


def _prepare_detector(
    detector: nn.Module, architecture: str, *, pretrained: bool, loader: Any
) -> None:
    """MMEngine Runner처럼 init_weights를 부른 뒤 pretrained 가중치를 덮어씁니다.

    ``MODELS.build``는 module을 만들기만 합니다. DINO의 query embedding이나 head처럼
    특별한 초기화가 필요한 자리가 그대로 남으므로, 처음부터 학습하는 실행도 pretrained
    실행도 먼저 초기화해야 합니다.
    """

    detector.init_weights()
    if pretrained:
        _load_pretrained(detector, architecture, loader)


def build_mmdetection_model(
    num_classes: int,
    *,
    architecture: str,
    pretrained: bool,
    input_size: int,
) -> nn.Module:
    """MMDetection을 늦게 import해 승인된 detector를 만듭니다."""
    try:
        import mmcv

        # mmdet은 import 도중 mmcv 버전을 재고 AssertionError를 냅니다. 그래서
        # ImportError만 잡아서는 부족합니다. 자세한 이유는 _shimmed_mmcv_version에.
        real_version = mmcv.__version__
        shim = _shimmed_mmcv_version(real_version)
        try:
            if shim is not None:
                mmcv.__version__ = shim
            from mmdet.registry import MODELS
            from mmdet.structures import DetDataSample
            from mmdet.utils import register_all_modules
            from mmengine.config import ConfigDict
            from mmengine.runner.checkpoint import CheckpointLoader
            from mmengine.structures import InstanceData
        finally:
            mmcv.__version__ = real_version
    except Exception as error:  # noqa: BLE001 - 설치 문제를 계약 오류로 바꿉니다.
        raise TrainError(
            "MMDetection architecture requires mmdet, mmcv, and mmengine "
            f"(install path check needed): {error!r}"
        ) from error

    register_all_modules(init_default_scope=True)
    # 평범한 dict로 넘기면 two-stage detector가 train_cfg.rpn을 속성으로 읽다 멈춥니다.
    # ConfigDict는 중첩된 dict까지 함께 바꿔 줍니다.
    detector = MODELS.build(
        ConfigDict(
            build_mmdetection_config(architecture, foreground_classes=num_classes - 1)
        )
    )
    _prepare_detector(
        detector, architecture, pretrained=pretrained, loader=CheckpointLoader
    )
    return MMDetectionAdapter(
        detector,
        input_size=input_size,
        data_sample_type=DetDataSample,
        instance_data_type=InstanceData,
    )
