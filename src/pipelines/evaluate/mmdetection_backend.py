"""MMDetection backend로 학습한 checkpoint를 추론에 쓸 수 있게 합니다.

계약은 `contracts/proposals/012-mmdetection-checkpoint-inference.md`입니다.
train을 import하지 않습니다(소유 경계). detector 설정과 전처리는 그 제안서가 권한
대로 **필요한 만큼 옮겨 적었습니다**.

## 옮겨 적은 값이 어긋나면

`model_config.schema_version`이 **detector 설정과 전처리 두 가지 모두**를 함께
가리킵니다. train이 detector 설정값이나 전처리를 바꾸면 이 번호를 올려야 하고, 그러면
옛 번호를 읽는 이쪽이 멈춥니다.

번호에 기대는 이유는 자동으로 잡히지 않기 때문입니다. module 구조가 달라지면
state_dict 모양이 맞지 않아 곧바로 실패하지만, NMS threshold·score threshold·정규화
상수·positional encoding처럼 **값만 달라진 경우에는 state_dict가 그대로 맞습니다.**
그때는 오류 없이 점수만 조용히 나빠집니다. 그래서 번호를 올리는 것이 train의 의무이고,
두 source를 직접 대조하는 test는 train의 adapter가 main에 들어온 뒤에 추가합니다.

`backend` key가 없는 checkpoint는 지금까지처럼 torchvision으로 읽습니다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .errors import PredictionError


BACKEND_NAME = "mmdetection"
DINO_ARCHITECTURE = "dino_r50_4scale"
CASCADE_ARCHITECTURE = "cascade_rcnn_swin_t_fpn"
SUPPORTED_ARCHITECTURES = (DINO_ARCHITECTURE, CASCADE_ARCHITECTURE)
MODEL_CONFIG_SCHEMA_VERSION = 1
PAD_MULTIPLE = 32
RESIZE_RULE = "longest_edge"
# train이 정규화에 쓴 값입니다. 다르면 예측이 조용히 나빠집니다.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# 학습 checkpoint는 detector를 감싼 adapter의 state_dict라 이 접두사가 붙습니다.
WRAPPER_PREFIX = "detector."


def _data_preprocessor() -> dict[str, Any]:
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
                "self_attn_cfg": {"embed_dims": 256, "num_levels": 4, "dropout": 0.0},
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
                "self_attn_cfg": {"embed_dims": 256, "num_heads": 8, "dropout": 0.0},
                "cross_attn_cfg": {"embed_dims": 256, "num_levels": 4, "dropout": 0.0},
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


def build_detector_config(
    architecture: str, *, foreground_classes: int
) -> dict[str, Any]:
    """train이 학습한 것과 같은 모양의 detector 설정을 만듭니다."""

    if architecture == DINO_ARCHITECTURE:
        return _dino_config(foreground_classes)
    if architecture == CASCADE_ARCHITECTURE:
        return _cascade_config(foreground_classes)
    raise PredictionError(
        f"허용하지 않는 MMDetection architecture입니다: {architecture}"
    )


def read_model_config(checkpoint: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """checkpoint의 `model_config`를 검증해 돌려줍니다.

    모르는 값을 만나면 기본값으로 넘어가지 않고 멈춥니다. 전처리가 학습과 달라지면
    점수만 조용히 나빠지기 때문입니다.

    `schema_version`은 전처리뿐 아니라 **detector 설정값까지** 함께 가리킵니다.
    자세한 내용은 이 module 첫머리 설명에 있습니다.
    """

    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise PredictionError(f"{source}: mmdetection checkpoint에 model_config가 필요합니다.")
    version = config.get("schema_version")
    if version != MODEL_CONFIG_SCHEMA_VERSION:
        raise PredictionError(
            f"{source}: 읽을 수 없는 model_config schema_version입니다: {version!r}"
        )
    resize = config.get("resize")
    if resize != RESIZE_RULE:
        raise PredictionError(f"{source}: 모르는 resize 규칙입니다: {resize!r}")
    input_size = config.get("input_size")
    if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size < 1:
        raise PredictionError(f"{source}: input_size는 1 이상의 정수여야 합니다: {input_size!r}")
    pad_multiple = config.get("pad_multiple")
    if pad_multiple != PAD_MULTIPLE:
        raise PredictionError(f"{source}: 모르는 pad_multiple입니다: {pad_multiple!r}")
    return {
        "schema_version": version,
        "input_size": input_size,
        "resize": resize,
        "pad_multiple": pad_multiple,
    }


def _validate_category_ids(
    checkpoint: Mapping[str, Any], *, source: str, num_classes: int
) -> None:
    """checkpoint의 `category_ids`가 class마다 하나씩 있는지 확인합니다.

    torchvision 경로는 `category_ids`가 없으면 model label을 그대로 COCO category id로
    씁니다. MMDetection 경로에서는 그렇게 넘어가면 안 됩니다. 없으면 label이 그대로
    category id가 되고, 목록이 짧으면 예측 label이 우연히 범위 안일 때 **다른 약의**
    category id가 나옵니다. 둘 다 오류처럼 보이지 않고 점수만 조용히 틀립니다.

    `category_ids[0]`은 background 자리라 예측에 쓰이지 않지만, 길이는 background를
    포함한 `num_classes`와 같아야 `1..N` 조회가 train이 저장한 것과 맞습니다.
    """

    category_ids = checkpoint.get("category_ids")
    if (
        not isinstance(category_ids, Sequence)
        or isinstance(category_ids, (str, bytes))
        or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in category_ids
        )
    ):
        raise PredictionError(
            f"{source}: mmdetection checkpoint에는 정수 list인 category_ids가 필요합니다: "
            f"{category_ids!r}"
        )
    if len(category_ids) != num_classes:
        raise PredictionError(
            f"{source}: category_ids 길이는 background를 포함한 num_classes와 같아야 "
            f"합니다: {len(category_ids)} != {num_classes}"
        )


def prepare_image(image: Any, *, input_size: int) -> tuple[Any, dict[str, Any]]:
    """학습과 같은 방식으로 resize·정규화·padding하고 metainfo를 만듭니다."""

    import torch
    from torch.nn import functional as F

    if image.ndim != 3 or image.shape[0] != 3:
        raise PredictionError("MMDetection 추론 입력은 [3, H, W] 모양이어야 합니다.")
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
    mean = resized.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = resized.new_tensor(IMAGENET_STD).view(3, 1, 1)
    normalized = (resized - mean) / std
    padded_height = ((resized_height + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    padded_width = ((resized_width + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    padded = F.pad(
        normalized,
        (0, padded_width - resized_width, 0, padded_height - resized_height),
    )
    metainfo = {
        "ori_shape": (height, width),
        "img_shape": (resized_height, resized_width),
        "pad_shape": (padded_height, padded_width),
        # DINO의 pre_transformer가 이 값을 읽습니다. 한 장씩 추론하므로 pad_shape과
        # 같지만, 없으면 AttributeError로 멈춥니다.
        "batch_input_shape": (padded_height, padded_width),
        "scale_factor": (resized_width / width, resized_height / height),
    }
    return torch.stack([padded]), metainfo


def to_output(instances: Any, *, metainfo: Mapping[str, Any]) -> dict[str, Any]:
    """MMDetection 예측을 원본 좌표와 1-based label로 되돌립니다.

    MMDetection foreground label은 `0..N-1`이고 저장소 label은 `1..N`이라 1을
    더합니다. 그래야 `category_ids[label]` 조회가 torchvision 경로와 같아집니다.
    """

    scale_x, scale_y = metainfo["scale_factor"]
    boxes = instances.bboxes.detach().float().cpu().clone()
    if boxes.numel():
        boxes[:, (0, 2)] /= scale_x
        boxes[:, (1, 3)] /= scale_y
    return {
        "boxes": boxes,
        "labels": instances.labels.detach().cpu() + 1,
        "scores": instances.scores.detach().float().cpu(),
    }


def _import_mmdetection() -> Any:
    """MMDetection을 늦게 import합니다. 없으면 이 backend를 고른 순간에 알립니다.

    `ImportError`만 잡으면 부족합니다. mmcv는 **컴파일된 확장**을 함께 싣기 때문에,
    설치는 됐지만 확장이 깨졌거나 torch 버전과 어긋나면 import 도중
    `OSError('DLL load failed')`처럼 다른 예외가 납니다. 그것이 그대로 올라가면
    `run()`이 잡지 못해 실행 자체가 죽습니다. 설치 문제라는 결론은 같으므로 같은
    오류로 바꿉니다.
    """

    from types import SimpleNamespace

    try:
        from mmdet.registry import MODELS
        from mmdet.structures import DetDataSample
        from mmdet.utils import register_all_modules
    except Exception as error:  # noqa: BLE001 - 설치 문제를 계약 오류로 바꿉니다.
        raise PredictionError(
            "MMDetection backend 추론에는 mmdet, mmcv, mmengine이 필요합니다. "
            f"requirements.txt의 설치 경로를 확인하세요: {error!r}"
        ) from error
    return SimpleNamespace(
        models=MODELS,
        data_sample_type=DetDataSample,
        register=lambda: register_all_modules(init_default_scope=True),
    )


def _detector_state(state_dict: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """adapter가 감싼 채로 저장한 state에서 detector 부분을 꺼냅니다.

    train은 detector를 감싼 adapter의 state를 저장하므로 **모든 key**가
    ``detector.``로 시작해야 합니다. 하나라도 다르면 멈춥니다. 접두사 없는 key를
    조용히 버리거나 그대로 쓰면 일부 가중치만 실린 model로 점수를 내게 되고,
    그 점수는 낮을 뿐 오류처럼 보이지 않아 아무도 눈치채지 못합니다.
    """

    names = list(state_dict)
    wrong = [
        name
        for name in names
        if not isinstance(name, str) or not name.startswith(WRAPPER_PREFIX)
    ]
    if not names or wrong:
        sample = ", ".join(str(name) for name in wrong[:3]) or "(비어 있음)"
        raise PredictionError(
            f"{source}: mmdetection checkpoint의 model_state_dict는 모든 key가 "
            f"'{WRAPPER_PREFIX}'로 시작해야 합니다: {sample}"
        )
    return {name[len(WRAPPER_PREFIX) :]: value for name, value in state_dict.items()}


class MMDetectionPredictor:
    """MMDetection detector를 torchvision detection model처럼 쓰게 감쌉니다.

    호출하면 원본 이미지 좌표의 `boxes`, 1-based `labels`, `scores`를 돌려주므로
    기존 예측 변환 코드가 그대로 동작합니다.
    """

    def __init__(self, detector: Any, *, input_size: int, data_sample_type: type) -> None:
        self._detector = detector
        self._input_size = input_size
        self._data_sample_type = data_sample_type

    def to(self, device: Any) -> "MMDetectionPredictor":
        # device 이동도 MMDetection module이 합니다. 바깥에서 잡는 오류 종류가
        # 정해져 있어서, 그 밖의 예외는 run()까지 새어 나갑니다.
        try:
            self._detector.to(device)
        except PredictionError:
            raise
        except Exception as error:  # noqa: BLE001 - 서드파티 오류를 계약 오류로 바꿉니다.
            raise PredictionError(
                f"MMDetection model을 device로 옮기지 못했습니다 ({device}): {error!r}"
            ) from error
        return self

    def eval(self) -> "MMDetectionPredictor":
        self._detector.eval()
        return self

    def __call__(self, images: Sequence[Any]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for image in images:
            # 전처리, DetDataSample 생성, metainfo 설정, 추론, 출력 변환이 모두
            # MMDetection과 torch의 몫입니다. 어떤 예외든 낼 수 있고 run()은
            # EvaluateError만 잡으므로, 이 구간 전체를 계약 오류로 바꿔야
            # 실행이 죽지 않고 status="error"로 보고됩니다.
            try:
                batch, metainfo = prepare_image(image, input_size=self._input_size)
                sample = self._data_sample_type()
                sample.set_metainfo(dict(metainfo))
                predicted = self._detector.predict(
                    batch.to(image.device), [sample], rescale=False
                )
                output = to_output(predicted[0].pred_instances, metainfo=metainfo)
            except PredictionError:
                raise
            except Exception as error:  # noqa: BLE001 - 서드파티 오류를 계약 오류로 바꿉니다.
                raise PredictionError(
                    f"MMDetection 추론에 실패했습니다: {error!r}"
                ) from error
            outputs.append(output)
        return outputs


def build_predictor(
    checkpoint: Mapping[str, Any],
    *,
    source: str,
    architecture: str,
    num_classes: int,
    state_dict: Mapping[str, Any],
) -> MMDetectionPredictor:
    """mmdetection checkpoint로 추론 model을 만듭니다."""

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise PredictionError(
            f"{source}: 허용하지 않는 MMDetection architecture입니다: {architecture}"
        )
    # num_classes는 background를 포함합니다. 1이면 foreground가 0개라 model을 만들 수
    # 없는데, 빼기만 하면 0이나 음수가 그대로 MMDetection에 들어갑니다.
    if num_classes < 2:
        raise PredictionError(
            f"{source}: mmdetection checkpoint의 num_classes는 background를 포함해 "
            f"2 이상이어야 합니다: {num_classes}"
        )
    # checkpoint 계약 확인은 모두 무거운 model 생성 **앞**에 둡니다.
    _validate_category_ids(checkpoint, source=source, num_classes=num_classes)
    model_config = read_model_config(checkpoint, source=source)
    detector_state = _detector_state(state_dict, source=source)
    dependencies = _import_mmdetection()
    # registry 등록, 생성, state 적용, eval 전환 모두 서드파티 코드입니다. 어떤 예외든
    # 낼 수 있으므로 계약 오류로 바꿉니다.
    try:
        dependencies.register()
        detector = dependencies.models.build(
            build_detector_config(architecture, foreground_classes=num_classes - 1)
        )
        detector.load_state_dict(detector_state)
        model = MMDetectionPredictor(
            detector,
            input_size=model_config["input_size"],
            data_sample_type=dependencies.data_sample_type,
        ).eval()
    except PredictionError:
        raise
    except Exception as error:  # noqa: BLE001 - 서드파티 오류를 계약 오류로 바꿉니다.
        raise PredictionError(
            f"{source}: checkpoint를 model에 적용하지 못했습니다: {error!r}"
        ) from error
    return model


__all__ = [
    "BACKEND_NAME",
    "SUPPORTED_ARCHITECTURES",
    "MMDetectionPredictor",
    "build_detector_config",
    "build_predictor",
    "prepare_image",
    "read_model_config",
    "to_output",
]
