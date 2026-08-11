"""MMDetection adapter와 그 architecture를 아직 고를 수 없게 막는 문 test입니다."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
import torchvision
from torch import nn

from src.pipelines.train.mmdetection_adapter import (
    MMDETECTION_ARCHITECTURES,
    MMDetectionAdapter,
    _prepare_detector,
    build_mmdetection_config,
    prepare_mmdetection_batch,
)
from src.pipelines.train.model import SUPPORTED_ARCHITECTURES, build_model
from src.pipelines.train.pipeline import _settings


@pytest.mark.parametrize(
    ("architecture", "detector_type"),
    [
        ("dino_r50_4scale", "DINO"),
        ("cascade_rcnn_swin_t_fpn", "CascadeRCNN"),
    ],
)
def test_mmdetection_architectures_build_allowlisted_bbox_configs(
    architecture: str, detector_type: str
):
    config = build_mmdetection_config(architecture, foreground_classes=3)

    assert architecture in MMDETECTION_ARCHITECTURES
    assert config["type"] == detector_type
    if architecture == "dino_r50_4scale":
        assert config["bbox_head"]["num_classes"] == 3
    else:
        assert [head["num_classes"] for head in config["roi_head"]["bbox_head"]] == [
            3,
            3,
            3,
        ]
        assert "mask_head" not in config["roi_head"]


@pytest.mark.parametrize("architecture", MMDETECTION_ARCHITECTURES)
def test_mmdetection_architectures_are_not_selectable_yet(architecture: str):
    """evaluate·web·requirements 통합 전까지는 설정으로 고를 수 없어야 합니다.

    checkpoint를 evaluate가 읽지 못하고 clean Colab에는 mmdet이 없으므로, 지금 고를 수
    있게 두면 학습만 되고 채점은 못 하는 실행이 공개됩니다. 통합이 끝나면
    ``contracts/proposals/012``대로 이 문을 엽니다.
    """

    assert architecture not in SUPPORTED_ARCHITECTURES
    with pytest.raises(ValueError, match="train.architecture must be one of"):
        _settings({"train": {"architecture": architecture}})
    with pytest.raises(ValueError, match="unsupported train architecture"):
        build_model(4, architecture=architecture)


def test_every_selectable_architecture_stays_loadable_by_evaluate():
    """evaluate의 predictor는 이름을 torchvision.models.detection에서 찾습니다.

    거기 없는 이름을 train이 고를 수 있게 되는 순간, 학습은 되는데 채점은 못 하는
    checkpoint가 공개됩니다. evaluate를 import하지 않고 같은 규칙만 확인합니다.
    """

    for architecture in SUPPORTED_ARCHITECTURES:
        assert getattr(torchvision.models.detection, architecture, None) is not None


def test_mmdetection_batch_zero_bases_first_and_last_label_then_resizes_and_pads():
    image = torch.ones((3, 4, 8), dtype=torch.float32)
    targets = (
        {
            "boxes": torch.tensor([[0.0, 0.0, 8.0, 4.0], [2.0, 1.0, 6.0, 3.0]]),
            "labels": torch.tensor([1, 3], dtype=torch.int64),
            "image_id": torch.tensor([7], dtype=torch.int64),
        },
    )

    images, converted, metadata = prepare_mmdetection_batch(
        (image,), targets, input_size=6
    )

    assert images.shape == (1, 3, 32, 32)
    assert torch.equal(converted[0]["labels"], torch.tensor([0, 2]))
    assert torch.allclose(
        converted[0]["boxes"],
        torch.tensor([[0.0, 0.0, 6.0, 3.0], [1.5, 0.75, 4.5, 2.25]]),
    )
    assert metadata == [
        {
            "img_id": 7,
            "ori_shape": (4, 8),
            "img_shape": (3, 6),
            "pad_shape": (32, 32),
            "batch_input_shape": (32, 32),
            "scale_factor": (0.75, 0.75),
        }
    ]


def test_mmdetection_batch_shares_one_batch_input_shape_across_images():
    """DINO의 pre_transformer가 batch 전체의 padding 크기를 읽습니다."""

    images, _, metadata = prepare_mmdetection_batch(
        (torch.ones((3, 8, 4)), torch.ones((3, 4, 40))),
        (
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        ),
        input_size=40,
    )

    assert images.shape == (2, 3, 64, 64)
    assert [entry["batch_input_shape"] for entry in metadata] == [(64, 64), (64, 64)]
    assert [entry["pad_shape"] for entry in metadata] == [(64, 32), (32, 64)]


class _FakeDataSample:
    def __init__(self) -> None:
        self.metainfo: dict[str, object] = {}
        self.gt_instances: object | None = None

    def set_metainfo(self, metainfo: Mapping[str, object]) -> None:
        self.metainfo.update(metainfo)


class _FakeInstanceData:
    def __init__(self, **fields: torch.Tensor) -> None:
        self.fields = fields


class _MetricAndLossDetector(nn.Module):
    """Cascade R-CNN처럼 loss와 정확도 지표를 함께 돌려줍니다."""

    def loss(self, batch, samples):
        return {
            "loss_rpn_cls": torch.tensor(1.0),
            "s0.loss_cls": [torch.tensor(2.0), torch.tensor(4.0)],
            "s0.acc": torch.tensor(97.0),
            "s1.acc": [torch.tensor(90.0)],
        }


def test_adapter_objective_ignores_accuracy_metrics_from_the_detector():
    """s0.acc를 더하면 정확도가 오를수록 loss가 커져 학습 목표가 뒤집힙니다."""

    adapter = MMDetectionAdapter(
        _MetricAndLossDetector(),
        input_size=32,
        data_sample_type=_FakeDataSample,
        instance_data_type=_FakeInstanceData,
    )
    target = {
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }

    losses = adapter([torch.ones((3, 8, 8))], [target])

    assert set(losses) == {"loss_rpn_cls", "s0.loss_cls"}
    assert float(losses["s0.loss_cls"]) == pytest.approx(6.0)


class _RecordingDetector(nn.Module):
    """init_weights와 load_state_dict 호출 순서를 기록하는 가짜 detector입니다."""

    def __init__(self, expected: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.expected = dict(expected)
        self.calls: list[str] = []
        self.loaded: dict[str, torch.Tensor] = {}

    def init_weights(self) -> None:
        self.calls.append("init_weights")

    def state_dict(self, *args, **kwargs):
        return dict(self.expected)

    def load_state_dict(self, state, strict=True):
        self.calls.append("load_state_dict")
        self.loaded = dict(state)


class _FakeLoader:
    def __init__(self, state: Mapping[str, torch.Tensor]) -> None:
        self.state = dict(state)

    def load_checkpoint(self, source: str, map_location: str = "cpu"):
        return {"state_dict": self.state}


def test_pretrained_run_initializes_the_detector_before_loading_weights():
    """MODELS.build는 init_weights를 부르지 않아 DINO head가 초기화되지 않습니다."""

    detector = _RecordingDetector({"backbone.conv1.weight": torch.zeros(2)})

    _prepare_detector(
        detector,
        "dino_r50_4scale",
        pretrained=True,
        loader=_FakeLoader({"backbone.conv1.weight": torch.ones(2)}),
    )

    assert detector.calls == ["init_weights", "load_state_dict"]
    assert torch.equal(detector.loaded["backbone.conv1.weight"], torch.ones(2))


def test_scratch_run_still_initializes_the_detector():
    detector = _RecordingDetector({})

    _prepare_detector(detector, "dino_r50_4scale", pretrained=False, loader=None)

    assert detector.calls == ["init_weights"]


def test_legacy_swin_checkpoint_keys_are_converted_before_filtering():
    """옛 Swin 이름을 그대로 두면 backbone이 거의 실리지 않고 조용히 scratch가 됩니다."""

    detector = _RecordingDetector(
        {
            "backbone.patch_embed.projection.weight": torch.zeros(2),
            "backbone.stages.0.blocks.0.norm1.weight": torch.zeros(2),
            "backbone.stages.0.blocks.0.attn.w_msa.qkv.weight": torch.zeros(2),
            "backbone.stages.0.blocks.0.ffn.layers.0.0.weight": torch.zeros(2),
            "backbone.stages.0.downsample.reduction.weight": torch.zeros((2, 4)),
        }
    )

    _prepare_detector(
        detector,
        "cascade_rcnn_swin_t_fpn",
        pretrained=True,
        loader=_FakeLoader(
            {
                "backbone.patch_embed.proj.weight": torch.ones(2),
                "backbone.layers.0.blocks.0.norm1.weight": torch.full((2,), 2.0),
                "backbone.layers.0.blocks.0.attn.qkv.weight": torch.full((2,), 3.0),
                "backbone.layers.0.blocks.0.mlp.fc1.weight": torch.full((2,), 4.0),
                "backbone.layers.0.downsample.reduction.weight": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
                ),
            }
        ),
    )

    assert set(detector.loaded) == set(detector.expected)
    assert torch.equal(
        detector.loaded["backbone.stages.0.blocks.0.attn.w_msa.qkv.weight"],
        torch.full((2,), 3.0),
    )
    # PatchMerging이 원본과 다른 순서로 4칸을 펼치므로 값도 같이 바뀌어야 합니다.
    assert torch.equal(
        detector.loaded["backbone.stages.0.downsample.reduction.weight"],
        torch.tensor([[1.0, 3.0, 2.0, 4.0], [5.0, 7.0, 6.0, 8.0]]),
    )
