"""MMDetection adapter와 그 architecture를 아직 고를 수 없게 막는 문 test입니다."""

from __future__ import annotations

import importlib.util

from collections.abc import Mapping

import pytest
import torch
import torchvision
from torch import nn

from src.pipelines.train.errors import TrainError
from src.pipelines.train.mmdetection_adapter import (
    DINO_CHECKPOINT,
    DINO_PRETRAINED_SOURCES,
    MMDETECTION_ARCHITECTURES,
    SWIN_B_CHECKPOINT,
    MMDetectionAdapter,
    _prepare_detector,
    _shimmed_mmcv_version,
    build_mmdetection_config,
    build_mmdetection_model,
    prepare_mmdetection_batch,
)
from src.pipelines.train.model import SUPPORTED_ARCHITECTURES, build_model
import ast
from pathlib import Path

from src.pipelines.train import model as model_module
from src.pipelines.train import pipeline as pipeline_module
from src.pipelines.train.pipeline import _checkpoint_payload, _settings


@pytest.mark.parametrize(
    ("architecture", "detector_type"),
    [
        ("dino_r50_4scale", "DINO"),
        ("dino_swin_b_4scale", "DINO"),
        ("cascade_rcnn_swin_t_fpn", "CascadeRCNN"),
    ],
)
def test_mmdetection_architectures_build_allowlisted_bbox_configs(
    architecture: str, detector_type: str
):
    config = build_mmdetection_config(architecture, foreground_classes=3)

    assert architecture in MMDETECTION_ARCHITECTURES
    assert config["type"] == detector_type
    if detector_type == "DINO":
        assert config["bbox_head"]["num_classes"] == 3
    else:
        assert [head["num_classes"] for head in config["roi_head"]["bbox_head"]] == [
            3,
            3,
            3,
        ]
        assert "mask_head" not in config["roi_head"]


def test_every_selectable_architecture_stays_loadable_by_evaluate():
    """고를 수 있는 이름은 모두 evaluate가 읽을 수 있어야 합니다.

    거기 없는 이름을 train이 고를 수 있게 되는 순간, 학습은 되는데 채점은 못 하는
    checkpoint가 공개됩니다. evaluate를 import하지 않고 같은 규칙만 확인합니다.
    torchvision 이름은 ``torchvision.models.detection``에서, MMDetection 이름은
    checkpoint의 ``backend``로 evaluate가 찾습니다.
    """

    for architecture in SUPPORTED_ARCHITECTURES:
        if architecture in MMDETECTION_ARCHITECTURES:
            continue
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


class _SourceLoader:
    """source마다 다른 state를 주는 loader입니다."""

    def __init__(self, states: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.states = dict(states)
        self.sources: list[str] = []

    def load_checkpoint(self, source: str, map_location: str = "cpu"):
        self.sources.append(source)
        return {"state_dict": dict(self.states[source])}


@pytest.mark.parametrize(
    ("architecture", "in_channels"),
    [
        ("dino_swin_t_4scale", [192, 384, 768]),
        ("dino_swin_b_4scale", [256, 512, 1024]),
    ],
)
def test_four_scale_swin_swaps_the_backbone_and_the_channels_it_feeds_only(
    architecture: str, in_channels: list[int]
):
    r50 = build_mmdetection_config("dino_r50_4scale", foreground_classes=3)
    swin = build_mmdetection_config(architecture, foreground_classes=3)

    assert swin["backbone"]["type"] == "SwinTransformer"
    assert swin["neck"]["in_channels"] == in_channels
    # 4scale을 그대로 두어야 R50 DINO의 encoder·decoder를 실을 수 있습니다.
    assert swin["neck"]["num_outs"] == r50["neck"]["num_outs"]
    assert swin["encoder"] == r50["encoder"]
    assert swin["decoder"] == r50["decoder"]
    assert "num_feature_levels" not in swin


def test_swin_l_is_five_scale_all_the_way_through():
    """한 자리만 넷으로 남으면 model이 만들어지지 않거나 조용히 다른 것이 됩니다."""

    swin = build_mmdetection_config("dino_swin_l_5scale", foreground_classes=3)

    assert swin["num_feature_levels"] == 5
    assert swin["backbone"]["out_indices"] == (0, 1, 2, 3)
    assert swin["neck"]["in_channels"] == [192, 384, 768, 1536]
    assert swin["neck"]["num_outs"] == 5
    assert swin["encoder"]["layer_cfg"]["self_attn_cfg"]["num_levels"] == 5
    assert swin["decoder"]["layer_cfg"]["cross_attn_cfg"]["num_levels"] == 5


def test_dino_swin_takes_backbone_and_transformer_from_different_checkpoints():
    """MMDetection이 내놓는 DINO는 Swin-L뿐이라 Swin-B는 두 곳에서 모읍니다."""

    detector = _RecordingDetector(
        {
            "backbone.patch_embed.projection.weight": torch.zeros(2),
            "backbone.stages.0.blocks.0.attn.w_msa.qkv.weight": torch.zeros(2),
            "neck.convs.0.conv.weight": torch.zeros((256, 256, 1, 1)),
            "encoder.layers.0.ffn.layers.0.0.weight": torch.zeros(2),
        }
    )
    loader = _SourceLoader(
        {
            SWIN_B_CHECKPOINT: {
                "backbone.patch_embed.proj.weight": torch.ones(2),
                "backbone.layers.0.blocks.0.attn.qkv.weight": torch.full((2,), 3.0),
            },
            DINO_CHECKPOINT: {
                "neck.convs.0.conv.weight": torch.zeros((256, 512, 1, 1)),
                "encoder.layers.0.ffn.layers.0.0.weight": torch.full((2,), 7.0),
            },
        }
    )

    _prepare_detector(detector, "dino_swin_b_4scale", pretrained=True, loader=loader)

    assert torch.equal(
        detector.loaded["backbone.stages.0.blocks.0.attn.w_msa.qkv.weight"],
        torch.full((2,), 3.0),
    )
    assert torch.equal(
        detector.loaded["encoder.layers.0.ffn.layers.0.0.weight"], torch.full((2,), 7.0)
    )
    # 입력 채널이 달라 모양이 어긋나는 neck은 멈추지 않고 빠집니다.
    assert "neck.convs.0.conv.weight" not in detector.loaded


def test_a_backbone_that_does_not_fill_stops_the_run_instead_of_training_scratch():
    detector = _RecordingDetector(
        {"backbone.stages.0.blocks.0.norm1.weight": torch.zeros(2)}
    )
    loader = _SourceLoader(
        {source: {"backbone.gone.weight": torch.ones(2)} for source in DINO_PRETRAINED_SOURCES["dino_swin_b_4scale"]}
    )

    with pytest.raises(TrainError):
        _prepare_detector(detector, "dino_swin_b_4scale", pretrained=True, loader=loader)


MMDETECTION_PACKAGES = ("mmcv", "mmdet", "mmengine")


def _require_mmdetection() -> None:
    """package가 **아예 없을 때만** 건너뜁니다. 그 밖의 실패는 실패로 둡니다.

    설치 실패까지 싸잡아 건너뛰면 잘못된 wheel, ``mmcv._ext`` 로딩 실패, 맞지 않는
    버전 조합, import 경로 회귀가 모두 **초록색 CI**로 보입니다. requirements가
    mmdet을 설치하기 시작한 뒤에는 그 구분이 특히 중요합니다. 설치가 깨진 것과
    애초에 설치 대상이 아닌 것은 다릅니다.

    ``find_spec``은 module을 실행하지 않으므로 설치 여부만 보고, import 실패는
    그대로 드러납니다.
    """

    for name in MMDETECTION_PACKAGES:
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"{name}이(가) 설치돼 있지 않습니다. requirements 밖의 선택 사항입니다.")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        pytest.param("2.1.0", None, id="below_is_left_alone"),
        pytest.param("2.2.0", "2.1.999", id="the_version_actually_checked"),
        pytest.param("2.2.0+a8073c7pt2.12.0cu126", "2.1.999", id="local_tag_is_stripped"),
        pytest.param("2.2.1", None, id="unchecked_patch_stays_closed"),
        pytest.param("2.3.0", None, id="unchecked_minor_stays_closed"),
        pytest.param("3.0.0", None, id="next_major_stays_closed"),
    ],
)
def test_mmcv_version_shim_only_covers_the_verified_range(version, expected):
    """범위로 열면 아직 나오지도 않은 2.2.1까지 통과해 엉뚱한 곳에서 깨집니다."""

    assert _shimmed_mmcv_version(version) == expected


@pytest.mark.parametrize("architecture", MMDETECTION_ARCHITECTURES)
def test_real_detector_is_built_and_produces_a_finite_loss(architecture):
    """진짜 mmdet detector로 설정과 loss 경로를 확인합니다.

    cascade_rcnn_swin_t_fpn은 평범한 dict를 넘기던 동안 **만들어지지도 않았는데**,
    가짜 detector를 쓰는 test는 모두 통과했습니다. train_cfg를 속성으로 읽기
    때문입니다.
    """

    _require_mmdetection()
    model = build_mmdetection_model(
        4, architecture=architecture, pretrained=False, input_size=320
    )

    images = [torch.rand(3, 240, 320)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 120.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([1]),
        }
    ]
    losses = model(images, targets)

    assert losses, "loss 항목이 하나도 없습니다"
    assert all("loss" in name for name in losses), sorted(losses)
    total = sum(losses.values())
    assert torch.isfinite(total), f"loss가 유한하지 않습니다: {total}"


@pytest.mark.parametrize("architecture", MMDETECTION_ARCHITECTURES)
def test_mmdetection_architectures_are_selectable(pretend_cuda, architecture: str):
    """이제 고를 수 있어야 합니다. 이 test가 그 문을 여는 표시입니다."""

    assert architecture in SUPPORTED_ARCHITECTURES
    settings = _settings(_mmdetection_raw(architecture=architecture))
    assert settings["architecture"] == architecture


@pytest.mark.parametrize("architecture", MMDETECTION_ARCHITECTURES)
def test_mmdetection_amp_uses_fp16_for_mmcv_cuda_ops(pretend_cuda, architecture: str):
    """GPU가 bf16을 지원해도 MMCV custom CUDA op는 fp16으로 실행합니다.

    DINO의 MultiScaleDeformableAttention을 비롯한 MMCV CUDA 확장은 bf16 dispatch가
    없습니다. GPU 지원만 보고 bf16을 고르면 첫 batch에서 실패하므로, MMDetection의
    amp는 fp16과 GradScaler를 써야 합니다.
    """

    settings = _settings(_mmdetection_raw(architecture=architecture))

    assert settings["precision"] == {
        "mode": "amp",
        "dtype": "fp16",
        "grad_scaler": True,
    }


@pytest.fixture
def pretend_cuda(monkeypatch):
    """GPU가 없는 곳에서도 같은 결과가 나오게 합니다.

    MMDetection model은 device가 cuda여야 하는데 CI runner에는 GPU가 없습니다. 실제로 CUDA를
    쓰는 test가 아니라 설정 검증만 보는 test이므로 확인 함수만 바꿉니다.
    """

    monkeypatch.setattr(pipeline_module.torch.cuda, "is_available", lambda: True)
    # 이 fixture를 MMDetection 외 precision test도 함께 쓸 수 있어 GPU 조회를 흉내 냅니다.
    # MMDetection의 amp 자체는 MMCV op가 지원하는 fp16으로 확정되어 이 값을 묻지 않습니다.
    monkeypatch.setattr(pipeline_module, "_native_bf16_supported", lambda: True)


def _mmdetection_raw(**overrides):
    """8GB 제약을 갖춘 최소 설정입니다. 그것 말고를 보고 싶을 때 씁니다."""

    raw = {
        "run_id": "t",
        "architecture": "dino_r50_4scale",
        "device": "cuda",
        "precision": "amp",
        "optimizer": "AdamW",
        "batch_size": 1,
    }
    raw.update(overrides)
    return {"train": raw}


def test_input_size_defaults_to_the_size_the_models_were_tuned_for(pretend_cuda):
    settings = _settings(_mmdetection_raw())

    assert settings["input_size"] == 640


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "640"])
def test_input_size_rejects_values_that_are_not_positive_integers(pretend_cuda, value):
    with pytest.raises(ValueError, match="input_size"):
        _settings(_mmdetection_raw(input_size=value))


def test_input_size_is_refused_with_a_torchvision_architecture():
    """torchvision model은 이 값을 쓰지 않습니다. 받으면 안 쓰고 버리는 셈입니다.

    조용히 무시하면 사용자는 크기를 정했다고 믿는데 학습은 원래 크기로 돕니다.
    """

    with pytest.raises(ValueError, match="input_size"):
        _settings(
            {
                "train": {
                    "run_id": "t",
                    "architecture": "retinanet_resnet50_fpn_v2",
                    "input_size": 640,
                }
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cpu"),
        ("precision", "fp32"),
        ("optimizer", "SGD"),
        ("batch_size", 2),
    ],
)
def test_mmdetection_refuses_combinations_that_do_not_fit_8gb(pretend_cuda, field, value):
    """작은 GPU에서 돌 가능성이 있는 조합만 받습니다. 시작한 뒤 터지면 밤을 버립니다."""

    raw = {
        "run_id": "t",
        "architecture": "dino_r50_4scale",
        "device": "cuda",
        "precision": "amp",
        "optimizer": "AdamW",
        "batch_size": 1,
    }
    raw[field] = value

    with pytest.raises(ValueError, match=field):
        _settings({"train": raw})


def test_mmdetection_checkpoint_carries_what_evaluate_needs(pretend_cuda):
    """제안서 012가 정한 값입니다. 없으면 evaluate가 torchvision으로 읽으려 듭니다."""

    settings = _settings(_mmdetection_raw(input_size=512))

    payload = _checkpoint_payload(
        {"epoch": 1, "model_state_dict": {}, "optimizer_state_dict": {}},
        settings,
        {"pill": 1},
        {1: 7},
    )

    assert payload["backend"] == "mmdetection"
    assert payload["architecture"] == "dino_r50_4scale"
    assert payload["model_config"] == {
        "schema_version": 1,
        "input_size": 512,
        "resize": "longest_edge",
        "pad_multiple": 32,
    }


def test_a_torchvision_checkpoint_carries_no_backend_key():
    """backend key가 없어야 evaluate가 지금까지처럼 torchvision으로 읽습니다.

    넣어 두면 옛 checkpoint와 모양이 달라지고, evaluate는 모르는 backend를 추측하지
    않고 멈춥니다.
    """

    settings = _settings({"train": {"run_id": "t"}})

    payload = _checkpoint_payload(
        {"epoch": 1, "model_state_dict": {}, "optimizer_state_dict": {}},
        settings,
        {"pill": 1},
        {1: 7},
    )

    assert "backend" not in payload
    assert "model_config" not in payload


def test_input_size_reaches_the_model_and_matches_the_checkpoint(pretend_cuda, monkeypatch):
    """설정한 크기로 학습해야 checkpoint에 적은 값과 같아집니다.

    adapter까지 닿지 않으면 학습은 기본값으로 돌고 checkpoint에는 설정값이 적힙니다.
    evaluate는 적힌 값으로 전처리하므로 학습과 추론이 조용히 갈라집니다. 오류는 나지
    않고 점수만 나빠집니다.
    """

    seen: dict[str, int] = {}
    monkeypatch.setattr(
        model_module,
        "build_mmdetection_model",
        lambda num_classes, *, architecture, pretrained, input_size: seen.update(
            {"input_size": input_size}
        ),
    )
    settings = _settings(_mmdetection_raw(input_size=512))

    model_module.build_model(
        4,
        architecture=settings["architecture"],
        pretrained=settings["pretrained"],
        input_size=settings["input_size"],
    )
    payload = _checkpoint_payload(
        {"epoch": 1, "model_state_dict": {}, "optimizer_state_dict": {}},
        settings,
        {"pill": 1},
        {1: 7},
    )

    assert seen["input_size"] == 512
    assert payload["model_config"]["input_size"] == seen["input_size"]


def test_the_pipeline_hands_the_configured_input_size_to_the_builder():
    """호출부가 이 값을 빠뜨리면 위 test는 통과하고 학습만 기본값으로 돕니다."""

    source = Path(pipeline_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_model"
    ]

    assert calls, "pipeline이 build_model을 부르지 않습니다."
    for call in calls:
        passed = {keyword.arg for keyword in call.keywords}
        assert "input_size" in passed, sorted(passed)


def test_training_config_records_input_size_only_for_mmdetection(pretend_cuda):
    """torchvision 실행은 이 값을 쓰지 않으므로 None으로 남깁니다."""

    mm = pipeline_module._training_config(_settings(_mmdetection_raw(input_size=512)))
    torchvision_run = pipeline_module._training_config(
        _settings({"train": {"run_id": "t"}})
    )

    assert mm["input_size"] == 512
    assert torchvision_run["input_size"] is None


def test_new_models_default_to_accumulating_eight_microbatches(pretend_cuda):
    """제안서 013이 정한 값입니다. 8GB에서 batch 1이라 그만큼 모아야 쓸 만합니다."""

    settings = _settings(_mmdetection_raw())

    assert settings["gradient_accumulation_steps"] == 8


def test_existing_models_keep_accumulating_one():
    """기존 모델의 학습 동작은 이 변경 전과 같아야 합니다."""

    settings = _settings({"train": {"run_id": "t"}})

    assert settings["gradient_accumulation_steps"] == 1
