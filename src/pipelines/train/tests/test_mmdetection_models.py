"""MMDetection 모델 adapter와 8GB 학습 계약 test입니다."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from src.pipelines.train.mmdetection_adapter import (
    build_mmdetection_config,
    prepare_mmdetection_batch,
)
from src.pipelines.train import model as model_module
from src.pipelines.train.model import MMDETECTION_ARCHITECTURES, SUPPORTED_ARCHITECTURES
from src.pipelines.train.pipeline import _checkpoint_payload, _settings
from src.pipelines.train import trainer as trainer_module


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

    assert architecture in SUPPORTED_ARCHITECTURES
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
            "scale_factor": (0.75, 0.75),
        }
    ]


def _new_model_config(architecture: str) -> dict[str, object]:
    return {
        "train": {
            "architecture": architecture,
            "optimizer": "AdamW",
            "device": "cuda",
            "precision": "amp",
            "batch_size": 1,
        }
    }


@pytest.mark.parametrize("architecture", MMDETECTION_ARCHITECTURES)
def test_mmdetection_settings_use_the_approved_8gb_defaults(monkeypatch, architecture):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: False)

    settings = _settings(_new_model_config(architecture))

    assert settings["input_size"] == 640
    assert settings["gradient_accumulation_steps"] == 8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device", "cpu", "train.device='cuda'"),
        ("precision", "fp32", "train.precision='amp'"),
        ("optimizer", "SGD", "train.optimizer='AdamW'"),
        ("batch_size", 2, "train.batch_size=1"),
    ],
)
def test_mmdetection_settings_reject_unverified_runtime_combinations(
    monkeypatch, field, value, message
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: False)
    config = _new_model_config("dino_r50_4scale")
    config["train"][field] = value

    with pytest.raises(ValueError, match=message):
        _settings(config)


def test_checkpoint_records_mmdetection_backend_and_json_safe_model_config():
    settings = {
        "architecture": "dino_r50_4scale",
        "input_size": 640,
        "gradient_accumulation_steps": 8,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "precision": {"mode": "amp", "dtype": "fp16", "grad_scaler": True},
    }

    payload = _checkpoint_payload(
        {"model_state_dict": {}}, settings, {"a": 1, "b": 2}, {1: 10, 2: 20}
    )

    assert payload["backend"] == "mmdetection"
    assert payload["model_config"] == {
        "schema_version": 1,
        "input_size": 640,
        "resize": "longest_edge",
        "pad_multiple": 32,
    }
    assert payload["category_ids"] == [0, 10, 20]
    assert payload["training_config"]["gradient_accumulation_steps"] == 8


def test_legacy_checkpoint_does_not_claim_mmdetection_metadata():
    payload = _checkpoint_payload(
        {"model_state_dict": {}},
        {"seed": 42},
        {"a": 1},
        {1: 10},
    )

    assert "backend" not in payload
    assert "model_config" not in payload
    assert payload["training_config"]["schema_version"] == 3


@pytest.mark.parametrize("architecture", MMDETECTION_ARCHITECTURES)
def test_model_builder_routes_mmdetection_class_count_and_input_size(
    monkeypatch, architecture
):
    sentinel = nn.Identity()
    calls = []

    def build(num_classes, **settings):
        calls.append((num_classes, settings))
        return sentinel

    monkeypatch.setattr(model_module, "build_mmdetection_model", build)

    result = model_module.build_model(
        4,
        architecture=architecture,
        pretrained=True,
        input_size=768,
    )

    assert result is sentinel
    assert calls == [
        (
            4,
            {
                "architecture": architecture,
                "pretrained": True,
                "input_size": 768,
            },
        )
    ]


class _ConstantDataset(Dataset):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int):
        return torch.ones((1, 1, 1)), {"labels": torch.tensor([1])}


class _ConstantLossDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):
        return {"loss": self.weight * torch.stack(images).mean()}


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_gradient_accumulation_updates_once_per_window_and_keeps_partial_window(
    monkeypatch,
):
    recorded: dict[str, _CountingSGD] = {}
    schedule_steps = []

    def build(parameters: list[nn.Parameter], settings: Mapping[str, object]):
        optimizer = _CountingSGD(parameters, lr=0.1, weight_decay=0.0, momentum=0.0)
        recorded["optimizer"] = optimizer
        return optimizer

    monkeypatch.setattr(trainer_module, "build_optimizer", build)
    monkeypatch.setattr(
        trainer_module,
        "build_lr_scheduler",
        lambda optimizer, settings, steps_per_epoch: type(
            "Schedule",
            (),
            {"step": lambda self: schedule_steps.append(steps_per_epoch)},
        )(),
    )
    model = _ConstantLossDetector()
    settings = {
        "seed": 7,
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "epochs": 1,
        "checkpoint_every": 1,
        "gradient_accumulation_steps": 2,
        "lr_scheduler": None,
        "early_stopping": None,
        "precision": {"mode": "fp32", "dtype": "fp32", "grad_scaler": False},
    }

    trainer_module.train_model(model, _ConstantDataset(), _ConstantDataset(), settings)

    assert recorded["optimizer"].step_calls == 3
    assert schedule_steps == [3, 3, 3]
    assert model.weight.item() == pytest.approx(0.7)
