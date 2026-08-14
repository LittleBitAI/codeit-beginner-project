import copy
import hashlib
import io
import json
import os
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from botocore.exceptions import ClientError
from PIL import Image
from torch import nn
from torchvision.models.detection import FasterRCNN, RetinaNet

from src.common import LocalStorage, S3Storage, StorageError
from src.common import train_contract
from src.pipelines import train
from src.pipelines.train import image_cache as image_cache_module
from src.pipelines.train import pipeline, trainer as trainer_module
from src.pipelines.train.dataset import DetectionAugmentation, REPOSITORY_ROOT, load_class_map
from src.pipelines.train.model import SUPPORTED_ARCHITECTURES, build_model
from src.pipelines.train import progress as progress_module
from src.pipelines.train.progress import SCHEMA, ProgressEmitter
from src.pipelines.train.trainer import build_optimizer, train_model


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(()))

    def forward(self, images, targets):
        image_term = sum(image.mean() * 0 for image in images)
        return {"loss_classifier": (self.weight - 0.25).square() + image_term}


class RetinaStyleDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(()))

    def forward(self, images, targets):
        image_term = sum(image.mean() * 0 for image in images)
        return {
            "classification": (self.weight - 0.25).square() + image_term,
            "bbox_regression": self.weight.square() * 0 + 0.5,
        }


class SequencedDetector(nn.Module):
    def __init__(self, validation_losses: list[float]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.validation_losses = iter(validation_losses)
        self.calls = 0

    def forward(self, images, targets):
        self.calls += 1
        value = 2.0 if self.calls % 2 else next(self.validation_losses)
        return {"sequence_loss": self.weight.square() * 0 + value}


class BatchNormDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):
        normalized = self.batch_norm(torch.stack(images))
        return {"loss_classifier": self.weight.square() + normalized.mean() * 0}


class InMemoryDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, value: float, size: int = 1) -> None:
        self.image = torch.full((3, 2, 2), value)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.image, {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _write_image(path: Path, *, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _manifest(image_name: str, images: int = 1) -> dict:
    """같은 이미지를 ``images``장 담은 manifest입니다.

    한 장짜리로는 microbatch를 모으는 경로를 지나지 못합니다. 학습에 쓰는 값이 아니라
    batch가 몇 개 나오는지가 필요한 test가 있어 같은 그림을 여러 번 넣습니다.
    """

    return {
        "images": [
            {"id": index, "file_name": image_name, "width": 16, "height": 12}
            for index in range(1, images + 1)
        ],
        "annotations": [
            {
                "id": index,
                "image_id": index,
                "category_id": 7,
                "bbox": [2, 3, 5, 4],
                "iscrowd": 0,
            }
            for index in range(1, images + 1)
        ],
        "categories": [{"id": 7, "name": "pill"}],
    }


def _absolute_storage_config(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    monkeypatch.setattr(pipeline, "build_model", lambda *args, **kwargs: TinyDetector())
    storage_root = tmp_path / "absolute-storage"
    storage = LocalStorage(storage_root)
    source_train = tmp_path / "source-train.png"
    source_validation = tmp_path / "source-validation.png"
    _write_image(source_train, color="red")
    _write_image(source_validation, color="blue")
    storage.upload_file(source_train, "datasets/train.png")
    storage.upload_file(source_validation, "datasets/validation.png")
    inputs = {
        "train_manifest_uri": storage.write_json(
            "datasets/train.json", _manifest("train.png")
        ),
        "validation_manifest_uri": storage.write_json(
            "datasets/validation.json", _manifest("validation.png")
        ),
        "class_map_uri": storage.write_json("datasets/class_map.json", {"pill": 1}),
        "dataset_summary_uri": storage.write_json(
            "datasets/summary.json", {"train_images": 1, "validation_images": 1}
        ),
    }
    config = {
        "storage": {"backend": "local", "local": {"root": _relative(storage_root)}},
        "inputs": {"data": inputs},
        "train": {
            "run_id": "absolute-inputs",
            "epochs": 1,
            "output_dir": _relative(tmp_path / "absolute-outputs"),
        },
    }
    return config, storage_root


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "build_model", lambda *args, **kwargs: TinyDetector())
    fixture_directory = tmp_path / "fixtures"
    train_image = fixture_directory / "train.png"
    validation_image = fixture_directory / "validation.png"
    _write_image(train_image, color="red")
    _write_image(validation_image, color="blue")
    train_manifest = fixture_directory / "train.json"
    validation_manifest = fixture_directory / "validation.json"
    class_map = fixture_directory / "class_map.json"
    dataset_summary = fixture_directory / "summary.json"
    _write_json(train_manifest, _manifest(train_image.name))
    _write_json(validation_manifest, _manifest(validation_image.name))
    _write_json(class_map, {"pill": 1})
    _write_json(dataset_summary, {"train_images": 1, "validation_images": 1})
    return {
        "storage": {"backend": "local", "local": {"root": _relative(tmp_path / "storage")}},
        "inputs": {
            "data": {
                "train_manifest_uri": _relative(train_manifest),
                "validation_manifest_uri": _relative(validation_manifest),
                "class_map_uri": _relative(class_map),
                "dataset_summary_uri": _relative(dataset_summary),
            }
        },
        "train": {
            "run_id": "cpu-smoke",
            "seed": 17,
            "epochs": 2,
            "batch_size": 1,
            "device": "cpu",
            "output_dir": _relative(tmp_path / "outputs"),
        },
    }


def test_run_trains_and_writes_contract_artifacts_without_mutating_inputs(local_config):
    original_inputs = copy.deepcopy(local_config["inputs"])

    result = train.run(local_config)

    assert set(result) == {"status", "artifacts", "summary", "message"}
    assert result["status"] == "ok"
    assert set(result["artifacts"]) == {
        "run_id",
        "best_checkpoint_uri",
        "last_checkpoint_uri",
        "training_history_uri",
    }
    assert local_config["inputs"] == original_inputs
    assert result["summary"]["train_images"] == 1
    assert result["summary"]["validation_images"] == 1
    assert result["summary"]["class_count"] == 1
    assert result["summary"]["epochs"] == 2
    assert result["summary"]["planned_epochs"] == 2
    assert result["summary"]["completed_epochs"] == 2
    assert result["summary"]["stopped_early"] is False

    for name in ("best_checkpoint_uri", "last_checkpoint_uri", "training_history_uri"):
        uri = result["artifacts"][name]
        assert not Path(uri).is_absolute()
        assert (REPOSITORY_ROOT / uri).is_file()
    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    assert [entry["epoch"] for entry in history] == [1, 2]
    for artifact_name in ("best_checkpoint_uri", "last_checkpoint_uri"):
        checkpoint = torch.load(
            REPOSITORY_ROOT / result["artifacts"][artifact_name],
            map_location="cpu",
            weights_only=True,
        )
        assert {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "validation_loss",
            "architecture",
            "num_classes",
            "class_map",
            "category_ids",
            "seed",
            "training_config",
        } <= set(checkpoint)
        assert checkpoint["architecture"] == "fasterrcnn_mobilenet_v3_large_320_fpn"
        assert checkpoint["class_map"] == {"pill": 1}
        assert checkpoint["category_ids"] == [0, 7]
        assert checkpoint["num_classes"] == 2
        assert checkpoint["seed"] == 17
        assert isinstance(checkpoint["model_state_dict"], dict)
        assert isinstance(checkpoint["optimizer_state_dict"], dict)
        assert checkpoint["optimizer_state_dict"]["param_groups"]

        recorded = checkpoint["training_config"]
        assert set(recorded) == {
            "schema_version",
            "run_id",
            "architecture",
            "optimizer",
            "augmentation",
            "lr_scheduler",
            "seed",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "input_size",
            "num_workers",
            "device",
            "precision",
            "pretrained",
            "early_stopping",
            "resume",
        }
        assert recorded["schema_version"] == 5
        assert recorded["resume"] is None
        assert recorded["run_id"] == "cpu-smoke"
        assert recorded["architecture"] == checkpoint["architecture"]
        assert recorded["optimizer"] == {
            "name": "SGD",
            "learning_rate": 0.005,
            "weight_decay": 0.0005,
            "momentum": 0.9,
        }
        assert recorded["augmentation"] == {
            "version": 1,
            "preset": "none",
            "horizontal_flip_probability": 0.0,
            "vertical_flip_probability": 0.0,
            "color_probability": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
        }
        assert recorded["seed"] == checkpoint["seed"]
        assert recorded["epochs"] == 2
        assert recorded["batch_size"] == 1
        assert recorded["num_workers"] == 0
        assert recorded["device"] == "cpu"
        assert recorded["precision"] == {
            "mode": "fp32",
            "dtype": "fp32",
            "grad_scaler": False,
        }
        assert recorded["pretrained"] is False
        assert recorded["early_stopping"] is None
        assert 1 <= checkpoint["epoch"] <= recorded["epochs"]

        if artifact_name == "best_checkpoint_uri":
            assert checkpoint["epoch"] == result["summary"]["best_epoch"]
            assert checkpoint["validation_loss"] == result["summary"][
                "best_validation_loss"
            ]
        else:
            assert checkpoint["epoch"] == recorded["epochs"]


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [("AdamW", torch.optim.AdamW), ("SGD", torch.optim.SGD), ("Adam", torch.optim.Adam)],
)
def test_optimizer_factory_uses_the_selected_implementation(name, expected_type):
    parameter = nn.Parameter(torch.tensor(1.0))
    settings = {
        "optimizer": name,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
    }
    if name == "SGD":
        settings["momentum"] = 0.8
    else:
        settings.update({"beta1": 0.85, "beta2": 0.95, "epsilon": 1e-7})

    optimizer = build_optimizer([parameter], settings)

    assert isinstance(optimizer, expected_type)


def test_explicit_adamw_run_records_effective_reproducibility_settings(local_config):
    local_config["train"].update(
        {
            "optimizer": "AdamW",
            "architecture": "fasterrcnn_resnet50_fpn_v2",
            "augmentation": {"preset": "pill_basic"},
            "learning_rate": 0.0002,
        }
    )

    result = train.run(local_config)

    assert result["status"] == "ok"
    checkpoint = torch.load(
        REPOSITORY_ROOT / result["artifacts"]["last_checkpoint_uri"],
        map_location="cpu",
        weights_only=True,
    )
    recorded = checkpoint["training_config"]
    assert checkpoint["architecture"] == "fasterrcnn_resnet50_fpn_v2"
    assert checkpoint["epoch"] == 2
    assert checkpoint["seed"] == 17
    assert recorded["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-08,
    }
    assert recorded["augmentation"] == {
        "version": 1,
        "preset": "pill_basic",
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.5,
        "color_probability": 0.3,
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.1,
        "hue": 0.02,
    }
    assert recorded["seed"] == 17
    assert recorded["epochs"] == 2
    optimizer_group = checkpoint["optimizer_state_dict"]["param_groups"][0]
    assert optimizer_group["lr"] == 0.0002
    assert optimizer_group["weight_decay"] == 0.01
    assert optimizer_group["betas"] == (0.9, 0.999)
    assert optimizer_group["eps"] == 1e-8
    assert result["summary"]["optimizer"] == "AdamW"
    assert result["summary"]["augmentation"] == "pill_basic"


def test_train_reads_exactly_the_setting_names_in_the_shared_contract(monkeypatch):
    """GUI가 그 이름으로 값을 실어 보냅니다. 여기가 그것을 정말 읽는 쪽입니다.

    값이 같은지는 계약의 표들이 지키지만, 값을 담아 보내는 **이름**은 지금까지 아무도
    지키지 않았습니다. web은 이 파일을 import할 수 없어 이름을 옮겨 적을 뿐이라, 한쪽이
    이름을 바꾸며 자기 test까지 함께 고치면 양쪽 다 초록인 채로 그 값이 조용히
    버려집니다. 여기서는 계약의 이름만 보고, web을 부르지 않습니다.

    이름 목록만 대조하면 부족합니다. 결과에 담는 이름은 그대로 두고 **입력에서 찾는
    이름만** 바꿔도(``raw.get("resume_from")`` → ``raw.get("resume_uri")``) 목록은 그대로
    맞기 때문입니다. 그래서 보낸 값이 결과에 **그대로 들어왔는지**까지 봅니다.

    보내는 값은 **하나도 기본값과 같지 않아야** 합니다. 기본값을 보내면 lookup 이름을
    바꿔도 fallback이 같은 값을 돌려주어 이 test가 그대로 통과합니다.

    optimizer마다 받는 칸이 다르므로(SGD의 momentum, AdamW의 beta) 나눠 읽고 합칩니다.
    ``input_size``와 절반 정밀도는 MMDetection model에서만 받고 그 조합은 CUDA를
    요구하므로, 마지막 한 벌은 CUDA가 있는 척하고 읽습니다. ``resume``은
    ``resume_from``을 보고 train이 만드는 값이라 뺍니다.
    """

    defaults = train_contract.SETTING_DEFAULTS
    common = {
        "run_id": "train-keys",
        "epochs": defaults["epochs"] + 2,
        "checkpoint_every": defaults["checkpoint_every"] + 1,
        "seed": defaults["seed"] + 1,
        "pretrained": not defaults["pretrained"],
        # CPU 기본값은 0입니다. 명시한 값은 그대로 쓰입니다.
        "num_workers": 2,
        "output_dir": f"{defaults['output_dir']}/nested",
        "output_prefix": f"{defaults['output_prefix']}/nested",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "augmentation": {"preset": "pill_basic"},
        "gradient_accumulation_steps": 2,
        "early_stopping": {"patience": 2, "min_delta": 0.0},
        "lr_scheduler": {"name": "cosine", "warmup_steps": 5, "min_lr_factor": 0.1},
        "resume_from": "artifacts/experiments/completed/.old.partial/last_checkpoint.pt",
    }
    sent = [
        {
            **common,
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "optimizer": "AdamW",
            "device": "cpu",
            "batch_size": defaults["batch_size"] + 1,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        },
        {
            **common,
            "architecture": "retinanet_resnet50_fpn_v2",
            "optimizer": "SGD",
            "device": "cpu",
            "batch_size": defaults["batch_size"] + 1,
            "momentum": 0.9,
        },
        {
            # `input_size`와 절반 정밀도는 이 model에서만 받습니다. 그 조합은 CUDA와
            # batch_size 1을 함께 요구하므로 나머지 값만 기본값과 다르게 보냅니다.
            **common,
            "architecture": train_contract.MMDETECTION_ARCHITECTURES[0],
            "optimizer": "AdamW",
            "device": "cuda",
            "precision": "amp",
            "batch_size": 1,
            "input_size": train_contract.DEFAULT_INPUT_SIZE + 160,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        },
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    read = [pipeline._settings({"train": dict(one)}) for one in sent]

    names = set().union(*(set(one) for one in read)) - {"resume"}
    assert names == set(train_contract.SETTING_KEYS)

    for one, got in zip(sent, read):
        for name, value in one.items():
            landed = got[name]
            if isinstance(value, dict):
                # 정규화가 빠진 값을 채워 넣으므로, 보낸 짝이 남아 있는지만 봅니다.
                assert {**landed, **value} == landed, f"{name}이 그대로 오지 않았습니다."
            elif isinstance(landed, dict):
                # precision처럼 고른 이름 하나가 여러 값으로 펼쳐지는 칸입니다.
                assert value in landed.values(), f"{name}이 그대로 오지 않았습니다."
            else:
                assert landed == value, f"{name}이 그대로 오지 않았습니다."


def test_every_mmdetection_name_in_the_shared_contract_has_a_config_here():
    """계약이 이름을 정하고, 그 이름으로 어떤 detector를 만들지는 여기서 정합니다.

    이름만 늘고 config가 없으면 GUI는 즉시 그 모델을 고를 수 있게 내놓는데 학습은
    엉뚱한 모델로 돌거나 죽습니다. 두 이름이 실제로 갈라지는지도 함께 봅니다.
    """

    from src.pipelines.train.mmdetection_adapter import (
        CASCADE_ARCHITECTURE,
        DINO_ARCHITECTURE,
        build_mmdetection_config,
    )

    assert {DINO_ARCHITECTURE, CASCADE_ARCHITECTURE} == set(
        train_contract.MMDETECTION_ARCHITECTURES
    )
    built = {
        name: build_mmdetection_config(name, foreground_classes=3)["type"]
        for name in train_contract.MMDETECTION_ARCHITECTURES
    }
    assert len(set(built.values())) == len(built), f"같은 detector로 갈립니다: {built}"


def test_a_contract_name_without_a_config_here_stops_instead_of_training_cascade(
    monkeypatch,
):
    """계약에 이름만 늘고 여기 config가 없을 때, 조용히 cascade로 학습하지 않는다.

    GUI는 계약을 읽어 그 이름을 곧바로 고를 수 있게 내놓습니다. 여기가 마지막 관문이라
    떨어뜨리면 사람은 다른 모델을 골랐다고 믿은 채 밤새 cascade를 학습합니다.
    """

    from src.pipelines.train import mmdetection_adapter

    monkeypatch.setattr(
        mmdetection_adapter,
        "MMDETECTION_ARCHITECTURES",
        (*train_contract.MMDETECTION_ARCHITECTURES, "later_added_detector"),
    )
    with pytest.raises(ValueError, match="later_added_detector"):
        mmdetection_adapter.build_mmdetection_config(
            "later_added_detector", foreground_classes=3
        )


def test_every_preset_name_in_the_shared_contract_is_implemented_here():
    """계약이 이름을 정하고, 그 이름이 실제로 무엇을 바꾸는지는 여기서 정합니다.

    나머지 값(model·optimizer·기본값)은 계약에서 그대로 가져다 쓰므로 어긋날 수
    없지만, 증강만은 이름과 내용이 나뉘어 있습니다. 계약에만 이름을 더하면 GUI는
    그것을 고를 수 있게 되고 학습은 KeyError로 죽습니다.
    """

    assert tuple(pipeline.AUGMENTATION_PRESETS) == train_contract.AUGMENTATIONS


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("optimizer", "Lion", "train.optimizer must be one of"),
        ("architecture", "unknown_detector", "train.architecture must be one of"),
        ("augmentation", {"preset": "unsafe_crop"}, "augmentation.preset must be one of"),
        ("precision", "fp8", "train.precision must be one of"),
        ("weight_decay", -1.0, "train.weight_decay must be a number"),
    ],
)
def test_invalid_config_fails_before_writing_run_artifacts(
    local_config, setting, value, message
):
    local_config["train"][setting] = value

    result = train.run(local_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert message in result["message"]
    output = REPOSITORY_ROOT / local_config["train"]["output_dir"] / "cpu-smoke"
    assert not output.exists()


@pytest.mark.parametrize(
    ("early_stopping", "message"),
    [
        ({}, "train.early_stopping.patience must be an integer >= 1"),
        ({"patience": True}, "train.early_stopping.patience must be an integer >= 1"),
        (
            {"patience": 1, "min_delta": float("nan")},
            "train.early_stopping.min_delta must be a number >= 0.0",
        ),
        (
            {"patience": 1, "unexpected": 3},
            "train.early_stopping contains unsupported settings: unexpected",
        ),
    ],
)
def test_invalid_early_stopping_fails_before_writing_artifacts(
    local_config, early_stopping, message
):
    local_config["train"]["early_stopping"] = early_stopping

    result = train.run(local_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert message in result["message"]
    output = REPOSITORY_ROOT / local_config["train"]["output_dir"] / "cpu-smoke"
    assert not output.exists()


@pytest.mark.parametrize(
    ("lr_scheduler", "message"),
    [
        ("cosine", "train.lr_scheduler must be an object"),
        ({"name": "sqrt"}, "train.lr_scheduler.name must be one of: none, cosine, step, linear"),
        (
            {"name": "cosine", "warmup_steps": -1},
            "train.lr_scheduler.warmup_steps must be an integer >= 0",
        ),
        (
            {"name": "cosine", "warmup_start_factor": 0.0},
            "train.lr_scheduler.warmup_start_factor must be a number > 0.0 and <= 1.0",
        ),
        (
            {"name": "cosine", "min_lr_factor": 1.5},
            "train.lr_scheduler.min_lr_factor must be a number >= 0.0 and <= 1.0",
        ),
        (
            {"name": "step", "step_size": 0},
            "train.lr_scheduler.step_size must be an integer >= 1",
        ),
        (
            {"name": "step", "gamma": 0.0},
            "train.lr_scheduler.gamma must be a number > 0.0 and <= 1.0",
        ),
        (
            {"name": "cosine", "step_size": 2},
            "train.lr_scheduler.step_size is not used by train.lr_scheduler.name=cosine",
        ),
        (
            {"name": "cosine", "unexpected": 3},
            "train.lr_scheduler contains unsupported settings: unexpected",
        ),
    ],
)
def test_invalid_lr_scheduler_fails_before_writing_artifacts(
    local_config, lr_scheduler, message
):
    """고른 schedule이 쓰지 않는 값은 조용히 무시하지 않고 학습 전에 거부합니다."""

    local_config["train"]["lr_scheduler"] = lr_scheduler

    result = train.run(local_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert message in result["message"]
    output = REPOSITORY_ROOT / local_config["train"]["output_dir"] / "cpu-smoke"
    assert not output.exists()


@pytest.mark.parametrize(
    ("optimizer", "irrelevant"),
    [
        ("AdamW", {"momentum": 0.9}),
        ("Adam", {"momentum": 0.9}),
        ("SGD", {"beta1": 0.9}),
        ("SGD", {"beta2": 0.999}),
        ("SGD", {"epsilon": 1e-8}),
    ],
)
def test_irrelevant_optimizer_settings_are_rejected_before_writing_artifacts(
    local_config, optimizer, irrelevant
):
    local_config["train"].update({"optimizer": optimizer, **irrelevant})

    result = train.run(local_config)

    field = next(iter(irrelevant))
    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert f"train.{field} is not used by train.optimizer={optimizer}" in result["message"]
    output = REPOSITORY_ROOT / local_config["train"]["output_dir"] / "cpu-smoke"
    assert not output.exists()


def test_adam_beta_bounds_are_validated_when_the_setting_is_meaningful(local_config):
    local_config["train"].update({"optimizer": "AdamW", "beta1": 1.0})

    result = train.run(local_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "train.beta1 must be a number >= 0.0 and < 1.0" in result["message"]


@pytest.mark.parametrize(
    ("bf16_supported", "dtype", "grad_scaler"),
    [(True, "bf16", False), (False, "fp16", True)],
)
def test_torchvision_amp_precision_chooses_only_native_bf16(
    monkeypatch, bf16_supported, dtype, grad_scaler
):
    """MMCV custom op를 쓰지 않는 모델은 GPU가 지원하면 bf16을 사용합니다."""

    support = Mock(return_value=bf16_supported)
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pipeline.torch.cuda, "is_bf16_supported", support)

    settings = pipeline._settings(
        {"train": {"device": "cuda", "precision": "amp"}}
    )

    assert settings["precision"] == {
        "mode": "amp",
        "dtype": dtype,
        "grad_scaler": grad_scaler,
    }
    support.assert_called_once_with(including_emulation=False)


@pytest.mark.parametrize("mode", ["amp", "fp16", "bf16"])
def test_every_mixed_precision_mode_requires_cuda(local_config, mode):
    """절반 정밀도는 CUDA에서만 의미가 있습니다. CPU면 시작 전에 막습니다."""

    local_config["train"]["precision"] = mode

    result = train.run(local_config)

    assert result["status"] == "error"
    assert f"train.precision='{mode}' requires train.device='cuda'" in result["message"]


def test_fp16_can_be_chosen_directly_and_never_asks_about_bf16(monkeypatch):
    """T4처럼 bf16이 없는 GPU에서 쓸 수 있는, 고르면 그대로 되는 선택지입니다.

    fp16은 어느 CUDA GPU에서나 되므로 bf16 지원 여부를 물어볼 이유가 없습니다.
    """

    support = Mock(return_value=False)
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pipeline.torch.cuda, "is_bf16_supported", support)

    settings = pipeline._settings({"train": {"device": "cuda", "precision": "fp16"}})

    assert settings["precision"] == {
        "mode": "fp16",
        "dtype": "fp16",
        "grad_scaler": True,
    }
    support.assert_not_called()


def test_gpu_training_reads_images_in_worker_processes(monkeypatch):
    """GPU가 도는 동안 다음 batch의 이미지를 미리 풀어 두게 합니다.

    Windows는 worker에 dataset을 pickle해 보내는데 그 안의 S3 client가 pickle되지
    않으므로, 거기서는 늘리지 않습니다. CPU 학습은 기다릴 GPU가 없어 그대로 0입니다.
    """

    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)

    def workers(device: str) -> int:
        return pipeline._settings({"train": {"device": device}})["num_workers"]

    # os.name을 직접 바꾸지 않습니다. 바꿔 두면 그 사이에 나는 실패를 pytest가
    # 출력하다 pathlib에서 죽어, 깨진 이유가 보이지 않습니다.
    monkeypatch.setattr(pipeline, "WORKERS_ARE_SPAWNED", False)
    # core보다 많이 띄우면 서로 느려지기만 하고, core를 셀 수 없으면 하나만 띄웁니다.
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: 2)
    assert workers("cuda") == 2
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: 64)
    assert workers("cuda") == 4
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: None)
    assert workers("cuda") == 1

    assert workers("cpu") == 0

    monkeypatch.setattr(pipeline, "WORKERS_ARE_SPAWNED", True)
    assert workers("cuda") == 0

    # 직접 적은 값은 어디서나 그대로 씁니다.
    assert (
        pipeline._settings({"train": {"device": "cpu", "num_workers": 2}})["num_workers"]
        == 2
    )


def test_bf16_is_accepted_on_a_gpu_that_supports_it_natively(monkeypatch):
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        pipeline.torch.cuda, "is_bf16_supported", Mock(return_value=True)
    )

    settings = pipeline._settings({"train": {"device": "cuda", "precision": "bf16"}})

    assert settings["precision"] == {
        "mode": "bf16",
        "dtype": "bf16",
        "grad_scaler": False,
    }


def test_bf16_is_refused_on_a_gpu_that_cannot_do_it_natively(monkeypatch):
    """T4에서 bf16을 고르면 조용히 느려지는 대신 이유를 말하고 멈춥니다.

    emulation으로 도는 bf16은 밤새 돌린 학습을 통째로 버리게 만들 만큼 느립니다.
    """

    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        pipeline.torch.cuda, "is_bf16_supported", Mock(return_value=False)
    )

    with pytest.raises(ValueError, match="native bfloat16"):
        pipeline._settings({"train": {"device": "cuda", "precision": "bf16"}})


def test_native_bf16_probe_falls_back_when_torch_cannot_answer(monkeypatch):
    """``including_emulation``이 없는 옛 torch에서도 emulation을 골라 주면 안 됩니다.

    그 인자가 없는 torch의 ``is_bf16_supported()``는 T4에서도 True를 돌려줍니다.
    그대로 믿으면 T4가 bf16 emulation으로 학습합니다. 인자를 거부하면 compute
    capability로 직접 판단합니다.
    """

    def old_torch(*_args, **kwargs):
        if "including_emulation" in kwargs:
            raise TypeError("unexpected keyword argument 'including_emulation'")
        return True

    monkeypatch.setattr(pipeline.torch.cuda, "is_bf16_supported", old_torch)
    monkeypatch.setattr(
        pipeline.torch.cuda, "get_device_capability", lambda: (7, 5)
    )
    assert pipeline._native_bf16_supported() is False

    monkeypatch.setattr(
        pipeline.torch.cuda, "get_device_capability", lambda: (8, 6)
    )
    assert pipeline._native_bf16_supported() is True


def _progress_events(captured_stderr: str) -> list[dict]:
    """torch 경고처럼 JSON이 아닌 줄은 건너뛰고 진행 event만 모읍니다."""

    events = []
    for line in captured_stderr.splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == SCHEMA:
            events.append(payload)
    return events


def test_run_emits_progress_events_on_stderr_and_leaves_stdout_empty(
    local_config, capsys
):
    result = train.run(local_config)
    captured = capsys.readouterr()

    assert result["status"] == "ok"
    assert captured.out == ""

    events = _progress_events(captured.err)
    assert [event["event"] for event in events] == [
        "run_started",
        "epoch_started",
        "step_progress",
        "step_progress",
        "epoch_completed",
        "epoch_started",
        "step_progress",
        "step_progress",
        "epoch_completed",
        "training_completed",
    ]
    assert all(event["run_id"] == "cpu-smoke" for event in events)
    for event in events:
        datetime.strptime(event["ts"], "%Y-%m-%dT%H:%M:%S.%fZ")

    started = events[0]
    first = events[4]
    last = events[8]
    completed = events[9]
    assert started == {
        "schema": SCHEMA,
        "event": "run_started",
        "run_id": "cpu-smoke",
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "device": "cpu",
        "epochs": 2,
        "train_images": 1,
        "validation_images": 1,
        "class_count": 1,
        "ts": started["ts"],
    }
    assert [
        {
            "epoch": event["epoch"],
            "epochs": event["epochs"],
            "phase": event["phase"],
            "step": event["step"],
            "total_steps": event["total_steps"],
        }
        for event in events
        if event["event"] == "step_progress"
    ] == [
        {"epoch": 1, "epochs": 2, "phase": "train", "step": 1, "total_steps": 1},
        {
            "epoch": 1,
            "epochs": 2,
            "phase": "validation",
            "step": 1,
            "total_steps": 1,
        },
        {"epoch": 2, "epochs": 2, "phase": "train", "step": 1, "total_steps": 1},
        {
            "epoch": 2,
            "epochs": 2,
            "phase": "validation",
            "step": 1,
            "total_steps": 1,
        },
    ]
    assert set(first) == {
        "schema",
        "event",
        "run_id",
        "epoch",
        "epochs",
        "train_loss",
        "validation_loss",
        "train_loss_components",
        "validation_loss_components",
        "best_validation_loss",
        "best_epoch",
        "is_best",
        "epoch_seconds",
        "learning_rate",
        "ts",
    }
    assert (first["epoch"], last["epoch"]) == (1, 2)
    assert first["is_best"] is True
    assert first["best_epoch"] == 1
    assert first["best_validation_loss"] == first["validation_loss"]
    assert isinstance(first["epoch_seconds"], float)
    assert first["epoch_seconds"] >= 0.0

    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert [entry["epoch"] for entry in history] == [1, 2]
    assert [entry["train_loss"] for entry in history] == [
        first["train_loss"],
        last["train_loss"],
    ]
    assert result["summary"]["best_epoch"] == last["best_epoch"]
    assert result["summary"]["best_validation_loss"] == last["best_validation_loss"]
    assert completed == {
        "schema": SCHEMA,
        "event": "training_completed",
        "run_id": "cpu-smoke",
        "planned_epochs": 2,
        "completed_epochs": 2,
        "stopped_early": False,
        "best_epoch": result["summary"]["best_epoch"],
        "best_validation_loss": result["summary"]["best_validation_loss"],
        "ts": completed["ts"],
    }


def test_history_and_progress_preserve_model_loss_names(local_config, monkeypatch, capsys):
    local_config["train"]["epochs"] = 1
    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: RetinaStyleDetector()
    )

    result = train.run(local_config)
    events = _progress_events(capsys.readouterr().err)
    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "ok"
    record = history[0]
    expected_names = {"classification", "bbox_regression"}
    assert set(record["train_loss_components"]) == expected_names
    assert set(record["validation_loss_components"]) == expected_names
    assert record["train_loss"] == pytest.approx(sum(record["train_loss_components"].values()))
    assert record["validation_loss"] == pytest.approx(
        sum(record["validation_loss_components"].values())
    )
    completed_epoch = next(event for event in events if event["event"] == "epoch_completed")
    assert completed_epoch["train_loss_components"] == record["train_loss_components"]
    assert completed_epoch["validation_loss_components"] == record[
        "validation_loss_components"
    ]


def test_history_and_progress_record_the_learning_rate_actually_used(
    local_config, capsys
):
    """warmup과 decay가 실제로 걸렸는지 사람이 눈으로 확인할 수 있어야 합니다."""

    local_config["train"].update(
        {
            "epochs": 3,
            "learning_rate": 0.01,
            "lr_scheduler": {"name": "linear", "min_lr_factor": 0.0},
        }
    )

    result = train.run(local_config)
    events = _progress_events(capsys.readouterr().err)

    assert result["status"] == "ok", result["message"]
    assert result["summary"]["lr_scheduler"] == "linear"
    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    # 이미지가 한 장이라 epoch당 batch도 하나입니다. 곧 epoch마다 한 번씩 내려갑니다.
    assert [entry["learning_rate"] for entry in history] == pytest.approx(
        [0.01, 0.005, 0.0]
    )
    assert [
        event["learning_rate"] for event in events if event["event"] == "epoch_completed"
    ] == pytest.approx([0.01, 0.005, 0.0])


def test_run_without_a_schedule_keeps_the_learning_rate_constant(local_config):
    """설정하지 않은 실행은 이 기능이 생기기 전과 완전히 같아야 합니다."""

    local_config["train"]["learning_rate"] = 0.01

    result = train.run(local_config)

    assert result["summary"]["lr_scheduler"] == "none"
    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert [entry["learning_rate"] for entry in history] == [0.01, 0.01]
    last = _load(REPOSITORY_ROOT / result["artifacts"]["last_checkpoint_uri"])
    assert last["training_config"]["lr_scheduler"] is None
    assert last["resume_state"]["scheduler_state_dict"] is None


def test_checkpoint_records_the_schedule_it_was_trained_with(local_config):
    local_config["train"]["lr_scheduler"] = {"name": "cosine", "warmup_steps": 2}

    result = train.run(local_config)

    last = _load(REPOSITORY_ROOT / result["artifacts"]["last_checkpoint_uri"])
    assert last["training_config"]["lr_scheduler"] == {
        "name": "cosine",
        "warmup_steps": 2,
        "warmup_start_factor": 0.001,
        "min_lr_factor": 0.01,
    }


def test_early_stopping_keeps_best_and_actual_last_checkpoint_separate(
    local_config, monkeypatch, capsys
):
    local_config["train"].update(
        {
            "epochs": 6,
            "early_stopping": {"patience": 2, "min_delta": 0.05},
        }
    )
    monkeypatch.setattr(
        pipeline,
        "build_model",
        lambda *args, **kwargs: SequencedDetector([1.0, 0.98, 0.94, 0.93, 0.95, 0.96]),
    )

    result = train.run(local_config)
    events = _progress_events(capsys.readouterr().err)

    assert result["status"] == "ok"
    assert result["summary"]["planned_epochs"] == 6
    assert result["summary"]["completed_epochs"] == 5
    assert result["summary"]["stopped_early"] is True
    assert result["summary"]["best_epoch"] == 4
    assert result["summary"]["best_validation_loss"] == pytest.approx(0.93)
    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert [record["epoch"] for record in history] == [1, 2, 3, 4, 5]
    best = torch.load(
        REPOSITORY_ROOT / result["artifacts"]["best_checkpoint_uri"],
        map_location="cpu",
        weights_only=True,
    )
    last = torch.load(
        REPOSITORY_ROOT / result["artifacts"]["last_checkpoint_uri"],
        map_location="cpu",
        weights_only=True,
    )
    assert best["epoch"] == 4
    assert best["validation_loss"] == pytest.approx(0.93)
    assert last["epoch"] == 5
    assert last["validation_loss"] == pytest.approx(0.95)
    assert last["training_config"]["early_stopping"] == {
        "patience": 2,
        "min_delta": 0.05,
    }
    completed = events[-1]
    assert completed["event"] == "training_completed"
    assert completed["completed_epochs"] == 5
    assert completed["stopped_early"] is True


def test_progress_stream_stays_silent_for_the_dummy_execution(capsys):
    result = train.run({"execution": {"mode": "dummy"}})
    captured = capsys.readouterr()

    assert result["status"] == "ok"
    assert captured.out == ""
    assert _progress_events(captured.err) == []


def test_progress_emitter_does_not_raise_when_the_reader_closed_the_pipe():
    class ClosedPipe:
        def write(self, line):
            raise BrokenPipeError("reader is gone")

        def flush(self):
            raise BrokenPipeError("reader is gone")

    emitter = ProgressEmitter("cancelled-run", ClosedPipe())

    assert emitter.emit("epoch_started", epoch=1, epochs=2) is None


def test_progress_emitter_stays_quiet_outside_the_creating_process(monkeypatch):
    stream = io.StringIO()
    emitter = ProgressEmitter("worker-run", stream)
    monkeypatch.setattr(progress_module.os, "getpid", lambda: -1)

    emitter.emit("epoch_started", epoch=1, epochs=2)

    assert stream.getvalue() == ""


def test_progress_emitter_writes_null_instead_of_invalid_json_numbers():
    stream = io.StringIO()

    ProgressEmitter("nan-run", stream).emit(
        "epoch_completed", train_loss=float("nan"), validation_loss=float("inf")
    )

    line = stream.getvalue()
    assert line.endswith("\n")
    assert "NaN" not in line and "Infinity" not in line
    payload = json.loads(line)
    assert payload["train_loss"] is None
    assert payload["validation_loss"] is None


def test_step_progress_emits_first_timed_last_and_new_phase_events():
    stream = io.StringIO()
    moments = iter([0.0, 4.9, 5.0, 5.1, 5.1])
    emitter = ProgressEmitter("step-run", stream, clock=lambda: next(moments))

    for step in range(1, 5):
        emitter.emit_step_progress(
            epoch=2,
            epochs=3,
            phase="train",
            step=step,
            total_steps=4,
        )
    emitter.emit_step_progress(
        epoch=2,
        epochs=3,
        phase="validation",
        step=1,
        total_steps=1,
    )

    events = _progress_events(stream.getvalue())
    assert [
        (event["phase"], event["step"], event["total_steps"])
        for event in events
    ] == [
        ("train", 1, 4),
        ("train", 3, 4),
        ("train", 4, 4),
        ("validation", 1, 1),
    ]
    assert all(event["event"] == "step_progress" for event in events)
    assert all((event["epoch"], event["epochs"]) == (2, 3) for event in events)


def test_load_class_map_converts_category_ids_in_sorted_order_without_mutating_input():
    document = {"11": " capsule ", "7": "pill"}
    original = copy.deepcopy(document)
    storage = Mock()
    storage.read_json.return_value = document

    class_map = load_class_map("s3://bucket/class-map.json", storage)

    assert class_map == {"pill": 1, "capsule": 2}
    assert document == original


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"7": "pill", "11": " pill "}, "names must be unique"),
        ({"category-seven": "pill"}, "category ids must be non-negative integers"),
        ({"-7": "pill"}, "category ids must be non-negative integers"),
        ({"7": "   "}, "names must be non-empty strings"),
        ({"1": "pill", "01": "capsule"}, "category ids must be unique"),
    ],
)
def test_load_class_map_rejects_invalid_category_id_format(document, message):
    storage = Mock()
    storage.read_json.return_value = document

    with pytest.raises(ValueError, match=message):
        load_class_map("s3://bucket/class-map.json", storage)


def test_run_accepts_category_id_to_name_class_map_and_keeps_checkpoint_contract(
    local_config,
):
    class_map_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["class_map_uri"]
    _write_json(class_map_path, {"7": "pill"})
    original_inputs = copy.deepcopy(local_config["inputs"])

    result = train.run(local_config)

    assert result["status"] == "ok"
    assert local_config["inputs"] == original_inputs
    checkpoint = torch.load(
        REPOSITORY_ROOT / result["artifacts"]["best_checkpoint_uri"],
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["class_map"] == {"pill": 1}
    assert checkpoint["category_ids"] == [0, 7]


def test_checkpoint_category_ids_are_indexed_by_model_label():
    checkpoint = {"model_state_dict": {"weight": torch.tensor(1.0)}}

    payload = pipeline._checkpoint_payload(
        checkpoint,
        {"seed": 17},
        {"capsule": 2, "pill": 1},
        {2: 11, 1: 7},
    )

    assert payload["category_ids"] == [0, 7, 11]
    assert payload["model_state_dict"] is checkpoint["model_state_dict"]
    assert payload["num_classes"] == 3


def test_run_accepts_absolute_paths_returned_by_local_storage(tmp_path, monkeypatch):
    config, _ = _absolute_storage_config(tmp_path, monkeypatch)

    result = train.run(config)

    assert result["status"] == "ok"


def test_run_rejects_absolute_input_outside_local_storage_root(tmp_path, monkeypatch):
    config, storage_root = _absolute_storage_config(tmp_path, monkeypatch)
    outside_class_map = storage_root.parent / "outside-class-map.json"
    _write_json(outside_class_map, {"pill": 1})
    config["inputs"]["data"]["class_map_uri"] = str(outside_class_map.resolve())

    result = train.run(config)

    assert result["status"] == "error"
    assert "leaves the storage root" in result["message"]


def test_run_refuses_to_overwrite_an_existing_run(local_config):
    first = train.run(local_config)
    second = train.run(local_config)

    assert first["status"] == "ok"
    assert second == {
        "status": "error",
        "artifacts": {},
        "summary": {},
        "message": "training failed: training run artifact already exists: cpu-smoke",
    }


def test_seed_reproduces_training_history(local_config):
    first_config = copy.deepcopy(local_config)
    second_config = copy.deepcopy(local_config)
    first_config["train"]["run_id"] = "seed-first"
    second_config["train"]["run_id"] = "seed-second"
    first_config["train"]["augmentation"] = {"preset": "pill_basic"}
    second_config["train"]["augmentation"] = {"preset": "pill_basic"}

    first = train.run(first_config)
    second = train.run(second_config)

    first_history = json.loads(
        (REPOSITORY_ROOT / first["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    second_history = json.loads(
        (REPOSITORY_ROOT / second["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    assert first_history == second_history


class RandomWalkDetector(nn.Module):
    """전역 RNG를 실제로 씁니다.

    ``TinyDetector``는 난수를 쓰지 않아서 난수 상태를 하나도 되돌리지 않아도 이어서 한
    결과가 똑같이 나옵니다. 그러면 이어서 학습 test가 아무것도 지키지 못합니다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(()))

    def forward(self, images, targets):
        image_term = sum(image.mean() * 0 for image in images)
        return {"loss_classifier": (self.weight - torch.rand(())).square() + image_term}


class ExplodingDetector(nn.Module):
    """정해진 호출에서 터집니다. 세션이 끊긴 학습을 흉내 냅니다."""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(()))
        self.fail_on_call = fail_on_call
        self.calls = 0

    def forward(self, images, targets):
        self.calls += 1
        if self.calls >= self.fail_on_call:
            raise RuntimeError("simulated session loss")
        image_term = sum(image.mean() * 0 for image in images)
        return {"loss_classifier": (self.weight - 0.25).square() + image_term}


def _working_directory(local_config) -> Path:
    train_settings = local_config["train"]
    return (
        REPOSITORY_ROOT
        / train_settings["output_dir"]
        / f".{train_settings['run_id']}.partial"
    )


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def test_interrupted_training_leaves_a_resumable_working_checkpoint(
    local_config, monkeypatch
):
    """이 하나가 나머지를 가능하게 합니다. 이어서 할 대상이 남아야 합니다."""

    # epoch마다 train 1번 + validation 1번이므로 5번째 호출은 epoch 3입니다.
    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: ExplodingDetector(5)
    )
    local_config["train"]["epochs"] = 5

    result = train.run(local_config)

    assert result["status"] == "error"
    working = _working_directory(local_config)
    last = _load(working / "last_checkpoint.pt")
    assert last["resume_state"]["completed_epoch"] == 2
    assert [entry["epoch"] for entry in last["resume_state"]["history"]] == [1, 2]
    # 끊긴 checkpoint도 evaluate가 읽을 수 있어야 합니다.
    assert last["architecture"] == "fasterrcnn_mobilenet_v3_large_320_fpn"
    assert last["num_classes"] == 2
    assert (working / "best_checkpoint.pt").is_file()


def test_local_last_stays_resumable_if_publishing_the_best_checkpoint_is_interrupted(
    local_config, monkeypatch
):
    """두 파일 사이에서 중단돼도 이미 쓴 last 하나만으로 이어갈 수 있어야 합니다."""

    replace_checkpoint = pipeline._replace_checkpoint

    def interrupt_best(path, value):
        if path.name == "best_checkpoint.pt":
            raise OSError("simulated interruption while publishing best")
        replace_checkpoint(path, value)

    monkeypatch.setattr(pipeline, "_replace_checkpoint", interrupt_best)
    local_config["train"]["epochs"] = 2

    interrupted = train.run(local_config)

    assert interrupted["status"] == "error"
    last_path = _working_directory(local_config) / "last_checkpoint.pt"
    last = _load(last_path)
    assert "model_state_dict" in last["resume_state"]["best"]
    assert not (_working_directory(local_config) / "best_checkpoint.pt").exists()

    monkeypatch.setattr(pipeline, "_replace_checkpoint", replace_checkpoint)
    resumed = _resume_config(local_config, last_path)
    resumed["train"]["epochs"] = 3

    result = train.run(resumed)

    assert result["status"] == "ok", result["message"]


def test_a_successful_run_leaves_no_working_directory_behind(local_config):
    result = train.run(local_config)

    assert result["status"] == "ok"
    assert not _working_directory(local_config).exists()


def test_no_working_directory_when_the_run_fails_before_the_first_epoch(local_config):
    local_config["inputs"]["data"]["train_manifest_uri"] = "artifacts/missing.json"

    result = train.run(local_config)

    assert result["status"] == "error"
    assert not _working_directory(local_config).exists()


def test_same_local_run_id_is_claimed_before_reading_inputs(
    local_config, monkeypatch
):
    """동시에 시작한 두 학습이 같은 partial checkpoint를 공유하면 안 됩니다."""

    original = pipeline.load_class_map
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def blocking_load_class_map(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            this_call = calls
        if this_call == 1:
            first_entered.set()
            if not release_first.wait(10):
                raise RuntimeError("동시 실행 test가 첫 실행을 놓아주지 않았습니다.")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "load_class_map", blocking_load_class_map)
    first_result = {}
    first_thread = threading.Thread(
        target=lambda: first_result.update(train.run(copy.deepcopy(local_config))),
        daemon=True,
    )
    first_thread.start()
    try:
        assert first_entered.wait(10), "첫 실행이 입력을 읽기 시작하지 않았습니다."
        second_result = train.run(copy.deepcopy(local_config))
    finally:
        release_first.set()
        first_thread.join(timeout=10)

    assert not first_thread.is_alive()
    assert second_result["status"] == "error"
    assert "already active" in second_result["message"]
    assert calls == 1
    assert first_result["status"] == "ok", first_result["message"]


def test_run_rejects_a_working_directory_link_before_reading_inputs(
    local_config, monkeypatch
):
    """외부를 가리키는 작업 폴더로 checkpoint가 저장소 밖에 쓰이면 안 됩니다."""

    working = _working_directory(local_config)
    working.parent.mkdir(parents=True, exist_ok=True)
    inputs_read = False

    def record_input_read(*args, **kwargs):
        nonlocal inputs_read
        inputs_read = True
        raise RuntimeError("작업 폴더 검사 전에 입력을 읽었습니다.")

    monkeypatch.setattr(pipeline, "load_class_map", record_input_read)
    with tempfile.TemporaryDirectory(prefix="train-working-link-") as directory:
        target = Path(directory)
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(working), str(target)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        else:
            working.symlink_to(target, target_is_directory=True)
        try:
            result = train.run(local_config)

            assert result["status"] == "error"
            assert "working directory" in result["message"]
            assert not inputs_read
            assert not any(target.iterdir())
        finally:
            if working.exists():
                if os.name == "nt":
                    os.rmdir(working)
                else:
                    working.unlink()


def test_run_refuses_to_start_when_an_interrupted_run_is_still_on_disk(local_config):
    working = _working_directory(local_config)
    working.mkdir(parents=True)
    (working / "last_checkpoint.pt").write_bytes(b"leftover")

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "interrupted run" in result["message"]


def _resume_config(local_config, source: Path) -> dict:
    resumed = copy.deepcopy(local_config)
    resumed["train"]["run_id"] = "cpu-resumed"
    resumed["train"]["resume_from"] = _relative(source)
    return resumed


def test_resumed_training_matches_an_uninterrupted_run(
    local_config, tmp_path, monkeypatch
):
    """이어서 한 학습이 끊기지 않은 학습과 같아야 합니다.

    난수 상태·조기 종료 상태·history를 하나라도 빠뜨리면 여기서 갈라집니다.
    """

    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: RandomWalkDetector()
    )
    local_config["train"]["augmentation"] = {"preset": "pill_basic"}

    straight = copy.deepcopy(local_config)
    straight["train"]["run_id"] = "cpu-straight"
    straight["train"]["epochs"] = 4
    straight_result = train.run(straight)

    interrupted = copy.deepcopy(local_config)
    interrupted["train"]["epochs"] = 2
    train.run(interrupted)
    source = REPOSITORY_ROOT / interrupted["train"]["output_dir"] / "cpu-smoke"

    resumed = _resume_config(local_config, source / "last_checkpoint.pt")
    resumed["train"]["epochs"] = 4
    resumed_result = train.run(resumed)

    assert resumed_result["status"] == "ok", resumed_result["message"]
    straight_history = json.loads(
        (REPOSITORY_ROOT / straight_result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    resumed_history = json.loads(
        (REPOSITORY_ROOT / resumed_result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert resumed_history == straight_history

    straight_weights = _load(
        REPOSITORY_ROOT / straight_result["artifacts"]["last_checkpoint_uri"]
    )["model_state_dict"]
    resumed_weights = _load(
        REPOSITORY_ROOT / resumed_result["artifacts"]["last_checkpoint_uri"]
    )["model_state_dict"]
    assert set(straight_weights) == set(resumed_weights)
    for name, value in straight_weights.items():
        assert torch.equal(value, resumed_weights[name]), name


@pytest.mark.parametrize("accumulation", [1, 2])
def test_resumed_training_matches_an_uninterrupted_run_while_accumulating(
    local_config, tmp_path, monkeypatch, accumulation
):
    """모으는 중에 끊겼다 이어서 해도 끊기지 않은 학습과 같아야 합니다.

    기본 fixture는 학습 이미지가 한 장이라 accumulation을 올려도 묶음이 생기지
    않습니다. 그래서 여러 장으로 바꿔 microbatch를 실제로 모으게 한 뒤, epoch 경계에서
    끊었다 이어 붙입니다. 모으는 도중 optimizer와 schedule의 걸음이 어긋나면 여기서
    갈라집니다.
    """

    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: RandomWalkDetector()
    )
    fixture_directory = tmp_path / "fixtures"
    _write_json(
        fixture_directory / "train.json", _manifest("train.png", images=5)
    )
    _write_json(fixture_directory / "summary.json", {"train_images": 5, "validation_images": 1})
    # 5장을 2개씩 모으면 마지막 묶음이 한 장만 남습니다. 못 채운 묶음까지 지나갑니다.
    local_config["train"]["gradient_accumulation_steps"] = accumulation
    # schedule은 일부러 켜지 않습니다. schedule 길이는 epochs에서 나오므로, 2 epoch로
    # 끊었다가 4로 이어 붙이면 곡선 자체가 곧게 4를 돈 실행과 달라집니다. 그것은
    # 의도된 동작이라 여기서 섞으면 accumulation이 원인인지 알 수 없게 됩니다.

    straight = copy.deepcopy(local_config)
    straight["train"]["run_id"] = "cpu-straight"
    straight["train"]["epochs"] = 4
    straight_result = train.run(straight)
    assert straight_result["status"] == "ok", straight_result["message"]

    interrupted = copy.deepcopy(local_config)
    interrupted["train"]["epochs"] = 2
    train.run(interrupted)
    source = REPOSITORY_ROOT / interrupted["train"]["output_dir"] / "cpu-smoke"

    resumed = _resume_config(local_config, source / "last_checkpoint.pt")
    resumed["train"]["epochs"] = 4
    resumed_result = train.run(resumed)

    assert resumed_result["status"] == "ok", resumed_result["message"]
    straight_weights = _load(
        REPOSITORY_ROOT / straight_result["artifacts"]["last_checkpoint_uri"]
    )["model_state_dict"]
    resumed_weights = _load(
        REPOSITORY_ROOT / resumed_result["artifacts"]["last_checkpoint_uri"]
    )["model_state_dict"]
    for name, value in straight_weights.items():
        assert torch.equal(value, resumed_weights[name]), name


def test_resume_without_restoring_the_random_state_diverges(
    local_config, tmp_path, monkeypatch
):
    """앞 test가 정말 무언가를 지키는지 확인하는 대조군입니다.

    난수 복원을 꺼 두면 결과가 갈라져야 합니다. 갈라지지 않으면 앞 test는 난수 복원을
    하나도 검사하지 못하고 있다는 뜻입니다.
    """

    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: RandomWalkDetector()
    )
    local_config["train"]["augmentation"] = {"preset": "pill_basic"}

    straight = copy.deepcopy(local_config)
    straight["train"]["run_id"] = "control-straight"
    straight["train"]["epochs"] = 4
    straight_result = train.run(straight)

    interrupted = copy.deepcopy(local_config)
    interrupted["train"]["run_id"] = "control-interrupted"
    interrupted["train"]["epochs"] = 2
    train.run(interrupted)
    source = (
        REPOSITORY_ROOT
        / interrupted["train"]["output_dir"]
        / "control-interrupted"
        / "last_checkpoint.pt"
    )

    monkeypatch.setattr(trainer_module, "restore_rng", lambda *args, **kwargs: None)
    resumed = _resume_config(local_config, source)
    resumed["train"]["run_id"] = "control-resumed"
    resumed["train"]["epochs"] = 4
    resumed_result = train.run(resumed)

    straight_history = json.loads(
        (REPOSITORY_ROOT / straight_result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    resumed_history = json.loads(
        (REPOSITORY_ROOT / resumed_result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert resumed_history != straight_history


def test_resume_keeps_a_best_epoch_from_before_the_interruption(
    local_config, tmp_path, monkeypatch
):
    """이어붙이기 이전이 더 좋았다면 그 epoch이 그대로 best로 공개되어야 합니다."""

    # 첫 실행은 epoch 1이 가장 좋고, 이어서 한 실행은 그보다 나빠집니다.
    detectors = iter([SequencedDetector([0.1, 0.2]), SequencedDetector([0.9, 0.9])])
    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: next(detectors)
    )
    local_config["train"]["epochs"] = 2
    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )

    resumed = _resume_config(local_config, source)
    resumed["train"]["epochs"] = 4
    result = train.run(resumed)

    assert result["status"] == "ok", result["message"]
    assert result["summary"]["best_epoch"] == 1
    best = _load(REPOSITORY_ROOT / result["artifacts"]["best_checkpoint_uri"])
    assert best["epoch"] == result["summary"]["best_epoch"]


def test_resumed_run_records_where_it_came_from(local_config, tmp_path):
    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )

    resumed = _resume_config(local_config, source)
    resumed["train"]["epochs"] = 4
    result = train.run(resumed)

    assert result["status"] == "ok", result["message"]
    assert result["summary"]["resumed_from"] == _relative(source)
    assert result["summary"]["resumed_at_epoch"] == 2
    last = _load(REPOSITORY_ROOT / result["artifacts"]["last_checkpoint_uri"])
    assert last["training_config"]["resume"] == {
        "resumed_from": _relative(source),
        "resumed_at_epoch": 2,
    }


def test_run_records_no_resume_block_when_it_starts_from_scratch(local_config):
    result = train.run(local_config)

    assert result["summary"]["resumed_from"] is None
    last = _load(REPOSITORY_ROOT / result["artifacts"]["last_checkpoint_uri"])
    assert last["training_config"]["resume"] is None


@pytest.mark.parametrize(
    ("resume_from", "message"),
    [
        (17, "train.resume_from must be a non-empty checkpoint path"),
        ("", "train.resume_from must be a non-empty checkpoint path"),
        ("../outside.pt", "leaves the repository"),
        ("artifacts/missing-checkpoint.pt", "does not exist"),
    ],
)
def test_run_rejects_unusable_resume_paths(local_config, resume_from, message):
    local_config["train"]["resume_from"] = resume_from

    result = train.run(local_config)

    assert result["status"] == "error"
    assert message in result["message"]
    assert not _working_directory(local_config).exists()


def test_run_rejects_a_checkpoint_that_predates_resume_support(local_config, tmp_path):
    legacy = tmp_path / "legacy_checkpoint.pt"
    torch.save({"epoch": 1, "model_state_dict": {}}, legacy)
    local_config["train"]["resume_from"] = _relative(legacy)

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "cannot be resumed" in result["message"]


def test_run_rejects_resuming_past_the_planned_epochs(local_config, tmp_path):
    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )

    resumed = _resume_config(local_config, source)
    resumed["train"]["epochs"] = 2

    result = train.run(resumed)

    assert result["status"] == "error"
    assert "train.epochs" in result["message"]


def test_run_rejects_changed_optimizer_settings_before_building_the_model(
    local_config, monkeypatch
):
    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )
    resumed = _resume_config(local_config, source)
    resumed["train"].update({"epochs": 4, "learning_rate": 0.123})
    build_model_spy = Mock()
    monkeypatch.setattr(pipeline, "build_model", build_model_spy)

    result = train.run(resumed)

    assert result["status"] == "error"
    assert "optimizer settings" in result["message"]
    build_model_spy.assert_not_called()


def test_resumed_training_continues_the_learning_rate_schedule(
    local_config, monkeypatch
):
    """이어서 한 실행이 warmup을 처음부터 다시 하면 안 됩니다.

    schedule 위치를 되돌리지 않으면 learning rate가 갈라지고, 그 뒤 가중치도 갈라집니다.
    """

    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: RandomWalkDetector()
    )
    local_config["train"]["lr_scheduler"] = {
        "name": "cosine",
        "warmup_steps": 3,
        "warmup_start_factor": 0.1,
    }

    straight = copy.deepcopy(local_config)
    straight["train"]["run_id"] = "schedule-straight"
    straight["train"]["epochs"] = 4
    straight_result = train.run(straight)

    interrupted = copy.deepcopy(local_config)
    interrupted["train"]["epochs"] = 2
    train.run(interrupted)
    source = REPOSITORY_ROOT / interrupted["train"]["output_dir"] / "cpu-smoke"

    resumed = _resume_config(local_config, source / "last_checkpoint.pt")
    resumed["train"]["epochs"] = 4
    resumed_result = train.run(resumed)

    assert resumed_result["status"] == "ok", resumed_result["message"]
    straight_history = json.loads(
        (REPOSITORY_ROOT / straight_result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    resumed_history = json.loads(
        (REPOSITORY_ROOT / resumed_result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    # warmup 3 step이 2 epoch째까지 이어지므로, 되돌리지 않으면 여기가 갈라집니다.
    assert [entry["learning_rate"] for entry in resumed_history] == [
        entry["learning_rate"] for entry in straight_history
    ]
    assert resumed_history == straight_history


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (None, {"name": "cosine"}),
        ({"name": "cosine"}, {"name": "linear"}),
        ({"name": "cosine"}, {"name": "cosine", "warmup_steps": 5}),
    ],
)
def test_run_rejects_resuming_into_a_different_schedule(
    local_config, monkeypatch, first, second
):
    """schedule이 바뀌면 learning rate 궤적이 조용히 달라집니다. optimizer와 같은 이유입니다."""

    if first is not None:
        local_config["train"]["lr_scheduler"] = first
    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )

    resumed = _resume_config(local_config, source)
    resumed["train"].update({"epochs": 4, "lr_scheduler": second})
    build_model_spy = Mock()
    monkeypatch.setattr(pipeline, "build_model", build_model_spy)

    result = train.run(resumed)

    assert result["status"] == "error"
    assert "learning rate schedule" in result["message"]
    build_model_spy.assert_not_called()


@pytest.mark.parametrize(
    ("first", "second"),
    [(1, 2), (2, 1)],
)
def test_run_rejects_resuming_into_a_different_accumulation(
    local_config, monkeypatch, first, second
):
    """모으는 수가 바뀌면 갱신 횟수와 schedule 걸음이 함께 달라집니다.

    optimizer나 schedule이 바뀐 것과 같은 이유입니다. 막지 않으면 이어붙인 실행이
    끊기지 않고 돈 실행과 달라지는데, 오류는 나지 않고 점수로만 드러납니다.
    """

    local_config["train"]["gradient_accumulation_steps"] = first
    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )

    resumed = _resume_config(local_config, source)
    resumed["train"].update({"epochs": 4, "gradient_accumulation_steps": second})
    build_model_spy = Mock()
    monkeypatch.setattr(pipeline, "build_model", build_model_spy)

    result = train.run(resumed)

    assert result["status"] == "error"
    assert "gradient_accumulation_steps" in result["message"]
    build_model_spy.assert_not_called()


def test_a_worker_gets_its_own_s3_connection_instead_of_the_parents(monkeypatch):
    """fork한 worker가 부모의 S3 client를 그대로 쓰면 안 됩니다.

    boto3 client는 열린 socket을 들고 있어 process 사이에 나눠 쓸 수 없습니다.
    나눠 쓰면 부모의 checkpoint 업로드와 worker의 이미지 다운로드가 같은 socket에서
    엉킵니다. 미리 받기에서 놓친 이미지는 학습 도중 worker가 받으므로 이 경로는
    실제로 지나갑니다.
    """

    # 진짜 boto3 client를 만들면 test가 AWS 자격 증명을 찾아 나섭니다. 부모에게는
    # 가짜를 쥐어 주고, 새로 만든 쪽은 아무 연결도 물려받지 않았는지만 봅니다.
    parent = S3Storage(
        "bucket", prefix="datasets", region="ap-northeast-2", client=Mock()
    )

    class _Dataset:
        storage = parent

    dataset = _Dataset()
    monkeypatch.setattr(
        trainer_module.torch.utils.data,
        "get_worker_info",
        lambda: SimpleNamespace(dataset=dataset),
    )

    trainer_module.give_worker_its_own_storage(0)

    assert dataset.storage is not parent
    # 부모가 쓰던 연결을 물려받지 않았습니다. 자기 것은 처음 쓸 때 엽니다.
    assert dataset.storage._provided_client is None
    assert dataset.storage._cached_client is None
    # bucket과 접속 설정은 그대로여야 같은 곳에서 같은 권한으로 받습니다.
    assert dataset.storage.bucket == parent.bucket
    assert dataset.storage.prefix == parent.prefix
    assert dataset.storage.region == parent.region


def test_run_rejects_resuming_into_a_different_worker_count(local_config, monkeypatch):
    """worker 수가 바뀌면 augmentation이 뽑는 무작위 수가 달라집니다.

    worker가 없으면 주 process의 RNG에서, 있으면 worker마다 따로 뿌린 RNG에서
    나옵니다. 막지 않으면 이어붙인 실행이 끊기지 않고 돈 실행과 다른 그림으로
    배우는데, 오류는 나지 않고 점수로만 드러납니다.
    """

    local_config["train"]["augmentation"] = {"preset": "pill_basic"}
    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )

    resumed = _resume_config(local_config, source)
    resumed["train"].update({"epochs": 4, "num_workers": 2})
    build_model_spy = Mock()
    monkeypatch.setattr(pipeline, "build_model", build_model_spy)

    result = train.run(resumed)

    assert result["status"] == "error"
    assert "num_workers" in result["message"]
    build_model_spy.assert_not_called()


def test_resume_accepts_a_checkpoint_that_never_recorded_its_worker_count(local_config):
    """이 key를 몰랐던 checkpoint는 0으로 읽어 그대로 이어서 할 수 있어야 합니다.

    그때는 worker 없이 주 process가 직접 읽었습니다. 0으로 읽지 않으면 이 PR 이전에
    저장한 checkpoint가 전부 이어서 할 수 없게 됩니다.
    """

    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    del checkpoint["training_config"]["num_workers"]
    torch.save(checkpoint, source)

    resumed = _resume_config(local_config, source)
    resumed["train"]["epochs"] = 4

    result = train.run(resumed)

    assert result["status"] == "ok", result["message"]


@pytest.mark.parametrize("recorded", [512, 640])
def test_run_rejects_resuming_into_a_different_input_size(
    local_config, monkeypatch, recorded
):
    """입력 크기가 바뀌면 resize와 padding이 달라져 다른 그림으로 배웁니다.

    MMDetection 실행을 실제로 돌리려면 GPU와 mmdet이 필요하므로, 여기서는 그 값을
    기록한 checkpoint를 만들어 같은 거부 경로를 지나게 합니다. 이 실행은 그 값을 쓰지
    않아 기대값이 ``None``이므로, 기록된 값이 무엇이든 달라 거부되어야 합니다.
    """

    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["training_config"]["input_size"] = recorded
    torch.save(checkpoint, source)

    resumed = _resume_config(local_config, source)
    resumed["train"]["epochs"] = 4
    build_model_spy = Mock()
    monkeypatch.setattr(pipeline, "build_model", build_model_spy)

    result = train.run(resumed)

    assert result["status"] == "error"
    assert "input_size" in result["message"]
    build_model_spy.assert_not_called()


def test_an_old_checkpoint_without_the_input_size_key_resumes(local_config):
    """이 key를 몰랐던 옛 checkpoint는 없음으로 읽습니다.

    그때는 MMDetection을 고를 수 없었으므로 그 실행은 이 값을 쓰지 않았습니다.
    없다고 거부하면 이미 돌던 학습을 이어서 할 수 없게 됩니다.
    """

    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["training_config"].pop("input_size")
    torch.save(checkpoint, source)

    resumed = _resume_config(local_config, source)
    resumed["train"]["epochs"] = 4

    result = train.run(resumed)

    assert result["status"] == "ok", result["message"]


def test_an_old_checkpoint_without_the_accumulation_key_resumes_as_one(
    local_config, tmp_path
):
    """이 key를 몰랐던 옛 checkpoint는 1로 읽습니다.

    그때는 모으지 않고 batch마다 갱신했으므로 1이 맞습니다. 없다고 거부하면 이미
    돌던 학습을 이어서 할 수 없게 됩니다.
    """

    train.run(local_config)
    source = (
        REPOSITORY_ROOT
        / local_config["train"]["output_dir"]
        / "cpu-smoke"
        / "last_checkpoint.pt"
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["training_config"].pop("gradient_accumulation_steps")
    torch.save(checkpoint, source)

    resumed = _resume_config(local_config, source)
    resumed["train"]["epochs"] = 4

    result = train.run(resumed)

    assert result["status"] == "ok", result["message"]


def test_run_rejects_absolute_resume_path_before_recording_it(local_config):
    local_config["train"]["resume_from"] = str(
        (REPOSITORY_ROOT / "artifacts" / "private-checkpoint.pt").resolve()
    )

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "repository-relative POSIX" in result["message"]


def test_run_reports_missing_input_artifacts_without_partial_success():
    result = train.run({"inputs": {"data": {}}, "train": {}})

    assert set(result) == {"status", "artifacts", "summary", "message"}
    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert result["summary"] == {}
    assert "missing artifacts" in result["message"]


def test_run_rejects_invalid_class_map(local_config):
    class_map = REPOSITORY_ROOT / local_config["inputs"]["data"]["class_map_uri"]
    _write_json(class_map, {"pill": 0})

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "0 is background" in result["message"]


def test_run_rejects_invalid_bbox_schema(local_config):
    manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["train_manifest_uri"]
    manifest = _manifest("train.png")
    manifest["annotations"][0]["bbox"] = [2, 3, -1, 4]
    _write_json(manifest_path, manifest)

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "positive size" in result["message"]


def test_run_rejects_corrupt_image(local_config):
    manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["train_manifest_uri"]
    image_path = manifest_path.parent / "train.png"
    image_path.write_bytes(b"not-an-image")

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "missing or corrupt" in result["message"]


def test_run_rejects_train_validation_image_overlap(local_config):
    train_manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["train_manifest_uri"]
    validation_manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["validation_manifest_uri"]
    _write_json(validation_manifest_path, _manifest("train.png"))

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "overlapping images" in result["message"]


def test_run_rejects_different_category_ids_between_splits(local_config):
    validation_manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["validation_manifest_uri"]
    manifest = _manifest("validation.png")
    manifest["categories"][0]["id"] = 9
    manifest["annotations"][0]["category_id"] = 9
    _write_json(validation_manifest_path, manifest)

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "category ids must match" in result["message"]


def test_model_builder_creates_faster_rcnn_without_downloading_weights():
    model = build_model(3, pretrained=False)

    assert isinstance(model, FasterRCNN)
    assert model.roi_heads.box_predictor.cls_score.out_features == 3


@pytest.mark.parametrize(
    ("architecture", "expected_type"),
    [
        ("fasterrcnn_mobilenet_v3_large_320_fpn", FasterRCNN),
        ("fasterrcnn_resnet50_fpn_v2", FasterRCNN),
        ("retinanet_resnet50_fpn_v2", RetinaNet),
    ],
)
def test_supported_model_builders_round_trip_state_without_download(
    architecture, expected_type
):
    assert architecture in SUPPORTED_ARCHITECTURES
    first = build_model(3, architecture=architecture, pretrained=False)
    second = build_model(3, architecture=architecture, pretrained=False)

    second.load_state_dict(first.state_dict())

    assert isinstance(first, expected_type)


def test_pill_basic_augmentation_updates_flip_boxes_without_mutating_target(monkeypatch):
    draws = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(next(draws)))
    image = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6)
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 3.0]]),
        "labels": torch.tensor([1]),
    }
    original = copy.deepcopy(target)
    augmentation = DetectionAugmentation(
        {
            "preset": "pill_basic",
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "color_probability": 0.3,
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.02,
        }
    )

    augmented_image, augmented_target = augmentation(image, target)

    assert torch.equal(augmented_image, torch.flip(image, dims=(-2, -1)))
    assert torch.equal(augmented_target["boxes"], torch.tensor([[2.0, 1.0, 5.0, 3.0]]))
    assert torch.equal(target["boxes"], original["boxes"])


def test_pill_basic_color_augmentation_keeps_detection_target(monkeypatch):
    draws = iter((1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(next(draws)))
    image = torch.full((3, 4, 6), 0.5)
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 3.0]]),
        "labels": torch.tensor([1]),
    }
    augmentation = DetectionAugmentation(
        {
            "preset": "pill_basic",
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "color_probability": 0.3,
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.02,
        }
    )

    _, augmented_target = augmentation(image, target)

    assert torch.equal(augmented_target["boxes"], target["boxes"])
    assert torch.equal(augmented_target["labels"], target["labels"])


def test_faster_rcnn_cpu_forward_and_backward_smoke():
    torch.manual_seed(17)
    model = build_model(2, pretrained=False).cpu().train()
    image = torch.rand(3, 32, 32)
    target = {
        "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([256.0]),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }

    losses = model([image], [target])
    total = sum(losses.values())
    total.backward()

    assert set(losses) == {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    assert torch.isfinite(total)


def test_validation_does_not_update_batch_norm_running_statistics():
    model = BatchNormDetector()
    settings = {
        "seed": 17,
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "epochs": 1,
    }

    train_model(
        model,
        InMemoryDetectionDataset(1.0),
        InMemoryDetectionDataset(0.0),
        settings,
    )

    assert torch.allclose(model.batch_norm.running_mean, torch.full((3,), 0.1))


def _trainer_settings(**overrides) -> dict:
    settings = {
        "seed": 17,
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "epochs": 2,
    }
    settings.update(overrides)
    return settings


def _collect_checkpoints(**overrides) -> list[tuple[dict, dict, dict]]:
    """학습 도중 trainer가 넘겨 준 (last, best, resume_state)를 모읍니다."""

    seen: list[tuple[dict, dict, dict]] = []
    train_model(
        TinyDetector(),
        InMemoryDetectionDataset(1.0),
        InMemoryDetectionDataset(0.0),
        _trainer_settings(**overrides),
        on_checkpoint=lambda last, best, resume_state: seen.append(
            (last, best, resume_state)
        ),
    )
    return seen


def test_trainer_hands_back_resumable_state_after_every_epoch():
    seen = _collect_checkpoints(epochs=2)

    assert [state["completed_epoch"] for _, _, state in seen] == [1, 2]
    assert [last["epoch"] for last, _, _ in seen] == [1, 2]
    # history는 지금까지의 전체입니다. 이어서 시작해도 손실 곡선이 끊기지 않아야 합니다.
    assert [entry["epoch"] for entry in seen[-1][2]["history"]] == [1, 2]


def test_checkpoint_every_thins_out_the_resumable_state():
    seen = _collect_checkpoints(epochs=4, checkpoint_every=2)

    assert [state["completed_epoch"] for _, _, state in seen] == [2, 4]


def test_resumable_state_carries_what_the_next_run_cannot_recompute():
    _, best, state = _collect_checkpoints(epochs=1)[-1]

    assert state["version"] == trainer_module.RESUME_STATE_VERSION
    assert set(state["rng"]) == {"python", "numpy", "torch", "cuda", "dataloader"}
    assert state["grad_scaler_state_dict"] is None
    assert set(state["early_stopping"]) == {
        "reference_loss",
        "epochs_without_improvement",
    }
    # best는 숫자만 담습니다. 가중치는 옆에 저장되는 best_checkpoint.pt가 들고 있습니다.
    assert state["best"] == {"epoch": 1, "validation_loss": best["validation_loss"]}
    assert isinstance(best["model_state_dict"], dict)


def _lr_curve(*, epochs: int, steps_per_epoch: int = 1, **scheduler) -> list[float]:
    """설정한 schedule이 step마다 실제로 쓰는 learning rate를 모읍니다.

    base learning rate를 1.0으로 두었으므로 값이 곧 배율입니다.
    """

    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    settings = _trainer_settings(
        epochs=epochs,
        # trainer는 언제나 정규화가 끝난 설정만 받습니다. 기본값도 같이 확인됩니다.
        lr_scheduler=pipeline._lr_scheduler({"lr_scheduler": dict(scheduler)}),
    )
    schedule = trainer_module.build_lr_scheduler(optimizer, settings, steps_per_epoch)
    seen = []
    for _ in range(epochs * steps_per_epoch):
        seen.append(optimizer.param_groups[0]["lr"])
        # 학습 loop와 같은 순서(optimizer 먼저, schedule 나중)로 돌립니다.
        parameter.grad = torch.zeros(())
        optimizer.step()
        schedule.step()
    return seen


def test_warmup_raises_the_learning_rate_linearly_and_then_stops():
    curve = _lr_curve(epochs=8, name="none", warmup_steps=4, warmup_start_factor=0.25)

    assert curve[:5] == pytest.approx([0.25, 0.4375, 0.625, 0.8125, 1.0])
    # warmup이 끝난 뒤 name="none"은 base learning rate를 그대로 유지합니다.
    assert curve[4:] == pytest.approx([1.0] * 4)


def test_cosine_falls_from_the_base_learning_rate_to_the_minimum_factor():
    curve = _lr_curve(epochs=5, name="cosine", min_lr_factor=0.1)

    # 마지막 batch가 실제로 min_lr_factor를 쓰는 것이 중요합니다. 마지막 step에서도
    # 아직 내려가는 중이면 설정한 최저 learning rate는 한 번도 쓰이지 않습니다.
    assert curve[0] == pytest.approx(1.0)
    assert curve[-1] == pytest.approx(0.1)
    assert curve == sorted(curve, reverse=True)


def test_linear_falls_straight_from_the_base_learning_rate_to_the_minimum_factor():
    curve = _lr_curve(epochs=5, name="linear", min_lr_factor=0.2)

    assert curve == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2])


def test_step_schedule_drops_once_every_step_size_epochs():
    curve = _lr_curve(epochs=4, steps_per_epoch=2, name="step", step_size=2, gamma=0.1)

    assert curve == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])


def test_warmup_runs_before_the_decay_and_the_decay_still_reaches_its_minimum():
    """warmup과 decay는 한 schedule입니다. 둘을 같이 켜도 끝은 min_lr_factor입니다."""

    curve = _lr_curve(
        epochs=6, name="linear", warmup_steps=2, warmup_start_factor=0.5, min_lr_factor=0.0
    )

    assert curve[:3] == pytest.approx([0.5, 0.75, 1.0])
    assert curve[-1] == pytest.approx(0.0)


def test_fp16_runtime_uses_autocast_and_grad_scaler(monkeypatch):
    calls: list[tuple[str, object]] = []

    class AutocastContext:
        def __enter__(self):
            calls.append(("autocast_enter", None))

        def __exit__(self, *args):
            calls.append(("autocast_exit", None))

    class FakeScaler:
        def scale(self, loss):
            calls.append(("scale", loss))
            return loss

        def step(self, optimizer):
            calls.append(("step", optimizer))
            optimizer.step()

        def update(self):
            calls.append(("update", None))

    scaler = FakeScaler()
    scaler_factory = Mock(return_value=scaler)
    autocast = Mock(return_value=AutocastContext())
    monkeypatch.setattr(trainer_module.torch.amp, "GradScaler", scaler_factory)
    monkeypatch.setattr(trainer_module.torch, "autocast", autocast)
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    runtime = trainer_module._PrecisionRuntime(
        {"mode": "amp", "dtype": "fp16", "grad_scaler": True},
        torch.device("cuda"),
    )

    with runtime.autocast():
        loss = parameter.square()
    runtime.backward(loss)
    runtime.step(optimizer)

    autocast.assert_called_once_with(device_type="cuda", dtype=torch.float16)
    scaler_factory.assert_called_once_with("cuda", enabled=True)
    assert [name for name, _ in calls] == [
        "autocast_enter",
        "autocast_exit",
        "scale",
        "step",
        "update",
    ]


def test_fp16_runtime_saves_and_restores_grad_scaler_state(monkeypatch):
    loaded: list[dict] = []

    class FakeScaler:
        def state_dict(self):
            return {"scale": 128.0, "growth_tracker": 7}

        def load_state_dict(self, state):
            loaded.append(dict(state))

    monkeypatch.setattr(
        trainer_module.torch.amp,
        "GradScaler",
        Mock(return_value=FakeScaler()),
    )
    runtime = trainer_module._PrecisionRuntime(
        {"mode": "amp", "dtype": "fp16", "grad_scaler": True},
        torch.device("cuda"),
    )

    saved = runtime.state_dict()
    runtime.load_state_dict({"scale": 32.0, "growth_tracker": 11})

    assert saved == {"scale": 128.0, "growth_tracker": 7}
    assert loaded == [{"scale": 32.0, "growth_tracker": 11}]


def test_training_uses_precision_context_for_train_and_validation(monkeypatch):
    calls = {"autocast": 0, "backward": 0, "step": 0}

    class Runtime:
        def autocast(self):
            class Context:
                def __enter__(self):
                    calls["autocast"] += 1

                def __exit__(self, *args):
                    return None

            return Context()

        def backward(self, loss):
            calls["backward"] += 1
            loss.backward()

        def step(self, optimizer):
            calls["step"] += 1
            optimizer.step()

        def state_dict(self):
            return None

        def load_state_dict(self, state):
            assert state is None

    monkeypatch.setattr(
        trainer_module,
        "_PrecisionRuntime",
        lambda precision, device: Runtime(),
    )
    settings = {
        "seed": 17,
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "epochs": 1,
        "precision": {"mode": "amp", "dtype": "fp16", "grad_scaler": True},
    }

    train_model(
        TinyDetector(),
        InMemoryDetectionDataset(1.0),
        InMemoryDetectionDataset(0.0),
        settings,
    )

    assert calls == {"autocast": 2, "backward": 1, "step": 1}


def _s3_storage_with(existing: set[str]) -> S3Storage:
    """주어진 key만 있는 것처럼 구는 S3Storage를 만듭니다."""

    client = Mock()

    def head_object(**request):
        if request["Key"] in existing:
            return {}
        raise ClientError(
            {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
        )

    client.head_object.side_effect = head_object
    return S3Storage("bucket", client=client)


def _s3_settings(tmp_path) -> dict:
    return {
        "run_id": "colab-run",
        "output_prefix": "experiments/completed",
        "output_dir": _relative(tmp_path / "outputs"),
    }


def test_s3_run_refuses_to_overwrite_an_interrupted_run_of_the_same_name(tmp_path):
    """Colab은 runtime이 바뀌면 로컬 작업 폴더가 없습니다.

    그때 같은 run_id로 다시 돌리면 S3의 running checkpoint를 덮어쓰게 되는데, 그게
    중단된 학습의 유일한 사본입니다. 로컬을 볼 수 없으니 S3를 직접 봐야 합니다.
    """

    storage = _s3_storage_with(
        {"experiments/completed/colab-run/running/last_checkpoint.pt"}
    )

    with pytest.raises(FileExistsError, match="still on S3"):
        pipeline._reject_existing_run(_s3_settings(tmp_path), storage)


def test_s3_run_reports_a_finished_run_of_the_same_name_before_training(tmp_path):
    """몇 시간 학습한 뒤가 아니라 시작하기 전에 알려 줘야 합니다."""

    storage = _s3_storage_with({"experiments/completed/colab-run/completed.json"})

    with pytest.raises(FileExistsError, match="already exists: colab-run"):
        pipeline._reject_existing_run(_s3_settings(tmp_path), storage)


def test_s3_run_starts_when_the_run_id_is_free(tmp_path):
    pipeline._reject_existing_run(_s3_settings(tmp_path), _s3_storage_with(set()))


def _recording_s3_storage() -> tuple[S3Storage, dict]:
    """올라간 object를 그대로 모아 두는 S3Storage를 만듭니다."""

    uploaded: dict[str, bytes] = {}
    client = Mock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )

    def put_object(**request):
        body = request["Body"]
        uploaded[request["Key"]] = body.read() if hasattr(body, "read") else body

    client.put_object.side_effect = put_object
    client.upload_file.side_effect = lambda source, bucket, key, **_: uploaded.update(
        {key: Path(source).read_bytes()}
    )
    return S3Storage("bucket", client=client), uploaded


def test_s3_mirror_writes_one_self_contained_object(tmp_path):
    """S3는 여러 object를 한 번에 바꿀 수 없습니다.

    best와 last를 따로 올리면 하나만 성공했을 때 짝이 어긋나고, bucket에 남은 유일한
    사본을 이어서 쓸 수 없게 됩니다. 그래서 한 파일에 best 가중치까지 담습니다.
    """

    storage, uploaded = _recording_s3_storage()
    best = {
        "epoch": 2,
        "validation_loss": 0.5,
        "model_state_dict": {"weight": torch.tensor(2.0)},
        "optimizer_state_dict": {"param_groups": []},
    }
    last_payload = {
        "epoch": 3,
        "model_state_dict": {"weight": torch.tensor(3.0)},
        "resume_state": {
            "completed_epoch": 3,
            "best": {"epoch": 2, "validation_loss": 0.5},
        },
    }

    pipeline._mirror_to_s3(storage, last_payload, best, _s3_settings(tmp_path))

    assert list(uploaded) == [
        "experiments/completed/colab-run/running/last_checkpoint.pt"
    ]
    mirrored = torch.load(
        io.BytesIO(next(iter(uploaded.values()))), map_location="cpu", weights_only=True
    )
    mirrored_best = mirrored["resume_state"]["best"]
    assert mirrored_best["epoch"] == 2
    assert torch.equal(mirrored_best["model_state_dict"]["weight"], torch.tensor(2.0))
    # 원래 payload를 건드리면 로컬 작업 파일까지 두 배가 됩니다.
    assert "model_state_dict" not in last_payload["resume_state"]["best"]


def test_s3_checkpoint_scratch_rejects_an_artifacts_directory_link(
    tmp_path, monkeypatch
):
    """S3 checkpoint를 받거나 쓸 때 저장소 밖 junction을 따라가면 안 됩니다."""

    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts = repository / "artifacts"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(artifacts), str(outside)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    else:
        artifacts.symlink_to(outside, target_is_directory=True)

    storage, _ = _recording_s3_storage()
    storage.download_file = Mock()
    best = {
        "epoch": 1,
        "model_state_dict": {},
        "optimizer_state_dict": {},
    }
    last = {
        "epoch": 1,
        "resume_state": {"best": {"epoch": 1}},
    }
    monkeypatch.setattr(pipeline, "REPOSITORY_ROOT", repository)
    try:
        with pytest.raises(ValueError, match="scratch directory"):
            pipeline._read_checkpoint(
                "s3://bucket/checkpoint.pt", storage, label="train.resume_from"
            )
        with pytest.raises(ValueError, match="scratch directory"):
            pipeline._mirror_to_s3(
                storage,
                last,
                best,
                {"run_id": "linked", "output_prefix": "experiments"},
            )

        storage.download_file.assert_not_called()
        storage.client.upload_file.assert_not_called()
        assert not any(outside.iterdir())
    finally:
        if os.path.lexists(artifacts):
            if os.name == "nt":
                os.rmdir(artifacts)
            else:
                artifacts.unlink()


def test_resume_reads_best_weights_embedded_in_a_single_checkpoint(
    local_config, tmp_path
):
    """S3에서 받은 한 파일짜리 checkpoint는 옆 파일 없이도 이어서 할 수 있어야 합니다."""

    train.run(local_config)
    finished = REPOSITORY_ROOT / local_config["train"]["output_dir"] / "cpu-smoke"
    last = _load(finished / "last_checkpoint.pt")
    best = _load(finished / "best_checkpoint.pt")
    last["resume_state"]["best"] = {
        **last["resume_state"]["best"],
        "model_state_dict": best["model_state_dict"],
        "optimizer_state_dict": best["optimizer_state_dict"],
    }
    alone = tmp_path / "downloaded" / "last_checkpoint.pt"
    alone.parent.mkdir(parents=True)
    torch.save(last, alone)

    resumed = _resume_config(local_config, alone)
    resumed["train"]["epochs"] = 4
    result = train.run(resumed)

    assert result["status"] == "ok", result["message"]
    assert result["summary"]["resumed_at_epoch"] == 2


def test_s3_run_publishes_a_last_checkpoint_that_can_be_resumed(
    local_config, monkeypatch
):
    """backend에 따라 최종 artifact의 내용이 달라지면 안 됩니다."""

    storage, uploaded = _recording_s3_storage()
    monkeypatch.setattr(pipeline, "create_storage", lambda config: storage)

    result = train.run(local_config)

    assert result["status"] == "ok", result["message"]
    # 학습 중 올린 running/ 사본이 아니라 공개된 artifact를 봐야 합니다.
    published = next(
        value
        for key, value in uploaded.items()
        if "/attempts/" in key and key.endswith("last_checkpoint.pt")
    )
    document = torch.load(io.BytesIO(published), map_location="cpu", weights_only=True)
    assert document["resume_state"]["completed_epoch"] == 2


def test_s3_run_claims_the_running_checkpoint_before_overwriting_it(
    local_config, monkeypatch
):
    storage, _ = _recording_s3_storage()
    monkeypatch.setattr(pipeline, "create_storage", lambda config: storage)

    result = train.run(local_config)

    assert result["status"] == "ok", result["message"]
    running_writes = [
        call.kwargs
        for call in storage.client.put_object.call_args_list
        if call.kwargs["Key"].endswith("/running/last_checkpoint.pt")
    ]
    assert len(running_writes) == 2
    assert running_writes[0]["IfNoneMatch"] == "*"
    assert "IfNoneMatch" not in running_writes[1]


def test_s3_publisher_returns_attempt_uris_and_writes_completion_marker(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pipeline,
        "uuid4",
        lambda: SimpleNamespace(hex="fixed-attempt"),
    )
    client = Mock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    uploaded = {}

    def put_object(**request):
        body = request["Body"]
        uploaded[request["Key"]] = body.read() if hasattr(body, "read") else body

    client.put_object.side_effect = put_object
    storage = S3Storage("bucket", client=client)
    checkpoint = {"model_state_dict": {"weight": torch.tensor(1.0)}}
    settings = {"run_id": "s3-smoke", "output_prefix": "experiments/completed"}

    artifacts = pipeline._publish_s3(
        storage,
        checkpoint,
        checkpoint,
        [{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}],
        settings,
    )

    assert artifacts == {
        "best_checkpoint_uri": "s3://bucket/experiments/completed/s3-smoke/attempts/fixed-attempt/best_checkpoint.pt",
        "last_checkpoint_uri": "s3://bucket/experiments/completed/s3-smoke/attempts/fixed-attempt/last_checkpoint.pt",
        "training_history_uri": "s3://bucket/experiments/completed/s3-smoke/attempts/fixed-attempt/training_history.json",
    }
    assert set(uploaded) == {
        "experiments/completed/s3-smoke/attempts/fixed-attempt/best_checkpoint.pt",
        "experiments/completed/s3-smoke/attempts/fixed-attempt/last_checkpoint.pt",
        "experiments/completed/s3-smoke/attempts/fixed-attempt/training_history.json",
        "experiments/completed/s3-smoke/completed.json",
    }
    assert all(call.kwargs["IfNoneMatch"] == "*" for call in client.put_object.call_args_list)


def test_s3_publisher_can_retry_same_run_id_after_intermediate_failure(monkeypatch):
    attempt_ids = iter(("failed-attempt", "successful-attempt"))
    monkeypatch.setattr(
        pipeline,
        "uuid4",
        lambda: SimpleNamespace(hex=next(attempt_ids)),
    )

    class RetryStorage:
        def __init__(self):
            self.objects = {}
            self.fail_once = True

        def exists(self, location):
            return location in self.objects

        def upload_file(self, source, destination):
            if self.fail_once and destination.endswith("last_checkpoint.pt"):
                self.fail_once = False
                raise StorageError("simulated intermediate upload failure")
            self.objects[destination] = Path(source).read_bytes()
            return f"s3://bucket/{destination}"

        def write_json(self, destination, value):
            self.objects[destination] = value
            return f"s3://bucket/{destination}"

    storage = RetryStorage()
    checkpoint = {"model_state_dict": {"weight": torch.tensor(1.0)}}
    history = [{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}]
    settings = {"run_id": "retry-smoke", "output_prefix": "experiments/completed"}

    with pytest.raises(StorageError, match="intermediate upload failure"):
        pipeline._publish_s3(storage, checkpoint, checkpoint, history, settings)

    completion = "experiments/completed/retry-smoke/completed.json"
    assert completion not in storage.objects
    artifacts = pipeline._publish_s3(storage, checkpoint, checkpoint, history, settings)

    assert "/attempts/successful-attempt/" in artifacts["best_checkpoint_uri"]
    assert storage.objects[completion] == {
        "run_id": "retry-smoke",
        "artifacts": artifacts,
    }
    with pytest.raises(FileExistsError, match="already exists"):
        pipeline._publish_s3(storage, checkpoint, checkpoint, history, settings)


def _bucket_storage(objects: dict[str, bytes]) -> S3Storage:
    """dict 하나를 bucket처럼 쓰는 S3Storage입니다."""

    client = Mock()

    def missing(operation: str) -> ClientError:
        return ClientError({"Error": {"Code": "404", "Message": "missing"}}, operation)

    def head_object(*, Bucket, Key, **_):
        if Key not in objects:
            raise missing("HeadObject")
        return {"ETag": f'"{hashlib.md5(objects[Key]).hexdigest()}"'}

    def get_object(*, Bucket, Key, **_):
        if Key not in objects:
            raise missing("GetObject")
        return {"Body": io.BytesIO(objects[Key])}

    def put_object(*, Bucket, Key, Body=b"", **request):
        if request.get("IfNoneMatch") == "*" and Key in objects:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "exists"}},
                "PutObject",
            )
        objects[Key] = Body.read() if hasattr(Body, "read") else Body

    def download_file(bucket, key, destination):
        if key not in objects:
            raise missing("GetObject")
        Path(destination).write_bytes(objects[key])

    client.head_object.side_effect = head_object
    client.get_object.side_effect = get_object
    client.put_object.side_effect = put_object
    client.download_file.side_effect = download_file
    return S3Storage("bucket", client=client)


DATASET_PREFIX = "datasets/pill_detection/processed/v1-seed42-8020"


def _s3_dataset_objects() -> dict[str, bytes]:
    """bucket에 올라가 있는 dataset artifact와 이미지입니다."""

    def image(color: str) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), color=color).save(buffer, format="PNG")
        return buffer.getvalue()

    def document(value) -> bytes:
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    train_manifest = document(_manifest("train.png"))
    validation_manifest = document(_manifest("validation.png"))
    return {
        f"{DATASET_PREFIX}/train.png": image("red"),
        f"{DATASET_PREFIX}/validation.png": image("blue"),
        f"{DATASET_PREFIX}/train_manifest.json": train_manifest,
        f"{DATASET_PREFIX}/validation_manifest.json": validation_manifest,
        f"{DATASET_PREFIX}/class_map.json": document({"pill": 1}),
        # image cache가 영구 cache를 쓰려면 version 조각과 두 manifest checksum이
        # 모두 있어야 합니다. 실제 data pipeline이 쓰는 형식 그대로입니다.
        f"{DATASET_PREFIX}/dataset_summary.json": document(
            {
                "train_images": 1,
                "validation_images": 1,
                "source_prefix": "datasets/pill_detection/raw/v1/original/",
                "split": {
                    "checksums": {
                        "algorithm": "sha256",
                        "train_manifest.json": {
                            "sha256": hashlib.sha256(train_manifest).hexdigest(),
                            "bytes": len(train_manifest),
                        },
                        "validation_manifest.json": {
                            "sha256": hashlib.sha256(validation_manifest).hexdigest(),
                            "bytes": len(validation_manifest),
                        },
                    }
                },
            }
        ),
    }


def _s3_train_config(tmp_path: Path, run_id: str) -> dict:
    return {
        "storage": {"backend": "s3", "s3": {"prefix": ""}},
        "inputs": {
            "data": {
                name: f"s3://bucket/{DATASET_PREFIX}/{name.removesuffix('_uri')}.json"
                for name in (
                    "train_manifest_uri",
                    "validation_manifest_uri",
                    "class_map_uri",
                    "dataset_summary_uri",
                )
            }
        },
        "train": {
            "run_id": run_id,
            "seed": 17,
            "epochs": 1,
            "batch_size": 1,
            "device": "cpu",
            "output_dir": _relative(tmp_path / "outputs"),
            "output_prefix": "experiments/completed",
        },
    }


def test_a_new_runtime_fills_its_image_cache_before_the_first_batch(tmp_path, monkeypatch):
    """Colab은 runtime이 바뀌면 빈 디스크로 시작합니다.

    그때 이미지를 batch마다 한 장씩 받으면 첫 epoch이 이미지 수만큼 기다립니다.
    첫 batch가 돌기 전에 전부 받아 두어야 그 기다림이 사라집니다.
    """

    objects = _s3_dataset_objects()
    storage = _bucket_storage(objects)
    monkeypatch.setattr(pipeline, "create_storage", lambda config: storage)
    downloads_before_training: list[int] = []

    class _CountingDetector(TinyDetector):
        def forward(self, images, targets):
            if not downloads_before_training:
                downloads_before_training.append(
                    storage.client.download_file.call_count
                )
            return super().forward(images, targets)

    monkeypatch.setattr(
        pipeline, "build_model", lambda *args, **kwargs: _CountingDetector()
    )
    monkeypatch.setattr(
        pipeline,
        "ImageCacheSession",
        # 실제 cache 위치는 저장소 안이라 test가 쓰면 안 됩니다.
        lambda summary: image_cache_module.ImageCacheSession(
            summary,
            cache_root=tmp_path / "runtime",
            temporary_root=tmp_path / "cache-temporary",
        ),
    )

    result = train.run(_s3_train_config(tmp_path, "colab-first"))

    assert result["status"] == "ok", result["message"]
    images = [key for key in objects if key.endswith(".png")]
    assert downloads_before_training == [len(images)]


class _AccumulationSpy:
    """microbatch마다 무엇이 불렸는지 순서대로 적어 둡니다."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.scaled_losses: list[float] = []
        self.learning_rates: list[float] = []

    def autocast(self):
        class Context:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *args):
                return None

        return Context()

    def backward(self, loss):
        self.events.append("backward")
        self.scaled_losses.append(float(loss.detach().cpu()))
        loss.backward()

    def step(self, optimizer):
        self.events.append("step")
        self.learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()

    def state_dict(self):
        return None

    def load_state_dict(self, state):
        assert state is None


def _train_with_accumulation(
    monkeypatch,
    spy,
    *,
    images: int,
    accumulation: int,
    lr_scheduler=None,
    learning_rate: float = 0.0,
):
    monkeypatch.setattr(
        trainer_module, "_PrecisionRuntime", lambda precision, device: spy
    )
    zeroed: list[int] = []
    real_zero = torch.optim.SGD.zero_grad

    def counting_zero_grad(self, set_to_none=True):
        zeroed.append(1)
        return real_zero(self, set_to_none=set_to_none)

    monkeypatch.setattr(torch.optim.SGD, "zero_grad", counting_zero_grad)
    spy.zeroed = zeroed
    settings = {
        "seed": 17,
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        # 0으로 두어 묶음 사이에 가중치가 변하지 않게 합니다. 변하면 손실도 함께
        # 달라져서, 나눗셈이 맞는지와 model이 바뀐 것이 뒤섞입니다.
        "learning_rate": learning_rate,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "epochs": 1,
        "gradient_accumulation_steps": accumulation,
        "lr_scheduler": lr_scheduler,
        "precision": {"mode": "fp32", "dtype": "fp32", "grad_scaler": False},
    }
    train_model(
        TinyDetector(),
        InMemoryDetectionDataset(1.0, size=images),
        InMemoryDetectionDataset(0.0, size=1),
        settings,
    )
    return len(zeroed)


def test_gradient_accumulation_steps_once_per_group(monkeypatch):
    """N개를 모아 한 번만 갱신해야 batch_size를 못 늘리는 GPU에서 의미가 있습니다."""

    spy = _AccumulationSpy()

    _train_with_accumulation(monkeypatch, spy, images=4, accumulation=2)

    assert spy.events == [
        "backward",
        "backward",
        "step",
        "backward",
        "backward",
        "step",
    ]
    # 묶음마다 한 번만 비워야 합니다. 매 microbatch마다 비우면 앞의 gradient가
    # 사라져 모으는 의미가 없어지는데, 학습은 그대로 돌아가 오류가 나지 않습니다.
    assert len(spy.zeroed) == 2


def test_gradient_accumulation_divides_the_loss_by_the_group_size(monkeypatch):
    """나누지 않으면 gradient가 N배가 되어 사실상 learning rate가 N배가 됩니다."""

    spy = _AccumulationSpy()

    _train_with_accumulation(monkeypatch, spy, images=2, accumulation=2)

    # InMemoryDetectionDataset(1.0)은 batch마다 같은 손실을 냅니다.
    assert spy.scaled_losses == pytest.approx(
        [spy.scaled_losses[0]] * 2
    ), spy.scaled_losses
    unscaled = _AccumulationSpy()
    _train_with_accumulation(monkeypatch, unscaled, images=2, accumulation=1)
    assert spy.scaled_losses[0] == pytest.approx(unscaled.scaled_losses[0] / 2)


def test_a_partial_last_group_still_updates_the_weights(monkeypatch):
    """마지막에 N개가 안 되면 그 gradient가 통째로 버려집니다.

    오류가 나지 않고 그 몇 장만 학습에서 빠집니다. epoch 수가 적을수록 비율이 큽니다.
    """

    spy = _AccumulationSpy()

    _train_with_accumulation(monkeypatch, spy, images=3, accumulation=2)

    assert spy.events.count("backward") == 3
    assert spy.events.count("step") == 2
    assert spy.events[-1] == "step"
    assert len(spy.zeroed) == 2
    # 마지막 묶음은 microbatch가 하나뿐입니다. 그런데도 accumulation으로 나누면
    # 그 몇 장의 gradient만 작아져, 오류 없이 그 데이터가 덜 반영됩니다.
    assert spy.scaled_losses[2] == pytest.approx(spy.scaled_losses[0] * 2)


def test_schedule_length_counts_updates_not_microbatches(monkeypatch):
    """schedule은 갱신할 때마다 한 걸음 갑니다. 길이도 갱신 수로 재야 합니다.

    microbatch 수로 재면 accumulation만큼 짧게 걸어, linear와 cosine이 설정한 최저
    learning rate에 닿지 못한 채 학습이 끝납니다. step schedule의 epoch 경계도
    늦어집니다. 오류는 나지 않고 learning rate 궤적만 조용히 달라집니다.
    """

    spy = _AccumulationSpy()

    _train_with_accumulation(
        monkeypatch,
        spy,
        images=4,
        accumulation=2,
        learning_rate=1.0,
        lr_scheduler={
            "name": "linear",
            "warmup_steps": 0,
            "warmup_start_factor": 0.001,
            "min_lr_factor": 0.1,
        },
    )

    # microbatch 4개를 2개씩 모으면 갱신은 2번입니다. 그 2번에 걸쳐 1.0에서
    # min_lr_factor까지 내려가야 합니다.
    assert spy.events.count("step") == 2
    # 갱신 2번에 걸쳐 1.0에서 min_lr_factor 0.1까지 내려갑니다. microbatch 수로
    # 재면 [1.0, 0.7]에서 끝나 최저값을 한 번도 쓰지 못합니다.
    assert spy.learning_rates == pytest.approx([1.0, 0.1])


def test_warmup_counts_updates_not_microbatches(monkeypatch):
    """warmup은 batch가 아니라 optimizer 갱신을 셉니다.

    accumulation이 1보다 크면 둘이 달라집니다. batch로 세면 warmup이 실제보다 빨리
    끝나 초반 learning rate가 설정보다 크게 올라갑니다. 사전학습 가중치로 시작할 때
    초반 손실이 튀는 것을 막으려고 켜는 기능이라, 빨리 끝나면 켠 의미가 옅어집니다.
    """

    spy = _AccumulationSpy()

    _train_with_accumulation(
        monkeypatch,
        spy,
        images=8,
        accumulation=2,
        learning_rate=1.0,
        lr_scheduler={
            "name": "linear",
            "warmup_steps": 2,
            "warmup_start_factor": 0.5,
            "min_lr_factor": 0.1,
        },
    )

    # microbatch 8개를 2개씩 모으면 갱신은 4번이고, warmup 2번은 그중 앞의 두 번
    # 곧 microbatch 4개에 걸칩니다. 0.5에서 출발해 0.75를 지나 1.0에 닿은 뒤 본곡선을
    # 타고 최저 배율 0.1로 끝납니다. batch로 세면 마지막이 0.82가 되어 최저값에 닿지
    # 못합니다.
    assert spy.events.count("step") == 4
    assert spy.learning_rates == pytest.approx([0.5, 0.75, 1.0, 0.1])
