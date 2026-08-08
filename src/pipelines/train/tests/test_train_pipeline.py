import copy
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
from src.pipelines import train
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
    def __init__(self, value: float) -> None:
        self.image = torch.full((3, 2, 2), value)

    def __len__(self):
        return 1

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


def _manifest(image_name: str) -> dict:
    return {
        "images": [
            {"id": 1, "file_name": image_name, "width": 16, "height": 12}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [2, 3, 5, 4],
                "iscrowd": 0,
            }
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
            "seed",
            "epochs",
            "batch_size",
            "num_workers",
            "device",
            "precision",
            "pretrained",
            "early_stopping",
            "resume",
        }
        assert recorded["schema_version"] == 2
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


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("optimizer", "Lion", "train.optimizer must be one of"),
        ("architecture", "unknown_detector", "train.architecture must be one of"),
        ("augmentation", {"preset": "unsafe_crop"}, "augmentation.preset must be one of"),
        ("precision", "bf16", "train.precision must be one of"),
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


def test_amp_precision_requires_cuda(local_config):
    local_config["train"]["precision"] = "amp"

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "train.precision='amp' requires train.device='cuda'" in result["message"]


@pytest.mark.parametrize(
    ("bf16_supported", "dtype", "grad_scaler"),
    [(True, "bf16", False), (False, "fp16", True)],
)
def test_amp_precision_chooses_only_native_bf16(
    monkeypatch, bf16_supported, dtype, grad_scaler
):
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
    runtime.backward_and_step(loss, optimizer)

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
    calls = {"autocast": 0, "backward_and_step": 0}

    class Runtime:
        def autocast(self):
            class Context:
                def __enter__(self):
                    calls["autocast"] += 1

                def __exit__(self, *args):
                    return None

            return Context()

        def backward_and_step(self, loss, optimizer):
            calls["backward_and_step"] += 1
            loss.backward()
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

    assert calls == {"autocast": 2, "backward_and_step": 1}


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
