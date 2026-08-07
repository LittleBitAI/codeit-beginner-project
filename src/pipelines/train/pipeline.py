"""설정 기반 object detection 학습 pipeline을 조정합니다."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from src.common import S3Storage, Storage, StorageError, create_storage

from .dataset import CocoDetectionDataset, REPOSITORY_ROOT, load_class_map, read_json_artifact
from .model import ARCHITECTURE, SUPPORTED_ARCHITECTURES, build_model
from .progress import ProgressEmitter
from .trainer import SUPPORTED_OPTIMIZERS, set_seed, train_model


RETURN_KEYS = {"status", "artifacts", "summary", "message"}
DATA_ARTIFACT_KEYS = {
    "train_manifest_uri",
    "validation_manifest_uri",
    "class_map_uri",
    "dataset_summary_uri",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
OPTIMIZER_PROFILES = {
    "AdamW": {
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    },
    "SGD": {
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    },
    "Adam": {
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    },
}
AUGMENTATION_PRESETS = {
    "none": {
        "version": 1,
        "preset": "none",
        "horizontal_flip_probability": 0.0,
        "vertical_flip_probability": 0.0,
        "color_probability": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "hue": 0.0,
    },
    "pill_basic": {
        "version": 1,
        "preset": "pill_basic",
        "horizontal_flip_probability": 0.5,
        "vertical_flip_probability": 0.5,
        "color_probability": 0.3,
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.1,
        "hue": 0.02,
    },
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(settings: Mapping[str, Any], name: str, default: int, *, minimum: int) -> int:
    value = settings.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"train.{name} must be an integer >= {minimum}")
    return value


def _float(settings: Mapping[str, Any], name: str, default: float, *, minimum: float) -> float:
    value = settings.get(name, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"train.{name} must be a number >= {minimum}")
    return float(value)


def _probability(settings: Mapping[str, Any], name: str, default: float) -> float:
    value = _float(settings, name, default, minimum=0.0)
    if value >= 1.0:
        raise ValueError(f"train.{name} must be a number >= 0.0 and < 1.0")
    return value


def _augmentation(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = raw.get("augmentation", {"preset": "none"})
    if not isinstance(value, Mapping):
        raise ValueError("train.augmentation must be an object")
    preset = value.get("preset", "none")
    if not isinstance(preset, str) or preset not in AUGMENTATION_PRESETS:
        choices = ", ".join(AUGMENTATION_PRESETS)
        raise ValueError(f"train.augmentation.preset must be one of: {choices}")
    unexpected = set(value) - {"preset"}
    if unexpected:
        raise ValueError(
            "train.augmentation contains unsupported settings: "
            + ", ".join(sorted(unexpected))
        )
    return dict(AUGMENTATION_PRESETS[preset])


def _early_stopping(raw: Mapping[str, Any]) -> dict[str, float | int] | None:
    value = raw.get("early_stopping")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("train.early_stopping must be an object")
    unexpected = set(value) - {"patience", "min_delta"}
    if unexpected:
        raise ValueError(
            "train.early_stopping contains unsupported settings: "
            + ", ".join(sorted(unexpected))
        )
    patience = value.get("patience")
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("train.early_stopping.patience must be an integer >= 1")
    min_delta = value.get("min_delta", 0.0)
    if (
        not isinstance(min_delta, (int, float))
        or isinstance(min_delta, bool)
        or not math.isfinite(float(min_delta))
        or float(min_delta) < 0.0
    ):
        raise ValueError("train.early_stopping.min_delta must be a number >= 0.0")
    return {"patience": patience, "min_delta": float(min_delta)}


def _reject_irrelevant_optimizer_settings(
    raw: Mapping[str, Any], optimizer: str
) -> None:
    irrelevant = {"momentum"} if optimizer in {"AdamW", "Adam"} else {
        "beta1",
        "beta2",
        "epsilon",
    }
    provided = sorted(irrelevant & set(raw))
    if provided:
        fields = ", ".join(f"train.{name}" for name in provided)
        raise ValueError(f"{fields} is not used by train.optimizer={optimizer}")


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(config.get("train", {}), "config.train")
    architecture = raw.get("architecture", ARCHITECTURE)
    if not isinstance(architecture, str) or architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "train.architecture must be one of: " + ", ".join(SUPPORTED_ARCHITECTURES)
        )
    optimizer = raw.get("optimizer", "SGD")
    if not isinstance(optimizer, str) or optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError("train.optimizer must be one of: " + ", ".join(SUPPORTED_OPTIMIZERS))
    _reject_irrelevant_optimizer_settings(raw, optimizer)
    profile = OPTIMIZER_PROFILES[optimizer]
    device = raw.get("device", "cpu")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("train.device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("train.device is 'cuda', but CUDA is not available")
    pretrained = raw.get("pretrained", False)
    if not isinstance(pretrained, bool):
        raise ValueError("train.pretrained must be a boolean")
    run_id = raw.get("run_id") or datetime.now(timezone.utc).strftime("train-%Y%m%dT%H%M%S%fZ")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("train.run_id contains unsupported characters")
    output_dir = raw.get("output_dir", "artifacts/experiments/completed")
    output_prefix = raw.get("output_prefix", "experiments/completed")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("train.output_dir must be a non-empty repository-relative path")
    if not isinstance(output_prefix, str) or not output_prefix.strip():
        raise ValueError("train.output_prefix must be a non-empty S3 prefix")
    normalized = {
        "run_id": run_id,
        "architecture": architecture,
        "optimizer": optimizer,
        "seed": _integer(raw, "seed", 42, minimum=0),
        "epochs": _integer(raw, "epochs", 1, minimum=1),
        "batch_size": _integer(raw, "batch_size", 1, minimum=1),
        "num_workers": _integer(raw, "num_workers", 0, minimum=0),
        "learning_rate": _float(
            raw, "learning_rate", profile["learning_rate"], minimum=0.0
        ),
        "weight_decay": _float(
            raw, "weight_decay", profile["weight_decay"], minimum=0.0
        ),
        "augmentation": _augmentation(raw),
        "device": device,
        "pretrained": pretrained,
        "early_stopping": _early_stopping(raw),
        "output_dir": output_dir,
        "output_prefix": output_prefix.strip("/"),
    }
    if optimizer == "SGD":
        normalized["momentum"] = _float(
            raw, "momentum", profile["momentum"], minimum=0.0
        )
    else:
        normalized["beta1"] = _probability(raw, "beta1", profile["beta1"])
        normalized["beta2"] = _probability(raw, "beta2", profile["beta2"])
        normalized["epsilon"] = _float(
            raw, "epsilon", profile["epsilon"], minimum=1e-16
        )
    return normalized


def _data_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    inputs = _mapping(config.get("inputs"), "config.inputs")
    data = _mapping(inputs.get("data"), "config.inputs.data")
    missing = sorted(DATA_ARTIFACT_KEYS - set(data))
    if missing:
        raise ValueError(f"config.inputs.data is missing artifacts: {', '.join(missing)}")
    resolved: dict[str, str] = {}
    for name in DATA_ARTIFACT_KEYS:
        value = data[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"config.inputs.data.{name} must be a non-empty string")
        resolved[name] = value
    return resolved


def _repo_output(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("train.output_dir must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError("train.output_dir leaves the repository") from error
    return resolved


def _checkpoint_payload(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
    class_map: Mapping[str, int],
    category_ids: Mapping[int, int],
) -> dict[str, Any]:
    category_ids_by_label = [0] + [
        category_ids[label] for label in range(1, len(class_map) + 1)
    ]
    return {
        **checkpoint,
        "architecture": settings.get("architecture", ARCHITECTURE),
        "num_classes": len(class_map) + 1,
        "class_map": dict(class_map),
        "category_ids": category_ids_by_label,
        "seed": settings["seed"],
        "training_config": _training_config(settings),
    }


def _training_config(settings: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = settings.get("optimizer", "SGD")
    profile = OPTIMIZER_PROFILES[optimizer]
    optimizer_settings: dict[str, Any] = {
        "name": optimizer,
        "learning_rate": settings.get("learning_rate", profile["learning_rate"]),
        "weight_decay": settings.get("weight_decay", profile["weight_decay"]),
    }
    if optimizer == "SGD":
        optimizer_settings["momentum"] = settings.get("momentum", profile["momentum"])
    else:
        optimizer_settings["betas"] = [
            settings.get("beta1", profile["beta1"]),
            settings.get("beta2", profile["beta2"]),
        ]
        optimizer_settings["epsilon"] = settings.get("epsilon", profile["epsilon"])
    augmentation = settings.get("augmentation", AUGMENTATION_PRESETS["none"])
    return {
        "schema_version": 1,
        "run_id": settings.get("run_id"),
        "architecture": settings.get("architecture", ARCHITECTURE),
        "optimizer": optimizer_settings,
        "augmentation": dict(augmentation),
        "seed": settings["seed"],
        "epochs": settings.get("epochs"),
        "batch_size": settings.get("batch_size"),
        "num_workers": settings.get("num_workers"),
        "device": settings.get("device"),
        "pretrained": settings.get("pretrained"),
        "early_stopping": (
            dict(settings["early_stopping"])
            if settings.get("early_stopping") is not None
            else None
        ),
    }


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as output:
        torch.save(dict(value), output)


def _publish_local(
    best: Mapping[str, Any],
    last: Mapping[str, Any],
    history: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, str]:
    output_parent = _repo_output(settings["output_dir"])
    final_directory = output_parent / settings["run_id"]
    if final_directory.exists():
        raise FileExistsError(f"training run artifact already exists: {settings['run_id']}")
    output_parent.mkdir(parents=True, exist_ok=True)
    stage_directory = Path(tempfile.mkdtemp(prefix=f".{settings['run_id']}-", dir=output_parent))
    try:
        _write_checkpoint(stage_directory / "best_checkpoint.pt", best)
        _write_checkpoint(stage_directory / "last_checkpoint.pt", last)
        (stage_directory / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        stage_directory.rename(final_directory)
    except Exception:
        shutil.rmtree(stage_directory, ignore_errors=True)
        raise

    def relative(name: str) -> str:
        return (final_directory / name).relative_to(REPOSITORY_ROOT).as_posix()

    return {
        "best_checkpoint_uri": relative("best_checkpoint.pt"),
        "last_checkpoint_uri": relative("last_checkpoint.pt"),
        "training_history_uri": relative("training_history.json"),
    }


def _publish_s3(
    storage: S3Storage,
    best: Mapping[str, Any],
    last: Mapping[str, Any],
    history: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, str]:
    prefix = f"{settings['output_prefix']}/{settings['run_id']}"
    completion_destination = f"{prefix}/completed.json"
    if storage.exists(completion_destination):
        raise FileExistsError(f"training run artifact already exists: {settings['run_id']}")
    attempt_prefix = f"{prefix}/attempts/{uuid4().hex}"
    destinations = {
        "best_checkpoint_uri": f"{attempt_prefix}/best_checkpoint.pt",
        "last_checkpoint_uri": f"{attempt_prefix}/last_checkpoint.pt",
        "training_history_uri": f"{attempt_prefix}/training_history.json",
    }

    scratch_root = REPOSITORY_ROOT / "artifacts"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="train-upload-", dir=scratch_root) as directory:
        temporary = Path(directory)
        best_path = temporary / "best_checkpoint.pt"
        last_path = temporary / "last_checkpoint.pt"
        _write_checkpoint(best_path, best)
        _write_checkpoint(last_path, last)
        best_uri = storage.upload_file(best_path, destinations["best_checkpoint_uri"])
        last_uri = storage.upload_file(last_path, destinations["last_checkpoint_uri"])
        history_uri = storage.write_json(destinations["training_history_uri"], history)
    artifacts = {
        "best_checkpoint_uri": best_uri,
        "last_checkpoint_uri": last_uri,
        "training_history_uri": history_uri,
    }
    storage.write_json(
        completion_destination,
        {"run_id": settings["run_id"], "artifacts": artifacts},
    )
    return artifacts


def _execute(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    data = _data_inputs(config)
    storage = create_storage(config)
    class_map = load_class_map(data["class_map_uri"], storage)
    dataset_summary = read_json_artifact(data["dataset_summary_uri"], storage)
    if not isinstance(dataset_summary, Mapping):
        raise ValueError("dataset summary must be a JSON object")

    cache_root = REPOSITORY_ROOT / "artifacts"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="train-images-", dir=cache_root) as directory:
        cache = Path(directory)
        train_dataset = CocoDetectionDataset(
            data["train_manifest_uri"],
            class_map,
            storage,
            cache / "train",
            augmentation=settings["augmentation"],
        )
        validation_dataset = CocoDetectionDataset(
            data["validation_manifest_uri"], class_map, storage, cache / "validation"
        )
        overlap = train_dataset.image_locations & validation_dataset.image_locations
        if overlap:
            raise ValueError("train and validation manifests contain overlapping images")
        if train_dataset.category_ids != validation_dataset.category_ids:
            raise ValueError("train and validation COCO category ids must match")
        progress = ProgressEmitter(settings["run_id"])
        progress.emit(
            "run_started",
            architecture=settings["architecture"],
            device=settings["device"],
            epochs=settings["epochs"],
            train_images=len(train_dataset),
            validation_images=len(validation_dataset),
            class_count=len(class_map),
        )
        set_seed(settings["seed"])
        model = build_model(
            len(class_map) + 1,
            architecture=settings["architecture"],
            pretrained=settings["pretrained"],
        )
        best, last, history = train_model(
            model, train_dataset, validation_dataset, settings, progress
        )

    category_ids = train_dataset.category_ids
    best_payload = _checkpoint_payload(best, settings, class_map, category_ids)
    last_payload = _checkpoint_payload(last, settings, class_map, category_ids)
    if isinstance(storage, S3Storage):
        artifact_uris = _publish_s3(storage, best_payload, last_payload, history, settings)
    else:
        artifact_uris = _publish_local(best_payload, last_payload, history, settings)
    artifacts = {"run_id": settings["run_id"], **artifact_uris}
    best_epoch = min(history, key=lambda entry: entry["validation_loss"])
    completed_epochs = len(history)
    stopped_early = completed_epochs < settings["epochs"]
    progress.emit(
        "training_completed",
        planned_epochs=settings["epochs"],
        completed_epochs=completed_epochs,
        stopped_early=stopped_early,
        best_epoch=best_epoch["epoch"],
        best_validation_loss=best_epoch["validation_loss"],
    )
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": {
            "architecture": settings["architecture"],
            "optimizer": settings["optimizer"],
            "augmentation": settings["augmentation"]["preset"],
            "device": settings["device"],
            "epochs": settings["epochs"],
            "planned_epochs": settings["epochs"],
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "train_images": len(train_dataset),
            "validation_images": len(validation_dataset),
            "class_count": len(class_map),
            "best_epoch": best_epoch["epoch"],
            "best_validation_loss": best_epoch["validation_loss"],
        },
        "message": (
            f"object detection training stopped early after {completed_epochs} of "
            f"{settings['epochs']} epochs"
            if stopped_early
            else "object detection training completed successfully"
        ),
    }


def run(config: dict) -> dict:
    """Data pipeline artifact와 train 설정으로 detector를 학습합니다."""
    try:
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        execution = config.get("execution")
        inputs = config.get("inputs")
        legacy_dummy = "train" not in config and inputs in (None, {})
        if (
            isinstance(execution, Mapping)
            and execution.get("mode") == "dummy"
        ) or legacy_dummy:
            return {
                "status": "ok",
                "artifacts": {},
                "summary": {"pipeline": "train", "mode": "dummy"},
                "message": "train pipeline dummy execution completed",
            }
        result = _execute(config)
    except (FileExistsError, OSError, RuntimeError, StorageError, ValueError) as error:
        result = {
            "status": "error",
            "artifacts": {},
            "summary": {},
            "message": f"training failed: {error}",
        }
    except Exception as error:  # Keep the public pipeline contract on unexpected library errors.
        result = {
            "status": "error",
            "artifacts": {},
            "summary": {},
            "message": f"training failed unexpectedly: {type(error).__name__}: {error}",
        }
    if set(result) != RETURN_KEYS:
        raise RuntimeError("train pipeline returned an invalid contract")
    return result
