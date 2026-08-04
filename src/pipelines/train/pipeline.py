"""Public orchestration for the Faster R-CNN train pipeline."""

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
from .model import ARCHITECTURE, build_model
from .trainer import set_seed, train_model


RETURN_KEYS = {"status", "artifacts", "summary", "message"}
DATA_ARTIFACT_KEYS = {
    "train_manifest_uri",
    "validation_manifest_uri",
    "class_map_uri",
    "dataset_summary_uri",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(config.get("train", {}), "config.train")
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
    return {
        "run_id": run_id,
        "seed": _integer(raw, "seed", 42, minimum=0),
        "epochs": _integer(raw, "epochs", 1, minimum=1),
        "batch_size": _integer(raw, "batch_size", 1, minimum=1),
        "num_workers": _integer(raw, "num_workers", 0, minimum=0),
        "learning_rate": _float(raw, "learning_rate", 0.005, minimum=0.0),
        "momentum": _float(raw, "momentum", 0.9, minimum=0.0),
        "weight_decay": _float(raw, "weight_decay", 0.0005, minimum=0.0),
        "device": device,
        "pretrained": pretrained,
        "output_dir": output_dir,
        "output_prefix": output_prefix.strip("/"),
    }


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
        "architecture": ARCHITECTURE,
        "num_classes": len(class_map) + 1,
        "class_map": dict(class_map),
        "category_ids": category_ids_by_label,
        "seed": settings["seed"],
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
            data["train_manifest_uri"], class_map, storage, cache / "train"
        )
        validation_dataset = CocoDetectionDataset(
            data["validation_manifest_uri"], class_map, storage, cache / "validation"
        )
        overlap = train_dataset.image_locations & validation_dataset.image_locations
        if overlap:
            raise ValueError("train and validation manifests contain overlapping images")
        if train_dataset.category_ids != validation_dataset.category_ids:
            raise ValueError("train and validation COCO category ids must match")
        set_seed(settings["seed"])
        model = build_model(len(class_map) + 1, pretrained=settings["pretrained"])
        best, last, history = train_model(model, train_dataset, validation_dataset, settings)

    category_ids = train_dataset.category_ids
    best_payload = _checkpoint_payload(best, settings, class_map, category_ids)
    last_payload = _checkpoint_payload(last, settings, class_map, category_ids)
    if isinstance(storage, S3Storage):
        artifact_uris = _publish_s3(storage, best_payload, last_payload, history, settings)
    else:
        artifact_uris = _publish_local(best_payload, last_payload, history, settings)
    artifacts = {"run_id": settings["run_id"], **artifact_uris}
    best_epoch = min(history, key=lambda entry: entry["validation_loss"])
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": {
            "architecture": ARCHITECTURE,
            "device": settings["device"],
            "epochs": settings["epochs"],
            "train_images": len(train_dataset),
            "validation_images": len(validation_dataset),
            "class_count": len(class_map),
            "best_epoch": best_epoch["epoch"],
            "best_validation_loss": best_epoch["validation_loss"],
        },
        "message": "Faster R-CNN training completed successfully",
    }


def run(config: dict) -> dict:
    """Train Faster R-CNN from data-pipeline artifact fixtures."""
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
