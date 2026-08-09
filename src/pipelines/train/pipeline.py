"""설정 기반 object detection 학습 pipeline을 조정합니다."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from src.common import S3Storage, Storage, StorageError, create_storage

from .dataset import (
    CocoDetectionDataset,
    REPOSITORY_ROOT,
    _is_s3,
    _local_artifact_path,
    _s3_relative,
    load_class_map,
    read_json_artifact,
)
from .image_cache import ImageCacheSession
from .model import ARCHITECTURE, SUPPORTED_ARCHITECTURES, build_model
from .progress import ProgressEmitter
from .trainer import (
    RESUME_STATE_VERSION,
    SUPPORTED_OPTIMIZERS,
    set_seed,
    train_model,
)


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
# ``amp``는 GPU를 보고 dtype을 대신 골라 주고, ``fp16``·``bf16``은 고른 그대로 씁니다.
# 자동 선택만 있으면 어떤 GPU에서 무엇으로 돌지 미리 알 수 없고, 그 GPU에 맞는 쪽을
# 사람이 고를 수도 없습니다.
PRECISION_MODES = ("fp32", "amp", "fp16", "bf16")
# 학습 중 작업 폴더에 두는 파일입니다. 마지막 것이 이어서 학습할 대상입니다.
WORKING_CHECKPOINT_NAMES = ("best_checkpoint.pt", "last_checkpoint.pt")


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


def _native_bf16_supported() -> bool:
    """Emulation을 뺀, 하드웨어가 직접 처리하는 bf16 지원 여부입니다.

    ``including_emulation``은 비교적 최근 torch에만 있는 인자입니다. 그 인자가 없는
    torch의 ``is_bf16_supported()``는 emulation까지 지원으로 세어 T4(sm_75)에서도
    True를 돌려줍니다. 그대로 믿으면 T4가 bf16 emulation으로 학습해 크게 느려집니다.
    인자를 거부하면 compute capability로 직접 판단합니다. bf16을 하드웨어로 처리하는
    것은 Ampere(8.0)부터입니다.
    """

    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


def _precision(raw: Mapping[str, Any], device: str) -> dict[str, str | bool]:
    """요청한 정밀도를 실제 CUDA가 빠르게 지원하는 dtype으로 확정합니다."""

    mode = raw.get("precision", "fp32")
    if not isinstance(mode, str) or mode not in PRECISION_MODES:
        raise ValueError("train.precision must be one of: " + ", ".join(PRECISION_MODES))
    if mode == "fp32":
        return {"mode": "fp32", "dtype": "fp32", "grad_scaler": False}
    if device != "cuda":
        raise ValueError(f"train.precision='{mode}' requires train.device='cuda'")
    if mode == "fp16":
        # fp16은 어느 CUDA GPU에서나 됩니다. 표현할 수 있는 수의 범위가 좁아 gradient가
        # 0으로 내려앉으므로 GradScaler가 필요합니다. bf16 지원 여부는 물을 필요가 없습니다.
        return {"mode": "fp16", "dtype": "fp16", "grad_scaler": True}
    native_bf16 = _native_bf16_supported()
    if mode == "bf16":
        if not native_bf16:
            # 조용히 fp16으로 바꾸지 않습니다. 고른 것과 다른 값으로 도는 학습은
            # 결과를 비교할 수 없고, emulation으로 도는 bf16은 밤새 돌린 시간을
            # 통째로 버리게 만듭니다.
            raise ValueError(
                "train.precision='bf16' requires a GPU with native bfloat16 support "
                "(compute capability 8.0 or newer); use 'fp16' or 'amp' instead"
            )
        return {"mode": "bf16", "dtype": "bf16", "grad_scaler": False}
    return {
        "mode": "amp",
        "dtype": "bf16" if native_bf16 else "fp16",
        "grad_scaler": not native_bf16,
    }


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
    resume_from = raw.get("resume_from")
    if resume_from is not None and (
        not isinstance(resume_from, str) or not resume_from.strip()
    ):
        raise ValueError("train.resume_from must be a non-empty checkpoint path")
    resume_path = Path(resume_from) if isinstance(resume_from, str) else None
    if isinstance(resume_from, str) and not _is_s3(resume_from) and (
        resume_path.is_absolute() or bool(resume_path.drive) or "\\" in resume_from
    ):
        raise ValueError(
            "train.resume_from must be a repository-relative POSIX path or an S3 URI"
        )
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
        "checkpoint_every": _integer(raw, "checkpoint_every", 1, minimum=1),
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
        "precision": _precision(raw, device),
        "pretrained": pretrained,
        "early_stopping": _early_stopping(raw),
        "output_dir": output_dir,
        "output_prefix": output_prefix.strip("/"),
        "resume_from": resume_from,
        # 이어서 학습한 출처는 checkpoint를 읽어 봐야 알 수 있으므로 _execute가 채웁니다.
        "resume": None,
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
        # 2: resume block이 생겼습니다. 처음부터 학습한 실행은 그 값이 None입니다.
        "schema_version": 2,
        "run_id": settings.get("run_id"),
        "architecture": settings.get("architecture", ARCHITECTURE),
        "optimizer": optimizer_settings,
        "augmentation": dict(augmentation),
        "seed": settings["seed"],
        "epochs": settings.get("epochs"),
        "batch_size": settings.get("batch_size"),
        "num_workers": settings.get("num_workers"),
        "device": settings.get("device"),
        "precision": dict(
            settings.get(
                "precision",
                {"mode": "fp32", "dtype": "fp32", "grad_scaler": False},
            )
        ),
        "pretrained": settings.get("pretrained"),
        "early_stopping": (
            dict(settings["early_stopping"])
            if settings.get("early_stopping") is not None
            else None
        ),
        # 처음부터 학습한 실행은 항상 None입니다. key 자체는 늘 있어야 읽는 쪽이 편합니다.
        "resume": (
            dict(settings["resume"]) if settings.get("resume") is not None else None
        ),
    }


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as output:
        torch.save(dict(value), output)


def _replace_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    """학습 중 checkpoint를 갱신합니다.

    쓰다가 죽어도 직전 checkpoint가 남도록 임시 파일에 먼저 쓰고 옮깁니다. 공개
    artifact는 여전히 ``_write_checkpoint``가 덮어쓰기 없이 씁니다.
    """

    temporary = path.with_name(f".{path.name}.writing")
    with temporary.open("wb") as output:
        torch.save(dict(value), output)
    os.replace(temporary, path)


def _final_directory(settings: Mapping[str, Any]) -> Path:
    return _repo_output(settings["output_dir"]) / settings["run_id"]


def _working_directory(settings: Mapping[str, Any]) -> Path:
    """학습 중인 checkpoint가 머무는 폴더입니다.

    점으로 시작하므로 완료된 실행과 섞이지 않고, 이름이 정해져 있어 나중에 사람이
    이어서 할 파일을 찾을 수 있습니다.
    """

    return _repo_output(settings["output_dir"]) / f".{settings['run_id']}.partial"


def _validate_working_directory(path: Path) -> None:
    """작업 폴더가 link를 거쳐 저장소 밖으로 나가지 않는지 확인합니다."""

    if os.path.lexists(path):
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if path.is_symlink() or attributes & reparse_point:
            raise ValueError(
                "training working directory must not be a symbolic link or reparse point"
            )
    try:
        path.resolve(strict=False).relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError("training working directory leaves the repository") from error


@contextmanager
def _temporary_checkpoint_directory(prefix: str) -> Iterator[Path]:
    """S3 checkpoint 임시 파일을 저장소 안의 일반 폴더에만 둡니다."""

    scratch_root = REPOSITORY_ROOT / "artifacts"
    if os.path.lexists(scratch_root):
        attributes = getattr(scratch_root.lstat(), "st_file_attributes", 0)
        reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if scratch_root.is_symlink() or attributes & reparse_point:
            raise ValueError(
                "training checkpoint scratch directory must not be a symbolic link "
                "or reparse point"
            )
        if not scratch_root.is_dir():
            raise ValueError("training checkpoint scratch directory must be a directory")
    else:
        scratch_root.mkdir()
    try:
        scratch_root.resolve(strict=True).relative_to(REPOSITORY_ROOT.resolve(strict=True))
    except ValueError as error:
        raise ValueError(
            "training checkpoint scratch directory leaves the repository"
        ) from error

    with tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root) as directory:
        temporary = Path(directory)
        try:
            temporary.resolve(strict=True).relative_to(
                REPOSITORY_ROOT.resolve(strict=True)
            )
        except ValueError as error:
            raise ValueError(
                "training checkpoint scratch directory leaves the repository"
            ) from error
        yield temporary


@contextmanager
def _claim_run(settings: Mapping[str, Any]) -> Iterator[None]:
    """같은 이름의 local 학습 하나만 입력을 읽도록 원자적으로 표시합니다."""

    parent = _repo_output(settings["output_dir"])
    parent.mkdir(parents=True, exist_ok=True)
    claim = parent / f".{settings['run_id']}.claim"
    try:
        with claim.open("xb"):
            pass
    except FileExistsError as error:
        raise FileExistsError(
            f"training run is already active: {settings['run_id']}"
        ) from error
    try:
        yield
    finally:
        claim.unlink(missing_ok=True)


@contextmanager
def _own_working_directory(settings: Mapping[str, Any]) -> Iterator[Path]:
    """검사한 local 작업 폴더를 원자적으로 만들고 빈 폴더만 정리합니다."""

    working = _working_directory(settings)
    _validate_working_directory(working)
    try:
        working.mkdir()
    except FileExistsError as error:
        _validate_working_directory(working)
        raise FileExistsError(
            "an interrupted run with the same run_id is still on disk; resume from "
            f"it or remove it: {working.relative_to(REPOSITORY_ROOT).as_posix()}"
        ) from error
    try:
        yield working
    finally:
        try:
            working.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # checkpoint가 하나라도 생긴 중단 실행은 이어서 할 수 있도록 남깁니다.
            pass


def _s3_run_prefix(settings: Mapping[str, Any]) -> str:
    return f"{settings['output_prefix']}/{settings['run_id']}"


def _reject_existing_run(settings: Mapping[str, Any], storage: Storage) -> None:
    """이름 충돌을 몇 시간 학습한 뒤가 아니라 시작 전에 알려 줍니다."""

    if isinstance(storage, S3Storage):
        prefix = _s3_run_prefix(settings)
        if storage.exists(f"{prefix}/completed.json"):
            raise FileExistsError(
                f"training run artifact already exists: {settings['run_id']}"
            )
        running = f"{prefix}/running/{WORKING_CHECKPOINT_NAMES[-1]}"
        if storage.exists(running):
            # 이 key를 덮어쓰면 중단된 학습의 유일한 사본이 사라집니다. Colab은
            # runtime이 바뀌면 로컬 작업 폴더가 없으므로 S3를 직접 봐야 압니다.
            raise FileExistsError(
                "an interrupted run with the same run_id is still on S3; resume from "
                f"it or remove it: {running}"
            )
    elif _final_directory(settings).exists():
        raise FileExistsError(
            f"training run artifact already exists: {settings['run_id']}"
        )
    working = _working_directory(settings)
    _validate_working_directory(working)
    if os.path.lexists(working):
        # 여기를 덮어쓰면 이 기능이 지키려던 바로 그 파일이 사라집니다.
        location = working.relative_to(REPOSITORY_ROOT).as_posix()
        raise FileExistsError(
            "an interrupted run with the same run_id is still on disk; resume from it "
            f"or remove it: {location}"
        )


def _read_checkpoint(location: str, storage: Storage, *, label: str) -> dict[str, Any]:
    """이어서 학습할 checkpoint를 읽습니다.

    사람이 config에 적어 준 경로이므로 ``weights_only=True``로만 읽습니다.
    """

    if _is_s3(location):
        with _temporary_checkpoint_directory("train-resume-") as temporary:
            path = temporary / "checkpoint.pt"
            storage.download_file(location, path)
            return _load_checkpoint(path, location, label=label)
    return _load_checkpoint(_local_artifact_path(location, storage), location, label=label)


def _load_checkpoint(path: Path, location: str, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {location}")
    document = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(document, Mapping):
        raise ValueError(f"{label} must be an object: {location}")
    return dict(document)


def _sibling_best_uri(location: str) -> str:
    """이어서 학습할 checkpoint 옆의 best_checkpoint.pt를 가리킵니다."""

    if _is_s3(location):
        return _s3_relative(location, "best_checkpoint.pt")
    return (Path(location).parent / "best_checkpoint.pt").as_posix()


def _load_resume(
    settings: dict[str, Any], storage: Storage, class_map: Mapping[str, int]
) -> dict[str, Any] | None:
    """이어서 학습할 상태를 읽고, 이어붙일 수 있는지 학습 전에 전부 확인합니다."""

    location = settings["resume_from"]
    if location is None:
        return None
    checkpoint = _read_checkpoint(location, storage, label="train.resume_from")
    state = checkpoint.get("resume_state")
    if not isinstance(state, Mapping):
        raise ValueError(
            f"checkpoint predates resume support and cannot be resumed: {location}"
        )
    if state.get("version") != RESUME_STATE_VERSION:
        raise ValueError(
            f"unsupported resume_state version: {state.get('version')}"
        )
    completed_epoch = state.get("completed_epoch")
    if not isinstance(completed_epoch, int) or isinstance(completed_epoch, bool):
        raise ValueError("resume_state.completed_epoch must be an integer")
    history = state.get("history")
    if not isinstance(history, list) or [
        entry.get("epoch") for entry in history
    ] != list(range(1, completed_epoch + 1)):
        # 구멍 난 history는 손실 곡선과 best_epoch을 조용히 틀리게 만듭니다.
        raise ValueError(
            "resume_state.history must cover epoch 1 to the resumed epoch without gaps"
        )
    if settings["epochs"] <= completed_epoch:
        raise ValueError(
            f"train.epochs must be greater than the resumed epoch {completed_epoch}"
        )
    if checkpoint.get("architecture") != settings["architecture"]:
        raise ValueError(
            "resume checkpoint was trained with a different train.architecture"
        )
    training_config = checkpoint.get("training_config")
    recorded_optimizer = (
        training_config.get("optimizer")
        if isinstance(training_config, Mapping)
        else None
    )
    expected_optimizer = _training_config(settings)["optimizer"]
    if not isinstance(recorded_optimizer, Mapping) or dict(
        recorded_optimizer
    ) != expected_optimizer:
        raise ValueError(
            "resume checkpoint optimizer settings do not match current train optimizer settings"
        )
    if checkpoint.get("num_classes") != len(class_map) + 1 or dict(
        checkpoint.get("class_map") or {}
    ) != dict(class_map):
        raise ValueError("resume checkpoint was trained with a different class map")
    early_stopping = settings.get("early_stopping")
    if "grad_scaler_state_dict" not in state:
        raise ValueError("resume_state is missing grad_scaler_state_dict")
    if (
        early_stopping is not None
        and state["early_stopping"]["epochs_without_improvement"]
        >= early_stopping["patience"]
    ):
        # 그대로 되돌리면 한 epoch만 돌고 다시 멈춥니다. 성공한 것처럼 보여서 더 나쁩니다.
        raise ValueError(
            "resume checkpoint already used up train.early_stopping.patience; "
            "raise patience or drop early_stopping"
        )
    recorded_best = state["best"]
    if "model_state_dict" in recorded_best:
        # S3에서 받은 사본은 한 파일 안에 best 가중치까지 들고 있습니다.
        best = dict(recorded_best)
    else:
        best = _read_checkpoint(
            _sibling_best_uri(location),
            storage,
            label="best_checkpoint.pt next to train.resume_from",
        )
        if best.get("epoch") != recorded_best["epoch"]:
            raise ValueError(
                "best_checkpoint.pt next to train.resume_from does not match "
                "resume_state.best"
            )
    settings["resume"] = {
        "resumed_from": location,
        "resumed_at_epoch": completed_epoch,
    }
    return {"checkpoint": checkpoint, "best": best}


def _with_embedded_best(
    last_payload: Mapping[str, Any], best: Mapping[str, Any]
) -> dict[str, Any]:
    """옆 best 파일이 없어도 이어서 할 수 있는 last checkpoint를 만듭니다."""

    return {
        **last_payload,
        "resume_state": {
            **last_payload["resume_state"],
            "best": {
                **last_payload["resume_state"]["best"],
                "model_state_dict": best["model_state_dict"],
                "optimizer_state_dict": best["optimizer_state_dict"],
            },
        },
    }


def _mirror_to_s3(
    storage: S3Storage,
    last_payload: Mapping[str, Any],
    best: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """이어서 할 상태를 S3에 **한 object로** 올립니다.

    S3는 여러 object를 한 번에 바꿀 수 없습니다. best와 last를 따로 올리면 하나만
    성공했을 때 두 파일의 epoch이 어긋나고, bucket에 남은 유일한 사본을 이어서 쓸 수
    없게 됩니다. 그래서 이 사본에는 best 가중치까지 담습니다. 로컬 작업 폴더는 두
    파일을 나란히 두지만 last도 같은 이유로 self-contained 상태를 유지합니다.
    """

    payload = _with_embedded_best(last_payload, best)
    destination = f"{_s3_run_prefix(settings)}/running/{WORKING_CHECKPOINT_NAMES[-1]}"
    with _temporary_checkpoint_directory("train-mirror-") as temporary:
        path = temporary / WORKING_CHECKPOINT_NAMES[-1]
        _write_checkpoint(path, payload)
        # 첫 checkpoint는 If-None-Match로 run_id 소유권까지 얻습니다. 같은 이름의 두
        # 작업이 사전 exists 검사를 동시에 통과해도 하나만 이 object를 만들 수 있습니다.
        storage.upload_file(path, destination, overwrite=overwrite)


def _publish_local(
    working_directory: Path,
    history: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, str]:
    """작업 폴더에 이미 있는 checkpoint에 history를 더해 폴더째 옮깁니다."""

    _validate_working_directory(working_directory)
    final_directory = _final_directory(settings)
    # 학습이 몇 시간 걸리는 동안 같은 이름이 생겼을 수 있어 옮기기 직전에 다시 봅니다.
    if final_directory.exists():
        raise FileExistsError(f"training run artifact already exists: {settings['run_id']}")
    (working_directory / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    working_directory.rename(final_directory)

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
    prefix = _s3_run_prefix(settings)
    completion_destination = f"{prefix}/completed.json"
    if storage.exists(completion_destination):
        raise FileExistsError(f"training run artifact already exists: {settings['run_id']}")
    attempt_prefix = f"{prefix}/attempts/{uuid4().hex}"
    destinations = {
        "best_checkpoint_uri": f"{attempt_prefix}/best_checkpoint.pt",
        "last_checkpoint_uri": f"{attempt_prefix}/last_checkpoint.pt",
        "training_history_uri": f"{attempt_prefix}/training_history.json",
    }

    with _temporary_checkpoint_directory("train-upload-") as temporary:
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
    with _claim_run(settings):
        _reject_existing_run(settings, storage)
        with _own_working_directory(settings) as working_directory:
            return _execute_claimed(settings, data, storage, working_directory)


def _execute_claimed(
    settings: dict[str, Any],
    data: dict[str, str],
    storage: Storage,
    working_directory: Path,
) -> dict[str, Any]:
    class_map = load_class_map(data["class_map_uri"], storage)
    dataset_summary = read_json_artifact(data["dataset_summary_uri"], storage)
    if not isinstance(dataset_summary, Mapping):
        raise ValueError("dataset summary must be a JSON object")
    resume = _load_resume(settings, storage, class_map)

    with ImageCacheSession(dataset_summary) as image_cache:
        train_dataset = CocoDetectionDataset(
            data["train_manifest_uri"],
            class_map,
            storage,
            image_cache,
            augmentation=settings["augmentation"],
        )
        validation_dataset = CocoDetectionDataset(
            data["validation_manifest_uri"], class_map, storage, image_cache
        )
        overlap = train_dataset.image_locations & validation_dataset.image_locations
        if overlap:
            raise ValueError("train and validation manifests contain overlapping images")
        if train_dataset.category_ids != validation_dataset.category_ids:
            raise ValueError("train and validation COCO category ids must match")
        category_ids = train_dataset.category_ids
        written_best: dict[str, int] = {}
        # 마지막으로 남긴 resume_state입니다. S3 게시가 이것을 그대로 써야 backend에
        # 따라 최종 last checkpoint의 내용이 달라지지 않습니다.
        latest: dict[str, Any] = {}
        s3_mirror_owned = False

        def write_checkpoints(
            last: dict[str, Any],
            best: dict[str, Any],
            resume_state: dict[str, Any],
        ) -> None:
            """Epoch이 끝날 때마다 이어서 할 수 있는 checkpoint를 남깁니다."""

            nonlocal s3_mirror_owned

            _validate_working_directory(working_directory)
            payload = _checkpoint_payload(last, settings, class_map, category_ids)
            payload["resume_state"] = {
                **resume_state,
                "source": {
                    "run_id": settings["run_id"],
                    "resume_from": settings["resume_from"],
                },
            }
            resumable_payload = _with_embedded_best(payload, best)
            # last를 먼저 바꿉니다. 이후 best 저장에서 중단돼도 last 안의 best 가중치로
            # 이어서 할 수 있고, last 자체를 쓰다 중단되면 os.replace 전 사본이 남습니다.
            _replace_checkpoint(
                working_directory / "last_checkpoint.pt", resumable_payload
            )
            if written_best.get("epoch") != best["epoch"]:
                _replace_checkpoint(
                    working_directory / "best_checkpoint.pt",
                    _checkpoint_payload(best, settings, class_map, category_ids),
                )
                written_best["epoch"] = best["epoch"]
            latest["resume_state"] = resumable_payload["resume_state"]
            if isinstance(storage, S3Storage):
                _mirror_to_s3(
                    storage,
                    resumable_payload,
                    best,
                    settings,
                    overwrite=s3_mirror_owned,
                )
                s3_mirror_owned = True

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
            model,
            train_dataset,
            validation_dataset,
            settings,
            progress,
            resume=resume,
            on_checkpoint=write_checkpoints,
        )

    if isinstance(storage, S3Storage):
        best_payload = _checkpoint_payload(best, settings, class_map, category_ids)
        last_payload = _checkpoint_payload(last, settings, class_map, category_ids)
        # 로컬은 작업 파일을 그대로 옮기므로 이미 들어 있습니다. S3도 같아야 합니다.
        last_payload["resume_state"] = latest["resume_state"]
        artifact_uris = _publish_s3(storage, best_payload, last_payload, history, settings)
        # 업로드가 끝났으므로 이 실행의 작업 폴더는 더 이상 이어서 할 대상이 아닙니다.
        shutil.rmtree(_working_directory(settings), ignore_errors=True)
    else:
        artifact_uris = _publish_local(_working_directory(settings), history, settings)
    artifacts = {"run_id": settings["run_id"], **artifact_uris}
    best_epoch = min(history, key=lambda entry: entry["validation_loss"])
    # 이어서 한 실행의 history는 1부터 이어지므로 마지막 epoch 번호가 곧 진행한 양입니다.
    completed_epochs = history[-1]["epoch"]
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
            "precision": settings["precision"]["dtype"],
            "epochs": settings["epochs"],
            "planned_epochs": settings["epochs"],
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "train_images": len(train_dataset),
            "validation_images": len(validation_dataset),
            "class_count": len(class_map),
            "best_epoch": best_epoch["epoch"],
            "best_validation_loss": best_epoch["validation_loss"],
            "resumed_from": (
                settings["resume"]["resumed_from"] if settings["resume"] else None
            ),
            "resumed_at_epoch": (
                settings["resume"]["resumed_at_epoch"] if settings["resume"] else None
            ),
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
