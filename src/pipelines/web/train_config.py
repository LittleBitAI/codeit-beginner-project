"""Train 설정 검증과 runtime config 생성.

이 module은 ``src/pipelines/train/pipeline.py``의 검증 규칙을 그대로 따라 합니다.
train을 import하지 않고 규칙만 복제하므로, GUI가 GPU 시간을 쓰기 전에 같은 이유로
같은 값을 거부합니다. 여기서 추가로 거부하는 것은 train보다 **먼저** 막는 경우뿐이고,
train이 거부하는 값을 여기서 통과시키지 않습니다.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from .errors import (
    FieldError,
    JobNotFoundError,
    WebPathError,
    WebValidationError,
    collect,
    raise_if_any,
)
from .gpu import cuda_is_available, native_bf16_supported
from .masking import redact
from .paths import (
    CONFIG_DIRNAME,
    config_dir,
    normalize_relative_posix,
    resolve_within_repo,
)
from .train_capabilities import (
    CUDA_ONLY_PRECISIONS,
    DEFAULT_AUGMENTATION,
    DEFAULT_PRECISION,
    LEGACY_ARCHITECTURE,
    LEGACY_OPTIMIZER,
    NEW_EXPERIMENT_OPTIMIZER,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_AUGMENTATIONS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_PRECISIONS,
)


__all__ = [
    "DATA_ARTIFACT_KEYS",
    "OPTIONAL_DATA_ARTIFACT_KEYS",
    "OPTIMIZER_PROFILES",
    "build_runtime_config",
    "field_specs",
    "normalize_data_inputs",
    "normalize_train_settings",
    "read_runtime_config",
    "validate_request",
    "write_runtime_config",
]


# train/pipeline.py:32 와 동일
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# train/pipeline.py:26-31 과 동일한 4개. 화면 표시 순서를 위해 tuple로 둡니다.
DATA_ARTIFACT_KEYS = (
    "train_manifest_uri",
    "validation_manifest_uri",
    "class_map_uri",
    "dataset_summary_uri",
)

# Train은 읽지 않지만, 성공한 학습 뒤 Evaluate가 대회 submission을 만들 때 씁니다.
# 기존 데이터셋에는 없어도 되므로 필수 4개와 분리합니다.
OPTIONAL_DATA_ARTIFACT_KEYS = ("test_manifest_uri",)

DEFAULT_OUTPUT_DIR = "artifacts/experiments/completed"
DEFAULT_OUTPUT_PREFIX = "experiments/completed"

# train은 patience에 기본값이 없습니다(있으면 필수). 화면이 안내하는 출발값입니다.
DEFAULT_EARLY_STOPPING_PATIENCE = 5
# train의 기본값과 같아야 합니다. test_web_train_contract.py가 대조합니다.
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
_EARLY_STOPPING_FIELDS = ("early_stopping_patience", "early_stopping_min_delta")

# (이름, 기본값, 최소값)
_INTEGER_FIELDS = (
    ("seed", 42, 0),
    ("epochs", 1, 1),
    ("batch_size", 1, 1),
    ("num_workers", 0, 0),
    ("checkpoint_every", 1, 1),
)
# 이어서 학습할 checkpoint의 파일 이름과 작업 폴더 규칙입니다. train이 정한 것을
# 그대로 옮겼습니다(`src/pipelines/train/pipeline.py`).
RESUME_CHECKPOINT_NAME = "last_checkpoint.pt"
WORKING_DIRECTORY_SUFFIX = ".partial"
RUNNING_PREFIX = "running"
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

_FIELD_LABELS = {
    "run_id": ("실행 이름", "실행 결과가 저장되는 directory 이름으로 그대로 쓰입니다."),
    "architecture": ("모델", "학습에 사용할 object detection architecture입니다."),
    "optimizer": ("Optimizer", "가중치를 갱신할 optimizer와 관련 수치 항목을 선택합니다."),
    "augmentation": (
        "증강 preset",
        "학습 split에만 적용합니다. 데이터가 적을 때 pill_basic이 과적합을 줄여 줍니다.",
    ),
    "precision": (
        "연산 정밀도",
        "fp32 외에는 device가 cuda여야 합니다. amp는 GPU를 보고 bf16과 fp16 중 되는 쪽을"
        " 골라 줍니다. fp16은 어느 GPU에서나 되고, bf16은 Ampere(RTX 30 시리즈, A100) 이상만"
        " 됩니다. T4처럼 bf16이 없는 GPU라면 amp나 fp16을 쓰세요.",
    ),
    "seed": ("Random seed", "같은 seed와 같은 데이터면 같은 결과가 나옵니다."),
    "checkpoint_every": (
        "Checkpoint 주기",
        "몇 epoch마다 이어서 할 수 있는 checkpoint를 남길지 정합니다. 1이면 매 epoch입니다.",
    ),
    "epochs": ("Epochs", "전체 학습 데이터를 몇 번 반복할지 정합니다."),
    "batch_size": ("Batch size", "한 번에 처리할 이미지 수. GPU 메모리에 가장 큰 영향을 줍니다."),
    "num_workers": ("DataLoader workers", "이미지를 읽어 오는 보조 process 수. 0이면 주 process가 직접 읽습니다."),
    "learning_rate": ("Learning rate", "한 번에 얼마나 크게 배울지 정합니다. 너무 크면 발산합니다."),
    "momentum": ("Momentum", "SGD가 이전 방향을 얼마나 유지할지 정합니다."),
    "weight_decay": ("Weight decay", "과적합을 억제하는 정규화 강도입니다."),
    "beta1": ("Beta 1", "Adam 계열의 1차 모멘트 감쇠율입니다."),
    "beta2": ("Beta 2", "Adam 계열의 2차 모멘트 감쇠율입니다."),
    "epsilon": ("Epsilon", "Adam 계열 계산에서 0으로 나누는 것을 막는 값입니다."),
    "early_stopping": (
        "조기 종료",
        "검증 손실이 나아지지 않으면 남은 epoch를 채우지 않고 학습을 끝냅니다.",
    ),
    "early_stopping_patience": (
        "Patience",
        "몇 epoch 연속으로 나아지지 않으면 끝낼지 정합니다. 1 이상의 정수입니다.",
    ),
    "early_stopping_min_delta": (
        "Min delta",
        "이만큼은 낮아져야 나아진 것으로 봅니다. 0이면 조금이라도 낮아지면 됩니다.",
    ),
    "device": ("Device", "학습을 CPU에서 할지 CUDA GPU에서 할지 정합니다."),
    "pretrained": ("Pretrained 가중치", "COCO로 미리 학습된 가중치에서 시작할지 정합니다."),
    "output_dir": ("Local 출력 directory", "저장소 기준 상대 경로여야 합니다."),
    "output_prefix": ("S3 출력 prefix", "S3 backend를 쓸 때 checkpoint를 올릴 위치입니다."),
}

_DATA_LABELS = {
    "train_manifest_uri": "학습 manifest",
    "validation_manifest_uri": "검증 manifest",
    "class_map_uri": "클래스 맵",
    "dataset_summary_uri": "데이터셋 요약",
    "test_manifest_uri": "테스트 manifest",
}


def _utc_now() -> datetime:
    """테스트에서 고정할 수 있도록 현재 시각을 한 곳에서만 읽습니다."""

    return datetime.now(timezone.utc)


def generate_run_id() -> str:
    """GUI가 만든 실행임을 알 수 있는 run_id를 만듭니다.

    train의 기본값은 ``train-`` 접두사를 쓰므로, CLI로 돌린 실행과 구분됩니다.
    """

    return _utc_now().strftime("web-%Y%m%dT%H%M%S%fZ")


def field_specs() -> list[dict[str, Any]]:
    """새 실험 화면이 form을 그릴 때 쓰는 필드 정의입니다."""

    specs: list[dict[str, Any]] = []
    for name, default, choices in (
        ("architecture", LEGACY_ARCHITECTURE, SUPPORTED_ARCHITECTURES),
        ("optimizer", NEW_EXPERIMENT_OPTIMIZER, SUPPORTED_OPTIMIZERS),
        ("augmentation", DEFAULT_AUGMENTATION, SUPPORTED_AUGMENTATIONS),
        ("precision", DEFAULT_PRECISION, SUPPORTED_PRECISIONS),
    ):
        label, hint = _FIELD_LABELS[name]
        specs.append(
            {
                "name": name,
                "type": "enum",
                "default": default,
                "choices": list(choices),
                "label": label,
                "hint": hint,
            }
        )
    label, hint = _FIELD_LABELS["run_id"]
    specs.append(
        {
            "name": "run_id",
            "type": "string",
            "default": None,
            "label": label,
            "hint": hint,
            "pattern": RUN_ID_PATTERN.pattern,
            "placeholder": "비워 두면 자동으로 만듭니다",
        }
    )
    for name, default, minimum in _INTEGER_FIELDS:
        label, hint = _FIELD_LABELS[name]
        specs.append(
            {
                "name": name,
                "type": "integer",
                "default": default,
                "minimum": minimum,
                "label": label,
                "hint": hint,
            }
        )
    default_profile = OPTIMIZER_PROFILES[NEW_EXPERIMENT_OPTIMIZER]
    for name in (
        "learning_rate",
        "weight_decay",
        "momentum",
        "beta1",
        "beta2",
        "epsilon",
    ):
        fallback_profile = OPTIMIZER_PROFILES["SGD"] if name == "momentum" else default_profile
        default = fallback_profile.get(name)
        minimum = 1e-16 if name == "epsilon" else 0.0
        label, hint = _FIELD_LABELS[name]
        specs.append(
            {
                "name": name,
                "type": "number",
                "default": default,
                "defaults_by_optimizer": {
                    optimizer: profile[name]
                    for optimizer, profile in OPTIMIZER_PROFILES.items()
                    if name in profile
                },
                "minimum": minimum,
                "label": label,
                "hint": hint,
            }
        )
    label, hint = _FIELD_LABELS["device"]
    specs.append(
        {
            "name": "device",
            "type": "enum",
            "default": "cpu",
            "choices": ["cpu", "cuda"],
            "label": label,
            "hint": hint,
        }
    )
    label, hint = _FIELD_LABELS["early_stopping"]
    specs.append(
        {"name": "early_stopping", "type": "boolean", "default": False, "label": label, "hint": hint}
    )
    label, hint = _FIELD_LABELS["early_stopping_patience"]
    specs.append(
        {
            "name": "early_stopping_patience",
            "type": "integer",
            "default": DEFAULT_EARLY_STOPPING_PATIENCE,
            "minimum": 1,
            "label": label,
            "hint": hint,
        }
    )
    label, hint = _FIELD_LABELS["early_stopping_min_delta"]
    specs.append(
        {
            "name": "early_stopping_min_delta",
            "type": "number",
            "default": DEFAULT_EARLY_STOPPING_MIN_DELTA,
            "minimum": 0.0,
            "label": label,
            "hint": hint,
        }
    )
    label, hint = _FIELD_LABELS["pretrained"]
    # 화면에서 시작하는 학습은 사전학습 가중치를 기본으로 씁니다. 아래
    # normalize_train_settings의 fallback은 train 기본값(False) 그대로 두어야 합니다.
    # 그쪽은 다른 소유 영역이고 test_web_train_contract.py가 두 값을 대조합니다.
    # 화면은 이 spec을 보고 pretrained를 명시적으로 실어 보냅니다.
    specs.append(
        {"name": "pretrained", "type": "boolean", "default": True, "label": label, "hint": hint}
    )
    for name, default in (
        ("output_dir", DEFAULT_OUTPUT_DIR),
        ("output_prefix", DEFAULT_OUTPUT_PREFIX),
    ):
        label, hint = _FIELD_LABELS[name]
        specs.append(
            {"name": name, "type": "string", "default": default, "label": label, "hint": hint}
        )
    return specs


def data_field_specs() -> list[dict[str, Any]]:
    """Data pipeline artifact 입력 칸 정의입니다."""

    return [
        {
            "name": key,
            "type": "uri",
            "label": _DATA_LABELS[key],
            "hint": "저장소 기준 상대 경로 또는 s3://bucket/key 형식입니다.",
            "required": True,
        }
        for key in DATA_ARTIFACT_KEYS
    ]


def _normalize_integer(
    raw: Any, name: str, default: int, minimum: int, errors: list[FieldError]
) -> int:
    value = raw.get(name, default)
    # train은 bool을 int 자리에서 명시적으로 거부합니다.
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        collect(errors, f"train.{name}", f"{minimum} 이상의 정수여야 합니다.")
        return default
    return value


def _normalize_float(
    raw: Any, name: str, default: float, minimum: float, errors: list[FieldError]
) -> float:
    value = raw.get(name, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        collect(errors, f"train.{name}", f"{minimum} 이상의 유한한 숫자여야 합니다.")
        return default
    return float(value)


def _normalize_probability(
    raw: Any, name: str, default: float, errors: list[FieldError]
) -> float:
    value = _normalize_float(raw, name, default, 0.0, errors)
    if value >= 1.0:
        collect(errors, f"train.{name}", "0 이상 1 미만의 유한한 숫자여야 합니다.")
        return default
    return value


def _normalize_early_stopping(
    raw: Any, errors: list[FieldError]
) -> dict[str, float | int] | None:
    """평평한 세 칸을 train이 받는 object 하나로 접습니다.

    검증 규칙은 train의 ``_early_stopping``(pipeline.py)을 그대로 옮긴 것입니다.
    꺼져 있으면 key 자체를 만들지 않아, 조기 종료를 쓰지 않는 사람의 config는
    이 기능이 생기기 전과 한 글자도 달라지지 않습니다.
    """

    enabled = raw.get("early_stopping", False)
    if not isinstance(enabled, bool):
        collect(errors, "train.early_stopping", "true 또는 false여야 합니다.")
        enabled = False

    if not enabled:
        # 끈 채로 값을 보내면 화면과 실제 학습이 달라 보입니다. optimizer에서 쓰지
        # 않는 값을 막는 것과 같은 이유입니다.
        for name in sorted(set(_EARLY_STOPPING_FIELDS) & set(raw)):
            collect(errors, f"train.{name}", "조기 종료를 사용할 때만 쓰는 값입니다.")
        return None

    # train은 patience를 필수로 받지만 그건 train이 받는 object의 규칙입니다. 다른 수치
    # 칸과 똑같이 비어 있으면 화면이 안내한 기본값을 채워 완성된 object를 보냅니다.
    patience = _normalize_integer(
        raw, "early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE, 1, errors
    )
    min_delta = _normalize_float(
        raw, "early_stopping_min_delta", DEFAULT_EARLY_STOPPING_MIN_DELTA, 0.0, errors
    )
    return {"patience": patience, "min_delta": min_delta}


def normalize_train_settings(raw: Any) -> dict[str, Any]:
    """``config["train"]`` 후보를 train과 같은 규칙으로 정규화합니다.

    문제를 하나 발견하면 멈추지 않고 모두 모아서 한 번에 보고합니다. 화면에서 여러
    칸의 오류를 동시에 보여줘야 하기 때문입니다.
    """

    errors: list[FieldError] = []
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise WebValidationError([FieldError("train", "train 설정은 object여야 합니다.")])

    architecture = raw.get("architecture", LEGACY_ARCHITECTURE)
    if not isinstance(architecture, str) or architecture not in SUPPORTED_ARCHITECTURES:
        collect(
            errors,
            "train.architecture",
            f"{', '.join(SUPPORTED_ARCHITECTURES)} 중 하나여야 합니다.",
        )
        architecture = LEGACY_ARCHITECTURE

    augmentation = raw.get("augmentation", DEFAULT_AUGMENTATION)
    if not isinstance(augmentation, str) or augmentation not in SUPPORTED_AUGMENTATIONS:
        collect(
            errors,
            "train.augmentation",
            f"{', '.join(SUPPORTED_AUGMENTATIONS)} 중 하나여야 합니다.",
        )
        augmentation = DEFAULT_AUGMENTATION

    precision = raw.get("precision", DEFAULT_PRECISION)
    if not isinstance(precision, str) or precision not in SUPPORTED_PRECISIONS:
        collect(
            errors,
            "train.precision",
            f"{', '.join(SUPPORTED_PRECISIONS)} 중 하나여야 합니다.",
        )
        precision = DEFAULT_PRECISION

    optimizer = raw.get("optimizer", LEGACY_OPTIMIZER)
    if not isinstance(optimizer, str) or optimizer not in SUPPORTED_OPTIMIZERS:
        collect(
            errors,
            "train.optimizer",
            f"{', '.join(SUPPORTED_OPTIMIZERS)} 중 하나여야 합니다.",
        )
        optimizer = LEGACY_OPTIMIZER
    irrelevant = {"momentum"} if optimizer in {"AdamW", "Adam"} else {
        "beta1",
        "beta2",
        "epsilon",
    }
    for name in sorted(irrelevant & set(raw)):
        collect(errors, f"train.{name}", f"{optimizer}에서 사용하지 않는 값입니다.")

    device = raw.get("device", "cpu")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        collect(errors, "train.device", "'cpu' 또는 'cuda'여야 합니다.")
        device = "cpu"
    elif device == "cuda" and not cuda_is_available():
        collect(errors, "train.device", "CUDA를 사용할 수 없는 환경입니다.")

    if precision in CUDA_ONLY_PRECISIONS and device != "cuda":
        # train이 같은 조건을 거부합니다. 여기서 잡지 않으면 subprocess가 뜬 뒤에야
        # 알게 되고, 화면에는 어느 칸이 잘못됐는지 남지 않습니다.
        collect(
            errors,
            "train.precision",
            f"{precision} 정밀도는 device가 cuda일 때만 쓸 수 있습니다.",
        )
        precision = DEFAULT_PRECISION
    elif precision == "bf16" and native_bf16_supported() is False:
        # T4 같은 Turing GPU는 bf16을 emulation으로만 처리해 아주 느립니다. 확실히
        # 안 되는 것을 아는 경우에만 막고, 알 수 없으면(None) train이 판단하게 둡니다.
        collect(
            errors,
            "train.precision",
            "이 컴퓨터의 GPU는 bf16을 지원하지 않습니다. fp16이나 amp를 쓰세요.",
        )
        precision = DEFAULT_PRECISION

    pretrained = raw.get("pretrained", False)
    if not isinstance(pretrained, bool):
        collect(errors, "train.pretrained", "true 또는 false여야 합니다.")
        pretrained = False

    resume_from = _normalize_resume_from(raw, errors)

    run_id = raw.get("run_id") or generate_run_id()
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        collect(
            errors,
            "train.run_id",
            "영문/숫자로 시작하고 영문·숫자·마침표·밑줄·붙임표만 쓸 수 있습니다(최대 128자).",
        )
        run_id = generate_run_id()

    output_dir = raw.get("output_dir", DEFAULT_OUTPUT_DIR)
    if not isinstance(output_dir, str) or not output_dir.strip():
        collect(errors, "train.output_dir", "비어 있지 않은 저장소 기준 상대 경로여야 합니다.")
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        try:
            output_dir = normalize_relative_posix(output_dir, label="출력 directory")
            resolve_within_repo(output_dir, label="출력 directory")
        except WebPathError as error:
            collect(errors, "train.output_dir", str(error))
            output_dir = DEFAULT_OUTPUT_DIR

    output_prefix = raw.get("output_prefix", DEFAULT_OUTPUT_PREFIX)
    if not isinstance(output_prefix, str) or not output_prefix.strip():
        collect(errors, "train.output_prefix", "비어 있지 않은 S3 prefix여야 합니다.")
        output_prefix = DEFAULT_OUTPUT_PREFIX

    settings: dict[str, Any] = {
        "run_id": run_id,
        "architecture": architecture,
        "optimizer": optimizer,
        "augmentation": augmentation,
        "precision": precision,
        "device": device,
        "pretrained": pretrained,
        "early_stopping": _normalize_early_stopping(raw, errors),
        "output_dir": output_dir,
        "output_prefix": output_prefix.strip("/"),
    }
    for name, default, minimum in _INTEGER_FIELDS:
        settings[name] = _normalize_integer(raw, name, default, minimum, errors)
    profile = OPTIMIZER_PROFILES[optimizer]
    for name in ("learning_rate", "weight_decay"):
        settings[name] = _normalize_float(raw, name, profile[name], 0.0, errors)
    if optimizer == "SGD":
        settings["momentum"] = _normalize_float(
            raw, "momentum", profile["momentum"], 0.0, errors
        )
    else:
        settings["beta1"] = _normalize_probability(
            raw, "beta1", profile["beta1"], errors
        )
        settings["beta2"] = _normalize_probability(
            raw, "beta2", profile["beta2"], errors
        )
        settings["epsilon"] = _normalize_float(
            raw, "epsilon", profile["epsilon"], 1e-16, errors
        )

    raise_if_any(errors)
    # train이 읽는 순서와 같게 정렬해 config를 읽기 쉽게 만듭니다.
    return {
        "run_id": settings["run_id"],
        "architecture": settings["architecture"],
        "optimizer": settings["optimizer"],
        # train은 preset key 하나만 든 object를 받고 다른 key가 있으면 거부합니다.
        "augmentation": {"preset": settings["augmentation"]},
        # 문자열 하나입니다. amp가 bf16이 될지 fp16이 될지는 train이 GPU를 보고 정합니다.
        "precision": settings["precision"],
        "seed": settings["seed"],
        "epochs": settings["epochs"],
        "checkpoint_every": settings["checkpoint_every"],
        "batch_size": settings["batch_size"],
        "num_workers": settings["num_workers"],
        # 처음부터 학습하는 실행은 key 자체를 넣지 않습니다. train은 없으면 지금과
        # 완전히 같게 동작합니다.
        **({"resume_from": resume_from} if resume_from is not None else {}),
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "device": settings["device"],
        "pretrained": settings["pretrained"],
        # 끄면 key 자체를 넣지 않습니다. train은 없으면 전체 epoch를 그대로 돕니다.
        **(
            {"early_stopping": settings["early_stopping"]}
            if settings["early_stopping"] is not None
            else {}
        ),
        "output_dir": settings["output_dir"],
        "output_prefix": settings["output_prefix"],
        **(
            {"momentum": settings["momentum"]}
            if optimizer == "SGD"
            else {
                "beta1": settings["beta1"],
                "beta2": settings["beta2"],
                "epsilon": settings["epsilon"],
            }
        ),
    }


def _normalize_resume_from(raw: dict, errors: list[FieldError]) -> str | None:
    """이어서 학습할 checkpoint 경로입니다. 없으면 처음부터 학습합니다.

    train은 저장소 안의 상대 경로와 ``s3://`` URI를 받습니다. 같은 규칙으로 미리
    거르지 않으면 subprocess가 뜬 뒤에야 알게 됩니다.
    """

    if "resume_from" not in raw:
        return None
    value = raw["resume_from"]
    if not isinstance(value, str) or not value.strip():
        collect(errors, "train.resume_from", "비어 있지 않은 문자열이어야 합니다.")
        return None

    text = value.strip()
    if text.lower().startswith("s3://"):
        split = urlsplit(text)
        if not split.netloc or not split.path.strip("/"):
            collect(errors, "train.resume_from", "s3://bucket/key 형식이어야 합니다.")
        return text
    if "://" in text:
        collect(
            errors,
            "train.resume_from",
            "저장소 기준 상대 경로 또는 s3:// URI만 쓸 수 있습니다.",
        )
        return text
    try:
        return normalize_relative_posix(text, label="이어서 학습할 checkpoint")
    except WebPathError as error:
        collect(errors, "train.resume_from", str(error))
        return text


def _normalize_data_uri(value: Any, key: str, errors: list[FieldError]) -> str:
    field = f"inputs.data.{key}"
    if not isinstance(value, str) or not value.strip():
        collect(errors, field, "비어 있지 않은 문자열이어야 합니다.")
        return ""

    text = value.strip()
    lowered = text.lower()
    if lowered.startswith("s3://"):
        split = urlsplit(text)
        if not split.netloc or not split.path.strip("/"):
            collect(errors, field, "s3://bucket/key 형식이어야 합니다.")
            return text
        if split.query or split.fragment:
            collect(errors, field, "s3 URI에 query나 fragment를 쓸 수 없습니다.")
        return text
    if "://" in text:
        collect(errors, field, "저장소 기준 상대 경로 또는 s3:// URI만 쓸 수 있습니다.")
        return text

    try:
        return normalize_relative_posix(text, label=_DATA_LABELS[key])
    except WebPathError as error:
        collect(errors, field, str(error))
        return text


def normalize_data_inputs(raw: Any) -> dict[str, str]:
    """필수 artifact 4개와 선택 test manifest URI를 검증합니다."""

    errors: list[FieldError] = []
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise WebValidationError([FieldError("inputs.data", "data 입력은 object여야 합니다.")])

    resolved: dict[str, str] = {}
    for key in DATA_ARTIFACT_KEYS:
        if key not in raw:
            collect(errors, f"inputs.data.{key}", f"{_DATA_LABELS[key]} 위치가 필요합니다.")
            continue
        resolved[key] = _normalize_data_uri(raw[key], key, errors)

    for key in OPTIONAL_DATA_ARTIFACT_KEYS:
        if key in raw:
            resolved[key] = _normalize_data_uri(raw[key], key, errors)

    raise_if_any(errors)
    ordered_keys = DATA_ARTIFACT_KEYS + OPTIONAL_DATA_ARTIFACT_KEYS
    return {key: resolved[key] for key in ordered_keys if key in resolved}


def uses_s3(data_inputs: dict[str, str]) -> bool:
    return any(value.lower().startswith("s3://") for value in data_inputs.values())


def run_id_output_path(settings: dict[str, Any]) -> Path:
    """Local backend에서 이 run이 결과를 쓸 directory입니다."""

    return resolve_within_repo(settings["output_dir"], label="출력 directory") / settings["run_id"]


def run_id_working_path(settings: dict[str, Any]) -> Path:
    """중단된 학습의 checkpoint가 남아 있는 자리입니다.

    train이 학습 중 checkpoint를 두는 폴더와 같은 이름이어야 합니다.
    """

    parent = resolve_within_repo(settings["output_dir"], label="출력 directory")
    return parent / f".{settings['run_id']}{WORKING_DIRECTORY_SUFFIX}"


def resume_checkpoint_uri(config: dict[str, Any]) -> str:
    """이 실행을 이어서 하려면 어느 checkpoint를 봐야 하는지 알려 줍니다."""

    train = config["train"]
    backend = config.get("storage", {}).get("backend")
    if backend == "s3":
        bucket = os.environ.get("PILL_STORAGE_S3_BUCKET", "").strip()
        if not bucket:
            raise WebValidationError(
                [
                    FieldError(
                        "train.resume_from",
                        "S3에 올라간 학습을 이어서 하려면 PILL_STORAGE_S3_BUCKET "
                        "환경 변수가 필요합니다.",
                    )
                ]
            )
        storage = config.get("storage")
        s3 = storage.get("s3") if isinstance(storage, dict) else None
        configured_prefix = s3.get("prefix") if isinstance(s3, dict) else None
        common_prefix = next(
            (
                str(value).strip().strip("/")
                for value in (
                    os.environ.get("PILL_STORAGE_S3_PREFIX"),
                    configured_prefix,
                )
                if value is not None and str(value).strip()
            ),
            "",
        )
        key = "/".join(
            part
            for part in (
                common_prefix,
                train["output_prefix"].strip("/"),
                train["run_id"],
                RUNNING_PREFIX,
                RESUME_CHECKPOINT_NAME,
            )
            if part
        )
        return f"s3://{bucket}/{key}"
    directory = train["output_dir"].strip("/")
    return (
        f"{directory}/.{train['run_id']}{WORKING_DIRECTORY_SUFFIX}"
        f"/{RESUME_CHECKPOINT_NAME}"
    )


def build_resume_config(
    config: dict[str, Any],
    *,
    run_id: str | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """중단된 실행의 설정을 그대로 두고 이어서 학습할 config를 만듭니다.

    이어서 하는 실행은 **새 이름**을 받습니다. 같은 이름을 다시 쓰면 train이 남아 있는
    작업 폴더를 보고 시작을 거부하고, 결과도 섞입니다.
    """

    resumed = copy.deepcopy(config)
    train = resumed["train"]
    train["resume_from"] = resume_checkpoint_uri(config)
    new_run_id = run_id or generate_run_id()
    if not isinstance(new_run_id, str) or not RUN_ID_PATTERN.fullmatch(new_run_id):
        raise WebValidationError(
            [FieldError("train.run_id", "실행 이름에 쓸 수 없는 글자가 있습니다.")]
        )
    train["run_id"] = new_run_id
    if epochs is not None:
        # train은 epochs를 남은 수가 아니라 전체 목표로 읽고, 이어붙일 epoch보다
        # 크지 않으면 시작 전에 거부합니다.
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise WebValidationError(
                [FieldError("train.epochs", "1 이상의 정수여야 합니다.")]
            )
        train["epochs"] = epochs
    return resumed


def preflight_warnings(
    settings: dict[str, Any], data_inputs: dict[str, str]
) -> list[dict[str, str]]:
    """실행을 막지는 않지만 미리 알려 주면 좋은 것들을 모읍니다."""

    warnings: list[dict[str, str]] = []
    if not uses_s3(data_inputs):
        for key, value in data_inputs.items():
            try:
                path = resolve_within_repo(value, label=_DATA_LABELS[key])
            except WebPathError:
                continue
            if not path.exists():
                warnings.append(
                    {
                        "field": f"inputs.data.{key}",
                        "message": f"{_DATA_LABELS[key]} 파일이 아직 없습니다. 학습 시작 직후 실패할 수 있습니다.",
                    }
                )
    if settings["device"] == "cpu":
        warnings.append(
            {
                "field": "train.device",
                "message": "CPU로 학습하면 GPU보다 매우 오래 걸립니다.",
            }
        )
    if settings["num_workers"] > 0 and os.name == "nt":
        warnings.append(
            {
                "field": "train.num_workers",
                "message": "Windows에서는 worker를 늘려도 효과가 작고 메모리를 더 씁니다.",
            }
        )
    return warnings


def check_run_id_collision(settings: dict[str, Any], data_inputs: dict[str, str]) -> None:
    """이미 같은 run_id의 결과가 있으면 시작 전에 막습니다.

    train은 학습을 **끝낸 뒤** 저장 단계에서야 ``FileExistsError``를 냅니다
    (``train/pipeline.py:154``). 몇 시간을 버리지 않도록 여기서 먼저 확인합니다.
    """

    if uses_s3(data_inputs):
        return  # S3 존재 확인은 credential이 필요하므로 시도하지 않습니다.
    if run_id_output_path(settings).exists():
        raise WebValidationError(
            [
                FieldError(
                    "train.run_id",
                    f"'{settings['run_id']}' 결과가 이미 있습니다. 다른 이름을 쓰세요.",
                )
            ]
        )
    working = run_id_working_path(settings)
    # train은 여기가 비어 있지 않으면 시작을 거부합니다. 중단된 학습의 유일한 사본이라
    # 덮어쓰지 않습니다.
    if working.is_dir() and any(working.iterdir()):
        raise WebValidationError(
            [
                FieldError(
                    "train.run_id",
                    f"'{settings['run_id']}' 학습이 중단된 채로 남아 있습니다. "
                    "이어서 학습하거나 다른 이름을 쓰세요.",
                )
            ]
        )


def validate_request(payload: Any) -> dict[str, Any]:
    """검증 결과를 화면이 그대로 쓸 수 있는 형태로 돌려줍니다. 아무것도 쓰지 않습니다."""

    if not isinstance(payload, dict):
        return {
            "valid": False,
            "errors": [{"field": "body", "message": "요청 본문은 object여야 합니다."}],
            "warnings": [],
            "normalized": None,
        }

    errors: list[dict[str, str]] = []
    settings: dict[str, Any] | None = None
    data_inputs: dict[str, str] | None = None

    try:
        settings = normalize_train_settings(payload.get("train"))
    except WebValidationError as error:
        errors.extend(error.as_list())

    inputs = payload.get("inputs")
    data_section = inputs.get("data") if isinstance(inputs, dict) else payload.get("data")
    try:
        data_inputs = normalize_data_inputs(data_section)
    except WebValidationError as error:
        errors.extend(error.as_list())

    warnings: list[dict[str, str]] = []
    if settings is not None and data_inputs is not None and not errors:
        try:
            check_run_id_collision(settings, data_inputs)
        except WebValidationError as error:
            errors.extend(error.as_list())
        warnings = preflight_warnings(settings, data_inputs)

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "normalized": None}
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "normalized": build_runtime_config(settings, data_inputs),
    }


def build_runtime_config(
    settings: dict[str, Any], data_inputs: dict[str, str]
) -> dict[str, Any]:
    """``--only train``에 넘길 완결된 config를 만듭니다.

    ``src/common/config.py``의 ``load_config``는 파일 하나만 읽고 병합하지 않으므로
    이 결과는 그 자체로 완전해야 합니다.
    """

    if uses_s3(data_inputs):
        # bucket 이름은 환경 변수(PILL_STORAGE_S3_BUCKET)에서 오므로 config에 넣지 않습니다.
        storage: dict[str, Any] = {"backend": "s3", "s3": {"prefix": ""}}
    else:
        storage = {"backend": "local", "local": {"root": "artifacts"}}

    return {
        "project": {"name": "pill-object-detection"},
        # configs/base.json은 execution.mode가 "dummy"입니다. 그 값이 들어오면 train이
        # 학습을 건너뛰고 dummy 결과만 돌려주므로, 여기서 항상 명시적으로 덮어씁니다.
        "execution": {"mode": "real"},
        "storage": storage,
        "train": dict(settings),
        "inputs": {"data": dict(data_inputs)},
    }


def _config_path(config_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", config_id):
        raise JobNotFoundError("설정을 찾을 수 없습니다.")
    return config_dir() / f"{config_id}.json"


def write_runtime_config(config: dict[str, Any]) -> str:
    """Runtime config를 gitignore된 위치에 원자적으로 기록하고 id를 돌려줍니다."""

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    config_id = uuid4().hex
    destination = directory / f"{config_id}.json"
    payload = json.dumps(config, ensure_ascii=False, indent=2) + "\n"

    handle, temporary_name = tempfile.mkstemp(
        dir=directory, prefix=f".{config_id}-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return config_id


def read_runtime_config(config_id: str) -> dict[str, Any]:
    """저장해 둔 runtime config를 읽습니다."""

    path = _config_path(config_id)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise JobNotFoundError("설정을 찾을 수 없습니다.") from error
    try:
        config = json.loads(content)
    except json.JSONDecodeError as error:
        raise JobNotFoundError("저장된 설정을 읽을 수 없습니다.") from error
    if not isinstance(config, dict):
        raise JobNotFoundError("저장된 설정 형식이 올바르지 않습니다.")
    return config


def config_relative_path(config_id: str) -> str:
    """subprocess ``--config`` 인자로 넘길 저장소 기준 상대 경로입니다."""

    _config_path(config_id)  # id 형식 검증
    return f"{CONFIG_DIRNAME}/{config_id}.json"


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """화면에 돌려줄 때 credential처럼 보이는 값을 가립니다."""

    return redact(config)
