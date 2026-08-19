"""Train capability 유무를 Web이 다룰 수 있는 한 형태로 맞춥니다.

Train pipeline의 현재 공개 interface는 ``run(config)``뿐이므로 Web이 Train 내부를
import하거나 임의의 함수를 호출하지 않습니다. 공개 capability 연결점이 생기기 전에는
검증된 현재 구성으로 fallback하고, 나중에는 ``reported_train_capability`` 한 곳만 공개
연결점에 맞춰 바꾸면 됩니다.

고를 수 있는 이름과 기본값은 ``src/common/train_contract``에서 그대로 가져옵니다.
예전에는 여기에 값을 복제해 두고 train의 source를 ``ast``로 읽어 어긋나는지 감시했는데,
값이 한 벌이면 어긋날 수가 없습니다. 이름을 더하고 빼는 것은 train 담당자입니다.

이 계층은 capability metadata만 호환합니다. 실제 runtime config 검증과 정규화는
``train_config``가 담당합니다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.common.train_contract import (
    ARCHITECTURE_BACKBONES,
    ARCHITECTURES as SUPPORTED_ARCHITECTURES,
    AUGMENTATIONS as SUPPORTED_AUGMENTATIONS,
    CUDA_ONLY_PRECISIONS,
    DEFAULT_ARCHITECTURE_BACKBONES,
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_ARCHITECTURE as LEGACY_ARCHITECTURE,
    DEFAULT_AUGMENTATION,
    DEFAULT_INPUT_SIZE,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_PRECISION,
    LEGACY_OPTIMIZER,
    MMDETECTION_ARCHITECTURES,
    MMDETECTION_REQUIRED,
    OPTIMIZERS as SUPPORTED_OPTIMIZERS,
    PRECISIONS as SUPPORTED_PRECISIONS,
)
from src.common.train_contract import LR_SCHEDULER_DEFAULTS as _LR_SCHEDULERS


__all__ = [
    "ARCHITECTURE_BACKBONES",
    "DEFAULT_ARCHITECTURE_BACKBONES",
    "CAPABILITY_SCHEMA_VERSION",
    "CUDA_ONLY_PRECISIONS",
    "DEFAULT_AUGMENTATION",
    "DEFAULT_LR_SCHEDULER",
    "DEFAULT_PRECISION",
    "LEGACY_ARCHITECTURE",
    "LEGACY_OPTIMIZER",
    "DEFAULT_ACCUMULATION_STEPS",
    "DEFAULT_INPUT_SIZE",
    "MMDETECTION_ARCHITECTURES",
    "MMDETECTION_REQUIRED",
    "SUPPORTED_ARCHITECTURES",
    "SUPPORTED_AUGMENTATIONS",
    "SUPPORTED_LR_SCHEDULERS",
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_PRECISIONS",
    "current_train_capability",
    "reported_train_capability",
    "resolve_train_capability",
]


CAPABILITY_SCHEMA_VERSION = 1

# 새 실험이 고르는 optimizer입니다. 값이 빠진 옛 기록만 LEGACY_OPTIMIZER로 읽습니다.
NEW_EXPERIMENT_OPTIMIZER = "AdamW"
# 고를 수 있는 schedule 이름입니다. 값 자체는 train이 씁니다.
SUPPORTED_LR_SCHEDULERS = tuple(_LR_SCHEDULERS)

_CHOICE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]{0,127}$")


def reported_train_capability() -> Any:
    """Train의 공개 capability 연결점입니다.

    아직 공용 capability 계약이 없으므로 항상 ``None``입니다. Train 내부를 탐색하거나
    import하는 우회는 소유 경계를 깨므로 하지 않습니다.
    """

    return None


def _fallback(reason: str) -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "source": "legacy_fallback",
        "fallback_reason": reason,
        "model": {
            "default": LEGACY_ARCHITECTURE,
            "choices": list(SUPPORTED_ARCHITECTURES),
            "selection_supported": True,
        },
        "optimizer": {
            "default": NEW_EXPERIMENT_OPTIMIZER,
            "choices": list(SUPPORTED_OPTIMIZERS),
            "selection_supported": True,
        },
    }


def _group(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    default = payload.get("default")
    choices = payload.get("choices")
    if (
        not isinstance(default, str)
        or _CHOICE_PATTERN.fullmatch(default) is None
        or not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
    ):
        return None

    normalized: list[str] = []
    for value in choices:
        if not isinstance(value, str) or _CHOICE_PATTERN.fullmatch(value) is None:
            return None
        if value not in normalized:
            normalized.append(value)
    if not normalized or default not in normalized:
        return None
    return {
        "default": default,
        "choices": normalized,
        "selection_supported": len(normalized) > 1,
    }


def resolve_train_capability(payload: Any) -> dict[str, Any]:
    """보고된 capability를 검증하고, 없거나 깨졌으면 현재 구성으로 되돌립니다."""

    if payload is None:
        return _fallback("train_capability_unavailable")
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        return _fallback("train_capability_invalid")

    model = _group(payload.get("model"))
    optimizer = _group(payload.get("optimizer"))
    if model is None or optimizer is None:
        return _fallback("train_capability_invalid")
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "source": "train",
        "fallback_reason": None,
        "model": model,
        "optimizer": optimizer,
    }


def current_train_capability() -> dict[str, Any]:
    """현재 Web 요청에서 사용할 정규화된 capability입니다."""

    return resolve_train_capability(reported_train_capability())
