"""학습 job 기록을 비교 화면용 experiment 요약으로 바꿉니다.

비교 화면은 원본 artifact URI나 전체 설정을 받을 필요가 없습니다. 이 adapter는 화면에
필요한 값만 고르고, 데이터셋은 URI 자체 대신 동일성 판정용 fingerprint만 돌려줍니다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from . import train_capabilities
from .jobs.model import STATUS_LABELS, JobRecord
from .masking import redact
from .train_config import DATA_ARTIFACT_KEYS


__all__ = ["experiment_summary"]


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _fingerprint(value: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _dataset(record: JobRecord) -> dict[str, Any]:
    """명시적 id를 우선하고, 없으면 완전한 artifact URI 묶음을 식별자로 씁니다."""

    explicit_id = _text(record.data_inputs.get("dataset_id"))
    if explicit_id is not None:
        return {
            "identity": _fingerprint({"dataset_id": explicit_id}),
            "identity_source": "dataset_id",
            "artifacts_complete": all(
                _text(record.data_inputs.get(key)) is not None for key in DATA_ARTIFACT_KEYS
            ),
        }

    artifacts: dict[str, str] = {}
    for key in DATA_ARTIFACT_KEYS:
        uri = _text(record.data_inputs.get(key))
        if uri is None:
            return {
                "identity": None,
                "identity_source": "unknown",
                "artifacts_complete": False,
            }
        artifacts[key] = uri.replace("\\", "/")

    return {
        "identity": _fingerprint(artifacts),
        "identity_source": "artifact_set",
        "artifacts_complete": True,
    }


def experiment_summary(
    record: JobRecord, capability: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """``JobRecord`` 한 건을 비교 화면의 안정된 최소 형태로 변환합니다.

    metric처럼 알 수 없는 값은 추정하지 않고 ``None``으로 둡니다. 다만 현재 Train에서
    고정된 model/optimizer는 capability 호환 계층의 값과 출처를 함께 기록합니다.
    """

    resolved_capability = (
        dict(capability) if capability is not None else train_capabilities.current_train_capability()
    )
    settings = record.settings
    train_summary = record.summary
    progress = record.progress
    metrics_value = train_summary.get("metrics")
    metrics = metrics_value if isinstance(metrics_value, Mapping) else {}

    recorded_architecture = _text(train_summary.get("architecture")) or _text(
        progress.get("architecture")
    )
    recorded_optimizer = _text(settings.get("optimizer"))
    model_capability = resolved_capability.get("model")
    optimizer_capability = resolved_capability.get("optimizer")
    architecture = recorded_architecture or (
        _text(model_capability.get("default"))
        if isinstance(model_capability, Mapping)
        else None
    )
    optimizer_name = recorded_optimizer or (
        _text(optimizer_capability.get("default"))
        if isinstance(optimizer_capability, Mapping)
        else None
    )
    reported_source = resolved_capability.get("source")
    capability_source = (
        reported_source
        if reported_source in {"train", "legacy_fallback"}
        else "legacy_fallback"
    )

    value = {
        "experiment_id": record.job_id,
        "run_id": record.run_id,
        "status": record.status,
        "status_label": STATUS_LABELS.get(record.status, record.status),
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "elapsed_seconds": record.elapsed_seconds(),
        "dataset": _dataset(record),
        "model": {
            "architecture": architecture,
            "pretrained": _boolean(settings.get("pretrained")),
            "source": "record" if recorded_architecture is not None else capability_source,
        },
        "optimizer": {
            "name": optimizer_name,
            "source": "record" if recorded_optimizer is not None else capability_source,
            "learning_rate": _number(settings.get("learning_rate")),
            "momentum": _number(settings.get("momentum")),
            "weight_decay": _number(settings.get("weight_decay")),
        },
        "training": {
            "device": _text(settings.get("device")),
            "epochs": _integer(settings.get("epochs")),
            "batch_size": _integer(settings.get("batch_size")),
            "num_workers": _integer(settings.get("num_workers")),
            "seed": _integer(settings.get("seed")),
        },
        "metrics": {
            "best_epoch": _integer(train_summary.get("best_epoch")),
            "best_validation_loss": _number(train_summary.get("best_validation_loss")),
            "map": _number(metrics.get("mAP")),
            "map50": _number(metrics.get("mAP50")),
        },
    }
    sanitized = redact(value)
    return sanitized if isinstance(sanitized, dict) else value
