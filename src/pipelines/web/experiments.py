"""Registry summary 목록과 선택한 experiment record를 Web 표현으로 바꿉니다."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from src.common import (
    ExperimentRegistryError,
    compare_experiment_summaries,
    list_experiment_summaries,
    read_experiment_record,
)

from . import train_capabilities
from .datasets import storage_environment
from .errors import FieldError, WebError, WebValidationError
from .masking import redact
from .paths import repository_root
from .train_config import DATA_ARTIFACT_KEYS, OPTIMIZER_PROFILES


__all__ = [
    "compare_registry_experiments",
    "list_registry_experiments",
    "registry_config",
]


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


def _dataset_from_artifacts(artifacts: Any) -> dict[str, Any]:
    if not isinstance(artifacts, Mapping):
        return {
            "identity": None,
            "identity_source": "unknown",
            "artifacts_complete": False,
        }
    selected: dict[str, str] = {}
    for key in DATA_ARTIFACT_KEYS:
        value = _text(artifacts.get(key))
        if value is None:
            return {
                "identity": None,
                "identity_source": "unknown",
                "artifacts_complete": False,
            }
        selected[key] = value.replace("\\", "/")
    return {
        "identity": _fingerprint(selected),
        "identity_source": "artifact_set",
        "artifacts_complete": True,
    }


def registry_config() -> dict[str, Any]:
    """현재 Web 환경이 선택한 storage의 Registry index를 읽을 설정입니다."""

    environment = storage_environment()
    backend = environment["default_backend"]
    storage = (
        {"backend": "s3", "s3": {"prefix": ""}}
        if backend == "s3"
        else {"backend": "local", "local": {"root": "artifacts"}}
    )
    return {"storage": storage, "registry": {"repo_root": str(repository_root())}}


def _summary_base(summary: Mapping[str, Any]) -> dict[str, Any]:
    run_id = _text(summary.get("run_id")) or ""
    created_at = _text(summary.get("created_at")) or ""
    metrics = summary.get("metrics")
    metric_values = metrics if isinstance(metrics, Mapping) else {}
    return {
        "experiment_id": run_id,
        "run_id": run_id,
        "status": "succeeded",
        "status_label": "등록 완료",
        "created_at": created_at,
        "started_at": None,
        "finished_at": created_at or None,
        "elapsed_seconds": None,
        "dataset": _dataset_from_artifacts(summary.get("artifacts")),
        "model": {"architecture": None, "pretrained": None, "source": "record"},
        "optimizer": {
            "name": None,
            "source": "record",
            "learning_rate": None,
            "momentum": None,
            "weight_decay": None,
            "beta1": None,
            "beta2": None,
            "epsilon": None,
        },
        "training": {
            "device": None,
            "epochs": None,
            "batch_size": None,
            "num_workers": None,
            "seed": _integer(summary.get("seed")),
        },
        "metrics": {
            "best_epoch": None,
            "best_validation_loss": None,
            "map": _number(metric_values.get("mAP")),
            "map50": _number(metric_values.get("mAP50")),
        },
    }


def _record_settings(record: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshot = record.get("config_snapshot")
    train = snapshot.get("train") if isinstance(snapshot, Mapping) else None
    if not isinstance(train, Mapping):
        return {}
    return train


def _enrich_summary(summary: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    value = _summary_base(summary)
    settings = _record_settings(record)
    recorded_architecture = _text(settings.get("architecture"))
    architecture = recorded_architecture or train_capabilities.LEGACY_ARCHITECTURE
    recorded_optimizer = _text(settings.get("optimizer"))
    optimizer = recorded_optimizer or train_capabilities.LEGACY_OPTIMIZER
    optimizer_source = "record" if recorded_optimizer else "legacy_fallback"
    if optimizer not in OPTIMIZER_PROFILES:
        optimizer = train_capabilities.LEGACY_OPTIMIZER
        optimizer_source = "legacy_fallback"
    profile = OPTIMIZER_PROFILES[optimizer]
    learning_rate = _number(settings.get("learning_rate"))
    seed = _integer(settings.get("seed"))
    value["model"] = {
        "architecture": architecture,
        "pretrained": _boolean(settings.get("pretrained")),
        "source": "record" if recorded_architecture else "legacy_fallback",
    }
    value["optimizer"] = {
        "name": optimizer,
        "source": optimizer_source,
        "learning_rate": (
            learning_rate if learning_rate is not None else profile["learning_rate"]
        ),
        "momentum": (
            _number(settings.get("momentum"))
            if optimizer == "SGD" and settings.get("momentum") is not None
            else profile.get("momentum")
        ),
        "weight_decay": (
            _number(settings.get("weight_decay"))
            if settings.get("weight_decay") is not None
            else profile["weight_decay"]
        ),
        "beta1": (
            _number(settings.get("beta1"))
            if optimizer != "SGD" and settings.get("beta1") is not None
            else profile.get("beta1")
        ),
        "beta2": (
            _number(settings.get("beta2"))
            if optimizer != "SGD" and settings.get("beta2") is not None
            else profile.get("beta2")
        ),
        "epsilon": (
            _number(settings.get("epsilon"))
            if optimizer != "SGD" and settings.get("epsilon") is not None
            else profile.get("epsilon")
        ),
    }
    value["training"] = {
        "device": _text(settings.get("device")),
        "epochs": _integer(settings.get("epochs")),
        "batch_size": _integer(settings.get("batch_size")),
        "num_workers": _integer(settings.get("num_workers")),
        "seed": seed if seed is not None else value["training"]["seed"],
    }
    sanitized = redact(value)
    return sanitized if isinstance(sanitized, dict) else value


def list_registry_experiments() -> list[dict[str, Any]]:
    try:
        return [_summary_base(item) for item in list_experiment_summaries(registry_config())]
    except ExperimentRegistryError as error:
        raise WebError(f"실험 목록을 읽지 못했습니다({type(error).__name__}).") from error


def compare_registry_experiments(run_ids: list[str]) -> dict[str, Any]:
    if not run_ids or not all(
        isinstance(run_id, str) and run_id.strip() for run_id in run_ids
    ):
        raise WebValidationError(
            [FieldError("run_ids", "비어 있지 않은 run_id 목록이 필요합니다.")]
        )
    config = registry_config()
    try:
        compared = compare_experiment_summaries(run_ids, config)
        summaries = {item["run_id"]: item for item in list_experiment_summaries(config)}
        uri_field = compared.get("fields", {}).get("experiment_record_uri", {})
        uri_values = uri_field.get("values", {}) if isinstance(uri_field, Mapping) else {}
        resolved: list[dict[str, Any]] = []
        for run_id in compared.get("run_ids", []):
            uri = uri_values.get(run_id) if isinstance(uri_values, Mapping) else None
            summary = summaries.get(run_id)
            if not isinstance(uri, str) or not isinstance(summary, Mapping):
                continue
            record = read_experiment_record(uri, config, expected_run_id=run_id)
            resolved.append(_enrich_summary(summary, record))
        return {"experiments": resolved, "missing": list(compared.get("missing", []))}
    except ExperimentRegistryError as error:
        raise WebError(f"실험 비교 정보를 읽지 못했습니다({type(error).__name__}).") from error
