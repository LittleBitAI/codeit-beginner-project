"""Data pipeline의 artifact를 만들거나 기존 artifact를 다음 pipeline에 연결합니다.

실행 경로는 세 가지입니다.

1. `execution.mode == "dummy"`: 기존 dummy 결과를 그대로 반환합니다.
2. `data.prepare == true`: 원본에서 artifact 5개를 만들거나, 이전 버전의 기존
   artifact 4개가 모두 있으면 누락된 test manifest만 보충합니다(`preparation.py`).
3. 그 외: `config["inputs"]["data"]`에 사전 제공된 URI 4개를 검증해 공개합니다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import DatasetPreparationError
from .preparation import (
    preparation_error_result,
    preparation_requested,
    run_preparation,
)


__all__ = ["run"]

_ARTIFACT_KEYS = (
    "train_manifest_uri",
    "validation_manifest_uri",
    "class_map_uri",
    "dataset_summary_uri",
)


def _dummy_result() -> dict[str, Any]:
    """기존 dummy 실행 결과를 그대로 반환합니다."""

    return {
        "status": "ok",
        "artifacts": {},
        "summary": {"pipeline": "data", "mode": "dummy"},
        "message": "data pipeline dummy 실행 완료",
    }


def _input_error_result() -> dict[str, Any]:
    """필수 artifact 입력이 잘못됐을 때 공통 오류 결과를 반환합니다."""

    required_keys = ", ".join(_ARTIFACT_KEYS)
    return {
        "status": "error",
        "artifacts": {},
        "summary": {
            "pipeline": "data",
            "mode": "integration",
            "required_artifact_keys": list(_ARTIFACT_KEYS),
        },
        "message": (
            "config['inputs']['data']에는 다음 네 key의 비어 있지 않은 문자열이 "
            f"필요합니다: {required_keys}"
        ),
    }


def _is_dummy(config: Mapping[str, Any]) -> bool:
    execution = config.get("execution")
    return isinstance(execution, Mapping) and execution.get("mode") == "dummy"


def _validated_artifacts(config: Any) -> dict[str, str] | None:
    """입력을 수정하지 않고 공개할 네 URI를 새 dict로 복사합니다."""

    if not isinstance(config, Mapping):
        return None
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    data = inputs.get("data")
    if not isinstance(data, Mapping):
        return None

    artifacts: dict[str, str] = {}
    for key in _ARTIFACT_KEYS:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
        artifacts[key] = value
    return artifacts


def run(config: dict) -> dict:
    """Dummy 유지, artifact 준비·백필 또는 기존 URI 공개를 실행합니다."""

    if isinstance(config, Mapping) and _is_dummy(config):
        return _dummy_result()

    try:
        requested = preparation_requested(config)
    except DatasetPreparationError as error:
        return preparation_error_result(str(error), config)
    if requested:
        return run_preparation(config)

    artifacts = _validated_artifacts(config)
    if artifacts is None:
        return _input_error_result()

    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": {"pipeline": "data", "mode": "integration"},
        "message": "사전 제공된 data artifact URI를 검증해 공개했습니다.",
    }
