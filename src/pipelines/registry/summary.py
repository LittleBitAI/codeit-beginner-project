"""Experiment record 하나를 공통 summary 문서로 요약하는 registry 내부 module입니다.

Summary는 Web을 포함한 모든 소비자가 목록·검색·비교에 쓰는 형식이며, 규격은
`contracts/proposals/002-experiment-index-and-summary.md`에 있습니다.

Evaluate의 `metrics.json` 내부 구조에 의존하는 곳은 이 file 하나뿐입니다. 다른
pipeline의 산출물 형식이 바뀌어도 registry가 통째로 깨지지 않도록, 지표 읽기는
실패해도 예외를 올리지 않고 `metrics_source`로만 알립니다. 파일은 읽기만 하며
고치거나 지우지 않습니다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .record import (
    OPTIONAL_ARTIFACT_KEYS,
    REQUIRED_ARTIFACT_KEYS,
    RegistryError,
    resolve_local_uri,
)


SUMMARY_VERSION = "1"

# Evaluate가 metrics.json에 쓰는 이름을 그대로 씁니다. 이름을 한 번 더 번역하면
# 어느 쪽이 진짜인지 알기 어려워집니다.
METRIC_KEYS: tuple[str, ...] = (
    "mAP",
    "mAP50",
    "mAP75",
    "precision50",
    "recall50",
)

# Train이 config에 쓰는 이름을 그대로 씁니다. 값 종류마다 통과시키는 타입이 달라서
# key 이름과 함께 적어 둡니다. run_id, seed, output_dir, output_prefix는 넣지
# 않습니다. seed는 summary 최상위에 이미 있고, 나머지 둘은 경로라 기록에 남기지
# 않습니다.
TRAINING_KEYS: tuple[tuple[str, str], ...] = (
    ("architecture", "text"),
    ("pretrained", "bool"),
    ("optimizer", "text"),
    ("learning_rate", "number"),
    ("momentum", "number"),
    ("weight_decay", "number"),
    ("beta1", "number"),
    ("beta2", "number"),
    ("epsilon", "number"),
    ("device", "text"),
    ("epochs", "integer"),
    ("batch_size", "integer"),
    ("num_workers", "integer"),
)


def artifact_key_names() -> tuple[str, ...]:
    """선언된 필수·선택 artifact 중 실제 파일을 가리키는 key 이름입니다."""

    names: list[str] = []
    for pipeline, required in REQUIRED_ARTIFACT_KEYS.items():
        names.extend(key for key in required if key.endswith("_uri"))
        names.extend(
            key
            for key in OPTIONAL_ARTIFACT_KEYS.get(pipeline, ())
            if key.endswith("_uri")
        )
    return tuple(names)


def _artifact_uris(record: Mapping[str, Any]) -> dict[str, str | None]:
    """record의 artifact를 평면 map으로 폅니다. 없는 선택 artifact는 null입니다."""

    pipelines = record.get("pipelines")
    entries: dict[str, str | None] = {name: None for name in artifact_key_names()}
    if not isinstance(pipelines, Mapping):
        return entries

    for artifacts in pipelines.values():
        if not isinstance(artifacts, Mapping):
            continue
        for key, value in artifacts.items():
            if key not in entries:
                continue
            if isinstance(value, Mapping):
                uri = value.get("uri")
                entries[key] = uri if isinstance(uri, str) else None
    return entries


def _empty_metrics() -> dict[str, None]:
    return {key: None for key in METRIC_KEYS}


def _number_or_none(value: Any) -> float | int | None:
    """JSON으로 다시 쓸 수 있는 숫자만 통과시킵니다."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # NaN과 Infinity는 표준 JSON이 아니라 그대로 두면 저장 시점에 깨집니다.
    if isinstance(value, float) and value != value:
        return None
    if value in (float("inf"), float("-inf")):
        return None
    return value


def _text_or_none(value: Any) -> str | None:
    """문자열만 통과시킵니다. 빈 문자열은 값이 없는 것과 같게 둡니다."""

    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _bool_or_none(value: Any) -> bool | None:
    """참·거짓만 통과시킵니다. 0이나 "true" 같은 값은 통과시키지 않습니다."""

    return value if isinstance(value, bool) else None


def _integer_or_none(value: Any) -> int | None:
    """정수만 통과시킵니다. bool은 파이썬에서 정수지만 여기서는 정수가 아닙니다."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _empty_training() -> dict[str, None]:
    return {key: None for key, _ in TRAINING_KEYS}


def read_training(record: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """record의 `config_snapshot.train`에서 학습 설정을 방어적으로 읽습니다.

    `config_snapshot.train`이 Mapping일 때만 `"config_snapshot"`을 함께 돌려줍니다.
    그 밖에는 모든 값을 None으로 두고 `"unavailable"`을 돌려줍니다. 타입이 어긋난
    값도 예외 없이 None이 되며, 기본값으로 채우지 않습니다. 기록에 없는 것과
    기본값을 쓴 것은 다르기 때문입니다.

    파일을 읽지 않으므로 `verify`와 무관합니다. `config_snapshot`은 record를 만들
    때 이미 redact를 거친 값입니다.
    """

    snapshot = record.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        return _empty_training(), "unavailable"
    train = snapshot.get("train")
    if not isinstance(train, Mapping):
        return _empty_training(), "unavailable"

    readers = {
        "text": _text_or_none,
        "bool": _bool_or_none,
        "integer": _integer_or_none,
        "number": _number_or_none,
    }
    return (
        {key: readers[kind](train.get(key)) for key, kind in TRAINING_KEYS},
        "config_snapshot",
    )


def read_metrics(
    metrics_uri: str | None,
    *,
    repo_root: Path,
    verify: bool,
) -> tuple[dict[str, Any], str]:
    """Evaluate의 metrics.json에서 지표를 방어적으로 읽습니다.

    파일이 없거나 원격이거나 JSON이 깨졌거나 키가 없으면 모든 지표를 None으로 두고
    `"unavailable"`을 함께 돌려줍니다. 지표를 못 읽었다는 이유로 실행이 실패하지는
    않습니다.
    """

    if not metrics_uri or not verify:
        return _empty_metrics(), "unavailable"

    try:
        path = resolve_local_uri(metrics_uri, repo_root=repo_root)
        document = json.loads(path.read_text(encoding="utf-8"))
    except (RegistryError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _empty_metrics(), "unavailable"

    if not isinstance(document, Mapping):
        return _empty_metrics(), "unavailable"
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping):
        return _empty_metrics(), "unavailable"

    return (
        {key: _number_or_none(metrics.get(key)) for key in METRIC_KEYS},
        "metrics_file",
    )


def build_summary(
    record: Mapping[str, Any],
    *,
    record_uri: str,
    repo_root: Path,
    verify: bool = True,
) -> dict[str, Any]:
    """Experiment record 하나를 공통 summary 문서로 만듭니다."""

    artifacts = _artifact_uris(record)
    metrics_uri = artifacts.get("metrics_uri")
    # 원격 artifact는 AWS 접근 없이 참조만 기록한다는 기존 정책을 그대로 따릅니다.
    is_local_metrics = isinstance(metrics_uri, str) and not metrics_uri.lower().startswith(
        "s3://"
    )
    metrics, metrics_source = read_metrics(
        metrics_uri if is_local_metrics else None,
        repo_root=repo_root,
        verify=verify,
    )

    training, training_source = read_training(record)

    verification = record.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}

    return {
        "summary_version": SUMMARY_VERSION,
        "run_id": record.get("run_id"),
        "created_at": record.get("created_at"),
        "seed": record.get("seed"),
        "schema_version": record.get("schema_version"),
        "experiment_record_uri": record_uri,
        "metrics": metrics,
        "metrics_source": metrics_source,
        "training": training,
        "training_source": training_source,
        "artifacts": artifacts,
        "verification": {
            "artifacts_checked": verification.get("artifacts_checked"),
            "artifacts_hashed": verification.get("artifacts_hashed"),
            "artifacts_skipped_remote": verification.get("artifacts_skipped_remote"),
        },
        "submission_check": record.get("submission_check"),
    }
