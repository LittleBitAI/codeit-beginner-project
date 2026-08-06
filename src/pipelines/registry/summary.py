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
        "artifacts": artifacts,
        "verification": {
            "artifacts_checked": verification.get("artifacts_checked"),
            "artifacts_hashed": verification.get("artifacts_hashed"),
            "artifacts_skipped_remote": verification.get("artifacts_skipped_remote"),
        },
        "submission_check": record.get("submission_check"),
    }
