"""Exact URI로 experiment record 하나를 조회하는 공용 facade."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .storage import LocalStorage, StorageError, create_storage


__all__ = ["ExperimentRegistryError", "read_experiment_record"]


class ExperimentRegistryError(RuntimeError):
    """Experiment record 조회 또는 최소 schema 검증이 실패한 경우입니다."""


def _local_read_source(uri: str, storage: LocalStorage) -> str:
    """Registry의 repo-relative URI를 LocalStorage root 기준으로 맞춥니다.

    Registry는 local 저장 결과를 repository 기준 상대 경로로 반환할 수 있습니다.
    예를 들어 storage root가 ``<repo>/artifacts``이면 반환 URI는
    ``artifacts/registry/...``입니다. URI 선두와 root 끝의 겹치는 경로만
    제거해 LocalStorage가 ``artifacts/artifacts``로 중복 해석하지 않게 합니다.
    """

    candidate = Path(uri)
    if candidate.is_absolute():
        return uri

    uri_parts = candidate.parts
    root_parts = storage.root.parts
    maximum = min(len(uri_parts), len(root_parts))
    for size in range(maximum, 0, -1):
        uri_prefix = tuple(part.casefold() for part in uri_parts[:size])
        root_suffix = tuple(part.casefold() for part in root_parts[-size:])
        if uri_prefix != root_suffix:
            continue
        remainder = uri_parts[size:]
        if remainder:
            return Path(*remainder).as_posix()
    return uri


def read_experiment_record(
    experiment_record_uri: str,
    config: Mapping[str, Any] | None,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """정확한 URI의 experiment record를 읽고 최소 identity를 검증합니다.

    Prefix listing이나 최신 record 탐색은 하지 않습니다. Storage 오류와 record
    schema 오류는 호출자가 backend와 무관하게 처리할 수 있도록
    :class:`ExperimentRegistryError`로 통일합니다.
    """

    if not isinstance(experiment_record_uri, str) or not experiment_record_uri.strip():
        raise ExperimentRegistryError(
            "experiment_record_uri는 비어 있지 않은 문자열이어야 합니다."
        )
    uri = experiment_record_uri.strip()

    if expected_run_id is not None and (
        not isinstance(expected_run_id, str) or not expected_run_id.strip()
    ):
        raise ExperimentRegistryError(
            "expected_run_id는 비어 있지 않은 문자열이어야 합니다."
        )
    expected = expected_run_id.strip() if isinstance(expected_run_id, str) else None

    try:
        storage = create_storage(config)
        source = (
            _local_read_source(uri, storage)
            if isinstance(storage, LocalStorage)
            else uri
        )
        record = storage.read_json(source)
    except StorageError as error:
        raise ExperimentRegistryError(
            f"experiment record를 읽지 못했습니다 ({uri}): {error}"
        ) from error

    if not isinstance(record, dict):
        raise ExperimentRegistryError(
            "experiment record는 object(dict)여야 하지만 "
            f"{type(record).__name__}을(를) 받았습니다."
        )

    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ExperimentRegistryError(
            "experiment record의 run_id는 비어 있지 않은 문자열이어야 합니다."
        )
    resolved_run_id = run_id.strip()
    if expected is not None and resolved_run_id != expected:
        raise ExperimentRegistryError(
            "experiment record의 run_id가 요청과 다릅니다: "
            f"expected={expected}, actual={resolved_run_id}"
        )

    return record
