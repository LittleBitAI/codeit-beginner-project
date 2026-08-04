"""Exact URI로 experiment record 하나를 조회하는 공용 facade."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .storage import LocalStorage, StorageError, create_storage


__all__ = ["ExperimentRegistryError", "read_experiment_record"]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ExperimentRegistryError(RuntimeError):
    """Experiment record 조회 또는 최소 schema 검증이 실패한 경우입니다."""


def _repo_root(config: Mapping[str, Any] | None) -> Path:
    """Registry와 같은 config 규칙으로 repository root를 결정합니다."""

    registry_config = config.get("registry") if isinstance(config, Mapping) else None
    if registry_config is None:
        return _REPOSITORY_ROOT
    if not isinstance(registry_config, Mapping):
        raise ExperimentRegistryError("config['registry']는 object여야 합니다.")

    configured = registry_config.get("repo_root")
    if configured is None:
        return _REPOSITORY_ROOT
    if not isinstance(configured, str) or not configured.strip():
        raise ExperimentRegistryError(
            "config['registry']['repo_root']는 비어 있지 않은 문자열이어야 합니다."
        )
    return Path(configured).expanduser().resolve()


def _local_read_source(uri: str, storage: LocalStorage, repo_root: Path) -> str:
    """Registry의 repo-relative URI를 LocalStorage root 기준으로 맞춥니다.

    이름이 같은 path segment를 추측해 제거하지 않습니다. URI를 registry의
    repository root 기준 절대 후보로 만든 뒤, 그 후보가 실제 LocalStorage root
    안에 있을 때만 storage 기준 상대 경로로 바꿉니다.
    """

    candidate = Path(uri)
    if candidate.is_absolute():
        return uri

    absolute_candidate = (repo_root / candidate).resolve()
    try:
        return absolute_candidate.relative_to(storage.root).as_posix()
    except ValueError:
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
            _local_read_source(uri, storage, _repo_root(config))
            if isinstance(storage, LocalStorage)
            else uri
        )
        record = storage.read_json(source)
    except StorageError as error:
        raise ExperimentRegistryError(
            "experiment record storage 조회에 실패했습니다 "
            f"({type(error).__name__})."
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
