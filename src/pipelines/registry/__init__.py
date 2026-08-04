"""Registry pipeline: data, train, evaluate artifact로 experiment record를 만듭니다.

외부에 공개하는 interface는 `run(config)` 하나입니다. 자기 설정은
`config["registry"]`에서, 이전 pipeline 결과는 `config["inputs"]`에서 읽으며
`config["inputs"]`는 수정하지 않습니다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.common import StorageError, create_storage

from .record import (
    REQUIRED_ARTIFACT_KEYS,
    RegistryError,
    build_record,
    resolve_run_id,
    validate_inputs,
)


__all__ = ["run"]

# src/pipelines/registry/__init__.py 기준으로 저장소 root를 찾습니다.
# 개인 PC 절대 경로를 코드에 넣지 않기 위한 방식입니다.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_RECORD_NAME = "experiment_record.json"
_DEFAULT_RECORD_PREFIX = "registry"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """설정 항목이 object인지 확인합니다."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RegistryError(f"{name} 설정은 object여야 합니다.")
    return value


def _is_dummy_run(inputs: Mapping[str, Any]) -> bool:
    """이전 pipeline artifact가 하나도 오지 않은 dummy 실행인지 확인합니다."""

    return not any(inputs.get(pipeline) for pipeline in REQUIRED_ARTIFACT_KEYS)


def _resolve_seed(registry_config: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    """Random 동작에 사용할 seed를 결정합니다. 기본값은 0입니다."""

    raw_seed = registry_config.get("seed", config.get("seed", 0))
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise RegistryError("seed는 정수여야 합니다.")
    return raw_seed


def _resolve_repo_root(registry_config: Mapping[str, Any]) -> Path:
    """Local artifact를 찾을 기준 경로를 결정합니다."""

    configured = registry_config.get("repo_root")
    if configured is None:
        return _REPO_ROOT
    if not isinstance(configured, str) or not configured.strip():
        raise RegistryError(
            "config['registry']['repo_root']는 비어 있지 않은 문자열이어야 합니다."
        )
    return Path(configured).expanduser().resolve()


def _relative_to_repo(location: str, repo_root: Path) -> str:
    """Local 저장 경로를 저장소 기준 상대 경로로 바꿉니다."""

    if location.lower().startswith("s3://"):
        return location
    try:
        return Path(location).resolve().relative_to(repo_root).as_posix()
    except ValueError:
        # 저장소 밖에 쓰도록 설정한 경우에는 storage가 준 경로를 그대로 씁니다.
        return Path(location).as_posix()


def run(config: dict) -> dict:
    """Experiment record를 만들고 저장합니다.

    이전 pipeline artifact가 하나도 없으면 기록할 실험이 없으므로 dummy 실행으로
    보고 통과시킵니다. artifact가 하나라도 들어오면 계약 schema를 엄격하게
    검증하고, 누락이나 손상이 있으면 status="error"를 반환합니다.
    """

    try:
        root_config = _mapping(config, "config")
        registry_config = _mapping(root_config.get("registry"), "config['registry']")
        inputs = _mapping(root_config.get("inputs"), "config['inputs']")

        if _is_dummy_run(inputs):
            return {
                "status": "ok",
                "artifacts": {},
                "summary": {"pipeline": "registry", "mode": "dummy"},
                "message": (
                    "이전 pipeline artifact가 없어 experiment record를 만들지 않았습니다."
                ),
            }

        seed = _resolve_seed(registry_config, root_config)
        repo_root = _resolve_repo_root(registry_config)
        verify = registry_config.get("verify_artifacts", True)
        if not isinstance(verify, bool):
            raise RegistryError(
                "config['registry']['verify_artifacts']는 true 또는 false여야 합니다."
            )
        overwrite = registry_config.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise RegistryError(
                "config['registry']['overwrite']는 true 또는 false여야 합니다."
            )

        validated_inputs = validate_inputs(inputs)
        run_id = resolve_run_id(
            validated_inputs,
            configured_run_id=registry_config.get("run_id"),
        )
        record = build_record(
            root_config,
            validated_inputs,
            run_id=run_id,
            seed=seed,
            repo_root=repo_root,
            verify=verify,
            created_at=registry_config.get("created_at"),
        )

        destination = registry_config.get("record_uri")
        if destination is None:
            destination = f"{_DEFAULT_RECORD_PREFIX}/{run_id}/{_DEFAULT_RECORD_NAME}"
        elif not isinstance(destination, str) or not destination.strip():
            raise RegistryError(
                "config['registry']['record_uri']는 비어 있지 않은 문자열이어야 합니다."
            )

        storage = create_storage(root_config)
        written = storage.write_json(destination, record, overwrite=overwrite)
        record_uri = _relative_to_repo(written, repo_root)

        verification = record["verification"]
        return {
            "status": "ok",
            "artifacts": {
                "run_id": run_id,
                "experiment_record_uri": record_uri,
            },
            "summary": {
                "pipeline": "registry",
                "run_id": run_id,
                "seed": seed,
                "schema_version": record["schema_version"],
                "created_at": record["created_at"],
                "artifacts_checked": verification["artifacts_checked"],
                "artifacts_hashed": verification["artifacts_hashed"],
                "artifacts_skipped_remote": verification["artifacts_skipped_remote"],
            },
            "message": f"experiment record를 저장했습니다: {record_uri}",
        }
    except RegistryError as error:
        return {
            "status": "error",
            "artifacts": {},
            "summary": {"pipeline": "registry", "error": type(error).__name__},
            "message": str(error),
        }
    except StorageError as error:
        return {
            "status": "error",
            "artifacts": {},
            "summary": {"pipeline": "registry", "error": type(error).__name__},
            "message": f"experiment record를 저장하지 못했습니다: {error}",
        }
