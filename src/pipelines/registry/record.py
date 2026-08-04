"""Experiment record를 검증하고 생성하는 registry pipeline 내부 module입니다.

이 module은 registry pipeline 안에서만 사용하며, 다른 pipeline은 import하지
않습니다. 외부에 공개하는 interface는 `src/pipelines/registry/__init__.py`의
`run(config)` 하나입니다.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"

# 각 pipeline이 registry에 넘겨야 하는 artifact key입니다. 값은 모두 str이며,
# `*_uri`로 끝나는 key만 실제 artifact 파일을 가리킵니다.
REQUIRED_ARTIFACT_KEYS: dict[str, tuple[str, ...]] = {
    "data": (
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
        "dataset_summary_uri",
    ),
    "train": (
        "run_id",
        "best_checkpoint_uri",
        "last_checkpoint_uri",
        "training_history_uri",
    ),
    "evaluate": (
        "run_id",
        "metrics_uri",
        "predictions_uri",
    ),
}

# config snapshot에 그대로 남기면 안 되는 값의 key 조각입니다.
_SECRET_KEY_HINTS = (
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "access_key",
    "secret_key",
    "api_key",
    "session",
)
_REDACTED = "***"

_HASH_CHUNK_SIZE = 1024 * 1024


class RegistryError(RuntimeError):
    """Registry pipeline이 실행을 중단해야 하는 경우에 사용하는 예외입니다."""


class MissingInputError(RegistryError):
    """이전 pipeline의 artifact가 없거나 key가 빠진 경우입니다."""


class InvalidSchemaError(RegistryError):
    """Artifact 값의 형식이 계약과 다른 경우입니다."""


class CorruptedArtifactError(RegistryError):
    """Artifact 파일이 없거나 읽을 수 없는 경우입니다."""


def is_remote_uri(uri: str) -> bool:
    """`s3://`로 시작하는 원격 artifact URI인지 확인합니다."""

    return uri.strip().lower().startswith("s3://")


def redact(value: Any) -> Any:
    """Config snapshot에서 credential로 보이는 값을 가립니다."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(hint in key_text.lower() for hint in _SECRET_KEY_HINTS):
                redacted[key_text] = _REDACTED
            else:
                redacted[key_text] = redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def validate_inputs(inputs: Any) -> dict[str, dict[str, str]]:
    """`config["inputs"]`가 artifact 계약을 지키는지 검사합니다.

    입력을 수정하지 않고, 검증된 값만 복사해서 돌려줍니다.
    """

    if not isinstance(inputs, Mapping):
        raise InvalidSchemaError("config['inputs']는 object여야 합니다.")

    validated: dict[str, dict[str, str]] = {}
    for pipeline, required_keys in REQUIRED_ARTIFACT_KEYS.items():
        artifacts = inputs.get(pipeline)
        if artifacts is None:
            raise MissingInputError(
                f"{pipeline} pipeline의 artifact가 없습니다. "
                f"config['inputs']['{pipeline}']에 {', '.join(required_keys)}가 필요합니다."
            )
        if not isinstance(artifacts, Mapping):
            raise InvalidSchemaError(
                f"config['inputs']['{pipeline}']는 object여야 합니다."
            )

        missing = [key for key in required_keys if key not in artifacts]
        if missing:
            raise MissingInputError(
                f"{pipeline} pipeline artifact에 필요한 key가 없습니다: {', '.join(missing)}"
            )

        entry: dict[str, str] = {}
        for key in required_keys:
            value = artifacts[key]
            if not isinstance(value, str):
                raise InvalidSchemaError(
                    f"config['inputs']['{pipeline}']['{key}']는 str이어야 하는데 "
                    f"{type(value).__name__}을(를) 받았습니다."
                )
            if not value.strip():
                raise InvalidSchemaError(
                    f"config['inputs']['{pipeline}']['{key}']는 비어 있지 않은 문자열이어야 합니다."
                )
            entry[key] = value.strip()
        validated[pipeline] = entry

    return validated


def resolve_run_id(
    validated_inputs: Mapping[str, Mapping[str, str]],
    *,
    configured_run_id: str | None = None,
) -> str:
    """Experiment record의 run_id를 결정합니다.

    설정값이 있으면 그대로 쓰고, 없으면 train과 evaluate의 run_id를 사용합니다.
    둘이 다르면 서로 다른 실험을 묶는 것이므로 오류로 처리합니다. run_id를
    새로 만들어 내지 않으므로 이 pipeline에는 random 동작이 없습니다.
    """

    if configured_run_id is not None:
        if not isinstance(configured_run_id, str) or not configured_run_id.strip():
            raise InvalidSchemaError(
                "config['registry']['run_id']는 비어 있지 않은 문자열이어야 합니다."
            )
        return configured_run_id.strip()

    train_run_id = validated_inputs["train"]["run_id"]
    evaluate_run_id = validated_inputs["evaluate"]["run_id"]
    if train_run_id != evaluate_run_id:
        raise InvalidSchemaError(
            "train과 evaluate의 run_id가 서로 다릅니다: "
            f"train={train_run_id}, evaluate={evaluate_run_id}. "
            "config['registry']['run_id']로 기록할 run_id를 직접 지정하세요."
        )
    return train_run_id


def _hash_local_file(path: Path) -> tuple[str, int]:
    """Local artifact file의 sha256과 byte 크기를 계산합니다."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def describe_artifact(uri: str, *, repo_root: Path, verify: bool = True) -> dict[str, Any]:
    """Artifact 하나의 위치와 무결성 정보를 만듭니다.

    local artifact는 저장소 root 기준 상대 경로로 보고 sha256을 계산합니다.
    `s3://` artifact는 AWS 접근 없이 기록만 하고 검증은 건너뜁니다.
    """

    if is_remote_uri(uri):
        return {
            "uri": uri,
            "location": "s3",
            "sha256": None,
            "size_bytes": None,
            "verified": False,
            "note": "원격 artifact는 AWS 접근 없이 참조만 기록했습니다.",
        }

    entry: dict[str, Any] = {
        "uri": uri,
        "location": "local",
        "sha256": None,
        "size_bytes": None,
        "verified": False,
    }
    if not verify:
        entry["note"] = "registry.verify_artifacts가 false여서 검증을 건너뛰었습니다."
        return entry

    candidate = Path(uri)
    if candidate.is_absolute():
        raise InvalidSchemaError(
            f"local artifact는 저장소 기준 상대 경로여야 합니다: {uri}"
        )

    path = (repo_root / candidate).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise InvalidSchemaError(
            f"local artifact 경로가 저장소 밖을 가리킵니다: {uri}"
        ) from error

    if not path.is_file():
        raise CorruptedArtifactError(f"local artifact file을 찾을 수 없습니다: {uri}")

    try:
        checksum, size = _hash_local_file(path)
    except OSError as error:
        raise CorruptedArtifactError(
            f"local artifact file을 읽지 못했습니다: {uri}"
        ) from error

    entry["sha256"] = checksum
    entry["size_bytes"] = size
    entry["verified"] = True
    return entry


def build_record(
    config: Mapping[str, Any],
    validated_inputs: Mapping[str, Mapping[str, str]],
    *,
    run_id: str,
    seed: int,
    repo_root: Path,
    verify: bool = True,
    created_at: str | None = None,
) -> dict[str, Any]:
    """재현에 필요한 정보를 모은 experiment record를 만듭니다."""

    pipelines: dict[str, dict[str, Any]] = {}
    checked = 0
    hashed = 0
    skipped_remote = 0

    for pipeline, artifacts in validated_inputs.items():
        entries: dict[str, Any] = {}
        for key, value in artifacts.items():
            if not key.endswith("_uri"):
                entries[key] = value
                continue
            described = describe_artifact(value, repo_root=repo_root, verify=verify)
            checked += 1
            if described["verified"]:
                hashed += 1
            elif described["location"] == "s3":
                skipped_remote += 1
            entries[key] = described
        pipelines[pipeline] = entries

    snapshot = {
        key: redact(copy.deepcopy(value))
        for key, value in config.items()
        if key != "inputs"
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config_snapshot": snapshot,
        "pipelines": pipelines,
        "verification": {
            "artifacts_checked": checked,
            "artifacts_hashed": hashed,
            "artifacts_skipped_remote": skipped_remote,
            "hash_algorithm": "sha256",
        },
    }
