"""Experiment record 하나를 공통 summary 문서로 요약하는 registry 내부 module입니다.

Summary는 Web을 포함한 모든 소비자가 목록·검색·비교에 쓰는 형식이며, 규격은
`contracts/proposals/002-experiment-index-and-summary.md`에 있습니다.

Evaluate의 `metrics.json`과 train의 `training_history.json` 내부 구조에 의존하는
곳은 이 file 하나뿐입니다. 다른 pipeline의 산출물 형식이 바뀌어도 registry가
통째로 깨지지 않도록, 읽기는 실패해도 예외를 올리지 않고 `metrics_source`와
`losses_source`로만 알립니다. 파일은 읽기만 하며 고치거나 지우지 않습니다.

`s3://` artifact는 `src/common/storage.py`가 준 storage로만 읽습니다. 이 file은
boto3를 직접 쓰지 않습니다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.common import Storage, StorageError

from .record import (
    OPTIONAL_ARTIFACT_KEYS,
    REQUIRED_ARTIFACT_KEYS,
    RegistryError,
    resolve_local_uri,
)


# "1" -> "2": metrics를 s3://에서도 읽고 losses와 losses_source를 더했습니다.
# "2" -> "3": training이 precision·checkpoint_every·gradient_accumulation_steps·
#   input_size와 중첩 설정 셋(augmentation·lr_scheduler·early_stopping)을 함께 담고,
#   per_class_summary가 생겼습니다. 기존 key는 이름도 뜻도 그대로입니다.
SUMMARY_VERSION = "3"

# Evaluate가 metrics.json에 쓰는 이름을 그대로 씁니다. 이름을 한 번 더 번역하면
# 어느 쪽이 진짜인지 알기 어려워집니다.
METRIC_KEYS: tuple[str, ...] = (
    "mAP",
    "mAP50",
    "mAP75",
    "precision50",
    "recall50",
)

# Train이 run() 요약과 training_history.json에 쓰는 이름을 그대로 씁니다.
# `best_*`는 validation loss가 가장 낮은 epoch, `final_*`는 마지막 epoch입니다.
LOSS_KEYS: tuple[str, ...] = (
    "best_epoch",
    "best_validation_loss",
    "final_train_loss",
    "final_validation_loss",
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
    ("precision", "text"),
    ("checkpoint_every", "integer"),
    ("gradient_accumulation_steps", "integer"),
    ("input_size", "integer"),
    # 중첩 object는 모양 그대로 옮깁니다. `augmentation`의 확률이나 `lr_scheduler`의
    # warmup처럼 안쪽 값까지 알아야 무엇으로 돌았는지 말할 수 있고, 화면이 그 설정으로
    # 새 실험을 다시 채우려면 train이 받는 모양 그대로여야 합니다.
    ("augmentation", "object"),
    ("lr_scheduler", "object"),
    ("early_stopping", "object"),
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


def _object_or_none(value: Any) -> dict[str, Any] | None:
    """중첩 설정을 평범한 dict로 옮깁니다. object가 아니면 None입니다."""

    if not isinstance(value, Mapping):
        return None
    return dict(value)


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
        "object": _object_or_none,
    }
    return (
        {key: readers[kind](train.get(key)) for key, kind in TRAINING_KEYS},
        "config_snapshot",
    )


def _is_remote(uri: str) -> bool:
    """원격 artifact를 가리키는 URI인지 확인합니다."""

    return uri.lower().startswith("s3://")


def _read_json_document(
    uri: str,
    *,
    repo_root: Path,
    storage: Storage | None,
) -> Any | None:
    """Artifact JSON 문서를 방어적으로 읽습니다. 못 읽으면 None입니다.

    `s3://`는 `src/common/storage.py`가 준 storage로만 읽습니다. storage가 없거나
    local backend라 원격을 다룰 수 없으면 값이 없는 것으로 봅니다. Local 경로는
    지금까지처럼 저장소 기준 상대 경로 규칙을 확인한 뒤 직접 읽습니다.

    파일이 없거나 권한이 없거나 JSON이 깨져도 예외를 올리지 않습니다. 지표나
    loss를 못 읽었다는 이유로 등록이 실패하면 안 되기 때문입니다.
    """

    try:
        if _is_remote(uri):
            if storage is None:
                return None
            return storage.read_json(uri)
        path = resolve_local_uri(uri, repo_root=repo_root)
        return json.loads(path.read_text(encoding="utf-8"))
    except (
        RegistryError,
        StorageError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None


def read_metrics(
    metrics_uri: str | None,
    *,
    repo_root: Path,
    verify: bool,
    storage: Storage | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Evaluate의 metrics.json에서 지표와 class별 요약을 방어적으로 읽습니다.

    파일이 없거나 JSON이 깨졌거나 키가 없으면 모든 지표를 None으로 두고
    `"unavailable"`을 함께 돌려줍니다. 지표를 못 읽었다는 이유로 실행이 실패하지는
    않습니다. `s3://`는 `storage`가 있을 때만 읽습니다.

    class별 요약은 evaluate가 이미 `analysis.per_class_summary`에 간추려 둔 것을
    그대로 옮깁니다. 여기서 다시 세지 않으므로 화면과 evaluate의 판정이 갈리지
    않고, 전체 class가 아니라 간추린 것이라 index가 커지지도 않습니다. 그 key가
    없던 옛 평가 결과는 None입니다.
    """

    if not metrics_uri or not verify:
        return _empty_metrics(), None, "unavailable"

    document = _read_json_document(metrics_uri, repo_root=repo_root, storage=storage)
    if not isinstance(document, Mapping):
        return _empty_metrics(), None, "unavailable"
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping):
        return _empty_metrics(), None, "unavailable"

    analysis = document.get("analysis")
    per_class = analysis.get("per_class_summary") if isinstance(analysis, Mapping) else None

    return (
        {key: _number_or_none(metrics.get(key)) for key in METRIC_KEYS},
        _object_or_none(per_class),
        "metrics_file",
    )


def _empty_losses() -> dict[str, None]:
    return {key: None for key in LOSS_KEYS}


def _loss_entries(document: Any) -> list[Mapping[str, Any]]:
    """training_history 문서에서 epoch 기록만 골라냅니다."""

    if isinstance(document, (str, bytes)) or not isinstance(document, Sequence):
        return []
    return [entry for entry in document if isinstance(entry, Mapping)]


def read_losses(
    training_history_uri: str | None,
    *,
    repo_root: Path,
    verify: bool,
    storage: Storage | None = None,
) -> tuple[dict[str, Any], str]:
    """Train의 training_history.json에서 loss를 방어적으로 읽습니다.

    `metrics` / `metrics_source`와 같은 짝 구조입니다. 값을 하나라도 읽었을 때만
    `"training_history"`를 돌려주고, 그 밖에는 전부 None과 `"unavailable"`입니다.
    타입이 어긋난 값은 그 값만 None이 되며 0.0 같은 기본값으로 채우지 않습니다.
    측정하지 않은 것과 0은 다르기 때문입니다.
    """

    if not training_history_uri or not verify:
        return _empty_losses(), "unavailable"

    document = _read_json_document(
        training_history_uri, repo_root=repo_root, storage=storage
    )
    entries = _loss_entries(document)
    if not entries:
        return _empty_losses(), "unavailable"

    # validation loss를 읽을 수 있는 epoch 중에서만 best를 고릅니다.
    scored = [
        (loss, entry)
        for entry in entries
        if (loss := _number_or_none(entry.get("validation_loss"))) is not None
    ]
    best = min(scored, key=lambda pair: pair[0])[1] if scored else {}
    final = entries[-1]

    losses = {
        "best_epoch": _integer_or_none(best.get("epoch")),
        "best_validation_loss": _number_or_none(best.get("validation_loss")),
        "final_train_loss": _number_or_none(final.get("train_loss")),
        "final_validation_loss": _number_or_none(final.get("validation_loss")),
    }
    if all(value is None for value in losses.values()):
        return _empty_losses(), "unavailable"
    return losses, "training_history"


def build_summary(
    record: Mapping[str, Any],
    *,
    record_uri: str,
    repo_root: Path,
    verify: bool = True,
    storage: Storage | None = None,
) -> dict[str, Any]:
    """Experiment record 하나를 공통 summary 문서로 만듭니다.

    `storage`를 주면 `s3://` artifact의 지표와 loss도 읽습니다. 팀이 전원 S3를 쓰는
    동안 summary가 계속 비어 있었기 때문입니다. 읽기에 실패해도 예외를 올리지 않고
    해당 값만 None으로 둡니다.
    """

    artifacts = _artifact_uris(record)
    metrics, per_class, metrics_source = read_metrics(
        artifacts.get("metrics_uri"),
        repo_root=repo_root,
        verify=verify,
        storage=storage,
    )
    losses, losses_source = read_losses(
        artifacts.get("training_history_uri"),
        repo_root=repo_root,
        verify=verify,
        storage=storage,
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
        # 평가를 못 읽었거나 그 key가 없던 옛 결과는 None입니다. 빈 목록으로 두면
        # 약한 class가 없다는 뜻이 되어 못 읽은 것과 구별되지 않습니다.
        "per_class_summary": per_class,
        "losses": losses,
        "losses_source": losses_source,
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
