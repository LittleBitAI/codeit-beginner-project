"""Registry summary 목록과 선택한 experiment record를 Web 표현으로 바꿉니다."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.common import (
    ExperimentRegistryError,
    list_experiment_summaries,
    read_experiment_record,
)

from . import experiment_detail, kaggle_scores, train_capabilities
from .datasets import storage_environment
from .errors import FieldError, JobNotFoundError, WebError, WebValidationError
from .masking import redact
from .paths import repository_root
from .train_config import DATA_ARTIFACT_KEYS, OPTIMIZER_PROFILES


__all__ = [
    "compare_registry_experiments",
    "list_registry_experiments",
    "read_registry_experiment",
    "registry_config",
    "registry_scope",
    "save_kaggle_score",
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


def _dataset_folder(artifacts: Any) -> str | None:
    """학습 manifest가 든 폴더 이름입니다. 표에 URI 대신 이것을 보여 줍니다.

    ``s3://bucket/datasets/pill_detection/processed/v3-seed42-8020-group/train_manifest.json``
    에서 ``v3-seed42-8020-group``을 얻습니다. 전체 URI는 100자가 넘어 표에 넣을 수
    없고, 팀이 데이터셋을 구별할 때 실제로 부르는 이름이 이 폴더 이름입니다.
    """

    if not isinstance(artifacts, Mapping):
        return None
    uri = _text(artifacts.get("train_manifest_uri"))
    if uri is None:
        return None
    parts = [part for part in uri.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        return None
    folder = parts[-2]
    # `s3://bucket/train_manifest.json`이면 앞이 scheme입니다. 그것은 이름이 아닙니다.
    return None if folder.endswith(":") else folder


def _dataset_from_artifacts(artifacts: Any) -> dict[str, Any]:
    label = _dataset_folder(artifacts)
    if not isinstance(artifacts, Mapping):
        return {
            "identity": None,
            "identity_source": "unknown",
            "artifacts_complete": False,
            "label": label,
        }
    selected: dict[str, str] = {}
    for key in DATA_ARTIFACT_KEYS:
        value = _text(artifacts.get(key))
        if value is None:
            return {
                "identity": None,
                "identity_source": "unknown",
                "artifacts_complete": False,
                "label": label,
            }
        selected[key] = value.replace("\\", "/")
    return {
        "identity": _fingerprint(selected),
        "identity_source": "artifact_set",
        "artifacts_complete": True,
        "label": label,
    }


def _completion_block(
    summary: Mapping[str, Any], kaggle_score: float | None
) -> dict[str, Any]:
    """평가, CSV 생성, 실제 Kaggle 제출을 서로 다른 사실로 둡니다.

    Registry의 ``submission_uri``는 CSV를 만들었다는 증거일 뿐 Kaggle에 올렸다는
    증거가 아닙니다. 사람이 실제 점수를 입력했을 때만 ``submitted``가 됩니다.
    """

    metrics = summary.get("metrics")
    metric_values = metrics if isinstance(metrics, Mapping) else {}
    artifacts = summary.get("artifacts")
    artifact_values = artifacts if isinstance(artifacts, Mapping) else {}
    check = summary.get("submission_check")
    check_values = check if isinstance(check, Mapping) else {}
    return {
        "evaluated": _number(metric_values.get("mAP")) is not None,
        "submission_generated": _text(artifact_values.get("submission_uri")) is not None,
        "submitted": kaggle_score is not None,
        "submission_checked": check_values.get("checked") is True,
        "submission_rows": _integer(check_values.get("row_count")),
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


def _training_blocks(
    settings: Mapping[str, Any], fallback_seed: int | None
) -> dict[str, Any]:
    """평면 학습 설정 하나를 화면이 쓰는 model/optimizer/training 세 블록으로 나눕니다.

    목록(index summary의 ``training``)과 비교(record의 ``config_snapshot.train``)가
    같은 실험에 다른 값을 보이지 않도록 두 경로가 이 함수 하나만 씁니다. 값이 없으면
    호환 기본값을 채우고 ``source``를 ``legacy_fallback``으로 표시합니다.
    ``seed``는 이 설정에 없으면 ``fallback_seed``(summary 최상위 seed)를 씁니다.
    """

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
    return {
        "model": {
            "architecture": architecture,
            "pretrained": _boolean(settings.get("pretrained")),
            "source": "record" if recorded_architecture else "legacy_fallback",
        },
        "optimizer": {
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
        },
        "training": {
            "device": _text(settings.get("device")),
            "epochs": _integer(settings.get("epochs")),
            "batch_size": _integer(settings.get("batch_size")),
            "num_workers": _integer(settings.get("num_workers")),
            # 이 두 값을 모르던 옛 기록은 None이라 화면에 -로 남습니다. 1이나 640으로
            # 지어내면 그때 무엇으로 돌았는지 모르는 채 숫자만 그럴듯해집니다.
            "gradient_accumulation_steps": _integer(
                settings.get("gradient_accumulation_steps")
            ),
            "input_size": _integer(settings.get("input_size")),
            "seed": seed if seed is not None else fallback_seed,
        },
    }


def _unknown_blocks(fallback_seed: int | None) -> dict[str, Any]:
    """학습 설정을 아예 모를 때의 세 블록입니다.

    ``training`` key가 없는 옛 index는 registry가 판단한 적이 없는 기록이므로
    호환 기본값을 지어내지 않고 전부 ``None``으로 두어 화면에 ``-``가 나오게 합니다.
    """

    return {
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
            "gradient_accumulation_steps": None,
            "input_size": None,
            "seed": fallback_seed,
        },
    }


def _index_blocks(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Index summary 하나를 세 블록으로 바꿉니다.

    ``training`` key가 있으면 ``training_source``가 ``config_snapshot``이든
    ``unavailable``이든 registry가 record를 보고 판단한 결과이므로 비교 경로와 같은
    helper를 태웁니다. key 자체가 없으면 이 기능 이전의 옛 index라 값을 모릅니다.
    """

    fallback_seed = _integer(summary.get("seed"))
    if "training" not in summary:
        return _unknown_blocks(fallback_seed)
    training = summary.get("training")
    settings = training if isinstance(training, Mapping) else {}
    return _training_blocks(settings, fallback_seed)


def _metrics_block(
    summary: Mapping[str, Any], kaggle_score: float | None
) -> dict[str, Any]:
    """Index summary 하나에서 화면이 비교할 결과값을 모두 꺼냅니다.

    지표는 registry가 evaluate의 이름(``mAP``, ``mAP50``, ...)으로 적어 두므로 화면이
    이미 쓰는 소문자 이름으로만 바꿔 담습니다. loss 4개는 registry가 나중에 붙인
    ``losses`` 블록에 있고, **그 블록이 없는 옛 summary도 그대로 읽힙니다.** 그때는
    호환 기본값을 지어내지 않고 전부 ``None``으로 두어 화면에 ``-``가 나오게 합니다.
    목록과 비교가 같은 실험에 다른 결과를 보이지 않도록 두 경로가 이 함수만 씁니다.
    """

    metrics = summary.get("metrics")
    metric_values = metrics if isinstance(metrics, Mapping) else {}
    losses = summary.get("losses")
    loss_values = losses if isinstance(losses, Mapping) else {}
    return {
        "best_epoch": _integer(loss_values.get("best_epoch")),
        "best_validation_loss": _number(loss_values.get("best_validation_loss")),
        "final_train_loss": _number(loss_values.get("final_train_loss")),
        "final_validation_loss": _number(loss_values.get("final_validation_loss")),
        "map": _number(metric_values.get("mAP")),
        "map50": _number(metric_values.get("mAP50")),
        "map75": _number(metric_values.get("mAP75")),
        "precision50": _number(metric_values.get("precision50")),
        "recall50": _number(metric_values.get("recall50")),
        "kaggle_score": kaggle_score,
    }


def _summary_base(
    summary: Mapping[str, Any], kaggle_score: float | None = None
) -> dict[str, Any]:
    run_id = _text(summary.get("run_id")) or ""
    created_at = _text(summary.get("created_at")) or ""
    blocks = _index_blocks(summary)
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
        "model": blocks["model"],
        "optimizer": blocks["optimizer"],
        "training": blocks["training"],
        "metrics": _metrics_block(summary, kaggle_score),
        "completion": _completion_block(summary, kaggle_score),
    }


def _record_settings(record: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshot = record.get("config_snapshot")
    train = snapshot.get("train") if isinstance(snapshot, Mapping) else None
    if not isinstance(train, Mapping):
        return {}
    return train


def _enrich_summary(
    summary: Mapping[str, Any],
    record: Mapping[str, Any],
    kaggle_score: float | None = None,
) -> dict[str, Any]:
    value = _summary_base(summary, kaggle_score)
    # record가 진실이므로 목록이 index로 채운 세 블록을 record 값으로 다시 계산합니다.
    value.update(
        _training_blocks(_record_settings(record), _integer(summary.get("seed")))
    )
    return _sanitized(value)


def _sanitized(value: dict[str, Any]) -> dict[str, Any]:
    """응답으로 나가기 전 credential처럼 보이는 값을 가립니다.

    목록과 비교가 같은 검사를 거쳐야 나중에 경로성 field가 늘어도 한쪽만 새지 않습니다.
    """

    sanitized = redact(value)
    return sanitized if isinstance(sanitized, dict) else value


def registry_scope() -> dict[str, Any]:
    """이 목록에 팀원의 실험도 들어오는지 알려 줍니다.

    Registry index는 storage backend를 그대로 따릅니다. S3 bucket이 설정돼 있으면
    팀이 공유하는 index를 읽으므로 팀원의 실험이 함께 나오고, local이면 이
    컴퓨터에서 등록한 것만 나옵니다. 화면이 "팀원 것까지 보인다"고 말하려면 그
    구분을 알아야 하는데, 목록만 봐서는 구별할 방법이 없습니다.
    """

    backend = storage_environment()["default_backend"]
    return {"backend": backend, "shared": backend == "s3"}


def list_registry_experiments() -> list[dict[str, Any]]:
    try:
        summaries = list_experiment_summaries(registry_config())
        if not summaries:
            return []
        scores = kaggle_scores.load_scores()
        return [
            _sanitized(_summary_base(item, scores.get(str(item.get("run_id", "")))))
            for item in summaries
        ]
    except ExperimentRegistryError as error:
        raise WebError(f"실험 목록을 읽지 못했습니다({type(error).__name__}).") from error


# Record 하나마다 storage 왕복이 한 번씩 붙습니다. 순서대로 읽으면 고른 실험 수에
# 비례해 기다리게 되므로 함께 읽습니다. 화면에서 한 번에 고르는 수가 많지 않아
# 상한은 낮게 둡니다.
_COMPARE_READ_WORKERS = 8


def _read_records(
    targets: list[tuple[str, str]], config: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """(run_id, record) 목록을 **요청한 순서 그대로** 돌려줍니다.

    읽기는 함께 하고 결과만 제출 순서로 모읍니다. 읽다가 실패하면 그 예외가
    그대로 올라와 호출자가 지금처럼 처리합니다.
    """

    if not targets:
        return []
    workers = max(1, min(_COMPARE_READ_WORKERS, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(read_experiment_record, uri, config, expected_run_id=run_id)
            for run_id, uri in targets
        ]
        return [(targets[index][0], future.result()) for index, future in enumerate(futures)]


def _artifact_uri(record: Mapping[str, Any], pipeline: str, key: str) -> str | None:
    """record의 `pipelines.<pipeline>.<key>.uri`를 방어적으로 꺼냅니다."""

    pipelines = record.get("pipelines")
    if not isinstance(pipelines, Mapping):
        return None
    stage = pipelines.get(pipeline)
    if not isinstance(stage, Mapping):
        return None
    entry = stage.get(key)
    if isinstance(entry, Mapping):
        return _text(entry.get("uri"))
    return _text(entry)


def read_registry_experiment(run_id: str) -> dict[str, Any]:
    """실험 하나의 설정과 평가 결과를 상세 화면이 쓸 모양으로 모읍니다.

    목록에 있는 값만으로는 "이 실험이 얼마나 잘 나왔나"를 판단할 수 없습니다.
    지표 9개 중 5개만 index에 있고 loss 곡선은 아예 없어서, record가 가리키는
    artifact를 직접 읽습니다. 평가나 학습 기록을 못 읽어도 설정은 보여 줍니다.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise WebValidationError([FieldError("run_id", "실험 이름이 필요합니다.")])

    wanted = run_id.strip()
    config = registry_config()
    try:
        summary = next(
            (
                item
                for item in list_experiment_summaries(config)
                if item.get("run_id") == wanted
            ),
            None,
        )
        if summary is None:
            raise JobNotFoundError(f"'{wanted}' 실험을 registry에서 찾을 수 없습니다.")

        uri = _text(summary.get("experiment_record_uri"))
        if uri is None:
            raise JobNotFoundError(f"'{wanted}' 실험 기록의 위치가 index에 없습니다.")
        record = read_experiment_record(uri, config, expected_run_id=wanted)
    except ExperimentRegistryError as error:
        raise WebError(f"실험 기록을 읽지 못했습니다({type(error).__name__}).") from error

    storage_config = config["storage"]
    score = kaggle_scores.load_scores().get(wanted)
    return {
        "experiment": _enrich_summary(summary, record, score),
        "evaluation": _sanitized(
            experiment_detail.evaluation_block(
                _artifact_uri(record, "evaluate", "metrics_uri"), storage_config
            )
        ),
        "history": _sanitized(
            experiment_detail.history_block(
                _artifact_uri(record, "train", "training_history_uri"), storage_config
            )
        ),
    }


def compare_registry_experiments(run_ids: list[str]) -> dict[str, Any]:
    if not run_ids or not all(
        isinstance(run_id, str) and run_id.strip() for run_id in run_ids
    ):
        raise WebValidationError(
            [FieldError("run_ids", "비어 있지 않은 run_id 목록이 필요합니다.")]
        )
    config = registry_config()
    requested = [run_id.strip() for run_id in run_ids]
    try:
        # Index는 한 번만 읽습니다. `compare_experiment_summaries`도 안에서 같은
        # 목록을 읽으므로 그것까지 부르면 전체 index를 두 번 읽게 되고, 실험
        # 하나만 골라도 등록된 전부를 훑게 됩니다. 화면이 실제로 쓰는 것은 그
        # 결과 중 `experiment_record_uri` 하나뿐이라 summary에서 바로 꺼냅니다.
        wanted = set(requested)
        summaries = {
            summary["run_id"]: summary
            for summary in list_experiment_summaries(config)
            if summary.get("run_id") in wanted
        }
        missing = [run_id for run_id in requested if run_id not in summaries]

        targets: list[tuple[str, str]] = []
        for run_id in requested:
            summary = summaries.get(run_id)
            if not isinstance(summary, Mapping):
                continue
            uri = summary.get("experiment_record_uri")
            if isinstance(uri, str) and uri.strip():
                targets.append((run_id, uri))

        scores = kaggle_scores.load_scores()
        resolved = [
            _enrich_summary(summaries[run_id], record, scores.get(run_id))
            for run_id, record in _read_records(targets, config)
        ]
        return {"experiments": resolved, "missing": missing}
    except ExperimentRegistryError as error:
        raise WebError(f"실험 비교 정보를 읽지 못했습니다({type(error).__name__}).") from error


def save_kaggle_score(
    run_id: str, score: float, overwrite: bool = False
) -> dict[str, Any]:
    """생성된 submission을 실제로 제출해 받은 점수를 기록합니다.

    ``overwrite``는 사람이 화면에서 "실제 mAP 수정"을 켜고 보낸 요청에만 붙습니다.
    표를 지나가다 누른 저장이 이미 적어 둔 점수를 갈아치우면 안 되므로, 고치겠다는
    말이 없는 요청은 지금까지처럼 400으로 막고 기존 기록을 그대로 둡니다.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise WebValidationError([FieldError("run_id", "실험 이름이 필요합니다.")])
    wanted = run_id.strip()
    try:
        summary = next(
            (
                item
                for item in list_experiment_summaries(registry_config())
                if item.get("run_id") == wanted
            ),
            None,
        )
    except ExperimentRegistryError as error:
        raise WebError(f"실험 목록을 읽지 못했습니다({type(error).__name__}).") from error
    if summary is None:
        raise JobNotFoundError(f"'{wanted}' 실험을 registry에서 찾을 수 없습니다.")
    artifacts = summary.get("artifacts")
    submission_uri = artifacts.get("submission_uri") if isinstance(artifacts, Mapping) else None
    if _text(submission_uri) is None:
        raise WebValidationError(
            [FieldError("kaggle_score", "submission.csv를 먼저 생성해야 합니다.")]
        )
    if not overwrite and wanted in kaggle_scores.load_scores():
        raise WebValidationError(
            [
                FieldError(
                    "kaggle_score",
                    "이미 기록된 실제 점수입니다. 고치려면 '실제 mAP 수정'을 켜세요.",
                )
            ]
        )
    if not kaggle_scores.save_score(wanted, score, overwrite=overwrite):
        raise WebValidationError(
            [
                FieldError(
                    "kaggle_score",
                    "이미 기록된 실제 점수입니다. 고치려면 '실제 mAP 수정'을 켜세요.",
                )
            ]
        )
    return {"run_id": wanted, "kaggle_score": score}
