"""Evaluate pipeline의 공개 실행 진입점입니다.

`config["evaluate"]`에서 자기 설정을 읽고, 이전 pipeline 결과는
`config["inputs"]`에서 읽습니다. `config["inputs"]`는 수정하지 않습니다.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .errors import ArtifactWriteError, ConfigurationError, EvaluateError
from .manifest import load_class_map, load_manifest, load_test_manifest
from .metrics import DEFAULT_IOU_THRESHOLDS, evaluate_detections, filter_predictions
from .predictor import load_predictions, predict_record_groups_with_checkpoint
from .storage_io import ArtifactStore, join_uri
from .submission import render_submission_csv


DEFAULT_OUTPUT_ROOT = "artifacts/evaluate"
DEFAULT_METRICS_FILENAME = "metrics.json"
DEFAULT_PREDICTIONS_FILENAME = "predictions.json"
DEFAULT_SUBMISSION_ROOT = "submissions"
DEFAULT_SUBMISSION_FILENAME = "submission.csv"
DEFAULT_MAX_DETECTIONS_PER_IMAGE = 4
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_metric(value: float | None) -> str:
    """계산하지 않은 metric은 숫자 대신 그대로 드러냅니다.

    ground truth가 하나도 없으면 평균 낼 대상이 없어 mAP가 None입니다.
    """
    return "null" if value is None else f"{value:.4f}"


@dataclass(frozen=True)
class Settings:
    """검증이 끝난 evaluate 실행 설정입니다."""

    run_id: str
    validation_manifest_uri: str
    test_manifest_uri: str | None
    class_map_uri: str | None
    checkpoint_uri: str | None
    predictions_input_uri: str | None
    metrics_uri: str
    predictions_uri: str
    submission_uri: str | None
    iou_thresholds: tuple[float, ...]
    score_threshold: float
    max_detections_per_image: int | None
    device: str
    seed: int
    overwrite: bool
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction_source(self) -> str:
        return "predictions_file" if self.predictions_input_uri else "checkpoint"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} 설정은 object여야 합니다.")
    return value


def _optional_uri(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name}는 비어 있지 않은 문자열이어야 합니다.")
    return value.strip()


def _resolve_uri(
    settings: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    key: str,
    stage: str,
    stage_key: str,
) -> str | None:
    """자기 설정을 우선 사용하고, 없으면 이전 pipeline artifact에서 찾습니다."""
    own_value = _optional_uri(settings.get(key), f"evaluate.{key}")
    if own_value is not None:
        return own_value
    stage_artifacts = _mapping(inputs.get(stage), f"inputs.{stage}")
    return _optional_uri(stage_artifacts.get(stage_key), f"inputs.{stage}.{stage_key}")


def _resolve_iou_thresholds(value: Any, *, competition: bool) -> tuple[float, ...]:
    """메인 지표 IoU 구간을 정합니다.

    이 구간은 mAP@[0.75:0.95]로 고정이며 설정으로 바꿀 수 없습니다. 설정 key가
    있는데 조용히 무시되는 상태를 만들지 않도록, 다른 값이 오면 거절합니다.
    """
    if value is None:
        return DEFAULT_IOU_THRESHOLDS
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        raise ConfigurationError("evaluate.iou_thresholds는 비어 있지 않은 숫자 list여야 합니다.")
    thresholds = tuple(float(item) for item in value)
    if any(not 0.0 < threshold <= 1.0 for threshold in thresholds):
        raise ConfigurationError("evaluate.iou_thresholds 값은 0보다 크고 1 이하여야 합니다.")
    if thresholds != DEFAULT_IOU_THRESHOLDS:
        if competition:
            raise ConfigurationError(
                "competition iou_thresholds는 [0.75, 0.80, 0.85, 0.90, 0.95]여야 합니다."
            )
        raise ConfigurationError(
            "evaluate.iou_thresholds는 지원하지 않습니다. 메인 지표 구간은 "
            f"{list(DEFAULT_IOU_THRESHOLDS)}로 고정입니다."
        )
    return thresholds


def _resolve_max_detections(value: Any) -> int | None:
    if value is None:
        return DEFAULT_MAX_DETECTIONS_PER_IMAGE
    if value is False:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(
            "evaluate.max_detections_per_image는 0보다 큰 정수이거나 제한 해제를 뜻하는 false여야 합니다."
        )
    return value


def resolve_settings(config: Mapping[str, Any]) -> Settings:
    """config를 검증하고 실행 설정을 만듭니다."""
    if not isinstance(config, Mapping):
        raise ConfigurationError("config는 object여야 합니다.")

    settings = _mapping(config.get("evaluate"), "evaluate")
    inputs = _mapping(config.get("inputs"), "inputs")
    train_inputs = _mapping(inputs.get("train"), "inputs.train")

    run_id = settings.get("run_id") or train_inputs.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ConfigurationError("run_id는 비어 있지 않은 문자열이어야 합니다.")
    resolved_run_id = (
        run_id.strip()
        if isinstance(run_id, str)
        else f"evaluate-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    manifest_uri = _resolve_uri(
        settings,
        inputs,
        key="validation_manifest_uri",
        stage="data",
        stage_key="validation_manifest_uri",
    )
    if manifest_uri is None:
        raise ConfigurationError(
            "validation manifest가 없습니다. evaluate.validation_manifest_uri 또는 "
            "inputs.data.validation_manifest_uri가 필요합니다."
        )

    test_manifest_uri = _resolve_uri(
        settings,
        inputs,
        key="test_manifest_uri",
        stage="data",
        stage_key="test_manifest_uri",
    )

    class_map_uri = _resolve_uri(
        settings, inputs, key="class_map_uri", stage="data", stage_key="class_map_uri"
    )
    predictions_input_uri = _optional_uri(
        settings.get("predictions_input_uri"), "evaluate.predictions_input_uri"
    )
    checkpoint_uri = _resolve_uri(
        settings, inputs, key="checkpoint_uri", stage="train", stage_key="best_checkpoint_uri"
    )
    if predictions_input_uri is None and checkpoint_uri is None:
        raise ConfigurationError(
            "예측을 만들 수 없습니다. evaluate.predictions_input_uri 또는 "
            "inputs.train.best_checkpoint_uri(evaluate.checkpoint_uri)가 필요합니다."
        )
    if test_manifest_uri is not None and checkpoint_uri is None:
        raise ConfigurationError(
            "test image 추론에는 checkpoint가 필요합니다. evaluate.checkpoint_uri 또는 "
            "inputs.train.best_checkpoint_uri를 설정하세요."
        )

    output_dir = _optional_uri(settings.get("output_dir"), "evaluate.output_dir") or join_uri(
        DEFAULT_OUTPUT_ROOT, resolved_run_id
    )
    metrics_filename = (
        _optional_uri(settings.get("metrics_filename"), "evaluate.metrics_filename")
        or DEFAULT_METRICS_FILENAME
    )
    predictions_filename = (
        _optional_uri(settings.get("predictions_filename"), "evaluate.predictions_filename")
        or DEFAULT_PREDICTIONS_FILENAME
    )

    score_threshold = settings.get("score_threshold", 0.0)
    if (
        not isinstance(score_threshold, (int, float))
        or isinstance(score_threshold, bool)
        or not 0.0 <= float(score_threshold) <= 1.0
    ):
        raise ConfigurationError("evaluate.score_threshold는 0 이상 1 이하의 숫자여야 합니다.")

    seed = settings.get("seed", config.get("seed", DEFAULT_SEED))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigurationError("seed는 정수여야 합니다.")

    device = _optional_uri(settings.get("device"), "evaluate.device") or DEFAULT_DEVICE
    overwrite = settings.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ConfigurationError("evaluate.overwrite는 true 또는 false여야 합니다.")

    metrics_uri = join_uri(output_dir, metrics_filename)
    predictions_uri = join_uri(output_dir, predictions_filename)
    if metrics_uri == predictions_uri:
        raise ConfigurationError(
            "metrics와 predictions는 같은 위치에 저장할 수 없습니다. "
            f"evaluate.metrics_filename과 evaluate.predictions_filename을 다르게 두세요: {metrics_uri}"
        )

    configured_submission_uri = _optional_uri(
        settings.get("submission_uri"), "evaluate.submission_uri"
    )
    if configured_submission_uri is not None and test_manifest_uri is None:
        raise ConfigurationError("evaluate.submission_uri에는 test_manifest_uri가 필요합니다.")
    submission_uri = None
    if test_manifest_uri is not None:
        submission_uri = configured_submission_uri or join_uri(
            join_uri(DEFAULT_SUBMISSION_ROOT, resolved_run_id),
            DEFAULT_SUBMISSION_FILENAME,
        )
    if submission_uri is not None and submission_uri in {metrics_uri, predictions_uri}:
        raise ConfigurationError(
            "metrics, predictions, submission은 같은 위치에 저장할 수 없습니다."
        )

    # 메인 지표 구간은 대회 실행 여부와 무관하게 [0.75, 0.80, 0.85, 0.90, 0.95]입니다.
    iou_thresholds = _resolve_iou_thresholds(
        settings.get("iou_thresholds"), competition=test_manifest_uri is not None
    )

    return Settings(
        run_id=resolved_run_id,
        validation_manifest_uri=manifest_uri,
        test_manifest_uri=test_manifest_uri,
        class_map_uri=class_map_uri,
        checkpoint_uri=checkpoint_uri,
        predictions_input_uri=predictions_input_uri,
        metrics_uri=metrics_uri,
        predictions_uri=predictions_uri,
        submission_uri=submission_uri,
        iou_thresholds=iou_thresholds,
        score_threshold=float(score_threshold),
        max_detections_per_image=_resolve_max_detections(settings.get("max_detections_per_image")),
        device=device,
        seed=seed,
        overwrite=overwrite,
        inputs=dict(inputs),
    )


def _error_result(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "artifacts": {},
        "summary": {"pipeline": "evaluate"},
        "message": message,
    }


def _public_prediction(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """평가에 사용한 값을 그대로 남깁니다.

    반올림하면 저장된 predictions로 다시 평가할 때 metric이 달라질 수 있습니다.
    """
    return {
        "image_id": prediction["image_id"],
        "category_id": prediction["category_id"],
        "bbox": [float(value) for value in prediction["bbox"]],
        "score": float(prediction["score"]),
    }


def _is_dummy_execution(config: Mapping[str, Any]) -> bool:
    """저장소 공통 dummy 실행이면서 evaluate 설정이 전혀 없는 경우를 알려줍니다."""
    if not isinstance(config, Mapping):
        return False
    execution = config.get("execution")
    mode = execution.get("mode") if isinstance(execution, Mapping) else None
    return mode == "dummy" and not config.get("evaluate")


def run(config: dict) -> dict:
    """Validation manifest와 checkpoint로 detection metric과 예측을 만듭니다."""
    if _is_dummy_execution(config):
        return {
            "status": "ok",
            "artifacts": {},
            "summary": {"pipeline": "evaluate", "mode": "dummy"},
            "message": "evaluate pipeline dummy 실행 완료 (evaluate 설정이 없어 평가를 건너뛰었습니다)",
        }

    started_at = _utc_now()
    created_uris: list[str] = []
    store: ArtifactStore | None = None

    try:
        settings = resolve_settings(config)
        random.seed(settings.seed)
        store = ArtifactStore(config)

        output_uris = [settings.metrics_uri, settings.predictions_uri]
        if settings.submission_uri is not None:
            output_uris.append(settings.submission_uri)
        for uri in output_uris:
            if not settings.overwrite and store.exists(uri):
                raise ArtifactWriteError(
                    f"artifact가 이미 있습니다. evaluate.overwrite를 true로 두어야 덮어씁니다: "
                    f"{store.normalize_uri(uri)}"
                )

        records = load_manifest(store, settings.validation_manifest_uri)
        class_map = (
            load_class_map(store, settings.class_map_uri) if settings.class_map_uri else {}
        )
        image_keys = {record["image_key"] for record in records}

        test_records: list[dict[str, Any]] | None = None
        test_category_ids: frozenset[int] = frozenset()
        if settings.test_manifest_uri is not None:
            test_records, test_category_ids = load_test_manifest(
                store, settings.test_manifest_uri
            )

        if settings.predictions_input_uri is not None:
            raw_predictions = load_predictions(
                store, settings.predictions_input_uri, known_image_keys=image_keys
            )
        else:
            raw_predictions = []

        inference_groups: list[list[dict[str, Any]]] = []
        validation_group_index: int | None = None
        test_group_index: int | None = None
        if settings.predictions_input_uri is None:
            validation_group_index = len(inference_groups)
            inference_groups.append(records)
        if test_records is not None:
            test_group_index = len(inference_groups)
            inference_groups.append(test_records)

        generated_groups: list[list[dict[str, Any]]] = []
        if inference_groups:
            generated_groups = predict_record_groups_with_checkpoint(
                store,
                inference_groups,
                checkpoint_uri=str(settings.checkpoint_uri),
                device=settings.device,
                seed=settings.seed,
            )
        if validation_group_index is not None:
            raw_predictions = generated_groups[validation_group_index]

        predictions = filter_predictions(
            raw_predictions,
            score_threshold=settings.score_threshold,
            max_detections_per_image=settings.max_detections_per_image,
        )
        report = evaluate_detections(
            records,
            predictions,
            iou_thresholds=settings.iou_thresholds,
            class_names=class_map,
            max_detections_per_image=settings.max_detections_per_image,
        )

        submission_text: str | None = None
        if test_group_index is not None:
            test_predictions = filter_predictions(
                generated_groups[test_group_index],
                score_threshold=settings.score_threshold,
                max_detections_per_image=settings.max_detections_per_image,
            )
            submission_text = render_submission_csv(
                test_predictions,
                category_ids=test_category_ids,
            )

        finished_at = _utc_now()
        common_fields = {
            "run_id": settings.run_id,
            "created_at": finished_at,
            "started_at": started_at,
            "validation_manifest_uri": settings.validation_manifest_uri,
            "class_map_uri": settings.class_map_uri,
            "checkpoint_uri": settings.checkpoint_uri,
            "predictions_input_uri": settings.predictions_input_uri,
            "prediction_source": settings.prediction_source,
            "score_threshold": settings.score_threshold,
            "max_detections_per_image": settings.max_detections_per_image,
            "seed": settings.seed,
            "device": settings.device,
        }

        predictions_document = {
            **common_fields,
            "raw_prediction_count": len(raw_predictions),
            "prediction_count": len(predictions),
            "bbox_format": "xywh",
            "predictions": [_public_prediction(prediction) for prediction in predictions],
        }
        submission_uri: str | None = None
        if settings.submission_uri is not None and submission_text is not None:
            submission_existed = store.exists(settings.submission_uri)
            submission_uri = store.write_text(
                settings.submission_uri, submission_text, overwrite=settings.overwrite
            )
            if not submission_existed:
                created_uris.append(settings.submission_uri)

        predictions_existed = store.exists(settings.predictions_uri)
        predictions_uri = store.write_json(
            settings.predictions_uri, predictions_document, overwrite=settings.overwrite
        )
        if not predictions_existed:
            created_uris.append(settings.predictions_uri)

        metrics_document = {**common_fields, **report}
        metrics_existed = store.exists(settings.metrics_uri)
        metrics_uri = store.write_json(
            settings.metrics_uri, metrics_document, overwrite=settings.overwrite
        )
        if not metrics_existed:
            created_uris.append(settings.metrics_uri)
    except EvaluateError as error:
        if store is not None:
            for uri in created_uris:
                store.remove_local(uri)
        return _error_result(str(error))

    artifacts = {
        "run_id": settings.run_id,
        "metrics_uri": metrics_uri,
        "predictions_uri": predictions_uri,
    }
    if submission_uri is not None:
        artifacts["submission_uri"] = submission_uri

    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": {
            "pipeline": "evaluate",
            "run_id": settings.run_id,
            "prediction_source": settings.prediction_source,
            "image_count": report["image_count"],
            "annotation_count": report["annotation_count"],
            "prediction_count": report["prediction_count"],
            "evaluated_class_count": report["evaluated_class_count"],
            "iou_thresholds": list(settings.iou_thresholds),
            "score_threshold": settings.score_threshold,
            "max_detections_per_image": settings.max_detections_per_image,
            "seed": settings.seed,
            "device": settings.device,
            "metrics": report["metrics"],
        },
        "message": (
            f"evaluate pipeline 실행 완료 (run_id={settings.run_id}, "
            f"image={report['image_count']}, mAP={_format_metric(report['metrics']['mAP'])})"
        ),
    }


__all__ = ["Settings", "resolve_settings", "run"]
