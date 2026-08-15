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
from .progress import ProgressEmitter
from .storage_io import ArtifactStore, join_uri
from .submission import render_submission_csv


DEFAULT_OUTPUT_ROOT = "artifacts/evaluate"
DEFAULT_METRICS_FILENAME = "metrics.json"
DEFAULT_PREDICTIONS_FILENAME = "predictions.json"
DEFAULT_TEST_PREDICTIONS_FILENAME = "test_predictions.json"
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
    # 제출한 것과 같은 test 예측을 JSON으로도 남길 위치입니다. test manifest가 있을
    # 때만 값이 있습니다.
    test_predictions_uri: str | None
    iou_thresholds: tuple[float, ...]
    score_threshold: float
    max_detections_per_image: int | None
    # 제출 CSV에서 뺄 category id입니다. 대회에 없는 보조 class를 여기에 넣습니다.
    submission_excluded_category_ids: frozenset[int]
    # validation 지표에서 뺄 category id입니다. 채점되지 않는 class를 평균에 넣으면
    # 로컬 mAP와 대회 점수가 서로 다른 class 집합을 재게 됩니다.
    metrics_excluded_category_ids: frozenset[int]
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


def _resolve_excluded_category_ids(
    value: Any, *, setting: str = "submission_excluded_category_ids"
) -> frozenset[int]:
    """뺄 category id 목록을 읽습니다.

    기본값은 빈 집합이라 설정하지 않으면 동작이 달라지지 않습니다. 제출과 지표가
    같은 형식을 쓰므로 검사도 하나로 둡니다. 어느 설정이 틀렸는지 알려면 이름이
    메시지에 들어가야 합니다.
    """

    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(
            f"evaluate.{setting}는 category id의 list여야 합니다."
        )
    identifiers = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ConfigurationError(
                f"evaluate.{setting}의 값은 0 이상의 정수여야 합니다."
            )
        identifiers.append(item)
    return frozenset(identifiers)


def _without_categories(
    records: Sequence[Mapping[str, Any]], excluded: frozenset[int]
) -> list[Mapping[str, Any]]:
    """정답에서 특정 category의 annotation만 뺍니다.

    이미지는 남깁니다. 그 이미지에 다른 class의 정답이 있을 수 있고, 이미지를 지우면
    거기서 나온 예측이 갈 곳을 잃어 false positive로 둔갑합니다.
    """

    if not excluded:
        return list(records)
    return [
        {
            **record,
            "annotations": [
                annotation
                for annotation in record["annotations"]
                if annotation["category_id"] not in excluded
            ],
        }
        for record in records
    ]


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
    test_predictions_filename = (
        _optional_uri(
            settings.get("test_predictions_filename"), "evaluate.test_predictions_filename"
        )
        or DEFAULT_TEST_PREDICTIONS_FILENAME
    )

    score_threshold = settings.get("score_threshold", 0.0)
    if (
        not isinstance(score_threshold, (int, float))
        or isinstance(score_threshold, bool)
        or not 0.0 <= float(score_threshold) <= 1.0
    ):
        raise ConfigurationError("evaluate.score_threshold는 0 이상 1 이하의 숫자여야 합니다.")

    submission_excluded_category_ids = _resolve_excluded_category_ids(
        settings.get("submission_excluded_category_ids")
    )
    metrics_excluded_category_ids = _resolve_excluded_category_ids(
        settings.get("metrics_excluded_category_ids"),
        setting="metrics_excluded_category_ids",
    )

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
    test_predictions_uri = (
        join_uri(output_dir, test_predictions_filename) if test_manifest_uri is not None else None
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
    # 출력이 넷으로 늘어 짝마다 따로 보면 빠뜨리기 쉽습니다. 한 번에 셉니다.
    written_uris = [
        uri
        for uri in (metrics_uri, predictions_uri, submission_uri, test_predictions_uri)
        if uri is not None
    ]
    if len(set(written_uris)) != len(written_uris):
        raise ConfigurationError(
            "metrics, predictions, submission, test predictions는 같은 위치에 "
            "저장할 수 없습니다."
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
        test_predictions_uri=test_predictions_uri,
        iou_thresholds=iou_thresholds,
        score_threshold=float(score_threshold),
        max_detections_per_image=_resolve_max_detections(settings.get("max_detections_per_image")),
        submission_excluded_category_ids=submission_excluded_category_ids,
        metrics_excluded_category_ids=metrics_excluded_category_ids,
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
    # 진행 로그는 stderr 전용 부가 출력입니다. 실패해도 평가는 그대로 진행됩니다.
    progress = ProgressEmitter()

    try:
        settings = resolve_settings(config)
        random.seed(settings.seed)
        store = ArtifactStore(config)

        output_uris = [settings.metrics_uri, settings.predictions_uri]
        if settings.submission_uri is not None:
            output_uris.append(settings.submission_uri)
        if settings.test_predictions_uri is not None:
            output_uris.append(settings.test_predictions_uri)
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

        test_image_count = len(test_records) if test_records is not None else 0
        progress.emit(
            "evaluate_started",
            run_id=settings.run_id,
            device=settings.device,
            validation_images=len(records),
            test_images=test_image_count,
        )

        if settings.predictions_input_uri is not None:
            raw_predictions = load_predictions(
                store, settings.predictions_input_uri, known_image_keys=image_keys
            )
        else:
            raw_predictions = []

        inference_groups: list[list[dict[str, Any]]] = []
        # 어느 group이 validation이고 어느 group이 test인지는 여기서만 알 수
        # 있으므로, 진행 로그의 stage 이름도 여기서 정합니다.
        group_stages: list[str] = []
        validation_group_index: int | None = None
        test_group_index: int | None = None
        if settings.predictions_input_uri is None:
            validation_group_index = len(inference_groups)
            inference_groups.append(records)
            group_stages.append("validation")
        if test_records is not None:
            test_group_index = len(inference_groups)
            inference_groups.append(test_records)
            group_stages.append("test")

        def report_predict_progress(index: int, done: int, total: int) -> None:
            """추론 진행을 stage 이름과 함께 알립니다. 실패해도 추론은 계속됩니다."""
            if 0 <= index < len(group_stages):
                progress.predict_progress(group_stages[index], done, total)

        generated_groups: list[list[dict[str, Any]]] = []
        if inference_groups:
            generated_groups = predict_record_groups_with_checkpoint(
                store,
                inference_groups,
                checkpoint_uri=str(settings.checkpoint_uri),
                device=settings.device,
                seed=settings.seed,
                on_progress=report_predict_progress,
            )
        if validation_group_index is not None:
            raw_predictions = generated_groups[validation_group_index]

        predictions = filter_predictions(
            raw_predictions,
            score_threshold=settings.score_threshold,
            max_detections_per_image=settings.max_detections_per_image,
        )
        # 채점되지 않는 class를 평균에 넣으면 로컬 mAP와 대회 점수가 서로 다른 집합을
        # 재게 됩니다. 저장되는 예측 원본은 건드리지 않습니다. 나중에 다른 집합으로
        # 다시 채점할 수 있어야 하기 때문입니다.
        #
        # 거르는 순서가 중요합니다. 이미지당 상한을 먼저 적용하면 제외 class의 고득점
        # 예측이 앞자리를 차지한 채 채점 대상 예측을 상한 밖으로 밀어내고, 그 뒤에
        # 제외해 봐야 밀려난 예측은 이미 없습니다. 제출 CSV가 상한보다 먼저 거르므로
        # (`006`) 그대로 두면 로컬 지표가 대회가 채점하는 목록과 달라집니다.
        scored_records = _without_categories(
            records, settings.metrics_excluded_category_ids
        )
        scored_predictions = (
            predictions
            if not settings.metrics_excluded_category_ids
            else filter_predictions(
                raw_predictions,
                score_threshold=settings.score_threshold,
                max_detections_per_image=settings.max_detections_per_image,
                excluded_category_ids=settings.metrics_excluded_category_ids,
            )
        )
        report = evaluate_detections(
            scored_records,
            scored_predictions,
            iou_thresholds=settings.iou_thresholds,
            class_names={
                category_id: name
                for category_id, name in class_map.items()
                if category_id not in settings.metrics_excluded_category_ids
            },
            max_detections_per_image=settings.max_detections_per_image,
        )
        if settings.metrics_excluded_category_ids:
            report["metrics_excluded_category_ids"] = sorted(
                settings.metrics_excluded_category_ids
            )
        progress.metrics_computed(report["metrics"])

        submission_text: str | None = None
        submission_rows = 0
        test_predictions: list[dict[str, Any]] | None = None
        if test_group_index is not None:
            # 제외는 test 경로에만 적용합니다. validation 지표에는 보조 class도
            # 남아 있어야 "대회 밖 알약을 알약으로 잡았는가"를 볼 수 있습니다.
            test_predictions = filter_predictions(
                generated_groups[test_group_index],
                score_threshold=settings.score_threshold,
                max_detections_per_image=settings.max_detections_per_image,
                excluded_category_ids=settings.submission_excluded_category_ids,
            )
            submission_text = render_submission_csv(
                test_predictions,
                category_ids=test_category_ids,
            )
            submission_rows = len(test_predictions)

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
            progress.emit("submission_written", rows=submission_rows)

        test_predictions_uri: str | None = None
        if settings.test_predictions_uri is not None and test_predictions is not None:
            test_predictions_document = {
                **common_fields,
                # 어느 test manifest를 본 예측인지 적습니다. 이것이 없으면 서로 다른
                # dataset 판의 예측을 image_id만 보고 섞어도 조용히 지나갑니다.
                "test_manifest_uri": settings.test_manifest_uri,
                "bbox_format": "xywh",
                # 제출에서 뺀 class가 있으면 이 파일에도 없습니다. 무엇이 빠졌는지
                # 적어 두지 않으면 나중에 읽는 쪽이 모델이 못 맞힌 것으로 읽습니다.
                "submission_excluded_category_ids": sorted(
                    settings.submission_excluded_category_ids
                ),
                "prediction_count": len(test_predictions),
                "predictions": [
                    _public_prediction(prediction) for prediction in test_predictions
                ],
            }
            test_predictions_existed = store.exists(settings.test_predictions_uri)
            test_predictions_uri = store.write_json(
                settings.test_predictions_uri,
                test_predictions_document,
                overwrite=settings.overwrite,
            )
            if not test_predictions_existed:
                created_uris.append(settings.test_predictions_uri)

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

        progress.emit(
            "evaluate_completed",
            validation_images=len(records),
            test_images=test_image_count,
        )
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
    if test_predictions_uri is not None:
        artifacts["test_predictions_uri"] = test_predictions_uri

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
            # 쓰지 않은 실행의 summary 모양은 그대로 두려고 값이 있을 때만 넣습니다.
            # `frozenset`은 JSON으로 직렬화할 수 없어 정렬한 list로 바꿉니다.
            **(
                {
                    "submission_excluded_category_ids": sorted(
                        settings.submission_excluded_category_ids
                    )
                }
                if settings.submission_excluded_category_ids
                else {}
            ),
            **(
                {
                    "metrics_excluded_category_ids": sorted(
                        settings.metrics_excluded_category_ids
                    )
                }
                if settings.metrics_excluded_category_ids
                else {}
            ),
        },
        "message": (
            f"evaluate pipeline 실행 완료 (run_id={settings.run_id}, "
            f"image={report['image_count']}, mAP={_format_metric(report['metrics']['mAP'])})"
        ),
    }


__all__ = ["Settings", "resolve_settings", "run"]
