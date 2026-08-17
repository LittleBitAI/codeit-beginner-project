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

from .errors import (
    ArtifactWriteError,
    ConfigurationError,
    EvaluateError,
    InputArtifactError,
)
from .fusion import DEFAULT_IOU_THRESHOLD as DEFAULT_FUSION_IOU_THRESHOLD
from .fusion import fuse_predictions
from .manifest import (
    load_class_map,
    load_manifest,
    load_test_manifest,
    sample_records,
)
from .metrics import DEFAULT_IOU_THRESHOLDS, evaluate_detections, filter_predictions
from .predictor import (
    load_predictions,
    parse_predictions,
    predict_record_groups_with_checkpoint,
)
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
    # 있으면 test 추론 대신 이 예측 파일들을 합쳐 씁니다. 실행 여럿을
    # 합치는 유일한 통로이며, 비어 있으면 지금까지처럼 추론합니다.
    test_predictions_input_uris: tuple[str, ...]
    # 사진이 다른 곳에 있어도 같은 시험지로 볼지입니다. 기본은 막습니다 — 위치가 다른
    # 것은 다른 사진이라는 증명은 아니지만, 잘못 합치면 조용히 틀린 제출이 나오고
    # 잘못 막히면 시끄럽게 멈춥니다. 같은 사진임을 **사람이 확인했을 때만** 켭니다.
    fusion_allow_copied_images: bool
    metrics_uri: str
    predictions_uri: str
    submission_uri: str | None
    # 제출한 것과 같은 test 예측을 JSON으로도 남길 위치입니다. test manifest가 있을
    # 때만 값이 있습니다.
    test_predictions_uri: str | None
    # 검증 image를 몇 장만 보고 잴지입니다. ``None``이면 지금까지처럼 전수입니다.
    validation_sample_size: int | None
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


def _prediction_source(
    store: ArtifactStore, document: Mapping[str, Any], uri: str
) -> str:
    """이 예측이 **어느 checkpoint의 증거인지** 돌려줍니다.

    합친 결과는 받지 않습니다. 그것을 다시 합치면 실행 수를 **입력 파일 수**로 세게
    되어, 안에 다섯 실행이 들어 있는 파일과 한 실행짜리 파일이 같은 한 표가 됩니다.
    확신도가 그만큼 어긋나고 제출이 조용히 달라집니다.

    평평하게 합치려면 원본 예측들을 한 번에 주면 됩니다. 중첩을 지원하지 않는 대신
    잘못 셀 일이 없습니다.
    """

    # 합친 결과라고 스스로 밝힌 파일은 checkpoint가 채워져 있어도 거부합니다.
    # 없는 것만 보면 표식과 checkpoint를 함께 담아 우회할 수 있습니다.
    # 빈 목록도 "합친 결과"라는 표식입니다. 진릿값으로 보면 `fused_from: []`에
    # checkpoint를 붙인 파일이 통과합니다.
    if (
        document.get("prediction_source") == "fusion"
        or document.get("fused_from") is not None
    ):
        raise InputArtifactError(
            f"{uri}: 합친 결과를 다시 합칠 수는 없습니다 — 원본 예측들을 한 번에 "
            "주세요."
        )

    checkpoint = document.get("checkpoint_uri")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise InputArtifactError(
            f"{uri}: 어느 checkpoint의 증거인지 말하지 않아 합칠 수 없습니다."
        )
    return str(store.artifact_identity(checkpoint.strip()))


def _load_fusion_inputs(
    store: ArtifactStore,
    uris: Sequence[str],
    *,
    test_records: Sequence[Mapping[str, Any]],
    test_category_ids: frozenset[int],
    test_manifest_uri: str,
    allow_copied_images: bool,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """합칠 예측을 읽고, 합쳐도 되는 것들인지 확인합니다.

    두 가지를 봅니다.

    **같은 시험지를 봤는가.** 예측에 나온 image id만 보면 안 됩니다. 아무것도 검출하지
    못한 이미지는 목록에 없는 것이 정상이라 멀쩡한 입력을 거절하고, 같은 id를 쓰는
    다른 시험지는 그대로 지나갑니다. 각 파일이 적어 둔 test manifest를 **실제로 읽어**
    image id와 사진의 위치와 크기와 category를 견줍니다. 위치까지 견주는 것은 위치가
    내용을 증명해서가 아니라 **틀리는 방향이 다르기** 때문입니다 — 잘못 합치면 조용히
    틀린 제출이 나오고, 잘못 막히면 시끄럽게 멈춥니다. 같은 시험지가 dataset 판마다
    다른 prefix에 복사돼 있는 것은 실제로 있는 일이라, 사람이 같은 사진임을 확인했으면
    `fusion_allow_copied_images`로 위치만 빼고 견줍니다.

    **서로 다른 실행인가.** 그 예측을 만든 **checkpoint**로 봅니다. 그것이 모델의
    신원이고, 같은 checkpoint는 몇 번을 넣어도 한 실행입니다. 표기가 달라도 같은
    파일이면 걸리도록 저장 계층에 물어봅니다. 합친 결과는 아예 받지 않습니다.
    """

    def manifest_content(
        records: Sequence[Mapping[str, Any]], category_ids: frozenset[int]
    ) -> tuple[Any, ...]:
        """시험지의 내용을 견줄 수 있는 값으로 만듭니다."""

        # 사진 **이름**은 따로 견줄 것이 없습니다. `parse_test_manifest`가 이미
        # `file_name`의 숫자 stem과 image id가 같기를 요구하므로, id가 같으면 이름도
        # 같습니다. 그래서 견주는 것은 **어느 파일인가**이고, 표기가 달라도 같은
        # 파일이면 걸리도록 저장 계층에 물어봅니다.
        #
        # ponytail: 위치는 내용의 증명이 아니라 보수적인 대리물입니다. 같은 자리에
        # 다른 사진이 놓이면 이것으로도 못 막습니다. manifest에 content digest가
        # 생기면 이 자리를 digest로 바꾸고 `fusion_allow_copied_images`를 없앱니다.
        return (
            frozenset(
                (
                    record["image_id"],
                    None
                    if allow_copied_images
                    else store.artifact_identity(record["image_uri"]),
                    record["width"],
                    record["height"],
                )
                for record in records
            ),
            category_ids,
        )

    own_content = manifest_content(test_records, test_category_ids)
    own_manifest = store.artifact_identity(test_manifest_uri)
    checked_manifests: set[tuple[str, ...]] = {own_manifest}

    groups: list[list[dict[str, Any]]] = []
    lineage: list[dict[str, Any]] = []
    # 어느 checkpoint를 이미 셌는지입니다.
    claimed_sources: dict[str, str] = {}
    image_keys = {record["image_key"] for record in test_records}

    for uri in uris:
        document = store.read_json(uri)
        if not isinstance(document, Mapping):
            raise InputArtifactError(f"{uri}: 합칠 예측은 object여야 합니다.")

        declared = document.get("test_manifest_uri")
        if not isinstance(declared, str) or not declared.strip():
            raise InputArtifactError(
                f"{uri}: 어느 test manifest를 본 예측인지 적혀 있지 않아 합칠 수 "
                "없습니다."
            )
        declared_target = store.artifact_identity(declared)
        if declared_target not in checked_manifests:
            other_records, other_categories = load_test_manifest(store, declared)
            if manifest_content(other_records, other_categories) != own_content:
                raise ConfigurationError(
                    f"{uri}는 다른 시험지를 본 예측입니다: {declared}\n"
                    "사진이 같은데 위치만 다른 것을 확인했다면 "
                    "evaluate.fusion_allow_copied_images를 true로 두세요."
                )
            checked_manifests.add(declared_target)

        source = _prediction_source(store, document, uri)
        if source in claimed_sources:
            raise ConfigurationError(
                f"같은 실행의 예측이 두 번 있습니다: {claimed_sources[source]}와 {uri}"
            )
        claimed_sources[source] = uri

        groups.append(parse_predictions(document, source=uri, known_image_keys=image_keys))
        lineage.append(
            {
                "uri": uri,
                "run_id": document.get("run_id"),
                "checkpoint_uri": document.get("checkpoint_uri"),
            }
        )
    return groups, lineage


def _resolve_prediction_inputs(value: Any) -> tuple[str, ...]:
    """합칠 예측 파일 목록을 읽습니다.

    기본값은 빈 tuple이라 설정하지 않으면 동작이 달라지지 않습니다. **둘 미만은
    막습니다.** 하나만 주면 합칠 것이 없는데도 겹치는 상자를 묶어 원본과 다른 예측이
    나옵니다. 추론 없이 제출을 다시 만드는 길로 쓸 수 없습니다.
    """

    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(
            "evaluate.test_predictions_input_uris는 URI의 list여야 합니다."
        )
    uris = [
        _optional_uri(item, "evaluate.test_predictions_input_uris의 항목")
        for item in value
    ]
    if any(uri is None for uri in uris):
        raise ConfigurationError(
            "evaluate.test_predictions_input_uris의 항목은 비어 있지 않은 문자열이어야 합니다."
        )
    if len(uris) < 2:
        raise ConfigurationError(
            "evaluate.test_predictions_input_uris에는 합칠 예측이 둘 이상 필요합니다. "
            "하나만 주면 합칠 것이 없는데도 겹치는 상자를 묶어 원본과 달라집니다."
        )
    return tuple(uri for uri in uris if uri is not None)


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


def _resolve_validation_sample_size(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(
            "evaluate.validation_sample_size는 0보다 큰 정수여야 합니다."
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
    test_predictions_input_uris = _resolve_prediction_inputs(
        settings.get("test_predictions_input_uris")
    )
    if (
        test_manifest_uri is not None
        and checkpoint_uri is None
        and not test_predictions_input_uris
    ):
        raise ConfigurationError(
            "test image 추론에는 checkpoint가 필요합니다. evaluate.checkpoint_uri, "
            "inputs.train.best_checkpoint_uri, 또는 합칠 예측을 가리키는 "
            "evaluate.test_predictions_input_uris를 설정하세요."
        )
    if test_predictions_input_uris and test_manifest_uri is None:
        raise ConfigurationError(
            "evaluate.test_predictions_input_uris에는 test_manifest_uri가 필요합니다."
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
    # 이름을 설정으로 열지 않습니다. 열면 다른 출력과 겹칠 수 있는 짝이 늘어나고,
    # 겹침을 저장 계층과 똑같이 판정하는 일은 이 변경이 감당할 범위가 아닙니다.
    # 필요해지면 그때 열면 됩니다.

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
    fusion_allow_copied_images = settings.get("fusion_allow_copied_images", False)
    if not isinstance(fusion_allow_copied_images, bool):
        raise ConfigurationError(
            "evaluate.fusion_allow_copied_images는 true 또는 false여야 합니다."
        )

    overwrite = settings.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ConfigurationError("evaluate.overwrite는 true 또는 false여야 합니다.")

    metrics_uri = join_uri(output_dir, metrics_filename)
    predictions_uri = join_uri(output_dir, predictions_filename)
    # 출력 위치가 겹치는지는 `run()`에서 저장 계층의 이름으로 견줍니다. 여기서는
    # 저장소 root를 모르므로 `./m.json` 같은 표기 차이를 가릴 수 없습니다.
    test_predictions_uri = (
        join_uri(output_dir, DEFAULT_TEST_PREDICTIONS_FILENAME)
        if test_manifest_uri is not None
        else None
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
        test_predictions_input_uris=test_predictions_input_uris,
        fusion_allow_copied_images=fusion_allow_copied_images,
        metrics_uri=metrics_uri,
        predictions_uri=predictions_uri,
        submission_uri=submission_uri,
        test_predictions_uri=test_predictions_uri,
        validation_sample_size=_resolve_validation_sample_size(
            settings.get("validation_sample_size")
        ),
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

        # 겹침은 **저장 계층에 물어서** 견줍니다. 표기 규칙을 여기서 지어내면
        # backend가 실제로 하는 해석과 어긋납니다 — `./m.json`과 `m.json`을 다르게
        # 보거나, S3에서 서로 다른 object인 `a//b`와 `a/b`를 같다고 막습니다.
        # 이미 있는 파일이면 hard link까지 가려 주므로 덮어쓰기를 켠 실행도 걸립니다.
        written_by_target: dict[tuple[str, ...], str] = {}
        for label, uri in (
            ("metrics", settings.metrics_uri),
            ("predictions", settings.predictions_uri),
            ("submission", settings.submission_uri),
            ("test predictions", settings.test_predictions_uri),
        ):
            if uri is None:
                continue
            target = store.artifact_identity(uri)
            if target in written_by_target:
                raise ConfigurationError(
                    f"{written_by_target[target]}와 {label}는 같은 위치에 저장할 수 "
                    f"없습니다: {store.normalize_uri(uri)}"
                )
            written_by_target[target] = label

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
        # 예측 file은 **전체** manifest를 기준으로 확인합니다. 표본에서 빠진 image를
        # 가리킨다고 그 file이 잘못된 것은 아닙니다.
        manifest_image_keys = {record["image_key"] for record in records}
        # 실제로 **줄어든** 실행만 표본입니다. 요청값이 manifest보다 크거나 같으면 전수를
        # 본 것이므로, 그때 요청값을 그대로 적으면 전수 점수가 표본 점수로 읽힙니다.
        sampled_size: int | None = None
        if (
            settings.validation_sample_size is not None
            and settings.validation_sample_size < len(records)
        ):
            # 표본으로 잰 점수는 전수와 같은 값이 아닙니다. 어느 파일에서 나온
            # 숫자인지 알 수 있도록 표본 크기를 결과 문서에도 함께 남깁니다.
            sampled_size = settings.validation_sample_size
            records = sample_records(records, sampled_size, seed=settings.seed)
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
            raw_predictions = [
                prediction
                for prediction in load_predictions(
                    store,
                    settings.predictions_input_uri,
                    known_image_keys=manifest_image_keys,
                )
                # 표본에 없는 image의 예측은 이번 채점에 들어가지 않습니다.
                if prediction["image_key"] in image_keys
            ]
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
        # 합칠 예측을 받았으면 test는 추론하지 않습니다. 파일마다 이 test manifest의
        # image를 가리키는지 `load_predictions`가 확인하므로, 다른 시험지를 본 예측을
        # 섞으면 거기서 걸립니다.
        fused_test_predictions: list[dict[str, Any]] | None = None
        fusion_lineage: list[dict[str, Any]] = []
        if test_records is not None and settings.test_predictions_input_uris:
            groups, fusion_lineage = _load_fusion_inputs(
                store,
                settings.test_predictions_input_uris,
                test_records=test_records,
                test_category_ids=test_category_ids,
                test_manifest_uri=str(settings.test_manifest_uri),
                allow_copied_images=settings.fusion_allow_copied_images,
            )
            fused_test_predictions = fuse_predictions(groups)
            progress.emit(
                "test_predictions_fused",
                sources=len(settings.test_predictions_input_uris),
                predictions=len(fused_test_predictions),
            )
        elif test_records is not None:
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
        # 추론했으면 그 결과를, 합쳤으면 합친 결과를 씁니다. 그 뒤 걸러 내는 규칙은
        # 어느 쪽이든 같습니다 — 합친 예측만 다른 상한이나 다른 제외를 타면 제출이
        # 두 가지 규칙으로 만들어집니다.
        raw_test_predictions = (
            generated_groups[test_group_index]
            if test_group_index is not None
            else fused_test_predictions
        )
        if raw_test_predictions is not None:
            # 제외는 test 경로에만 적용합니다. validation 지표에는 보조 class도
            # 남아 있어야 "대회 밖 알약을 알약으로 잡았는가"를 볼 수 있습니다.
            test_predictions = filter_predictions(
                raw_test_predictions,
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
            # 표본으로 잰 점수를 전수 점수로 읽으면 서로 다른 시험지의 숫자를
            # 나란히 놓게 됩니다. 전수로 잰 실행은 `None`입니다.
            "validation_sample_size": sampled_size,
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
        # 네 출력 모두 쓴 **뒤에** 정리 목록에 넣습니다. 쓰기 전에 넣으면 `exists()`와
        # 쓰기 사이에 다른 실행이 만든 파일까지 이 실행이 지웁니다. 그 대신 쓰다가
        # 끊기면 반쯤 쓰인 파일이 남는데, 이는 이 pipeline이 처음부터 갖고 있던 성질이고
        # 고치려면 저장 계층을 원자적으로 바꿔야 해서 따로 다룹니다.
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
                # 이 파일이 어디서 왔는지 적습니다. validation이 저장된 예측을
                # 읽었더라도 추론한 test 예측은 checkpoint에서 옵니다. 합친 것이면
                # 무엇을 합쳤는지까지 남겨야 재현할 수 있습니다 — 그것이 없으면
                # 어느 실행들의 앙상블인지 되짚을 방법이 없습니다.
                "prediction_source": (
                    "fusion" if fused_test_predictions is not None else "checkpoint"
                ),
                # 합친 결과는 어느 checkpoint의 것도 아닙니다. 설정에 있던 값을 그대로
                # 남기면 다음 융합이 그 무관한 checkpoint를 모델 신원으로 씁니다.
                **({"checkpoint_uri": None} if fused_test_predictions is not None else {}),
                "predictions_input_uri": None,
                "fused_from": fusion_lineage or None,
                "fusion_iou_threshold": (
                    DEFAULT_FUSION_IOU_THRESHOLD if fusion_lineage else None
                ),
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
            # 전수로 잰 실행의 summary 모양은 그대로 둡니다. 요청값이 manifest보다
            # 커서 실제로는 전수를 본 실행도 여기 들어가면 안 됩니다.
            **({"validation_sample_size": sampled_size} if sampled_size else {}),
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
