"""COCO 스타일 object detection metric을 pycocotools로 계산합니다.

AP와 mAP는 pycocotools ``COCOeval``이 계산합니다. numpy는 평가할 IoU 기준값
배열을 만들고 ``evalImgs``를 집계하는 보조 도구로만 씁니다. 매칭 로직은 직접
구현하지 않습니다.

- COCOeval에는 항상 IoU 0.50~0.95 10점을 넘기고, 메인 지표는 그중
  0.75~0.95 구간(:data:`MAIN_SLICE`)을 잘라 평균 냅니다.
- 계산하지 않았거나 ground truth가 없는 값은 ``0.0``이 아니라 ``None``입니다.
- 보조 지표(TP/FP/FN, precision, recall, F1)는 IoU 0.50과 0.75 두 기준에서
  score 0.5 이상인 예측만으로 집계합니다.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .errors import MetricError


#: COCOeval에 항상 넘기는 고정 IoU 기준값. numpy로 만드는 유일한 기준값 배열입니다.
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(
    float(value) for value in np.round(np.linspace(0.50, 0.95, 10), 2)
)

#: 메인 지표 mAP@[0.75:0.95]가 차지하는 구간입니다.
MAIN_SLICE = slice(5, 10)

#: 보조 지표를 뽑는 IoU 기준의 인덱스입니다.
IDX_50 = 0
IDX_75 = 5

#: 설정이 비었을 때 쓰는 메인 지표 구간입니다.
DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = COCO_IOU_THRESHOLDS[MAIN_SLICE]

#: 보조 지표(TP/FP/FN, precision, recall)를 집계할 때 쓰는 confidence 기준입니다.
ANALYSIS_SCORE_THRESHOLD = 0.5

#: 보조 지표를 보고할 IoU 기준과 그 인덱스입니다.
ANALYSIS_IOU_INDICES: tuple[tuple[str, int], ...] = (("0.50", IDX_50), ("0.75", IDX_75))

#: max_detections_per_image에 제한이 없을 때 COCOeval에 넘기는 값입니다.
_UNLIMITED_DETECTIONS = 1000

#: 전체 크기 구간 하나만 사용합니다.
_AREA_RANGE_ALL = [0.0, 1e10]

# 인덱스 상수는 위 배열이 정확히 이 10점일 때만 유효합니다.
assert COCO_IOU_THRESHOLDS[IDX_50] == 0.50
assert COCO_IOU_THRESHOLDS[IDX_75] == 0.75
assert DEFAULT_IOU_THRESHOLDS == (0.75, 0.80, 0.85, 0.90, 0.95)


def filter_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    score_threshold: float = 0.0,
    max_detections_per_image: int | None = None,
) -> list[dict[str, Any]]:
    """score threshold와 이미지당 최대 detection 수를 적용합니다."""
    kept = [
        dict(prediction)
        for prediction in predictions
        if float(prediction["score"]) >= score_threshold
    ]
    kept.sort(key=lambda prediction: -float(prediction["score"]))
    if max_detections_per_image is None:
        return kept

    counts: dict[str, int] = {}
    limited: list[dict[str, Any]] = []
    for prediction in kept:
        image_key = prediction["image_key"]
        if counts.get(image_key, 0) >= max_detections_per_image:
            continue
        counts[image_key] = counts.get(image_key, 0) + 1
        limited.append(prediction)
    return limited


def _mean_or_none(values: np.ndarray) -> float | None:
    """COCOeval의 -1(해당 조건에 ground truth 없음)을 뺀 평균입니다.

    유효한 값이 하나도 없으면 ``0.0``이 아니라 ``None``을 돌려줍니다.
    """
    valid = np.asarray(values, dtype=float)
    valid = valid[valid > -1]
    return float(valid.mean()) if valid.size else None


def _ratio(numerator: int, denominator: int) -> float | None:
    """분모가 0이면 ``0.0``이 아니라 ``None``입니다."""
    return float(numerator) / float(denominator) if denominator > 0 else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def summarize_matches(
    eval_imgs: Sequence[Any],
    t_index: int,
    score_threshold: float,
    *,
    category_id: int | None = None,
) -> dict[str, Any]:
    """IoU 임계값 ``t_index``, score 임계값에서의 TP/FP/FN을 집계합니다.

    ``dtMatches``는 score 내림차순 greedy 매칭 결과이므로, 매칭 이후에 저score
    예측을 걸러내는 것은 매칭 이전에 걸러내는 것과 동일합니다. (낮은 score 예측은
    높은 score 예측이 이미 가져간 ground truth를 뺏을 수 없습니다.)

    ``evalImgs``의 detection 목록은 ``evaluateImg``가 이미 maxDet으로 잘라 두었기
    때문에 이미지당 최대 detection 수가 자동으로 반영됩니다.
    """
    true_positive = 0
    false_positive = 0
    truth_count = 0

    for entry in eval_imgs:
        if entry is None:
            continue
        if list(entry["aRng"]) != _AREA_RANGE_ALL:
            continue
        if category_id is not None and entry["category_id"] != category_id:
            continue

        gt_ignore = np.asarray(entry["gtIgnore"], dtype=bool)
        truth_count += int((~gt_ignore).sum())

        scores = np.asarray(entry["dtScores"], dtype=float)
        if scores.size == 0:
            continue
        dt_ignore = np.asarray(entry["dtIgnore"], dtype=bool)
        dt_matches = np.asarray(entry["dtMatches"], dtype=float)
        keep = (scores >= score_threshold) & ~dt_ignore[t_index]
        matched = dt_matches[t_index][keep]
        true_positive += int((matched > 0).sum())
        false_positive += int((matched == 0).sum())

    false_negative = truth_count - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _image_index(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """image_key를 1부터의 정수로 바꿉니다.

    JSONL manifest의 image_id는 문자열일 수 있지만 pycocotools는 정수 id를
    요구하기 때문입니다.
    """
    return {record["image_key"]: index for index, record in enumerate(records, start=1)}


def _ground_truth_document(
    records: Sequence[Mapping[str, Any]],
    image_index: Mapping[str, int],
    category_ids: Sequence[int],
    class_names: Mapping[int, str],
) -> tuple[dict[str, Any], int]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for record in records:
        images.append(
            {
                "id": image_index[record["image_key"]],
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )
        for annotation in record["annotations"]:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index[record["image_key"]],
                    "category_id": int(annotation["category_id"]),
                    "bbox": [x, y, width, height],
                    # pycocotools가 요구하는 field입니다.
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    document = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": class_names.get(category_id, str(category_id))}
            for category_id in category_ids
        ],
    }
    return document, len(annotations)


def _detection_document(
    predictions: Sequence[Mapping[str, Any]], image_index: Mapping[str, int]
) -> list[dict[str, Any]]:
    return [
        {
            "image_id": image_index[prediction["image_key"]],
            "category_id": int(prediction["category_id"]),
            "bbox": [float(value) for value in prediction["bbox"]],
            "score": float(prediction["score"]),
        }
        for prediction in predictions
    ]


def _truth_counts(records: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        for annotation in record["annotations"]:
            category_id = int(annotation["category_id"])
            counts[category_id] = counts.get(category_id, 0) + 1
    return counts


def _prediction_counts(predictions: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for prediction in predictions:
        category_id = int(prediction["category_id"])
        counts[category_id] = counts.get(category_id, 0) + 1
    return counts


def _empty_analysis(truth_total: int) -> dict[str, Any]:
    """예측이 하나도 없을 때의 보조 지표입니다."""
    return {
        "score_threshold": ANALYSIS_SCORE_THRESHOLD,
        "by_iou": {
            label: {
                "tp": 0,
                "fp": 0,
                "fn": truth_total,
                "precision": None,
                "recall": _ratio(0, truth_total),
                "f1": None,
            }
            for label, _ in ANALYSIS_IOU_INDICES
        },
    }


def _report_without_predictions(
    records: Sequence[Mapping[str, Any]],
    iou_thresholds: Sequence[float],
    max_detections_per_image: int | None,
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    """예측이 0건이면 COCOeval을 돌리지 않고 직접 report를 만듭니다.

    ``loadRes``가 빈 목록에서 실패하기 때문입니다.
    """
    truth_counts = _truth_counts(records)
    annotation_count = sum(truth_counts.values())
    per_class = [
        {
            "category_id": category_id,
            "name": class_names.get(category_id, str(category_id)),
            # ground truth가 있는데 맞힌 게 없으므로 미계산(None)이 아니라 0.0입니다.
            "ap": 0.0,
            "ap50_95": 0.0,
            "ap50": 0.0,
            "ap75": 0.0,
            "precision50": None,
            "recall50": 0.0,
            "precision75": None,
            "recall75": 0.0,
            "tp50": 0,
            "fp50": 0,
            "fn50": truth_counts[category_id],
            "tp75": 0,
            "fp75": 0,
            "fn75": truth_counts[category_id],
            "truth_count": truth_counts[category_id],
            "prediction_count": 0,
        }
        for category_id in sorted(truth_counts)
    ]
    zero_if_scored: float | None = 0.0 if per_class else None
    return {
        "iou_thresholds": [float(value) for value in iou_thresholds],
        "iou_thresholds_all": list(COCO_IOU_THRESHOLDS),
        "max_detections_per_image": max_detections_per_image,
        "image_count": len(records),
        "annotation_count": annotation_count,
        "prediction_count": 0,
        "evaluated_class_count": len(per_class),
        "metrics": {
            "mAP": zero_if_scored,
            "mAP75_95": zero_if_scored,
            "mAP50_95": zero_if_scored,
            "mAP50": zero_if_scored,
            "mAP75": zero_if_scored,
            "precision50": None,
            "recall50": zero_if_scored,
            "precision75": None,
            "recall75": zero_if_scored,
        },
        "analysis": _empty_analysis(annotation_count),
        "per_class": per_class,
    }


def _run_cocoeval(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    max_detections: int,
) -> COCOeval:
    """COCOeval을 한 번 실행합니다.

    ``COCO()`` 생성자와 ``evaluate()``/``accumulate()``가 진행 상황을 표준출력으로
    찍기 때문에 삼킵니다. web pipeline이 subprocess 로그를 파싱하므로 오염되면
    안 됩니다. ``self.eval``은 ``accumulate()``가 채우므로 ``summarize()``는
    호출하지 않고 배열을 직접 인덱싱합니다.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = dict(ground_truth)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(detections))

        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.iouThrs = np.asarray(COCO_IOU_THRESHOLDS, dtype=float)
        # 단일 원소이므로 아래에서 M축 인덱스는 항상 0입니다.
        evaluator.params.maxDets = [max_detections]
        evaluator.params.areaRng = [list(_AREA_RANGE_ALL)]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


def evaluate_detections(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    class_names: Mapping[int, str] | None = None,
    max_detections_per_image: int | None = None,
) -> dict[str, Any]:
    """Ground truth와 예측을 비교해 COCO 스타일 metric을 계산합니다.

    ``predictions``는 이미 score threshold와 이미지당 최대 detection 수가 적용된
    목록이라고 가정합니다.
    """
    main_thresholds = tuple(float(value) for value in iou_thresholds)
    if main_thresholds != DEFAULT_IOU_THRESHOLDS:
        raise MetricError(
            "메인 지표 IoU 구간은 "
            f"{list(DEFAULT_IOU_THRESHOLDS)}로 고정입니다: {list(main_thresholds)}"
        )

    names = dict(class_names or {})
    if not predictions:
        return _report_without_predictions(
            records, main_thresholds, max_detections_per_image, names
        )

    truth_counts = _truth_counts(records)
    prediction_counts = _prediction_counts(predictions)
    category_ids = sorted(set(truth_counts) | set(prediction_counts))
    image_index = _image_index(records)

    ground_truth, annotation_count = _ground_truth_document(
        records, image_index, category_ids, names
    )
    detections = _detection_document(predictions, image_index)
    max_detections = (
        _UNLIMITED_DETECTIONS
        if max_detections_per_image is None
        else int(max_detections_per_image)
    )

    try:
        evaluator = _run_cocoeval(ground_truth, detections, max_detections)
    except MetricError:
        raise
    except Exception as error:  # pycocotools가 던지는 예외 유형이 다양합니다.
        raise MetricError(f"pycocotools metric 계산에 실패했습니다: {error}") from error

    # precision: [T, R, K, A, M] / recall: [T, K, A, M]. A와 M은 크기 1이라 0 고정.
    precision = evaluator.eval["precision"]
    evaluated_ids = list(evaluator.params.catIds)

    analysis_by_iou: dict[str, Any] = {}
    per_class_matches: dict[str, dict[int, dict[str, Any]]] = {}
    for label, t_index in ANALYSIS_IOU_INDICES:
        analysis_by_iou[label] = summarize_matches(
            evaluator.evalImgs, t_index, ANALYSIS_SCORE_THRESHOLD
        )
        per_class_matches[label] = {
            category_id: summarize_matches(
                evaluator.evalImgs,
                t_index,
                ANALYSIS_SCORE_THRESHOLD,
                category_id=category_id,
            )
            for category_id in category_ids
        }

    per_class: list[dict[str, Any]] = []
    for category_id in category_ids:
        if category_id in evaluated_ids:
            k = evaluated_ids.index(category_id)
            class_precision = precision[:, :, k, 0, 0]
            entry_ap = _mean_or_none(class_precision[MAIN_SLICE])
            entry_ap50_95 = _mean_or_none(class_precision)
            entry_ap50 = _mean_or_none(class_precision[IDX_50])
            entry_ap75 = _mean_or_none(class_precision[IDX_75])
        else:
            entry_ap = entry_ap50_95 = entry_ap50 = entry_ap75 = None

        matches_50 = per_class_matches["0.50"][category_id]
        matches_75 = per_class_matches["0.75"][category_id]
        per_class.append(
            {
                "category_id": category_id,
                "name": names.get(category_id, str(category_id)),
                "ap": entry_ap,
                "ap50_95": entry_ap50_95,
                "ap50": entry_ap50,
                "ap75": entry_ap75,
                "precision50": matches_50["precision"],
                "recall50": matches_50["recall"],
                "precision75": matches_75["precision"],
                "recall75": matches_75["recall"],
                "tp50": matches_50["tp"],
                "fp50": matches_50["fp"],
                "fn50": matches_50["fn"],
                "tp75": matches_75["tp"],
                "fp75": matches_75["fp"],
                "fn75": matches_75["fn"],
                "truth_count": truth_counts.get(category_id, 0),
                "prediction_count": prediction_counts.get(category_id, 0),
            }
        )

    return {
        "iou_thresholds": list(main_thresholds),
        "iou_thresholds_all": list(COCO_IOU_THRESHOLDS),
        "max_detections_per_image": max_detections_per_image,
        "image_count": len(records),
        "annotation_count": annotation_count,
        "prediction_count": len(predictions),
        "evaluated_class_count": sum(1 for count in truth_counts.values() if count > 0),
        "metrics": {
            "mAP": _mean_or_none(precision[MAIN_SLICE, :, :, 0, 0]),
            "mAP75_95": _mean_or_none(precision[MAIN_SLICE, :, :, 0, 0]),
            "mAP50_95": _mean_or_none(precision[:, :, :, 0, 0]),
            "mAP50": _mean_or_none(precision[IDX_50, :, :, 0, 0]),
            "mAP75": _mean_or_none(precision[IDX_75, :, :, 0, 0]),
            # score>=0.5 기준 집계값입니다 (이전: PR 곡선의 마지막 점).
            "precision50": analysis_by_iou["0.50"]["precision"],
            "recall50": analysis_by_iou["0.50"]["recall"],
            "precision75": analysis_by_iou["0.75"]["precision"],
            "recall75": analysis_by_iou["0.75"]["recall"],
        },
        "analysis": {
            "score_threshold": ANALYSIS_SCORE_THRESHOLD,
            "by_iou": analysis_by_iou,
        },
        "per_class": per_class,
    }


__all__ = [
    "ANALYSIS_SCORE_THRESHOLD",
    "COCO_IOU_THRESHOLDS",
    "DEFAULT_IOU_THRESHOLDS",
    "evaluate_detections",
    "filter_predictions",
    "summarize_matches",
]
