"""COCO 스타일 object detection metric을 numpy로 계산합니다.

pycocotools를 추가하지 않고 같은 정의를 따릅니다.

- IoU threshold 0.50부터 0.95까지 0.05 간격, 총 10개
- 101-point interpolated average precision
- class별 AP를 평균해 mAP를 계산하며, ground truth가 없는 class는 제외
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def _to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(float).copy()
    converted[:, 2] = converted[:, 0] + converted[:, 2]
    converted[:, 3] = converted[:, 1] + converted[:, 3]
    return converted


def iou_matrix(prediction_boxes: np.ndarray, truth_boxes: np.ndarray) -> np.ndarray:
    """[x, y, width, height] box 사이의 IoU 행렬을 계산합니다."""
    if len(prediction_boxes) == 0 or len(truth_boxes) == 0:
        return np.zeros((len(prediction_boxes), len(truth_boxes)), dtype=float)

    predictions = _to_xyxy(np.asarray(prediction_boxes, dtype=float).reshape(-1, 4))
    truths = _to_xyxy(np.asarray(truth_boxes, dtype=float).reshape(-1, 4))

    top_left = np.maximum(predictions[:, None, :2], truths[None, :, :2])
    bottom_right = np.minimum(predictions[:, None, 2:], truths[None, :, 2:])
    side = np.clip(bottom_right - top_left, 0.0, None)
    intersection = side[..., 0] * side[..., 1]

    prediction_area = (
        (predictions[:, 2] - predictions[:, 0]) * (predictions[:, 3] - predictions[:, 1])
    )[:, None]
    truth_area = ((truths[:, 2] - truths[:, 0]) * (truths[:, 3] - truths[:, 1]))[None, :]
    union = prediction_area + truth_area - intersection
    return np.where(union > 0, intersection / np.where(union > 0, union, 1.0), 0.0)


def _average_precision(matched: np.ndarray, truth_count: int) -> tuple[float, float, float]:
    """정렬된 매칭 결과에서 AP, precision, recall을 계산합니다."""
    if truth_count <= 0:
        return 0.0, 0.0, 0.0
    if matched.size == 0:
        return 0.0, 0.0, 0.0

    true_positive = np.cumsum(matched)
    false_positive = np.cumsum(~matched)
    recall = true_positive / truth_count
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)

    monotone_precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(
        indices < monotone_precision.size,
        monotone_precision[np.clip(indices, 0, monotone_precision.size - 1)],
        0.0,
    )
    return float(sampled.mean()), float(precision[-1]), float(recall[-1])


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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


def _match_class(
    detections: list[dict[str, Any]],
    truths_by_image: Mapping[str, np.ndarray],
    iou_thresholds: Sequence[float],
) -> dict[float, np.ndarray]:
    """Score 내림차순으로 detection을 ground truth에 greedy 매칭합니다."""
    order = sorted(range(len(detections)), key=lambda index: -float(detections[index]["score"]))
    rows_by_image: dict[str, list[int]] = {}
    for index in order:
        rows_by_image.setdefault(detections[index]["image_key"], []).append(index)

    ious_by_image: dict[str, np.ndarray] = {}
    row_of_detection: dict[int, int] = {}
    for image_key, indices in rows_by_image.items():
        truth_boxes = truths_by_image.get(image_key)
        if truth_boxes is None or len(truth_boxes) == 0:
            continue
        boxes = np.asarray([detections[index]["bbox"] for index in indices], dtype=float)
        ious_by_image[image_key] = iou_matrix(boxes, truth_boxes)
        for row, index in enumerate(indices):
            row_of_detection[index] = row

    matched_by_threshold: dict[float, np.ndarray] = {}
    for threshold in iou_thresholds:
        used = {key: np.zeros(len(boxes), dtype=bool) for key, boxes in truths_by_image.items()}
        matched = np.zeros(len(order), dtype=bool)
        for position, index in enumerate(order):
            image_key = detections[index]["image_key"]
            ious = ious_by_image.get(image_key)
            if ious is None:
                continue
            candidates = np.where(used[image_key], -1.0, ious[row_of_detection[index]])
            if candidates.size == 0:
                continue
            best = int(np.argmax(candidates))
            if candidates[best] >= threshold:
                used[image_key][best] = True
                matched[position] = True
        matched_by_threshold[threshold] = matched
    return matched_by_threshold


def evaluate_detections(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    class_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Ground truth와 예측을 비교해 COCO 스타일 metric을 계산합니다.

    `predictions`는 이미 score threshold와 이미지당 최대 detection 수가 적용된
    목록이라고 가정합니다.
    """
    thresholds = [float(threshold) for threshold in iou_thresholds]
    names = dict(class_names or {})

    truths_by_class: dict[int, dict[str, list[list[float]]]] = {}
    annotation_count = 0
    for record in records:
        for annotation in record["annotations"]:
            category_id = int(annotation["category_id"])
            per_image = truths_by_class.setdefault(category_id, {})
            per_image.setdefault(record["image_key"], []).append(list(annotation["bbox"]))
            annotation_count += 1

    detections_by_class: dict[int, list[dict[str, Any]]] = {}
    for prediction in predictions:
        detections_by_class.setdefault(int(prediction["category_id"]), []).append(dict(prediction))

    per_class: list[dict[str, Any]] = []
    for category_id in sorted(set(truths_by_class) | set(detections_by_class)):
        truths_by_image = {
            image_key: np.asarray(boxes, dtype=float)
            for image_key, boxes in truths_by_class.get(category_id, {}).items()
        }
        truth_count = sum(len(boxes) for boxes in truths_by_image.values())
        detections = detections_by_class.get(category_id, [])
        matched_by_threshold = (
            _match_class(detections, truths_by_image, thresholds) if detections else {}
        )

        results = {}
        for threshold in thresholds:
            matched = matched_by_threshold.get(threshold, np.zeros(0, dtype=bool))
            results[threshold] = _average_precision(matched, truth_count)

        average_precisions = [results[threshold][0] for threshold in thresholds]
        precision_50, recall_50 = results.get(0.5, (0.0, 0.0, 0.0))[1:]
        per_class.append(
            {
                "category_id": category_id,
                "name": names.get(category_id, str(category_id)),
                "ap": _mean(average_precisions) if truth_count else None,
                "ap50": results[0.5][0] if 0.5 in results and truth_count else None,
                "ap75": results[0.75][0] if 0.75 in results and truth_count else None,
                "precision50": precision_50 if truth_count else None,
                "recall50": recall_50 if truth_count else None,
                "truth_count": truth_count,
                "prediction_count": len(detections),
            }
        )

    scored = [entry for entry in per_class if entry["truth_count"] > 0]
    return {
        "iou_thresholds": thresholds,
        "image_count": len(records),
        "annotation_count": annotation_count,
        "prediction_count": len(predictions),
        "evaluated_class_count": len(scored),
        "metrics": {
            "mAP": _mean([entry["ap"] for entry in scored]),
            "mAP50": _mean([entry["ap50"] for entry in scored if entry["ap50"] is not None]),
            "mAP75": _mean([entry["ap75"] for entry in scored if entry["ap75"] is not None]),
            "precision50": _mean([entry["precision50"] for entry in scored]),
            "recall50": _mean([entry["recall50"] for entry in scored]),
        },
        "per_class": per_class,
    }


__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "evaluate_detections",
    "filter_predictions",
    "iou_matrix",
]
