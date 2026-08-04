"""Detection metric 계산 test입니다."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipelines.evaluate.metrics import (
    DEFAULT_IOU_THRESHOLDS,
    evaluate_detections,
    filter_predictions,
    iou_matrix,
)


def _record(image_key: str, annotations: list[dict]) -> dict:
    return {
        "image_id": image_key,
        "image_key": image_key,
        "image_uri": f"data/val/{image_key}.jpg",
        "width": 100,
        "height": 100,
        "annotations": annotations,
    }


def _prediction(image_key: str, category_id: int, bbox: list[float], score: float) -> dict:
    return {
        "image_id": image_key,
        "image_key": image_key,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
    }


def test_default_iou_thresholds_follow_coco():
    assert DEFAULT_IOU_THRESHOLDS[0] == 0.5
    assert DEFAULT_IOU_THRESHOLDS[-1] == 0.95
    assert len(DEFAULT_IOU_THRESHOLDS) == 10


def test_iou_matrix_matches_manual_calculation():
    predictions = np.array([[0, 0, 10, 10]], dtype=float)
    truths = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [50, 50, 10, 10]], dtype=float)

    result = iou_matrix(predictions, truths)

    assert result[0][0] == pytest.approx(1.0)
    assert result[0][1] == pytest.approx(50 / 150)
    assert result[0][2] == pytest.approx(0.0)


def test_perfect_predictions_reach_full_map():
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]
    predictions = [_prediction("img-1", 1, [10, 10, 20, 20], 0.9)]

    report = evaluate_detections(records, predictions)

    assert report["metrics"]["mAP"] == pytest.approx(1.0)
    assert report["metrics"]["mAP50"] == pytest.approx(1.0)
    assert report["metrics"]["precision50"] == pytest.approx(1.0)
    assert report["metrics"]["recall50"] == pytest.approx(1.0)


def test_no_predictions_gives_zero_metrics():
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]

    report = evaluate_detections(records, [])

    assert report["metrics"]["mAP"] == 0.0
    assert report["prediction_count"] == 0
    assert report["per_class"][0]["truth_count"] == 1


def test_partial_overlap_counts_only_at_low_iou_thresholds():
    records = [_record("img-1", [{"category_id": 1, "bbox": [0, 0, 10, 10]}])]
    predictions = [_prediction("img-1", 1, [2, 0, 10, 10], 0.9)]  # IoU = 8 / 12

    report = evaluate_detections(records, predictions, iou_thresholds=[0.5, 0.75])

    per_class = report["per_class"][0]
    assert per_class["ap50"] == pytest.approx(1.0)
    assert per_class["ap75"] == pytest.approx(0.0)
    assert report["metrics"]["mAP"] == pytest.approx(0.5)


def test_duplicate_detection_is_counted_as_false_positive():
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]
    predictions = [
        _prediction("img-1", 1, [10, 10, 20, 20], 0.9),
        _prediction("img-1", 1, [10, 10, 20, 20], 0.8),
    ]

    report = evaluate_detections(records, predictions, iou_thresholds=[0.5])

    assert report["per_class"][0]["precision50"] == pytest.approx(0.5)
    assert report["per_class"][0]["recall50"] == pytest.approx(1.0)


def test_class_without_ground_truth_is_excluded_from_map():
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]
    predictions = [
        _prediction("img-1", 1, [10, 10, 20, 20], 0.9),
        _prediction("img-1", 2, [50, 50, 20, 20], 0.7),
    ]

    report = evaluate_detections(records, predictions, iou_thresholds=[0.5])

    assert report["evaluated_class_count"] == 1
    assert report["metrics"]["mAP"] == pytest.approx(1.0)
    ghost_class = next(entry for entry in report["per_class"] if entry["category_id"] == 2)
    assert ghost_class["ap"] is None
    assert ghost_class["prediction_count"] == 1


def test_class_names_are_taken_from_class_map():
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]

    report = evaluate_detections(records, [], class_names={1: "tylenol"})

    assert report["per_class"][0]["name"] == "tylenol"


def test_filter_predictions_applies_score_threshold_and_top_k():
    predictions = [
        _prediction("img-1", 1, [0, 0, 10, 10], 0.9),
        _prediction("img-1", 1, [10, 0, 10, 10], 0.8),
        _prediction("img-1", 1, [20, 0, 10, 10], 0.2),
        _prediction("img-2", 1, [0, 0, 10, 10], 0.7),
    ]

    filtered = filter_predictions(predictions, score_threshold=0.5, max_detections_per_image=1)

    assert [(item["image_key"], item["score"]) for item in filtered] == [
        ("img-1", 0.9),
        ("img-2", 0.7),
    ]


def test_filter_predictions_without_limit_keeps_every_detection():
    predictions = [_prediction("img-1", 1, [0, 0, 10, 10], score) for score in (0.9, 0.8, 0.7)]

    assert len(filter_predictions(predictions, max_detections_per_image=None)) == 3
