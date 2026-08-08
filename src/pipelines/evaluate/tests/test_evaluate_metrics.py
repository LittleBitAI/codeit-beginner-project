"""Detection metric 계산 test입니다."""

from __future__ import annotations

import pytest

from src.pipelines.evaluate.metrics import (
    ANALYSIS_SCORE_THRESHOLD,
    COCO_IOU_THRESHOLDS,
    DEFAULT_IOU_THRESHOLDS,
    evaluate_detections,
    filter_predictions,
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


# --- fixture 3종 -----------------------------------------------------------


@pytest.fixture
def perfect() -> tuple[list[dict], list[dict]]:
    """Ground truth와 예측이 완전히 일치합니다."""
    records = [
        _record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
        _record("img-2", [{"category_id": 2, "bbox": [50, 50, 20, 20]}]),
    ]
    predictions = [
        _prediction("img-1", 1, [10, 10, 20, 20], 0.9),
        _prediction("img-2", 2, [50, 50, 20, 20], 0.8),
    ]
    return records, predictions


@pytest.fixture
def offset() -> tuple[list[dict], list[dict]]:
    """박스를 의도적으로 어긋나게 두어 IoU 구간별로 결과가 갈립니다."""
    records = [_record("img-1", [{"category_id": 1, "bbox": [0, 0, 10, 10]}])]
    predictions = [_prediction("img-1", 1, [2, 0, 10, 10], 0.9)]  # IoU = 8 / 12
    return records, predictions


@pytest.fixture
def messy() -> tuple[list[dict], list[dict]]:
    """실패 유형을 한곳에 모은 fixture입니다.

    - img-1: 오분류 (class 1 ground truth를 class 2로 예측)
    - img-2: 미탐지 (예측 없음)
    - img-3: ground truth에 붙었지만 score가 기준 미만 → P0 회귀에 필요합니다
    - img-4: 이미지당 최대 detection 수(4)를 넘는 예측 6개
    """
    records = [
        _record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
        _record("img-2", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
        _record("img-3", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
        _record("img-4", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
    ]
    predictions = [
        _prediction("img-1", 2, [10, 10, 20, 20], 0.9),
        _prediction("img-3", 1, [10, 10, 20, 20], 0.3),
        *[
            _prediction("img-4", 1, [10 + step, 10, 20, 20], 0.9 - step / 100)
            for step in range(6)
        ],
    ]
    return records, predictions


def _evaluate(bundle, **kwargs) -> dict:
    records, predictions = bundle
    return evaluate_detections(records, predictions, **kwargs)


# --- High 테스트 -----------------------------------------------------------


def test_main_metric_defaults_to_competition_interval(perfect):
    """옵션 없이 실행하면 메인 지표가 mAP@[0.75:0.95]여야 합니다."""
    report = _evaluate(perfect)

    assert report["iou_thresholds"] == [0.75, 0.8, 0.85, 0.9, 0.95]
    assert DEFAULT_IOU_THRESHOLDS == (0.75, 0.80, 0.85, 0.90, 0.95)
    assert report["iou_thresholds_all"] == list(COCO_IOU_THRESHOLDS)
    assert len(COCO_IOU_THRESHOLDS) == 10


def test_perfect_predictions_reach_full_map(perfect):
    """계산된(None이 아닌) 모든 mAP가 1.0이고 헛짚음도 놓침도 없어야 합니다."""
    report = _evaluate(perfect)

    computed = {
        key: value
        for key, value in report["metrics"].items()
        if value is not None and key.startswith("mAP")
    }
    assert computed, "mAP가 하나도 계산되지 않았습니다"
    for key, value in computed.items():
        assert value == pytest.approx(1.0), key

    for label in ("0.50", "0.75"):
        assert report["analysis"]["by_iou"][label]["fp"] == 0
        assert report["analysis"]["by_iou"][label]["fn"] == 0


def test_map50_95_averages_all_ten_coco_points(offset):
    """mAP@[0.50:0.95]는 0.55·0.65를 포함한 10점 평균이어야 합니다."""
    report = _evaluate(offset)
    metrics = report["metrics"]

    # IoU = 8/12 ≈ 0.667 → 0.50~0.65에서만 맞고 0.70 이상에서는 틀립니다.
    # 10점 중 4점만 1.0이므로 부분집합 배열로는 나올 수 없는 값입니다.
    assert metrics["mAP50_95"] == pytest.approx(0.4, abs=1e-6)
    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["mAP75"] == pytest.approx(0.0)
    assert metrics["mAP"] == pytest.approx(0.0)


def test_missing_ground_truth_reports_null_not_zero():
    """ground truth가 없는 class는 0.0이 아니라 None이어야 합니다."""
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]
    predictions = [
        _prediction("img-1", 1, [10, 10, 20, 20], 0.9),
        _prediction("img-1", 2, [50, 50, 20, 20], 0.7),
    ]

    report = evaluate_detections(records, predictions)

    ghost = next(item for item in report["per_class"] if item["category_id"] == 2)
    assert ghost["ap"] is None
    assert ghost["ap50"] is None
    assert ghost["prediction_count"] == 1
    assert report["evaluated_class_count"] == 1

    # 반대 방향: 계산은 했지만 맞힌 게 없으면 None이 아니라 0.0입니다.
    scored = next(item for item in report["per_class"] if item["category_id"] == 1)
    assert scored["ap"] == pytest.approx(1.0)
    miss = evaluate_detections(
        [_record("img-1", [{"category_id": 1, "bbox": [0, 0, 10, 10]}])],
        [_prediction("img-1", 1, [80, 80, 10, 10], 0.9)],
    )
    assert miss["metrics"]["mAP50"] == 0.0
    assert miss["metrics"]["mAP50"] is not None


def test_match_counts_are_consistent(messy):
    """evalImgs 집계와 manifest에서 따로 센 값이 일치해야 합니다."""
    records, predictions = messy
    max_detections = 4
    report = evaluate_detections(
        records, predictions, max_detections_per_image=max_detections
    )

    # ground truth 쪽: COCOeval을 거치지 않은 manifest 계산과 비교합니다.
    annotation_count = sum(len(record["annotations"]) for record in records)
    assert report["annotation_count"] == annotation_count

    # 예측 쪽: score 기준과 이미지당 상위 N개를 적용한 뒤의 수와 비교합니다.
    surviving = len(
        filter_predictions(
            [item for item in predictions if item["score"] >= ANALYSIS_SCORE_THRESHOLD],
            max_detections_per_image=max_detections,
        )
    )
    for label in ("0.50", "0.75"):
        counts = report["analysis"]["by_iou"][label]
        assert counts["tp"] + counts["fp"] == surviving, label
        assert counts["tp"] + counts["fn"] == annotation_count, label


# --- Medium 테스트 ---------------------------------------------------------


def test_iou_interval_separates_loose_and_strict_matches(offset):
    report = _evaluate(offset)

    per_class = report["per_class"][0]
    assert per_class["ap50"] == pytest.approx(1.0)
    assert per_class["ap75"] == pytest.approx(0.0)
    assert report["metrics"]["mAP50"] > report["metrics"]["mAP75"]


def test_zero_denominator_reports_null():
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]

    no_predictions = evaluate_detections(records, [])
    assert no_predictions["metrics"]["precision50"] is None
    assert no_predictions["prediction_count"] == 0
    assert no_predictions["per_class"][0]["truth_count"] == 1

    no_truth = evaluate_detections(
        [_record("img-1", [])], [_prediction("img-1", 1, [10, 10, 20, 20], 0.9)]
    )
    assert no_truth["metrics"]["recall50"] is None


def test_cocoeval_does_not_write_to_stdout(perfect, capsys):
    """pycocotools 출력이 새면 web이 파싱하는 subprocess 로그가 오염됩니다."""
    _evaluate(perfect)

    captured = capsys.readouterr()
    assert captured.out == ""


# --- 유지되는 기존 동작 ----------------------------------------------------


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


def test_filter_predictions_drops_excluded_categories_before_the_per_image_limit():
    """제외 대상은 이미지당 상한을 적용하기 **전에** 버립니다.

    상한 뒤에 버리면 제외 대상이 상한 칸을 차지한 뒤 사라져, 그 이미지에서 남는
    예측이 상한보다 적어집니다. 제출 행 수가 그만큼 줄어듭니다.
    """

    predictions = [
        _prediction("img-1", 999999, [0, 0, 10, 10], 0.95),
        _prediction("img-1", 1, [10, 0, 10, 10], 0.9),
        _prediction("img-1", 2, [20, 0, 10, 10], 0.8),
    ]

    filtered = filter_predictions(
        predictions, max_detections_per_image=2, excluded_category_ids={999999}
    )

    assert [item["category_id"] for item in filtered] == [1, 2]


def test_filter_predictions_keeps_every_category_by_default():
    """제외 목록을 주지 않으면 동작이 지금과 같습니다."""

    predictions = [
        _prediction("img-1", 999999, [0, 0, 10, 10], 0.9),
        _prediction("img-1", 1, [10, 0, 10, 10], 0.8),
    ]

    assert len(filter_predictions(predictions)) == 2
