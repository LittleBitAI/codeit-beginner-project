"""평가 결과 분석 도구 test입니다."""

from __future__ import annotations

import json

import pytest

from src.pipelines.evaluate.analysis import (
    MIN_TRUTH_COUNT_FOR_WEAK,
    SWEEP_THRESHOLDS,
    TOP_N_PER_CLASS_SUMMARY,
    best_f1_threshold,
    summarize_per_class,
)
from src.pipelines.evaluate.metrics import ANALYSIS_SCORE_THRESHOLD, evaluate_detections


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


@pytest.fixture
def failures() -> dict:
    """실패 유형을 하나씩 담은 평가 결과입니다.

    - img-1: class를 틀림 (classification)
    - img-2: 예측 없음 (미탐지)
    - img-3: 같은 class인데 박스가 어긋남 (localization)
    - img-4: 정탐 하나 + 아무것과도 안 겹치는 예측 하나 (background)
    - img-5: ground truth에 붙었지만 score가 기준 미만 (P0 회귀에 필요)
    """
    records = [
        _record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
        _record("img-2", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
        _record("img-3", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
        _record("img-4", [{"category_id": 2, "bbox": [10, 10, 20, 20]}]),
        _record("img-5", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
    ]
    predictions = [
        _prediction("img-1", 2, [10, 10, 20, 20], 0.90),
        _prediction("img-3", 1, [16, 16, 20, 20], 0.80),
        _prediction("img-4", 2, [10, 10, 20, 20], 0.95),
        _prediction("img-4", 1, [70, 70, 10, 10], 0.70),
        _prediction("img-5", 1, [10, 10, 20, 20], 0.30),
    ]
    return evaluate_detections(
        records,
        predictions,
        class_names={1: "tylenol", 2: "aspirin"},
        max_detections_per_image=4,
    )


# --- 1. Confusion Matrix ---------------------------------------------------


def test_confusion_matrix_separates_confusion_from_miss(failures):
    """class를 틀린 것과 아예 못 찾은 것이 다른 칸에 들어가야 합니다."""
    confusion = failures["analysis"]["confusion_matrix"]["0.50"]
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    tylenol = labels.index("tylenol")
    aspirin = labels.index("aspirin")

    assert labels[0] == "background"
    # tylenol ground truth를 aspirin으로 부른 건이 비대각 칸에 있습니다.
    assert matrix[tylenol][aspirin] == 1
    # 정탐은 대각선에 있습니다.
    assert matrix[aspirin][aspirin] == 1
    # 못 찾은 tylenol은 background 열(FN)에 있습니다.
    assert matrix[tylenol][0] > 0
    # 헛짚은 예측은 background 행(FP)에 있습니다.
    assert matrix[0][tylenol] > 0


def test_low_score_match_does_not_erase_ground_truth(failures):
    """score 기준에 걸린 예측이 붙은 ground truth가 행렬에서 사라지면 안 됩니다.

    gtMatches는 score 기준 적용 전 결과라, 그것으로 미탐지를 판정하면 img-5의
    ground truth가 어느 칸에도 들어가지 않고 증발합니다.
    """
    confusion = failures["analysis"]["confusion_matrix"]["0.50"]
    counts = failures["analysis"]["by_iou"]["0.50"]
    total = sum(sum(row) for row in confusion["matrix"])

    surviving_predictions = counts["tp"] + counts["fp"]
    unmatched_truth = sum(row[0] for row in confusion["matrix"])
    assert total == surviving_predictions + unmatched_truth

    # img-5의 ground truth 하나가 실제로 미탐지로 세어졌습니다.
    assert unmatched_truth == 3


# --- 2. Score Threshold Sweep ----------------------------------------------


def test_sweep_covers_every_threshold(failures):
    sweep = failures["analysis"]["score_sweep"]["0.50"]

    assert len(SWEEP_THRESHOLDS) == 19
    assert SWEEP_THRESHOLDS[0] == 0.05
    assert SWEEP_THRESHOLDS[-1] == 0.95
    assert sorted(sweep) == [f"{value:.2f}" for value in SWEEP_THRESHOLDS]
    for counts in sweep.values():
        assert set(counts) == {"tp", "fp", "fn", "precision", "recall", "f1"}


def test_raising_threshold_never_adds_true_positives(failures):
    """threshold를 올리는 것은 뒤쪽 detection을 잘라 내는 것과 같습니다."""
    sweep = failures["analysis"]["score_sweep"]["0.50"]
    ordered = [sweep[f"{value:.2f}"] for value in SWEEP_THRESHOLDS]

    for lower, higher in zip(ordered, ordered[1:]):
        assert higher["tp"] <= lower["tp"]
        assert higher["fp"] <= lower["fp"]
        assert higher["fn"] >= lower["fn"]


def test_sweep_matches_the_fixed_threshold_summary(failures):
    """sweep의 0.50 지점은 by_iou가 쓰는 고정 기준과 같은 값이어야 합니다."""
    sweep = failures["analysis"]["score_sweep"]["0.50"]
    fixed = failures["analysis"]["by_iou"]["0.50"]

    assert ANALYSIS_SCORE_THRESHOLD == 0.5
    assert sweep["0.50"] == fixed


# --- 3. Best F1 Threshold --------------------------------------------------


def test_best_f1_picks_the_highest_score(failures):
    best = failures["analysis"]["best_f1"]["0.50"]
    sweep = failures["analysis"]["score_sweep"]["0.50"]

    assert best["f1"] == max(
        counts["f1"] for counts in sweep.values() if counts["f1"] is not None
    )
    assert sweep[f"{best['threshold']:.2f}"]["precision"] == best["precision"]
    assert sweep[f"{best['threshold']:.2f}"]["recall"] == best["recall"]


def test_best_f1_tie_breaks_on_precision_then_lower_threshold():
    """F1이 같으면 precision이 높은 쪽, 그래도 같으면 threshold가 낮은 쪽입니다."""
    sweep = {
        0.10: {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        0.20: {"precision": 0.8, "recall": 0.36, "f1": 0.5},  # 같은 F1, 높은 precision
        0.30: {"precision": 0.8, "recall": 0.36, "f1": 0.5},  # 완전 동점, 더 높은 threshold
        0.40: {"precision": 0.9, "recall": 0.1, "f1": 0.4},
    }

    assert best_f1_threshold(sweep)["threshold"] == 0.20


def test_best_f1_is_null_when_nothing_is_computable():
    assert best_f1_threshold({0.5: {"precision": None, "recall": None, "f1": None}}) is None


# --- 4. 실패 이미지 분석 ---------------------------------------------------


def test_per_image_rows_are_sorted_by_failure_count(failures):
    rows = failures["analysis"]["per_image"]["0.50"]

    assert [row["image_id"] for row in rows][:1] == ["img-1"]
    failure_counts = [row["failure_count"] for row in rows]
    assert failure_counts == sorted(failure_counts, reverse=True)
    for row in rows:
        assert row["failure_count"] == row["fp"] + row["fn"]


def test_per_image_counts_are_consistent(failures):
    """이미지별 수치의 합이 전체 집계와 맞아야 합니다."""
    rows = failures["analysis"]["per_image"]["0.50"]
    total = failures["analysis"]["by_iou"]["0.50"]

    assert sum(row["tp"] for row in rows) == total["tp"]
    assert sum(row["fp"] for row in rows) == total["fp"]
    assert sum(row["fn"] for row in rows) == total["fn"]
    assert sum(row["truth_count"] for row in rows) == failures["annotation_count"]
    for row in rows:
        assert row["prediction_count"] == row["tp"] + row["fp"]


# --- 5. Localization vs Classification Error -------------------------------


def test_false_positives_are_split_by_cause(failures):
    """준비한 실패 유형이 각각 제 칸으로 가야 합니다."""
    breakdown = failures["analysis"]["error_breakdown"]["0.50"]

    assert breakdown["classification"] == 1  # img-1: 위치는 맞고 class가 틀림
    assert breakdown["localization"] == 1  # img-3: class는 맞고 박스가 어긋남
    assert breakdown["background"] == 1  # img-4: 아무것과도 안 겹침
    assert breakdown["duplicate"] == 0


def test_error_breakdown_accounts_for_every_false_positive(failures):
    """분류에서 새는 false positive가 없어야 합니다."""
    breakdown = failures["analysis"]["error_breakdown"]["0.50"]

    assert sum(breakdown.values()) == failures["analysis"]["by_iou"]["0.50"]["fp"]


def test_duplicate_detection_is_counted_separately():
    """같은 class를 잘 맞혔는데 중복인 예측은 별도 칸으로 갑니다.

    요구된 세 가지 중 어디에도 맞지 않아, 이 칸이 없으면 다른 항목이 부풀려집니다.
    """
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]
    predictions = [
        _prediction("img-1", 1, [10, 10, 20, 20], 0.9),
        _prediction("img-1", 1, [10, 10, 20, 20], 0.8),  # 같은 곳을 한 번 더
    ]

    report = evaluate_detections(records, predictions, max_detections_per_image=4)
    breakdown = report["analysis"]["error_breakdown"]["0.50"]

    assert breakdown["duplicate"] == 1
    assert breakdown["localization"] == 0
    assert breakdown["background"] == 0
    assert sum(breakdown.values()) == report["analysis"]["by_iou"]["0.50"]["fp"]


# --- 6. class별 요약 -------------------------------------------------------


def _class_entry(
    category_id: int, *, ap: float | None, truth_count: int, prediction_count: int = 0
) -> dict:
    """per_class 항목 중 요약이 읽는 field만 갖춘 최소 dict입니다."""
    return {
        "category_id": category_id,
        "name": f"class-{category_id}",
        "ap": ap,
        "ap50": ap,
        "ap75": ap,
        "truth_count": truth_count,
        "prediction_count": prediction_count,
    }


def test_truth_count_boundary_splits_weak_from_sparse():
    """정답 수 4는 weak, 1~3은 sparse입니다. 둘은 겹치지 않습니다.

    sparse의 AP는 표본이 적어 믿을 수 없으므로, AP가 아니라 표본 수가 앞자리
    기준입니다. 그래서 AP 순서와 표본 순서를 일부러 어긋나게 두었습니다.
    """
    assert MIN_TRUTH_COUNT_FOR_WEAK == 4
    summary = summarize_per_class(
        [
            _class_entry(1, ap=0.2, truth_count=4),
            _class_entry(2, ap=0.1, truth_count=3),
            _class_entry(3, ap=0.9, truth_count=1),
        ]
    )

    assert [row["category_id"] for row in summary["weak"]] == [1]
    # AP가 가장 높아도 표본이 가장 적은 3이 앞입니다.
    assert [row["category_id"] for row in summary["sparse"]] == [3, 2]


def test_unmeasured_separates_classes_without_truth():
    """정답이 없어 측정되지 않은 class는 약한 class로 취급하지 않습니다.

    unmeasured는 줄 세울 기준이 없어(category_id 순) 잘라 내면 정보만 사라지므로
    top N을 적용하지 않습니다. 그래서 일부러 top N보다 많이 넣습니다.
    """
    ghosts = [
        _class_entry(order, ap=None, truth_count=0, prediction_count=1)
        for order in range(2, 2 + TOP_N_PER_CLASS_SUMMARY + 1)
    ]

    summary = summarize_per_class([_class_entry(1, ap=0.4, truth_count=8), *ghosts])

    assert [row["category_id"] for row in summary["weak"]] == [1]
    assert [row["category_id"] for row in summary["unmeasured"]] == [
        row["category_id"] for row in ghosts
    ]
    assert summary["counts"] == {"weak": 1, "sparse": 0, "unmeasured": len(ghosts)}
    # ap=0으로 치환하지 않고 원래 값을 그대로 냅니다.
    assert summary["unmeasured"][0]["ap"] is None


def test_null_ap_never_leads_weak_or_sparse():
    """ap=null을 0으로 보면 측정되지 않은 class가 항상 최악으로 보입니다.

    정답 수가 충분한데 ap가 null인 경우는 실제로는 나오지 않지만, 그 가정이
    깨져도 null이 최저 AP 자리를 차지하지 않아야 합니다.
    """
    summary = summarize_per_class(
        [
            _class_entry(1, ap=None, truth_count=10),
            _class_entry(2, ap=0.3, truth_count=10),
            _class_entry(3, ap=None, truth_count=2),
            _class_entry(4, ap=0.3, truth_count=2),
        ]
    )

    assert [row["category_id"] for row in summary["weak"]] == [2, 1]
    assert [row["category_id"] for row in summary["sparse"]] == [4, 3]
    assert summary["weak"][-1]["ap"] is None


def test_tie_breakers_follow_the_documented_order():
    """AP가 같으면 표본이 많은 쪽, 완전 동점이면 category_id가 작은 쪽입니다."""
    summary = summarize_per_class(
        [
            _class_entry(7, ap=0.2, truth_count=5),
            _class_entry(3, ap=0.2, truth_count=20),
            _class_entry(1, ap=0.2, truth_count=5),
        ]
    )

    assert [row["category_id"] for row in summary["weak"]] == [3, 1, 7]


def test_top_n_keeps_the_lowest_ap_and_counts_report_the_total():
    """AP가 가장 낮은 쪽만 남기고, 몇 개가 잘렸는지는 알 수 있어야 합니다.

    category_id 순서와 AP 순서를 일부러 어긋나게 두어, 잘라 내기 전에 AP
    오름차순으로 정렬했는지까지 이 test가 함께 지킵니다.
    """
    entries = [
        _class_entry(order, ap=(8 - order) / 100, truth_count=10) for order in range(1, 8)
    ]

    summary = summarize_per_class(entries)
    assert TOP_N_PER_CLASS_SUMMARY == 5
    # AP가 가장 낮은 다섯 개입니다. category_id 순서가 아닙니다.
    assert [row["category_id"] for row in summary["weak"]] == [7, 6, 5, 4, 3]
    assert summary["counts"]["weak"] == 7

    smaller = summarize_per_class(entries, top_n=2)
    assert [row["category_id"] for row in smaller["weak"]] == [7, 6]
    assert smaller["top_n"] == 2
    assert smaller["counts"]["weak"] == 7


def test_summary_reuses_per_class_without_reordering_it():
    """요약을 붙여도 per_class 배열은 category_id 순서를 유지해야 합니다."""
    records = [
        _record("img-1", [{"category_id": 2, "bbox": [10, 10, 20, 20]}]),
        _record("img-2", [{"category_id": 1, "bbox": [10, 10, 20, 20]}]),
    ]
    predictions = [_prediction("img-1", 2, [10, 10, 20, 20], 0.9)]

    report = evaluate_detections(records, predictions, class_names={1: "a", 2: "b"})
    summary = report["analysis"]["per_class_summary"]

    assert [item["category_id"] for item in report["per_class"]] == [1, 2]
    # 요약 row는 per_class의 값을 그대로 옮긴 것입니다.
    for group in ("weak", "sparse", "unmeasured"):
        for row in summary[group]:
            source = next(
                item
                for item in report["per_class"]
                if item["category_id"] == row["category_id"]
            )
            assert row == {field: source[field] for field in row}
    # background는 per_class에 없으므로 요약에도 없습니다.
    assert all(
        row["name"] != "background"
        for group in ("weak", "sparse", "unmeasured")
        for row in summary[group]
    )


def test_per_class_summary_survives_the_empty_prediction_path():
    """예측이 0건이어도 같은 모양이어야 소비자가 분기하지 않습니다."""
    records = [
        _record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}] * 4),
        _record("img-2", [{"category_id": 2, "bbox": [10, 10, 20, 20]}] * 2),
    ]

    summary = evaluate_detections(records, [])["analysis"]["per_class_summary"]

    # 정답이 있는데 하나도 못 맞혔으므로 미계산(None)이 아니라 0.0입니다.
    assert [row["category_id"] for row in summary["weak"]] == [1]
    assert summary["weak"][0]["ap"] == 0.0
    assert [row["category_id"] for row in summary["sparse"]] == [2]
    assert summary["unmeasured"] == []


# --- 산출물 계약 -----------------------------------------------------------


def test_analysis_survives_json_serialization(failures):
    """metrics.json으로 저장되므로 직렬화가 가능해야 합니다."""
    text = json.dumps(failures, allow_nan=False)

    assert "NaN" not in text
    assert "Infinity" not in text


def test_analysis_shape_is_the_same_without_predictions():
    """예측이 없어도 같은 모양을 내야 소비자가 분기하지 않습니다."""
    records = [_record("img-1", [{"category_id": 1, "bbox": [10, 10, 20, 20]}])]

    empty = evaluate_detections(records, [])["analysis"]
    filled = evaluate_detections(
        records, [_prediction("img-1", 1, [10, 10, 20, 20], 0.9)]
    )["analysis"]

    assert set(empty) == set(filled)
    for key in ("confusion_matrix", "score_sweep", "best_f1", "per_image", "error_breakdown"):
        assert set(empty[key]) == set(filled[key]) == {"0.50", "0.75"}
    assert empty["best_f1"]["0.50"] is None
    assert empty["per_image"]["0.50"][0]["fn"] == 1
    # class별 요약은 IoU에 종속되지 않아 IoU별로 나누지 않습니다.
    for report in (empty, filled):
        assert set(report["per_class_summary"]) == {
            "min_truth_count",
            "top_n",
            "counts",
            "weak",
            "sparse",
            "unmeasured",
        }
