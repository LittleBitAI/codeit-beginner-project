"""평가 결과를 원인별로 파고드는 분석 도구입니다.

`metrics.py`가 낸 점수가 "얼마나 좋은가"를 답한다면, 이 모듈은 "왜 그런가"를
답합니다. 여섯 가지를 냅니다.

- score threshold sweep과 best F1 threshold: 운영에 쓸 confidence 기준 고르기
- confusion matrix: 어떤 class를 어떤 class로 잘못 불렀는지
- 이미지별 실패 분석: FP·FN이 몰린 이미지 찾기
- false positive 원인 분류: 위치가 틀렸는지, class가 틀렸는지, 헛본 것인지
- class별 요약: 표본이 충분한데 약한 class와 표본이 적어 못 믿을 class 가르기

매칭은 전부 pycocotools가 한 결과(`evalImgs`)를 읽어서 집계하고, IoU가 더 필요한
곳은 pycocotools의 `maskUtils.iou`를 씁니다. 매칭이나 IoU 기하 계산을 직접
구현하지 않습니다. numpy는 집계와 행렬 구성에 씁니다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from pycocotools import mask as maskUtils

from .errors import MetricError


#: confusion matrix에서 background가 차지하는 행·열 index입니다.
BACKGROUND_INDEX = 0

#: confusion matrix의 background 표시 이름입니다.
BACKGROUND_LABEL = "background"

#: score threshold sweep 지점입니다. 0.05부터 0.95까지 0.05 간격입니다.
SWEEP_THRESHOLDS: tuple[float, ...] = tuple(
    float(value) for value in np.round(np.arange(0.05, 1.0, 0.05), 2)
)

#: localization 실패로 볼 최소 IoU입니다.
#:
#: 이 값이 없으면 같은 class ground truth가 이미지 어딘가에 있기만 해도 전부
#: localization으로 분류되어 background 실패가 사라집니다. 겹침이 이 값에도 못
#: 미치면 "박스를 조금 빗맞힌 것"이 아니라 "없는 것을 본 것"으로 봅니다.
LOCALIZATION_IOU_FLOOR = 0.1

#: 전체 크기 구간입니다. evalImgs에서 이 구간 항목만 집계합니다.
AREA_RANGE_ALL = [0.0, 1e10]

#: weak로 보려면 필요한 최소 ground truth 수입니다.
#:
#: 이보다 정답이 적으면 AP가 실력이 아니라 표본 수에 흔들립니다. 정답이 하나뿐인
#: class는 그 하나를 맞히면 AP 1.0, 놓치면 0.0이라 "약한 class" 목록에 섞이면
#: 개선할 곳을 잘못 가리킵니다.
MIN_TRUTH_COUNT_FOR_WEAK = 4

#: weak·sparse 목록에 남길 class 수입니다.
TOP_N_PER_CLASS_SUMMARY = 5

#: 요약 row에 옮겨 담을 per_class field입니다. 여기 없는 field는 복제하지 않습니다.
_SUMMARY_FIELDS = (
    "category_id",
    "name",
    "ap",
    "ap50",
    "ap75",
    "truth_count",
    "prediction_count",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    """분모가 0이면 ``0.0``이 아니라 ``None``입니다."""
    return float(numerator) / float(denominator) if denominator > 0 else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    """Precision과 recall의 조화평균입니다. 한쪽이라도 없으면 ``None``입니다."""
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _relevant_entries(
    eval_imgs: Sequence[Any],
    *,
    category_id: int | None = None,
    image_id: int | None = None,
) -> Iterable[Mapping[str, Any]]:
    """전체 크기 구간에 해당하는 evalImgs 항목만 골라 냅니다."""
    for entry in eval_imgs:
        if entry is None:
            continue
        if list(entry["aRng"]) != AREA_RANGE_ALL:
            continue
        if category_id is not None and entry["category_id"] != category_id:
            continue
        if image_id is not None and entry["image_id"] != image_id:
            continue
        yield entry


def _kept_mask(entry: Mapping[str, Any], t_index: int, score_threshold: float) -> np.ndarray:
    """score 기준을 넘고 무시 대상이 아닌 detection을 고릅니다."""
    scores = np.asarray(entry["dtScores"], dtype=float)
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    dt_ignore = np.asarray(entry["dtIgnore"], dtype=bool)
    return (scores >= score_threshold) & ~dt_ignore[t_index]


def summarize_matches(
    eval_imgs: Sequence[Any],
    t_index: int,
    score_threshold: float,
    *,
    category_id: int | None = None,
    image_id: int | None = None,
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

    for entry in _relevant_entries(eval_imgs, category_id=category_id, image_id=image_id):
        gt_ignore = np.asarray(entry["gtIgnore"], dtype=bool)
        truth_count += int((~gt_ignore).sum())

        keep = _kept_mask(entry, t_index, score_threshold)
        if not keep.any():
            continue
        matched = np.asarray(entry["dtMatches"], dtype=float)[t_index][keep]
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
        "f1": f1_score(precision, recall),
    }


def sweep_score_thresholds(
    eval_imgs: Sequence[Any],
    t_index: int,
    *,
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
) -> dict[float, dict[str, Any]]:
    """score threshold를 바꿔 가며 precision, recall, F1을 계산합니다.

    ``COCOeval``을 다시 돌리지 않습니다. 매칭 결과는 score 내림차순으로 이미
    확정되어 있어서, threshold를 올리는 것은 뒤쪽 detection을 잘라 내는 것과
    같기 때문입니다.

    이미지당 최대 detection 수가 적용된 뒤의 예측만 대상이므로, threshold를
    낮춰도 잘려 나간 예측은 되살아나지 않습니다. 낮은 threshold 구간에서 recall이
    천장에 막히면 그 때문입니다.
    """
    if not len(thresholds):
        raise MetricError("score threshold sweep 지점이 비어 있습니다.")
    return {
        float(threshold): summarize_matches(eval_imgs, t_index, float(threshold))
        for threshold in thresholds
    }


def best_f1_threshold(sweep: Mapping[float, Mapping[str, Any]]) -> dict[str, Any] | None:
    """sweep 결과에서 F1이 가장 높은 threshold를 고릅니다.

    F1이 같으면 precision이 높은 쪽을, 그래도 같으면 threshold가 낮은 쪽을
    고릅니다. F1을 계산할 수 있는 지점이 하나도 없으면 ``None``입니다.
    """
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold, counts in sweep.items():
        if counts["f1"] is None:
            continue
        # F1 최대 → precision 최대 → threshold 최소 순으로 고릅니다.
        key = (-float(counts["f1"]), -float(counts["precision"] or 0.0), float(threshold))
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "threshold": float(threshold),
                "precision": counts["precision"],
                "recall": counts["recall"],
                "f1": counts["f1"],
            }
    return best


def _label_index(category_ids: Sequence[int]) -> dict[int, int]:
    """category_id를 confusion matrix의 행·열 index로 바꿉니다. 0은 background입니다."""
    return {category_id: order for order, category_id in enumerate(category_ids, start=1)}


def build_confusion_matrix(
    eval_imgs: Sequence[Any],
    coco_gt: Any,
    coco_dt: Any,
    category_ids: Sequence[int],
    t_index: int,
    score_threshold: float,
    *,
    class_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """class를 무시하고 매칭한 결과로 confusion matrix를 만듭니다.

    ``matrix[정답][예측]`` 형태이고 index 0은 background입니다. 대각선은 TP,
    비대각선은 class를 혼동한 건수, 0행은 FP, 0열은 FN입니다.

    ``eval_imgs``는 반드시 ``params.useCats = 0``으로 돌린 pass의 결과여야
    합니다. 기본 설정의 ``COCOeval``은 같은 class 안에서만 매칭하므로 class를
    혼동한 경우가 보이지 않습니다.

    매칭되지 않은 ground truth는 ``gtMatches``로 판정하지 않습니다. ``gtMatches``는
    score 기준을 적용하기 **전**의 매칭 결과라서, 낮은 score 예측이 붙은 ground
    truth가 "매칭됨"으로 남습니다. 그 예측은 score 기준에서 잘려 어느 칸에도 들어가지
    않으므로 그 ground truth가 행렬에서 사라집니다. 그래서 살아남은 예측이 실제로
    붙잡은 id 집합으로 판정합니다.
    """
    names = dict(class_names or {})
    index_of = _label_index(category_ids)
    size = len(category_ids) + 1
    matrix = np.zeros((size, size), dtype=int)

    for entry in _relevant_entries(eval_imgs):
        keep = _kept_mask(entry, t_index, score_threshold)
        matched_gt_ids: set[int] = set()

        if keep.any():
            dt_matches = np.asarray(entry["dtMatches"], dtype=float)[t_index]
            for position, detection_id in enumerate(entry["dtIds"]):
                if not keep[position]:
                    continue
                predicted = coco_dt.anns[detection_id]["category_id"]
                predicted_index = index_of.get(predicted)
                if predicted_index is None:
                    raise MetricError(
                        f"예측 category_id {predicted}가 category 목록에 없습니다."
                    )
                truth_id = int(dt_matches[position])
                if truth_id <= 0:
                    matrix[BACKGROUND_INDEX][predicted_index] += 1  # FP
                    continue
                matched_gt_ids.add(truth_id)
                truth = coco_gt.anns[truth_id]["category_id"]
                matrix[index_of[truth]][predicted_index] += 1

        # gtMatches가 아니라 살아남은 예측이 붙잡은 id로 판정합니다.
        gt_ignore = np.asarray(entry["gtIgnore"], dtype=bool)
        for truth_id, ignored in zip(entry["gtIds"], gt_ignore):
            # gtIds와 gtIgnore는 evaluateImg가 gtind로 정렬한 동일 gt 목록에서
            # 나오므로 index가 대응합니다. pycocotools 구현에 기대는 전제입니다.
            if ignored or truth_id in matched_gt_ids:
                continue
            truth = coco_gt.anns[truth_id]["category_id"]
            matrix[index_of[truth]][BACKGROUND_INDEX] += 1  # FN

    labels = [BACKGROUND_LABEL] + [
        names.get(category_id, str(category_id)) for category_id in category_ids
    ]
    return {
        "labels": labels,
        "category_ids": [None, *[int(value) for value in category_ids]],
        "matrix": matrix.tolist(),
    }


def per_image_breakdown(
    eval_imgs: Sequence[Any],
    t_index: int,
    score_threshold: float,
    *,
    image_ids: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """이미지별 TP/FP/FN과 precision·recall을 내고 실패가 큰 순으로 정렬합니다.

    ``FP + FN``이 큰 이미지가 앞에 옵니다. 같으면 FN이 많은 쪽, 그래도 같으면
    image_id 순입니다.

    ``prediction_count``는 score 기준과 이미지당 최대 detection 수를 적용한 뒤의
    수입니다. 따라서 항상 ``tp + fp``와 같습니다.
    """
    original = dict(image_ids or {})
    seen: list[int] = []
    for entry in _relevant_entries(eval_imgs):
        if entry["image_id"] not in seen:
            seen.append(entry["image_id"])

    rows: list[dict[str, Any]] = []
    for image_id in seen:
        counts = summarize_matches(eval_imgs, t_index, score_threshold, image_id=image_id)
        rows.append(
            {
                "image_id": original.get(image_id, image_id),
                "truth_count": counts["tp"] + counts["fn"],
                "prediction_count": counts["tp"] + counts["fp"],
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": counts["precision"],
                "recall": counts["recall"],
                "failure_count": counts["fp"] + counts["fn"],
            }
        )

    rows.sort(key=lambda row: (-row["failure_count"], -row["fn"], str(row["image_id"])))
    return rows


def _summary_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    """per_class 항목에서 진단에 쓰는 field만 새 dict로 옮깁니다."""
    return {field: entry[field] for field in _SUMMARY_FIELDS}


def _ap_sort_key(row: Mapping[str, Any]) -> tuple[bool, float]:
    """AP 오름차순 정렬 key입니다. ``None``은 항상 뒤로 갑니다.

    ``None``을 ``0.0``으로 치환하면 측정되지 않은 class가 최저 AP 자리를 차지해
    "가장 약한 class"로 잘못 보입니다. 그래서 값 대신 ``None`` 여부를 앞자리 key로
    둡니다. 뒤따르는 실수는 같은 ``None`` 여부끼리만 비교되므로 순서에 영향이
    없는 자리 채움입니다.
    """
    ap = row["ap"]
    return (ap is None, float(ap) if ap is not None else 0.0)


def summarize_per_class(
    per_class: Sequence[Mapping[str, Any]],
    *,
    min_truth_count: int = MIN_TRUTH_COUNT_FOR_WEAK,
    top_n: int = TOP_N_PER_CLASS_SUMMARY,
) -> dict[str, Any]:
    """이미 계산된 ``per_class``를 약한·희소·미측정 class로 나눠 정렬합니다.

    새 지표를 계산하지 않습니다. ``per_class``에 있는 ``ap``와 ``truth_count``를
    다시 배열할 뿐이고, 입력 리스트와 그 안의 dict는 건드리지 않습니다.

    ``truth_count``와 ``ap``는 뜻이 다르므로 한 점수로 합치지 않습니다.
    ``truth_count``로 그룹을 가르고 그룹 안에서만 ``ap``로 줄을 세웁니다.

    - ``weak``: ``truth_count >= min_truth_count``. 표본이 충분한데 AP가 낮은
      class로, AP 오름차순 → truth_count 내림차순 → category_id 오름차순입니다.
    - ``sparse``: ``0 < truth_count < min_truth_count``. AP가 표본 부족으로
      흔들릴 수 있어 weak와 섞지 않습니다. truth_count 오름차순 → AP 오름차순 →
      category_id 오름차순입니다.
    - ``unmeasured``: ``truth_count == 0``. 성능이 낮은 것이 아니라 잴 정답이
      없는 상태입니다(``ap``가 ``None``일 수 있습니다). category_id 순으로 전부
      냅니다. 줄 세울 기준이 없어 잘라 내면 정보만 사라집니다.

    ``counts``는 자르기 전 개수입니다. 목록 길이만 보면 "다섯 개뿐"인지 "다섯 개로
    잘렸는지" 구분할 수 없습니다. background는 ``per_class``에 없으므로 여기에도
    없습니다.
    """
    weak: list[dict[str, Any]] = []
    sparse: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    for entry in per_class:
        row = _summary_row(entry)
        if row["truth_count"] >= min_truth_count:
            weak.append(row)
        elif row["truth_count"] > 0:
            sparse.append(row)
        else:
            unmeasured.append(row)

    weak.sort(key=lambda row: (*_ap_sort_key(row), -row["truth_count"], row["category_id"]))
    sparse.sort(key=lambda row: (row["truth_count"], *_ap_sort_key(row), row["category_id"]))
    unmeasured.sort(key=lambda row: row["category_id"])

    return {
        "min_truth_count": int(min_truth_count),
        "top_n": int(top_n),
        "counts": {
            "weak": len(weak),
            "sparse": len(sparse),
            "unmeasured": len(unmeasured),
        },
        "weak": weak[:top_n],
        "sparse": sparse[:top_n],
        "unmeasured": unmeasured,
    }


def _max_iou(box: Sequence[float], candidates: Sequence[Sequence[float]]) -> float:
    """box와 후보들 사이 IoU의 최댓값입니다. 후보가 없으면 0.0입니다."""
    if not candidates:
        return 0.0
    scores = np.asarray(
        maskUtils.iou([list(box)], [list(item) for item in candidates], [0] * len(candidates))
    )
    return float(scores.max()) if scores.size else 0.0


def classify_false_positives(
    eval_imgs: Sequence[Any],
    coco_gt: Any,
    coco_dt: Any,
    t_index: int,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, int]:
    """False positive를 원인별로 나눕니다.

    ``eval_imgs``는 기본 설정(class별 매칭) pass의 결과를 씁니다. 각 false
    positive에 대해 같은 class ground truth와의 최대 IoU, 다른 class ground
    truth와의 최대 IoU를 보고 아래 순서로 판정합니다.

    - ``duplicate``: 같은 class와 충분히 겹치지만 그 ground truth를 더 높은 score
      예측이 이미 가져갔습니다. 요구된 세 가지 중 어디에도 맞지 않아 따로 셉니다.
      이 칸이 없으면 중복 검출이 다른 항목에 섞여 수치를 부풀립니다.
    - ``classification``: 다른 class ground truth와 충분히 겹칩니다. 위치는
      맞았지만 class를 틀렸습니다.
    - ``localization``: 같은 class ground truth와 조금 겹치지만 기준에 못
      미칩니다. class는 맞았지만 박스가 어긋났습니다.
    - ``background``: 어떤 ground truth와도 의미 있게 겹치지 않습니다.

    겹침 근거가 강한 쪽을 먼저 보기 때문에 이 순서입니다.
    """
    counts = {"localization": 0, "classification": 0, "background": 0, "duplicate": 0}

    for entry in _relevant_entries(eval_imgs):
        keep = _kept_mask(entry, t_index, score_threshold)
        if not keep.any():
            continue
        dt_matches = np.asarray(entry["dtMatches"], dtype=float)[t_index]
        for position, detection_id in enumerate(entry["dtIds"]):
            if not keep[position] or dt_matches[position] > 0:
                continue  # 걸러졌거나 true positive입니다.

            detection = coco_dt.anns[detection_id]
            truths = coco_gt.imgToAnns.get(detection["image_id"], [])
            same_class = [item["bbox"] for item in truths if item["category_id"] == detection["category_id"]]
            other_class = [
                item["bbox"] for item in truths if item["category_id"] != detection["category_id"]
            ]
            same_iou = _max_iou(detection["bbox"], same_class)
            other_iou = _max_iou(detection["bbox"], other_class)

            if same_iou >= iou_threshold:
                counts["duplicate"] += 1
            elif other_iou >= iou_threshold:
                counts["classification"] += 1
            elif same_iou >= LOCALIZATION_IOU_FLOOR:
                counts["localization"] += 1
            else:
                counts["background"] += 1

    return counts


__all__ = [
    "BACKGROUND_LABEL",
    "LOCALIZATION_IOU_FLOOR",
    "MIN_TRUTH_COUNT_FOR_WEAK",
    "SWEEP_THRESHOLDS",
    "TOP_N_PER_CLASS_SUMMARY",
    "best_f1_threshold",
    "build_confusion_matrix",
    "classify_false_positives",
    "f1_score",
    "per_image_breakdown",
    "summarize_matches",
    "summarize_per_class",
    "sweep_score_thresholds",
]
