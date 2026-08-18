"""등록된 실험 하나의 평가 결과와 loss 곡선을 화면이 쓸 크기로 줄여 읽습니다.

`metrics.json`은 58x58 confusion matrix와 이미지 2천여 장의 `per_image`까지 들어
650KB가 넘습니다. 화면이 필요한 것은 그중 지표 9개와 요약 몇 블록뿐이라, **여기서
골라 담고 나머지는 버립니다.** 그대로 흘려보내면 상세 화면 한 번에 650KB가 오갑니다.

evaluate나 train을 import하지 않습니다. 그들이 artifact로 공개한 URI를 `src/common`의
storage로 읽을 뿐입니다. **이 module의 함수는 어떤 이유로도 예외를 던지지 않습니다.**
지표 하나를 못 읽었다고 상세 화면 전체가 안 보이면 곤란하기 때문입니다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.common import create_storage

from .paths import resolve_within_repo


__all__ = ["evaluation_block", "history_block"]


# evaluate가 metrics.json에 쓰는 이름 그대로입니다. 화면이 한 번 더 번역하면 어느
# 쪽이 진짜인지 알기 어려워집니다. 지금까지 화면은 이 중 5개만 보여 줬습니다.
METRIC_KEYS: tuple[str, ...] = (
    "mAP",
    "mAP50_95",
    "mAP75_95",
    "mAP50",
    "mAP75",
    "precision50",
    "recall50",
    "precision75",
    "recall75",
)

COUNT_KEYS: tuple[str, ...] = (
    "image_count",
    "annotation_count",
    "prediction_count",
    "evaluated_class_count",
)

# per_class_summary가 이 모양이 아니면 버립니다. evaluate의 summarize_per_class 결과입니다.
_SUMMARY_KEYS = frozenset(
    {"min_truth_count", "top_n", "counts", "weak", "sparse", "unmeasured"}
)

#: 화면에 보낼 헷갈린 쌍의 수입니다. 대회 118종이면 행렬은 119x119이고 비대각선
#: 칸만 14,042개입니다. 그대로 보내면 칸 하나가 1px이라 아무것도 못 읽습니다.
#: 사람이 묻는 것은 "무엇을 무엇으로 착각하는가"이므로 잦은 것부터 자릅니다.
CONFUSION_TOP_N = 20

#: evaluate가 false positive를 나누는 이름입니다. 값이 없으면 0으로 채우지 않고
#: 그 IoU를 통째로 뺍니다 — 0건과 "안 쟀다"는 다른 말입니다.
_ERROR_CAUSES: tuple[str, ...] = ("localization", "classification", "background", "duplicate")


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _read_document(uri: str, storage_config: Mapping[str, Any]) -> Any | None:
    """artifact 문서 하나를 읽습니다. 못 읽으면 None입니다.

    로컬 실행의 URI는 `artifacts/evaluate/<run>/metrics.json`처럼 **저장소 기준 상대
    경로**입니다. 그것을 그대로 storage backend에 넘기면 backend가 자기 root를 앞에
    다시 붙여 `artifacts/artifacts/...`를 찾다가 실패합니다. 그래서 로컬은 저장소
    안으로 확정한 경로에서 직접 읽고, 원격만 backend에 맡깁니다.
    """

    try:
        if "://" in uri:
            return create_storage({"storage": dict(storage_config)}).read_json(uri)
        return json.loads(
            resolve_within_repo(uri, label="artifact_uri").read_text(encoding="utf-8")
        )
    except Exception:
        # storage 오류, 권한, 깨진 JSON, 저장소 밖 경로 무엇이든 여기서 멈춥니다.
        return None


def _unavailable_evaluation(reason: str) -> dict[str, Any]:
    """못 읽었을 때도 성공했을 때와 **같은 key**를 채워 돌려줍니다.

    key를 통째로 빼면 화면이 available을 확인하기 전에 값을 만지는 순간 죽습니다.
    실제로 학습만 하고 평가를 돌리지 않은 기록에서 상세 화면이 흰 채로 멈췄습니다.
    빈 값과 "없다"를 구분하는 것은 ``available``과 ``reason``의 몫입니다.
    """

    return {
        "available": False,
        "reason": reason,
        "metrics": {key: None for key in METRIC_KEYS},
        "counts": {key: None for key in COUNT_KEYS},
        "score_threshold": None,
        "max_detections_per_image": None,
        "score_sweep": {},
        "best_f1": {},
        "confusions": {},
        "confusion_counts": {},
        "error_breakdown": {},
        "per_class_summary": None,
    }


def _unavailable_history(reason: str) -> dict[str, Any]:
    """loss 곡선도 같은 이유로 빈 목록까지 채워 돌려줍니다."""

    return {"available": False, "reason": reason, "epochs": []}


def _sweep_rows(sweep: Any) -> list[dict[str, Any]]:
    """`{"0.05": {...}}` 형태를 threshold 순 배열로 폅니다.

    화면은 이것을 곡선으로 그리므로 순서가 있어야 합니다. dict의 key 순서를 믿지
    않고 threshold 숫자로 다시 세웁니다.
    """

    if not isinstance(sweep, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for key, counts in sweep.items():
        threshold = _number(float(key)) if _looks_numeric(key) else None
        if threshold is None or not isinstance(counts, Mapping):
            continue
        rows.append(
            {
                "threshold": threshold,
                "precision": _number(counts.get("precision")),
                "recall": _number(counts.get("recall")),
                "f1": _number(counts.get("f1")),
            }
        )
    rows.sort(key=lambda row: row["threshold"])
    return rows


def _looks_numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _confused_pairs(block: Any, top_n: int) -> tuple[list[dict[str, Any]], int]:
    """confusion matrix에서 **헷갈린 쌍만** 골라 잦은 순으로 냅니다.

    돌려주는 것은 (상위 목록, 전체 쌍 수)입니다. 행렬 자체는 내지 않습니다.

    대각선은 맞힌 것이라 뺍니다. 0번 행·열(background)은 **뺄 수 없습니다** —
    0행은 없는 것을 찾은 것이고 0열은 놓친 것이라, 빼면 왜 틀렸는지의 절반이
    사라집니다.
    """

    if not isinstance(block, Mapping):
        return [], 0
    matrix = block.get("matrix")
    labels = block.get("labels")
    if not isinstance(matrix, Sequence) or not isinstance(labels, Sequence):
        return [], 0
    ids = block.get("category_ids")
    id_list = list(ids) if isinstance(ids, Sequence) else []

    def name_of(index: int) -> str:
        return str(labels[index]) if index < len(labels) else str(index)

    def id_of(index: int) -> Any:
        return id_list[index] if index < len(id_list) else None

    pairs: list[dict[str, Any]] = []
    for truth, row in enumerate(matrix):
        if not isinstance(row, Sequence):
            continue
        for predicted, count in enumerate(row):
            if truth == predicted or not isinstance(count, int) or count <= 0:
                continue
            pairs.append(
                {
                    "truth_id": id_of(truth),
                    "truth": name_of(truth),
                    "predicted_id": id_of(predicted),
                    "predicted": name_of(predicted),
                    "count": count,
                }
            )
    # 건수 내림차순, 같으면 이름 순입니다. 같은 자료로 두 번 그리면 순서가 같아야
    # 사람이 "바뀐 것"과 "다시 그린 것"을 구별합니다.
    pairs.sort(key=lambda row: (-row["count"], str(row["truth"]), str(row["predicted"])))
    return pairs[:top_n], len(pairs)


def _error_causes(value: Any) -> dict[str, int] | None:
    """false positive 원인 4개를 정수로 냅니다. 하나라도 없으면 `None`입니다."""

    if not isinstance(value, Mapping):
        return None
    counts: dict[str, int] = {}
    for cause in _ERROR_CAUSES:
        number = value.get(cause)
        if isinstance(number, bool) or not isinstance(number, int):
            return None
        counts[cause] = number
    return counts


def _best_f1(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    threshold = _number(value.get("threshold"))
    if threshold is None:
        return None
    return {
        "threshold": threshold,
        "precision": _number(value.get("precision")),
        "recall": _number(value.get("recall")),
        "f1": _number(value.get("f1")),
    }


def evaluation_block(
    metrics_uri: str | None, storage_config: Mapping[str, Any]
) -> dict[str, Any]:
    """`metrics.json`에서 화면이 쓸 것만 골라 옵니다.

    confusion matrix와 per_image, per_class 전체는 **담지 않습니다.** 약한 class는
    evaluate가 이미 골라 둔 `per_class_summary`로 충분하고, 나머지는 650KB를 브라우저로
    옮길 만한 값이 아닙니다.
    """

    if not metrics_uri:
        return _unavailable_evaluation("이 실험에는 평가 결과 파일이 기록돼 있지 않습니다.")

    document = _read_document(metrics_uri, storage_config)
    if not isinstance(document, Mapping):
        return _unavailable_evaluation("평가 결과 파일을 읽지 못했습니다.")

    metrics = document.get("metrics")
    metric_values = metrics if isinstance(metrics, Mapping) else {}
    analysis = document.get("analysis")
    analysis_values = analysis if isinstance(analysis, Mapping) else {}

    per_class = analysis_values.get("per_class_summary")
    if not isinstance(per_class, Mapping) or set(per_class) != _SUMMARY_KEYS:
        # 이 계약 이전 평가에는 블록 자체가 없습니다. 빈 표를 지어내지 않습니다.
        per_class = None

    sweep = analysis_values.get("score_sweep")
    best = analysis_values.get("best_f1")
    confusion = analysis_values.get("confusion_matrix")
    confusion_items = confusion.items() if isinstance(confusion, Mapping) else []
    picked = {
        label: _confused_pairs(block, CONFUSION_TOP_N) for label, block in confusion_items
    }
    breakdown = analysis_values.get("error_breakdown")
    breakdown_items = breakdown.items() if isinstance(breakdown, Mapping) else []
    causes = {label: _error_causes(value) for label, value in breakdown_items}
    return {
        "available": True,
        "reason": None,
        "metrics": {key: _number(metric_values.get(key)) for key in METRIC_KEYS},
        "counts": {key: _number(document.get(key)) for key in COUNT_KEYS},
        "score_threshold": _number(analysis_values.get("score_threshold")),
        "max_detections_per_image": _number(document.get("max_detections_per_image")),
        # IoU label("0.50"/"0.75")별로 나뉩니다. 화면이 어느 쪽을 볼지 고릅니다.
        "score_sweep": {
            label: _sweep_rows(rows)
            for label, rows in (sweep.items() if isinstance(sweep, Mapping) else [])
        },
        "best_f1": {
            label: _best_f1(value)
            for label, value in (best.items() if isinstance(best, Mapping) else [])
        },
        # 행렬이 아니라 **헷갈린 쌍**입니다. 자른 개수를 함께 말하지 않으면 잘린
        # 목록이 전부로 읽힙니다.
        "confusions": {label: rows for label, (rows, _) in picked.items()},
        "confusion_counts": {
            label: {"pairs": total, "shown": len(rows)}
            for label, (rows, total) in picked.items()
        },
        "error_breakdown": {label: value for label, value in causes.items()},
        "per_class_summary": dict(per_class) if per_class is not None else None,
    }


def history_block(
    history_uri: str | None, storage_config: Mapping[str, Any]
) -> dict[str, Any]:
    """`training_history.json`에서 epoch별 loss만 꺼냅니다.

    숫자만 보고는 과적합이 언제 시작됐는지 알기 어렵습니다. 곡선으로 보면 한눈에
    드러나므로, 화면이 그릴 수 있게 epoch 순서대로 폅니다.
    """

    if not history_uri:
        return _unavailable_history("이 실험에는 학습 기록 파일이 없습니다.")

    document = _read_document(history_uri, storage_config)
    if isinstance(document, (str, bytes)) or not isinstance(document, Sequence):
        return _unavailable_history("학습 기록 파일을 읽지 못했습니다.")

    epochs: list[dict[str, Any]] = []
    for entry in document:
        if not isinstance(entry, Mapping):
            continue
        epoch = _number(entry.get("epoch"))
        if epoch is None:
            continue
        epochs.append(
            {
                "epoch": int(epoch),
                "train_loss": _number(entry.get("train_loss")),
                "validation_loss": _number(entry.get("validation_loss")),
                "epoch_seconds": _number(entry.get("epoch_seconds")),
                "is_best": entry.get("is_best") if isinstance(entry.get("is_best"), bool) else None,
                # schedule이 생기기 전에 학습한 실행에는 이 값이 없습니다.
                "learning_rate": _number(entry.get("learning_rate")),
            }
        )
    if not epochs:
        return _unavailable_history("학습 기록에 epoch이 하나도 없습니다.")
    epochs.sort(key=lambda item: item["epoch"])
    return {"available": True, "reason": None, "epochs": epochs}
