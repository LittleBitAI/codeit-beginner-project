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


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


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
        return _unavailable("이 실험에는 평가 결과 파일이 기록돼 있지 않습니다.")

    document = _read_document(metrics_uri, storage_config)
    if not isinstance(document, Mapping):
        return _unavailable("평가 결과 파일을 읽지 못했습니다.")

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
        return _unavailable("이 실험에는 학습 기록 파일이 없습니다.")

    document = _read_document(history_uri, storage_config)
    if isinstance(document, (str, bytes)) or not isinstance(document, Sequence):
        return _unavailable("학습 기록 파일을 읽지 못했습니다.")

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
            }
        )
    if not epochs:
        return _unavailable("학습 기록에 epoch이 하나도 없습니다.")
    epochs.sort(key=lambda item: item["epoch"])
    return {"available": True, "reason": None, "epochs": epochs}
