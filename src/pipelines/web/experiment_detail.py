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

#: confusion matrix의 0번 행·열 이름입니다. evaluate의 `BACKGROUND_LABEL`을 베낀
#: 값입니다 — 그쪽을 import할 수 없습니다. 어긋나면 목록이 비지, 틀리지는 않습니다.
BACKGROUND_LABEL = "background"


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _in_unit_range(value: Any) -> bool:
    """0과 1 사이인지 봅니다.

    threshold도 precision·recall·F1도 evaluate에서는 전부 이 범위입니다
    (`SWEEP_THRESHOLDS`는 0.05~0.95, 나머지는 분자/분모). 범위를 안 보면 화면이
    `기준 1.50`, `-50%`, `200%`를 **정상 측정값처럼** 그립니다.
    """

    number = _number(value)
    return number is not None and 0.0 <= number <= 1.0


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


def _sweep_rows(sweep: Any) -> list[dict[str, Any]] | None:
    """`{"0.05": {...}}` 형태를 threshold 순 배열로 폅니다.

    화면은 이것을 곡선으로 그리므로 순서가 있어야 합니다. dict의 key 순서를 믿지
    않고 threshold 숫자로 다시 세웁니다.

    **읽지 못하면 `None`입니다.** 빈 배열로 내면 "재 봤는데 잴 지점이 없었다"와
    같아져, 깨진 기록이 정상 결과로 읽힙니다.
    """

    if not isinstance(sweep, Mapping):
        return None
    rows: list[dict[str, Any]] = []
    for key, counts in sweep.items():
        if not _looks_numeric(key) or not _in_unit_range(float(key)):
            return None
        if not isinstance(counts, Mapping):
            return None
        row = {"threshold": _number(float(key))}
        for name in ("precision", "recall", "f1"):
            # key가 **없는** 것도 깨진 것입니다. evaluate는 셋을 언제나 씁니다.
            # `counts.get()`으로 꺼내면 없는 key와 일부러 쓴 `None`이 같아집니다.
            if name not in counts:
                return None
            value = counts[name]
            # `None`은 evaluate가 일부러 쓴 값입니다 — 분모가 0이라 못 잰 것입니다.
            # 그 밖의 값은 **깨진 것**이라, 그냥 두면 둘이 똑같이 `-`로 그려집니다.
            if value is not None and not _in_unit_range(value):
                return None
            row[name] = _number(value)
        rows.append(row)
    rows.sort(key=lambda row: row["threshold"])
    return rows


def _looks_numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _readable_block(
    analysis: Mapping[str, Any] | None, key: str
) -> Mapping[str, Any] | None:
    """IoU별 블록을 꺼냅니다. **없으면 빈 것, 깨졌으면 `None`입니다.**

    둘을 합치면 깨진 파일이 "이 기능 이전에 돌린 평가"로 설명됩니다. `analysis`
    자체를 읽지 못한 경우(`None`)도 마찬가지로 읽지 못한 것입니다.

    **key가 없는 것과 값이 `null`인 것도 다릅니다.** evaluate는 이 자리에 언제나
    dict를 씁니다. `null`이 보이면 파일이 상한 것이지 옛 평가가 아닙니다.
    """

    if analysis is None:
        return None
    if key not in analysis:
        return {}
    value = analysis[key]
    return value if isinstance(value, Mapping) else None


def _is_list(value: Any) -> bool:
    """목록인지 봅니다. **글자열은 목록이 아닙니다.**

    `str`과 `bytes`도 `Sequence`라, 그냥 `Sequence`로 보면 `"abc"`가 세 칸짜리
    목록으로 통과합니다. 그러면 글자 하나하나가 class 이름이 되어, 깨진 파일에서
    그럴듯한 진단이 나옵니다.
    """

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _confused_pairs(block: Any, top_n: int) -> tuple[list[dict[str, Any]], int] | None:
    """confusion matrix에서 **헷갈린 쌍만** 골라 잦은 순으로 냅니다.

    돌려주는 것은 (상위 목록, 전체 쌍 수)입니다. 행렬 자체는 내지 않습니다.
    **읽지 못하면 `None`입니다** — 빈 목록으로 내면 "재 보니 하나도 안 헷갈렸다"와
    같아져, 깨진 파일이 좋은 결과로 읽힙니다.

    대각선은 맞힌 것이라 뺍니다. 0번 행·열(background)은 **뺄 수 없습니다** —
    0행은 없는 것을 찾은 것이고 0열은 놓친 것이라, 빼면 왜 틀렸는지의 절반이
    사라집니다.
    """

    if not isinstance(block, Mapping):
        return None
    matrix = block.get("matrix")
    labels = block.get("labels")
    # 빈 행렬은 evaluate가 내지 않습니다 — 적어도 background 한 칸은 있습니다.
    # 그냥 두면 "헷갈린 쌍이 하나도 없습니다"로 그려집니다.
    if not _is_list(matrix) or not _is_list(labels) or not labels:
        return None
    # **정사각이어야 합니다.** evaluate는 `labels`와 같은 크기로 씁니다.
    # 이름이 모자라면 없는 자리를 index로 채우게 되어 `'1'` 같은 그럴듯한 이름이
    # 나오고, 남으면 어느 칸이 어느 class인지 어긋납니다.
    if len(matrix) != len(labels):
        return None
    # 0번은 background 자리입니다. 이름이 다르면 화면이 그 행·열을 **보통 class로**
    # 그립니다 — 없는 것을 찾은 것과 놓친 것이 class끼리의 착각으로 읽힙니다.
    if str(labels[0]) != BACKGROUND_LABEL:
        return None
    ids = block.get("category_ids")
    id_list = list(ids) if _is_list(ids) and len(ids) >= len(matrix) else []

    def name_of(index: int) -> str:
        return str(labels[index])

    def id_of(index: int) -> Any:
        return id_list[index] if index < len(id_list) else None

    pairs: list[dict[str, Any]] = []
    for truth, row in enumerate(matrix):
        # 행 길이도 같아야 합니다. 길면 없는 index를 조회해 `IndexError`가 상세
        # 응답 밖으로 나가고, 짧으면 없는 칸이 조용히 0건으로 읽힙니다.
        if not _is_list(row) or len(row) != len(labels):
            return None
        for predicted, count in enumerate(row):
            # 칸은 0 이상의 정수뿐입니다. `bool`은 `int`를 물려받으므로 따로
            # 막습니다 — `True`를 1건으로 세면 없던 혼동이 생깁니다. 그 밖의
            # 값이 보이면 이 파일은 우리가 아는 모양이 아닙니다.
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                return None
            if truth == predicted or count == 0:
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
    """false positive 원인 4개를 정수로 냅니다. 읽지 못하면 `None`입니다.

    건수는 **0 이상**입니다. 음수를 그대로 내보내면 화면이 음수 건수와 100%를
    넘는 비율을 그립니다 — 깨진 파일이 이상한 숫자로만 드러납니다.
    """

    if not isinstance(value, Mapping):
        return None
    counts: dict[str, int] = {}
    for cause in _ERROR_CAUSES:
        number = value.get(cause)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            return None
        counts[cause] = number
    return counts


def _best_f1(value: Any) -> dict[str, Any] | None:
    """F1이 가장 높았던 지점입니다. 읽지 못하거나 없으면 `None`입니다.

    여기서 `None`은 화면에서 "최고점 표시 없음"이 됩니다. 표시가 없는 것은
    **아무 말도 하지 않는 것**이라, 다른 블록과 달리 세 상태를 따로 두지
    않았습니다. 대신 범위를 벗어난 값은 반드시 막습니다 — `F1 최고 200%`는
    표시가 없는 것과 달리 **틀린 말**입니다.
    """

    if not isinstance(value, Mapping):
        return None
    if not _in_unit_range(value.get("threshold")):
        return None
    for name in ("precision", "recall", "f1"):
        number = value.get(name)
        if number is not None and not _in_unit_range(number):
            return None
    return {
        "threshold": _number(value.get("threshold")),
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
    # `analysis`가 통째로 이상하면 **읽지 못한 것**입니다. 빈 dict로 두면 그 아래
    # 모든 블록이 "이 기능 이전 평가"로 설명됩니다.
    analysis_values: Mapping[str, Any] | None
    if "analysis" not in document:
        analysis_values = {}
    else:
        analysis_values = analysis if isinstance(analysis, Mapping) else None
    readable = analysis_values if analysis_values is not None else {}

    per_class = readable.get("per_class_summary")
    if not isinstance(per_class, Mapping) or set(per_class) != _SUMMARY_KEYS:
        # 이 계약 이전 평가에는 블록 자체가 없습니다. 빈 표를 지어내지 않습니다.
        per_class = None

    best = _readable_block(analysis_values, "best_f1")
    # 블록 **자체**가 깨진 경우입니다. 없는 것과 같게 `{}`로 내면 화면이 "이 기능
    # 이전 평가"라고 잘못 설명합니다. 어느 IoU가 있었는지도 알 수 없으므로
    # 블록째 `None`으로 답합니다.
    sweep = _readable_block(analysis_values, "score_sweep")
    confusion = _readable_block(analysis_values, "confusion_matrix")
    breakdown = _readable_block(analysis_values, "error_breakdown")

    picked = (
        None
        if confusion is None
        else {
            label: _confused_pairs(block, CONFUSION_TOP_N)
            for label, block in confusion.items()
        }
    )
    causes = (
        None
        if breakdown is None
        else {label: _error_causes(value) for label, value in breakdown.items()}
    )
    return {
        "available": True,
        "reason": None,
        "metrics": {key: _number(metric_values.get(key)) for key in METRIC_KEYS},
        "counts": {key: _number(document.get(key)) for key in COUNT_KEYS},
        "score_threshold": _number(readable.get("score_threshold")),
        "max_detections_per_image": _number(document.get("max_detections_per_image")),
        # IoU label("0.50"/"0.75")별로 나뉩니다. 화면이 어느 쪽을 볼지 고릅니다.
        "score_sweep": (
            None
            if sweep is None
            else {label: _sweep_rows(rows) for label, rows in sweep.items()}
        ),
        "best_f1": (
            None if best is None else {label: _best_f1(value) for label, value in best.items()}
        ),
        # 행렬이 아니라 **헷갈린 쌍**입니다. 자른 개수를 함께 말하지 않으면 잘린
        # 목록이 전부로 읽힙니다. 읽지 못한 IoU는 개수가 `null`입니다 — 목록에서
        # 빼 버리면 "이 기능 이전 평가"와, 빈 목록으로 두면 "0건"과 뒤섞입니다.
        "confusions": (
            None
            if picked is None
            else {label: value[0] for label, value in picked.items() if value is not None}
        ),
        "confusion_counts": (
            None
            if picked is None
            else {
                label: (
                    None if value is None else {"pairs": value[1], "shown": len(value[0])}
                )
                for label, value in picked.items()
            }
        ),
        "error_breakdown": causes,
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
