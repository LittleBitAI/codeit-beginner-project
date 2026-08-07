"""평가 subprocess가 stderr로 흘리는 진행 로그를 읽습니다.

`docs/evaluate_progress_contract.md`의 ``evaluate.progress/1`` 계약을 따릅니다.
``data_progress.py``(data 쪽 소비자)와 같은 모양이고, 가장 중요한 성질도 같습니다.
**어떤 입력에도 예외를 던지지 않고**, 진행 줄이 하나도 없으면 진행률을 지어내지 않고
``available: False``를 그대로 보고합니다.

I/O가 없는 순수 함수만 있습니다. 남은 시간도 event가 들고 온 ``ts``로만 계산하며,
관측이 부족하면 추정하지 않습니다.

``COCOeval``이 stdout에 쓰는 요약도 stderr로 섞여 들어올 수 있습니다. 그런 줄은
버리지 않고 원문 로그로 남깁니다.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from .masking import sanitize_line


__all__ = ["EvaluateProgressState", "consume_line", "snapshot"]


SCHEMA_PREFIX = "evaluate.progress/"
SUPPORTED_MAJOR = "1"

_WARNING_MARKERS = ("[w]", "warning", "warn:", "userwarning", "futurewarning", "deprecat")
_ERROR_MARKERS = ("[e]", "error", "traceback", "exception", "failed", "credential")

# 추론 단계 이름과 화면에 쓸 문구입니다. 계약에 없는 이름은 받아들이지 않습니다.
PREDICT_STAGES = {
    "validation": "validation 추론 중",
    "test": "test 추론 중",
}

# 이미지 개수를 세는 필드입니다. 시작과 완료가 같은 이름을 씁니다.
IMAGE_KEYS = ("validation_images", "test_images")

# 계약이 정한 지표 이름입니다. 계산되지 않은 값은 ``0.0``이 아니라 ``None``입니다.
METRIC_KEYS = ("mAP", "mAP50", "mAP75")

STAGE_LABELS = {
    "started": "평가 시작",
    "metrics": "지표 계산 중",
    "submission": "submission 쓰는 중",
    "completed": "평가 완료",
    **PREDICT_STAGES,
}


class EvaluateProgressState:
    """평가 실행 한 건의 진행 상태. 순수 데이터만 담습니다."""

    __slots__ = (
        "completed",
        "device",
        "done",
        "images",
        "last_seconds",
        "malformed_lines",
        "metrics",
        "predict_stage",
        "run_id",
        "saw_progress",
        "speed_done",
        "speed_seconds",
        "stage",
        "submission_rows",
        "total",
    )

    def __init__(self) -> None:
        self.saw_progress = False
        self.stage: str | None = None
        self.run_id: str | None = None
        self.device: str | None = None
        self.images: dict[str, int] | None = None
        self.predict_stage: str | None = None
        self.done: int | None = None
        self.total: int | None = None
        self.metrics: dict[str, float | None] | None = None
        self.submission_rows: int | None = None
        self.completed: dict[str, int] | None = None
        # 남은 시간을 재기 위한 관측 기준점입니다. (추론한 장 수, ts) 한 쌍이며 추론
        # 단계가 바뀌면 버립니다. validation의 속도로 test를 재면 거짓말이 됩니다.
        self.speed_done: int | None = None
        self.speed_seconds: float | None = None
        # 가장 마지막으로 해석한 ``ts``입니다. 해석하지 못하면 ``None``이 되어 그
        # 순간에는 남은 시간을 내지 않습니다.
        self.last_seconds: float | None = None
        self.malformed_lines = 0


def _level_for(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in _ERROR_MARKERS):
        return "error"
    if any(marker in lowered for marker in _WARNING_MARKERS):
        return "warn"
    return "info"


def _log(text: str, level: str | None = None) -> dict[str, Any]:
    return {"level": level or _level_for(text), "text": sanitize_line(text)}


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _epoch_seconds(value: Any) -> float | None:
    """계약의 UTC ISO-8601 ``ts``를 초로 바꿉니다. 해석하지 못하면 ``None``입니다."""

    text = _text(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (ValueError, OSError, OverflowError):
        return None


def _count_field(state: EvaluateProgressState, event: dict[str, Any], key: str) -> int | None:
    """개수 필드 하나를 읽습니다. 타입이 틀리면 그 필드만 버립니다."""

    if key not in event:
        return None
    number = _non_negative_int(event[key])
    if number is None:
        state.malformed_lines += 1
    return number


def _metric_field(state: EvaluateProgressState, event: dict[str, Any], key: str) -> float | None:
    """지표 하나를 읽습니다. 계산되지 않았거나 유한하지 않으면 ``None``입니다.

    browser의 ``JSON.parse``는 맨 ``NaN``에서 실패하므로 ``NaN``/``inf``는 반드시
    ``None``으로 바꿉니다. 명시된 ``null``은 계약이 정한 값이라 흠으로 세지 않습니다.
    """

    value = event.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        state.malformed_lines += 1
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clear_predict(state: EvaluateProgressState) -> None:
    """추론이 끝난 뒤에는 막대와 속도 기준점을 함께 버립니다."""

    state.predict_stage = None
    state.done = state.total = None
    state.speed_done = state.speed_seconds = None


def _image_counts(state: EvaluateProgressState, event: dict[str, Any]) -> dict[str, int] | None:
    numbers = {key: _count_field(state, event, key) for key in IMAGE_KEYS}
    return {key: value for key, value in numbers.items() if value is not None} or None


def _apply_evaluate_started(
    state: EvaluateProgressState, event: dict[str, Any]
) -> dict[str, Any]:
    state.run_id = _text(event.get("run_id")) or state.run_id
    state.device = _text(event.get("device")) or state.device
    state.images = _image_counts(state, event) or state.images
    state.stage = "started"

    found = state.images or {}
    pieces = ["평가 시작"]
    if state.run_id:
        pieces.append(f"run {state.run_id}")
    if state.device:
        pieces.append(f"device {state.device}")
    if "validation_images" in found:
        pieces.append(f"검증 이미지 {found['validation_images']}장")
    if "test_images" in found:
        pieces.append(f"test 이미지 {found['test_images']}장")
    return _log(" · ".join(pieces), level="info")


def _apply_predict_progress(
    state: EvaluateProgressState, event: dict[str, Any]
) -> dict[str, Any]:
    stage = _text(event.get("stage"))
    if stage not in PREDICT_STAGES:
        # 모르는 단계 이름은 그 필드만 버리고 개수는 살립니다.
        state.malformed_lines += 1
        stage = None

    if stage != state.predict_stage:
        # 단계가 바뀌면 이전 단계의 속도 기준점을 버립니다.
        state.speed_done = None
        state.speed_seconds = None
    state.predict_stage = stage
    if stage is not None:
        state.stage = stage

    # ``done``이 ``total``보다 커도 고치지 않습니다. 관측한 값을 그대로 보고합니다.
    state.done = _count_field(state, event, "done")
    state.total = _count_field(state, event, "total")

    seconds = _epoch_seconds(event.get("ts"))
    state.last_seconds = seconds
    if state.done is not None and seconds is not None and state.speed_seconds is None:
        state.speed_done, state.speed_seconds = state.done, seconds

    label = PREDICT_STAGES.get(stage or "", "추론 중")
    done_text = "?" if state.done is None else str(state.done)
    total_text = "?" if state.total is None else str(state.total)
    return _log(f"{label} · {done_text} / {total_text}", level="info")


def _apply_metrics_computed(
    state: EvaluateProgressState, event: dict[str, Any]
) -> dict[str, Any]:
    state.metrics = {key: _metric_field(state, event, key) for key in METRIC_KEYS}
    state.stage = "metrics"
    _clear_predict(state)

    found = state.metrics
    return _log(
        "지표 계산 중 · "
        + " · ".join(
            f"{key} {'-' if found[key] is None else round(found[key], 4)}" for key in METRIC_KEYS
        ),
        level="info",
    )


def _apply_submission_written(
    state: EvaluateProgressState, event: dict[str, Any]
) -> dict[str, Any]:
    rows = _count_field(state, event, "rows")
    if rows is not None:
        state.submission_rows = rows
    state.stage = "submission"
    _clear_predict(state)

    tail = "" if state.submission_rows is None else f" · {state.submission_rows}줄"
    return _log(f"submission 쓰는 중{tail}", level="info")


def _apply_evaluate_completed(
    state: EvaluateProgressState, event: dict[str, Any]
) -> dict[str, Any]:
    state.completed = _image_counts(state, event)
    state.stage = "completed"
    _clear_predict(state)

    found = state.completed or {}
    return _log(
        "평가 완료 · "
        + " · ".join(
            f"{label} {found[key]}"
            for key, label in (("validation_images", "검증"), ("test_images", "test"))
            if key in found
        ),
        level="info",
    )


_HANDLERS = {
    "evaluate_started": _apply_evaluate_started,
    "predict_progress": _apply_predict_progress,
    "metrics_computed": _apply_metrics_computed,
    "submission_written": _apply_submission_written,
    "evaluate_completed": _apply_evaluate_completed,
}


def consume_line(state: EvaluateProgressState, line: object) -> dict[str, Any] | None:
    """한 줄을 해석해 상태를 갱신하고, 화면에 남길 log 항목을 돌려줍니다.

    진행 줄이면 사람이 읽을 문장으로 바꿔 주고, 그 밖의 줄은 원문 그대로 둡니다.
    빈 줄이면 ``None``을 돌려줍니다. 어떤 경우에도 예외를 던지지 않습니다.
    """

    if not isinstance(line, str):
        return None
    text = line.rstrip("\r\n")
    if not text.strip():
        return None
    if not text.lstrip().startswith("{"):
        return _log(text)

    try:
        event = json.loads(text)
    except (ValueError, TypeError):
        return _log(text)
    if not isinstance(event, dict):
        return _log(text)

    schema = event.get("schema")
    if not isinstance(schema, str) or not schema.startswith(SCHEMA_PREFIX):
        return _log(text)
    if schema[len(SCHEMA_PREFIX) :].split(".", 1)[0] != SUPPORTED_MAJOR:
        # 모르는 major 버전은 상태를 바꾸지 않고 원문만 남깁니다.
        return _log(text)

    handler = _HANDLERS.get(event.get("event"))
    if handler is None:
        # 계약에 새 event가 추가돼도 깨지지 않습니다.
        return _log(text)

    state.saw_progress = True
    try:
        return handler(state, event)
    except Exception:  # 파서가 평가를 죽이는 일은 없어야 합니다.
        state.malformed_lines += 1
        return _log(text)


def _percent(done: int | None, total: int | None) -> float | None:
    if done is None or not total:
        return None
    # 막대가 넘치지 않도록 표시용 값만 자릅니다. ``done`` 자체는 고치지 않습니다.
    return round(min(done / total, 1.0) * 100, 1)


def _eta_seconds(state: EvaluateProgressState) -> float | None:
    """실제로 관측된 추론 속도로만 남은 시간을 계산합니다.

    같은 추론 단계에서 두 번 이상 관측했고 그 사이에 장 수와 시간이 모두 늘었을
    때만 값을 냅니다. 그 밖에는 추정하지 않습니다.
    """

    if state.predict_stage is None or state.done is None or not state.total:
        return None
    if state.speed_done is None or state.speed_seconds is None:
        return None
    predicted = state.done - state.speed_done
    if state.last_seconds is None or predicted <= 0:
        return None
    seconds = state.last_seconds - state.speed_seconds
    if seconds <= 0:
        return None
    remaining = state.total - state.done
    if remaining <= 0:
        return 0.0
    return round(remaining / (predicted / seconds), 1)


def snapshot(state: EvaluateProgressState) -> dict[str, Any]:
    """화면에 그대로 넘길 수 있는 JSON 안전한 진행 상태를 만듭니다."""

    if not state.saw_progress:
        # 지어내지 않습니다. evaluate가 진행 로그를 내보내기 전에는 여기에 머뭅니다.
        return {
            "available": False,
            "reason": "evaluate_pipeline_no_progress_stream",
            "message": "evaluate pipeline이 진행 로그를 제공하지 않아 진행률을 알 수 없습니다.",
            "stage": None,
            "stage_label": None,
            "images": None,
            "predict": None,
            "metrics": None,
            "submission_rows": None,
            "completed": None,
            "eta_seconds": None,
            "malformed_lines": state.malformed_lines,
        }

    predict = None
    if state.predict_stage is not None or state.done is not None or state.total is not None:
        predict = {
            "stage": state.predict_stage,
            "done": state.done,
            "total": state.total,
            "percent": _percent(state.done, state.total),
        }

    return {
        "available": True,
        "reason": None,
        "message": None,
        "stage": state.stage,
        "stage_label": STAGE_LABELS.get(state.stage or ""),
        "run_id": state.run_id,
        "device": state.device,
        "images": dict(state.images) if state.images else None,
        "predict": predict,
        "metrics": dict(state.metrics) if state.metrics else None,
        "submission_rows": state.submission_rows,
        "completed": dict(state.completed) if state.completed else None,
        "eta_seconds": _eta_seconds(state),
        "malformed_lines": state.malformed_lines,
    }
