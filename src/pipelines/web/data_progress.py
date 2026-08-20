"""데이터 준비 subprocess가 stderr로 흘리는 진행 로그를 읽습니다.

`docs/data_progress_contract.md`의 ``data.progress/1`` 계약을 따릅니다.
``progress.py``(train 쪽 소비자)와 같은 모양이고, 가장 중요한 성질도 같습니다.
**어떤 입력에도 예외를 던지지 않고**, 진행 줄이 하나도 없으면 진행률을 지어내지 않고
``available: False``를 그대로 보고합니다.

I/O가 없는 순수 함수만 있습니다. 남은 시간도 event가 들고 온 ``ts``로만 계산하며,
관측이 부족하면 추정하지 않습니다.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .masking import sanitize_line


__all__ = ["DataProgressState", "consume_line", "snapshot"]


SCHEMA_PREFIX = "data.progress/"
SUPPORTED_MAJOR = "1"

_WARNING_MARKERS = ("[w]", "warning", "warn:", "userwarning", "futurewarning", "deprecat")
_ERROR_MARKERS = ("[e]", "error", "traceback", "exception", "failed", "credential")

# 읽기 단계 이름과 화면에 쓸 문구입니다. 계약에 없는 이름은 받아들이지 않습니다.
READ_STAGES = {
    "annotations": "annotation 읽는 중",
    "test_images": "test 이미지 읽는 중",
    # 은행은 단계 시작도 알리고 개수도 셉니다. 그래서 아래 STEPS에도 있습니다.
    # 한쪽만 알면 준비에서 가장 오래 걸리는 단계가 통째로 "모르는 단계"가 됩니다.
    "crop_bank": "참조 crop 자르는 중",
}

# 읽기가 끝난 뒤의 단계입니다.
STEPS = {
    "split": "나누는 중",
    "manifests": "manifest 만드는 중",
    "crop_bank": READ_STAGES["crop_bank"],
    "publish": "올리는 중",
}

STAGE_LABELS = {
    "listing": "원본 목록 확인",
    "completed": "준비 완료",
    **READ_STAGES,
    **STEPS,
}


class DataProgressState:
    """준비 실행 한 건의 진행 상태. 순수 데이터만 담습니다."""

    __slots__ = (
        "completed",
        "done",
        "last_seconds",
        "malformed_lines",
        "raw_prefix",
        "read_stage",
        "seed",
        "saw_progress",
        "sources",
        "speed_done",
        "speed_seconds",
        "split_method",
        "split_ratio",
        "stage",
        "total",
    )

    def __init__(self) -> None:
        self.saw_progress = False
        self.stage: str | None = None
        self.raw_prefix: str | None = None
        self.split_ratio: str | None = None
        self.seed: int | None = None
        self.split_method: str | None = None
        self.sources: dict[str, int] | None = None
        self.read_stage: str | None = None
        self.done: int | None = None
        self.total: int | None = None
        self.completed: dict[str, int] | None = None
        # 남은 시간을 재기 위한 관측 기준점입니다. (읽은 개수, ts) 한 쌍이며 읽기
        # 단계가 바뀌면 버립니다. 직전 단계의 속도로 다음 단계를 재면 거짓말이 됩니다.
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


def _count_field(state: DataProgressState, event: dict[str, Any], key: str) -> int | None:
    """개수 필드 하나를 읽습니다. 타입이 틀리면 그 필드만 버립니다."""

    if key not in event:
        return None
    number = _non_negative_int(event[key])
    if number is None:
        state.malformed_lines += 1
    return number


def _apply_prepare_started(state: DataProgressState, event: dict[str, Any]) -> dict[str, Any]:
    state.raw_prefix = _text(event.get("raw_prefix")) or state.raw_prefix
    state.split_ratio = _text(event.get("split_ratio")) or state.split_ratio
    state.split_method = _text(event.get("split_method")) or state.split_method
    seed = _count_field(state, event, "seed")
    if seed is not None:
        state.seed = seed
    state.stage = "listing"

    pieces = ["데이터 준비 시작"]
    if state.raw_prefix:
        pieces.append(f"원본 {state.raw_prefix}")
    if state.split_ratio:
        pieces.append(f"분할 {state.split_ratio}")
    if state.seed is not None:
        pieces.append(f"seed {state.seed}")
    return _log(" · ".join(pieces), level="info")


def _apply_sources_listed(state: DataProgressState, event: dict[str, Any]) -> dict[str, Any]:
    sources = {
        key: _count_field(state, event, key)
        for key in ("train_images", "annotations", "test_images")
    }
    state.sources = {key: value for key, value in sources.items() if value is not None} or None
    state.stage = state.stage or "listing"

    found = state.sources or {}
    return _log(
        "원본 목록 확인 · "
        + " · ".join(
            f"{label} {found[key]}개"
            for key, label in (
                ("train_images", "학습 이미지"),
                ("annotations", "annotation"),
                ("test_images", "test 이미지"),
            )
            if key in found
        ),
        level="info",
    )


def _apply_read_progress(state: DataProgressState, event: dict[str, Any]) -> dict[str, Any]:
    stage = _text(event.get("stage"))
    if stage not in READ_STAGES:
        # 모르는 단계 이름은 그 필드만 버리고 개수는 살립니다.
        state.malformed_lines += 1
        stage = None

    if stage != state.read_stage:
        # 단계가 바뀌면 이전 단계의 속도 기준점을 버립니다.
        state.speed_done = None
        state.speed_seconds = None
    state.read_stage = stage
    if stage is not None:
        state.stage = stage

    # ``done``이 ``total``보다 커도 고치지 않습니다. 관측한 값을 그대로 보고합니다.
    state.done = _count_field(state, event, "done")
    state.total = _count_field(state, event, "total")

    seconds = _epoch_seconds(event.get("ts"))
    state.last_seconds = seconds
    if state.done is not None and seconds is not None and state.speed_seconds is None:
        state.speed_done, state.speed_seconds = state.done, seconds

    label = READ_STAGES.get(stage or "", "읽는 중")
    done_text = "?" if state.done is None else str(state.done)
    total_text = "?" if state.total is None else str(state.total)
    return _log(f"{label} · {done_text} / {total_text}", level="info")


def _apply_step_started(state: DataProgressState, event: dict[str, Any]) -> dict[str, Any]:
    step = _text(event.get("step"))
    if step not in STEPS:
        # 모르는 단계는 지어내지 않습니다. 화면은 직전 단계를 그대로 둡니다.
        state.malformed_lines += 1
        return _log(f"모르는 준비 단계입니다: {step}", level="info")
    state.stage = step
    state.read_stage = None
    state.done = state.total = None
    state.speed_done = state.speed_seconds = None
    return _log(STEPS[step], level="info")


def _apply_prepare_completed(state: DataProgressState, event: dict[str, Any]) -> dict[str, Any]:
    numbers = {
        key: _count_field(state, event, key)
        for key in ("train_images", "validation_images", "category_count")
    }
    state.completed = {key: value for key, value in numbers.items() if value is not None} or None
    state.stage = "completed"
    state.read_stage = None
    state.done = state.total = None

    found = state.completed or {}
    return _log(
        "준비 완료 · "
        + " · ".join(
            f"{label} {found[key]}"
            for key, label in (
                ("train_images", "학습"),
                ("validation_images", "검증"),
                ("category_count", "클래스"),
            )
            if key in found
        ),
        level="info",
    )


_HANDLERS = {
    "prepare_started": _apply_prepare_started,
    "sources_listed": _apply_sources_listed,
    "read_progress": _apply_read_progress,
    "step_started": _apply_step_started,
    "prepare_completed": _apply_prepare_completed,
}


def consume_line(state: DataProgressState, line: object) -> dict[str, Any] | None:
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
    except Exception:  # 파서가 준비를 죽이는 일은 없어야 합니다.
        state.malformed_lines += 1
        return _log(text)


def _percent(done: int | None, total: int | None) -> float | None:
    if done is None or not total:
        return None
    # 막대가 넘치지 않도록 표시용 값만 자릅니다. ``done`` 자체는 고치지 않습니다.
    return round(min(done / total, 1.0) * 100, 1)


def _eta_seconds(state: DataProgressState) -> float | None:
    """실제로 관측된 읽기 속도로만 남은 시간을 계산합니다.

    같은 읽기 단계에서 두 번 이상 관측했고 그 사이에 개수와 시간이 모두 늘었을
    때만 값을 냅니다. 그 밖에는 추정하지 않습니다.
    """

    if state.read_stage is None or state.done is None or not state.total:
        return None
    if state.speed_done is None or state.speed_seconds is None:
        return None
    read = state.done - state.speed_done
    if state.last_seconds is None or read <= 0:
        return None
    seconds = state.last_seconds - state.speed_seconds
    if seconds <= 0:
        return None
    remaining = state.total - state.done
    if remaining <= 0:
        return 0.0
    return round(remaining / (read / seconds), 1)


def snapshot(state: DataProgressState) -> dict[str, Any]:
    """화면에 그대로 넘길 수 있는 JSON 안전한 진행 상태를 만듭니다."""

    if not state.saw_progress:
        # 지어내지 않습니다. data가 진행 로그를 내보내기 전에는 여기에 머뭅니다.
        return {
            "available": False,
            "reason": "data_pipeline_no_progress_stream",
            "message": "data pipeline이 진행 로그를 제공하지 않아 진행률을 알 수 없습니다.",
            "stage": None,
            "stage_label": None,
            "sources": None,
            "read": None,
            "completed": None,
            "eta_seconds": None,
            "malformed_lines": state.malformed_lines,
        }

    read = None
    if state.read_stage is not None or state.done is not None or state.total is not None:
        read = {
            "stage": state.read_stage,
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
        "raw_prefix": state.raw_prefix,
        "split_ratio": state.split_ratio,
        "seed": state.seed,
        "split_method": state.split_method,
        "sources": dict(state.sources) if state.sources else None,
        "read": read,
        "completed": dict(state.completed) if state.completed else None,
        "eta_seconds": _eta_seconds(state),
        "malformed_lines": state.malformed_lines,
    }
