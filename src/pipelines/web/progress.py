"""Train subprocess가 stderr로 흘리는 진행 로그를 읽습니다.

`docs/train_progress_contract.md`의 ``train.progress/1`` 계약을 따릅니다. 이 module의
가장 중요한 성질은 **어떤 입력에도 예외를 던지지 않는다**는 것과, 진행 줄이 하나도
없을 때 진행률을 지어내지 않고 ``available: False``를 그대로 보고한다는 것입니다.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .masking import sanitize_line


__all__ = ["ProgressState", "consume_line", "snapshot"]


SCHEMA_PREFIX = "train.progress/"
SUPPORTED_MAJOR = "1"

_WARNING_MARKERS = ("[w]", "warning", "warn:", "userwarning", "futurewarning", "deprecat")
_ERROR_MARKERS = ("[e]", "error", "traceback", "exception", "failed", "cuda out of memory")

# ``12.3%``처럼 숫자와 퍼센트만 있는 줄입니다. 터미널에서는 한 줄에서 숫자만 바뀌지만,
# pipe로 받으면 갱신마다 새 줄이 되어 수백 줄이 쌓입니다. 실제로 모델 가중치를 한 번
# 내려받는 데 598줄 중 590줄이 이것이었습니다.
_PERCENT_ONLY = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*%\s*$")

# 이만큼 진행했을 때만 한 줄 남깁니다.
PERCENT_STEP = 20.0


class ProgressState:
    """한 job의 진행 상태. 순수 데이터만 담습니다."""

    __slots__ = (
        "architecture",
        "class_count",
        "current_epoch",
        "device",
        "epochs",
        "epochs_by_number",
        "last_percent",
        "malformed_lines",
        "run_id",
        "saw_progress",
        "suppressed_lines",
        "train_images",
        "validation_images",
    )

    def __init__(self) -> None:
        self.saw_progress = False
        self.last_percent: float | None = None
        self.suppressed_lines = 0
        self.run_id: str | None = None
        self.architecture: str | None = None
        self.device: str | None = None
        self.epochs: int | None = None
        self.train_images: int | None = None
        self.validation_images: int | None = None
        self.class_count: int | None = None
        self.current_epoch: int | None = None
        # epoch 번호가 중복되거나 역순으로 와도 안전하도록 dict에 담습니다.
        self.epochs_by_number: dict[int, dict[str, Any]] = {}
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


def _finite_number(value: Any) -> float | None:
    """JSON은 NaN/Infinity를 허용하지만 브라우저의 JSON.parse는 거부합니다."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _apply_run_started(state: ProgressState, event: dict[str, Any]) -> dict[str, Any]:
    state.run_id = _text(event.get("run_id")) or state.run_id
    state.architecture = _text(event.get("architecture")) or state.architecture
    state.device = _text(event.get("device")) or state.device
    state.epochs = _positive_int(event.get("epochs")) or state.epochs
    state.train_images = _non_negative_int(event.get("train_images"))
    state.validation_images = _non_negative_int(event.get("validation_images"))
    state.class_count = _non_negative_int(event.get("class_count"))

    pieces = [f"학습 시작 · {state.architecture or '모델 미상'}"]
    if state.device:
        pieces.append(f"device {state.device}")
    if state.epochs:
        pieces.append(f"총 {state.epochs} epoch")
    if state.train_images is not None:
        pieces.append(f"학습 {state.train_images}장 / 검증 {state.validation_images}장")
    if state.class_count is not None:
        pieces.append(f"클래스 {state.class_count}개")
    return _log(" · ".join(pieces), level="info")


def _apply_epoch_started(state: ProgressState, event: dict[str, Any]) -> dict[str, Any] | None:
    epoch = _positive_int(event.get("epoch"))
    total = _positive_int(event.get("epochs"))
    if total is not None:
        state.epochs = total
    if epoch is None:
        state.malformed_lines += 1
        return None
    state.current_epoch = epoch
    total_text = f"/{state.epochs}" if state.epochs else ""
    return _log(f"epoch {epoch}{total_text} 시작", level="info")


def _apply_epoch_completed(state: ProgressState, event: dict[str, Any]) -> dict[str, Any] | None:
    epoch = _positive_int(event.get("epoch"))
    total = _positive_int(event.get("epochs"))
    if total is not None:
        state.epochs = total
    if epoch is None:
        state.malformed_lines += 1
        return None

    record = {
        "epoch": epoch,
        "train_loss": _finite_number(event.get("train_loss")),
        "validation_loss": _finite_number(event.get("validation_loss")),
        "epoch_seconds": _finite_number(event.get("epoch_seconds")),
        "is_best": bool(event.get("is_best")) if isinstance(event.get("is_best"), bool) else None,
    }
    # 같은 epoch이 다시 오면 나중 값으로 덮어씁니다.
    state.epochs_by_number[epoch] = record
    state.current_epoch = epoch

    total_text = f"/{state.epochs}" if state.epochs else ""
    best_mark = " · 최고 기록" if record["is_best"] else ""
    return _log(
        f"epoch {epoch}{total_text} 완료 · train {_format_number(record['train_loss'])}"
        f" · val {_format_number(record['validation_loss'])}{best_mark}",
        level="info",
    )


_HANDLERS = {
    "run_started": _apply_run_started,
    "epoch_started": _apply_epoch_started,
    "epoch_completed": _apply_epoch_completed,
}


def _consume_plain_line(state: ProgressState, text: str) -> dict[str, Any] | None:
    """진행률 표시만 있는 줄은 일정 간격으로만 남깁니다.

    한 번의 내려받기가 수백 줄이 되면 정작 중요한 경고와 오류가 묻힙니다.
    처음, 마지막(100%), 그리고 ``PERCENT_STEP``만큼 나아갔을 때만 남깁니다.
    """

    match = _PERCENT_ONLY.match(text)
    if match is None:
        state.last_percent = None
        return _log(text)

    try:
        value = float(match.group(1))
    except ValueError:
        state.last_percent = None
        return _log(text)

    previous = state.last_percent
    state.last_percent = value
    keep = (
        previous is None  # 시작을 알림
        or value >= 100.0  # 끝을 알림
        or value < previous  # 새 내려받기가 시작됨
        or value - previous >= PERCENT_STEP
    )
    if keep:
        return _log(text, level="info")
    state.suppressed_lines += 1
    return None


def consume_line(state: ProgressState, line: str) -> dict[str, Any] | None:
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
        return _consume_plain_line(state, text)

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
        # 모르는 event는 무시합니다. 계약에 새 event가 추가돼도 깨지지 않습니다.
        return _log(text)

    state.saw_progress = True
    try:
        return handler(state, event)
    except Exception:  # 파서가 job을 죽이는 일은 없어야 합니다.
        state.malformed_lines += 1
        return _log(text)


def _eta_seconds(state: ProgressState) -> float | None:
    """실제로 측정된 epoch 시간이 2건 이상일 때만 남은 시간을 추정합니다."""

    if not state.epochs or state.current_epoch is None:
        return None
    durations = [
        record["epoch_seconds"]
        for record in state.epochs_by_number.values()
        if record["epoch_seconds"] is not None
    ]
    if len(durations) < 2:
        return None
    remaining = state.epochs - max(state.epochs_by_number)
    if remaining <= 0:
        return 0.0
    return round(sum(durations) / len(durations) * remaining, 1)


def snapshot(state: ProgressState) -> dict[str, Any]:
    """화면에 그대로 넘길 수 있는 JSON 안전한 진행 상태를 만듭니다."""

    if not state.saw_progress:
        # 지어내지 않습니다. train이 진행 로그를 내보내기 전에는 여기에 머뭅니다.
        return {
            "available": False,
            "reason": "train_pipeline_no_progress_stream",
            "message": "train pipeline이 진행 로그를 제공하지 않아 진행률을 알 수 없습니다.",
            "epochs": [],
            "total_epochs": None,
            "current_epoch": None,
            "eta_seconds": None,
            "suppressed_lines": state.suppressed_lines,
        }

    ordered = [state.epochs_by_number[key] for key in sorted(state.epochs_by_number)]
    completed = len(ordered)
    total = state.epochs
    best = None
    scored = [record for record in ordered if record["validation_loss"] is not None]
    if scored:
        best_record = min(scored, key=lambda record: record["validation_loss"])
        best = {
            "epoch": best_record["epoch"],
            "validation_loss": best_record["validation_loss"],
        }

    return {
        "available": True,
        "reason": None,
        "message": None,
        "run_id": state.run_id,
        "architecture": state.architecture,
        "device": state.device,
        "train_images": state.train_images,
        "validation_images": state.validation_images,
        "class_count": state.class_count,
        "total_epochs": total,
        "current_epoch": state.current_epoch,
        "completed_epochs": completed,
        "percent": round(completed / total * 100, 1) if total else None,
        "eta_seconds": _eta_seconds(state),
        "epochs": ordered,
        "best": best,
        "malformed_lines": state.malformed_lines,
        "suppressed_lines": state.suppressed_lines,
    }
