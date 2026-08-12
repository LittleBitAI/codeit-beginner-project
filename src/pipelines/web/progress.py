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


__all__ = [
    "ProgressState",
    "consume_line",
    "seed_epochs",
    "snapshot",
    "take_quiet_change",
]


SCHEMA_PREFIX = "train.progress/"
SUPPORTED_MAJOR = "1"

# ``step_progress``가 쓰는 phase입니다. 계약에 없는 이름은 화면에 보여 주지 않습니다.
STEP_PHASES = ("train", "validation")

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
        "finished",
        "last_percent",
        "malformed_lines",
        "quiet_change",
        "reported_completed_epochs",
        "run_id",
        "saw_progress",
        "step",
        "stopped_early",
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
        # 지금 지나고 있는 epoch 안의 batch 위치입니다. epoch 경계에서 지웁니다.
        self.step: dict[str, Any] | None = None
        # log 줄 없이 상태만 바뀌었다는 표시입니다. `take_quiet_change`가 읽고 지웁니다.
        self.quiet_change = False
        self.finished = False
        # train이 알려 준 실제 수행 횟수입니다. 없으면 받은 epoch 수로 셉니다.
        self.reported_completed_epochs: int | None = None
        self.stopped_early: bool | None = None


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


def _loss_components(value: Any) -> dict[str, float] | None:
    """모델이 돌려준 개별 loss를 이름 그대로 담습니다.

    이름은 모델마다 다르므로(Faster R-CNN과 RetinaNet이 서로 다릅니다) 열거하지도
    공통 이름으로 바꾸지도 않습니다. 쓸 수 있는 항목이 하나도 없으면 빈 dict 대신
    ``None``을 돌려주어 화면이 없는 값을 지어내지 않게 합니다.
    """

    if not isinstance(value, dict):
        return None
    components: dict[str, float] = {}
    for name, item in value.items():
        number = _finite_number(item)
        if name and number is not None:
            components[str(name)] = number
    return components or None


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
    # 새 epoch은 batch 0부터 다시 셉니다.
    state.step = None
    total_text = f"/{state.epochs}" if state.epochs else ""
    return _log(f"epoch {epoch}{total_text} 시작", level="info")


def _apply_step_progress(state: ProgressState, event: dict[str, Any]) -> dict[str, Any] | None:
    """지금 epoch 안에서 몇 번째 batch를 지나고 있는지 기록합니다.

    **log 줄을 만들지 않습니다.** 이 event는 phase마다 5초에 한 번씩 나오므로 그대로
    남기면 긴 학습에서 정작 중요한 경고와 오류가 묻힙니다. 위치는 진행 막대로 보여
    줍니다. 대신 화면이 갱신되도록 `quiet_change`를 세워 둡니다.
    """

    total_epochs = _positive_int(event.get("epochs"))
    if total_epochs is not None:
        state.epochs = total_epochs
    epoch = _positive_int(event.get("epoch"))
    if epoch is not None:
        # epoch_started를 놓쳤어도 몇 번째 epoch인지는 알 수 있습니다.
        state.current_epoch = epoch

    phase = _text(event.get("phase"))
    step = _positive_int(event.get("step"))
    total_steps = _positive_int(event.get("total_steps"))
    if phase not in STEP_PHASES or step is None or total_steps is None or step > total_steps:
        # 위치를 지어내느니 batch 표시를 접습니다. epoch 진행률은 그대로 남습니다.
        state.malformed_lines += 1
        return None

    state.step = {
        "phase": phase,
        "step": step,
        "total_steps": total_steps,
        "percent": round(step / total_steps * 100, 1),
    }
    state.quiet_change = True
    return None


def _epoch_record(epoch: int, event: dict[str, Any]) -> dict[str, Any]:
    """끝난 epoch 하나를 화면이 그릴 수 있는 모양으로 만듭니다."""

    return {
        "epoch": epoch,
        "train_loss": _finite_number(event.get("train_loss")),
        "validation_loss": _finite_number(event.get("validation_loss")),
        "train_loss_components": _loss_components(event.get("train_loss_components")),
        "validation_loss_components": _loss_components(event.get("validation_loss_components")),
        "epoch_seconds": _finite_number(event.get("epoch_seconds")),
        "is_best": bool(event.get("is_best")) if isinstance(event.get("is_best"), bool) else None,
        # schedule을 쓰지 않는 실행과 이 계약 이전의 옛 실행은 None입니다.
        "learning_rate": _finite_number(event.get("learning_rate")),
    }


def seed_epochs(state: ProgressState, entries: Any) -> None:
    """앞선 실행이 이미 끝낸 epoch를 진행 상태에 미리 채웁니다.

    이어서 학습한 실행은 checkpoint에서 출발하므로 그전 epoch가 진행 log로 오지
    않습니다. 채우지 않으면 손실 그래프가 이어붙인 지점에서 시작해, 앞의 곡선이
    사라진 것처럼 보입니다. 저장돼 있던 기록이라 key가 빠져 있을 수 있으므로 지금
    계약의 모양으로 다시 맞춥니다.
    """

    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        epoch = _positive_int(entry.get("epoch"))
        if epoch is not None:
            state.epochs_by_number[epoch] = _epoch_record(epoch, entry)


def _apply_epoch_completed(state: ProgressState, event: dict[str, Any]) -> dict[str, Any] | None:
    epoch = _positive_int(event.get("epoch"))
    total = _positive_int(event.get("epochs"))
    if total is not None:
        state.epochs = total
    if epoch is None:
        state.malformed_lines += 1
        return None

    record = _epoch_record(epoch, event)
    # 같은 epoch이 다시 오면 나중 값으로 덮어씁니다.
    state.epochs_by_number[epoch] = record
    state.current_epoch = epoch
    # 끝난 epoch의 batch 위치가 남아 있으면 아직 도는 것처럼 읽힙니다.
    state.step = None

    total_text = f"/{state.epochs}" if state.epochs else ""
    best_mark = " · 최고 기록" if record["is_best"] else ""
    return _log(
        f"epoch {epoch}{total_text} 완료 · train {_format_number(record['train_loss'])}"
        f" · val {_format_number(record['validation_loss'])}{best_mark}",
        level="info",
    )


def _apply_training_completed(state: ProgressState, event: dict[str, Any]) -> dict[str, Any]:
    """학습이 끝났다는 사실과 실제로 돈 epoch 수를 기록합니다.

    조기 종료로 계획보다 일찍 끝나면 받은 epoch 수만으로는 진행률이 24%에 멈춘 것처럼
    보입니다. 끝났다는 사실을 알아야 100%와 남은 시간 0을 말할 수 있습니다.
    """

    state.finished = True
    state.step = None
    planned = _positive_int(event.get("planned_epochs"))
    if planned is not None and state.epochs is None:
        # run_started를 놓쳤을 때만 씁니다. 계획 epoch는 원래 그 event가 알려 줍니다.
        state.epochs = planned
    state.reported_completed_epochs = _positive_int(event.get("completed_epochs"))
    stopped_early = event.get("stopped_early")
    state.stopped_early = stopped_early if isinstance(stopped_early, bool) else None

    completed = state.reported_completed_epochs
    if completed is None:
        completed = len(state.epochs_by_number)
    total_text = f"/{state.epochs}" if state.epochs else ""
    pieces = [f"학습 완료 · {completed}{total_text} epoch"]
    if state.stopped_early:
        pieces.append("조기 종료")
    best_epoch = _positive_int(event.get("best_epoch"))
    best_loss = _finite_number(event.get("best_validation_loss"))
    if best_epoch is not None and best_loss is not None:
        pieces.append(f"최고 val {_format_number(best_loss)} (epoch {best_epoch})")
    return _log(" · ".join(pieces), level="info")


_HANDLERS = {
    "run_started": _apply_run_started,
    "epoch_started": _apply_epoch_started,
    "step_progress": _apply_step_progress,
    "epoch_completed": _apply_epoch_completed,
    "training_completed": _apply_training_completed,
}


def take_quiet_change(state: ProgressState) -> bool:
    """log 줄 없이 상태만 바뀌었는지 알려 주고 표시를 지웁니다.

    `consume_line`이 ``None``을 돌려주는 경우는 두 가지입니다. 하나는 버릴 줄이고,
    다른 하나는 log에 남기지 않는 batch 진행입니다. 뒤쪽은 화면을 갱신해야 합니다.
    """

    changed = state.quiet_change
    state.quiet_change = False
    return changed


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

    if state.finished:
        # 조기 종료로 계획 epoch가 남아 있어도 더 기다릴 시간은 없습니다.
        return 0.0
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
            "step": None,
            "eta_seconds": None,
            "suppressed_lines": state.suppressed_lines,
        }

    ordered = [state.epochs_by_number[key] for key in sorted(state.epochs_by_number)]
    # train이 알려 준 실제 수행 횟수를 우선합니다. 없으면 지금까지 본 epoch **번호**로
    # 셉니다. 개수로 세면 11 epoch부터 이어서 학습한 실행이 로그에는 `epoch 11/12`라고
    # 뜨는데 화면은 0/12에서 다시 시작한 것처럼 보입니다.
    completed = state.reported_completed_epochs
    if completed is None:
        completed = max(
            max(state.epochs_by_number, default=0),
            (state.current_epoch or 1) - 1,
        )
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
        # 지금 지나는 batch 위치입니다. 계약 이전의 실행과 epoch 사이에서는 None입니다.
        "step": state.step,
        "completed_epochs": completed,
        # 끝난 학습은 계획 epoch가 남아 있어도 100%입니다.
        "finished": state.finished,
        "stopped_early": state.stopped_early,
        "percent": 100.0 if state.finished else (round(completed / total * 100, 1) if total else None),
        "eta_seconds": _eta_seconds(state),
        "epochs": ordered,
        "best": best,
        "malformed_lines": state.malformed_lines,
        "suppressed_lines": state.suppressed_lines,
    }
