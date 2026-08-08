"""``train.progress/1`` 진행 로그를 stderr에 JSON Lines로 내보냅니다.

이 stream은 학습을 관찰하기 위한 부가 출력일 뿐이며, 학습 결과나
``run(config)`` 반환값에 어떤 영향도 주지 않습니다. 그래서 emitter는 어떤
예외도 밖으로 내보내지 않고, stdout에는 아무것도 쓰지 않습니다.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Callable


SCHEMA = "train.progress/1"
STEP_PROGRESS_INTERVAL_SECONDS = 5.0


def _timestamp() -> str:
    """UTC ISO-8601 시각을 마이크로초와 ``Z`` 접미사까지 포함해 만듭니다."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_safe(value: Any) -> Any:
    """``NaN``/``inf``는 유효한 JSON이 아니므로 ``null``로 바꿉니다."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class ProgressEmitter:
    """학습을 방해하지 않고 진행 상황 한 건을 한 줄로 기록합니다."""

    def __init__(
        self,
        run_id: str,
        stream: Any | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run_id = run_id
        self._stream = stream
        self._clock = clock
        self._step_phase: tuple[int, str] | None = None
        self._last_step_emitted_at: float | None = None
        # DataLoader worker는 stderr를 상속하므로, emitter를 만든 process에서만
        # 출력해 중복 줄이 나오지 않게 합니다.
        self._pid = os.getpid()

    def emit(self, event: str, **fields: Any) -> None:
        """진행 event 한 줄을 stderr에 flush까지 마쳐 기록합니다."""

        try:
            if os.getpid() != self._pid:
                return
            stream = sys.stderr if self._stream is None else self._stream
            if stream is None:
                return
            payload = {
                "schema": SCHEMA,
                "event": event,
                "run_id": self._run_id,
                **{name: _json_safe(value) for name, value in fields.items()},
                "ts": _timestamp(),
            }
            line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
        except Exception:
            # 학습 취소로 pipe가 닫히면 BrokenPipeError(Windows에서는 OSError)가
            # 납니다. 진행 로그 실패가 학습이나 exit code를 바꾸면 안 됩니다.
            return

    def emit_step_progress(
        self,
        *,
        epoch: int,
        epochs: int,
        phase: str,
        step: int,
        total_steps: int,
    ) -> None:
        """각 phase의 첫·마지막 step과 5초 간격의 중간 step을 기록합니다."""

        try:
            now = self._clock()
            phase_key = (epoch, phase)
            if phase_key != self._step_phase:
                self._step_phase = phase_key
                self._last_step_emitted_at = None
            should_emit = (
                step == 1
                or step == total_steps
                or self._last_step_emitted_at is None
                or now - self._last_step_emitted_at >= STEP_PROGRESS_INTERVAL_SECONDS
            )
            if not should_emit:
                return
            self._last_step_emitted_at = now
            self.emit(
                "step_progress",
                epoch=epoch,
                epochs=epochs,
                phase=phase,
                step=step,
                total_steps=total_steps,
            )
        except Exception:
            # clock이나 출력 stream이 실패해도 학습 결과에는 영향을 주지 않습니다.
            return
