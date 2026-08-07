"""``evaluate.progress/1`` 진행 로그를 stderr에 JSON Lines로 내보냅니다.

평가가 지금 무엇을 하고 있는지 알리기 위한 부가 출력일 뿐이며, 산출물이나
``run(config)`` 반환값에 어떤 영향도 주지 않습니다. 그래서 emitter는 어떤
예외도 밖으로 내보내지 않습니다.

**stdout에는 아무것도 쓰지 않습니다.** ``COCOeval``이 stdout에 쓰기 때문에 그
호출을 ``redirect_stdout``으로 감싸 두었고, web이 그 subprocess 로그를
파싱합니다. 진행 로그가 stdout에 한 글자라도 더하면 그 파싱이 깨집니다.

``src/pipelines/data/progress.py``의 ``data.progress/1`` emitter와 같은
구조이며, 계약 정본은 web의 진행 로그 문서입니다.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


SCHEMA = "evaluate.progress/1"

# 추론 event를 이미지마다 내보내면 대회 test 842장이 그대로 pipe에 쏟아집니다.
# 전체의 2% 이상 진행했거나 마지막 event로부터 1초 이상 지났을 때만 내보냅니다.
PREDICT_PROGRESS_MIN_RATIO = 0.02
PREDICT_PROGRESS_MIN_SECONDS = 1.0

# 진행 로그에 싣는 주요 지표입니다. metrics 문서의 이름을 그대로 씁니다.
METRIC_FIELDS = ("mAP", "mAP50", "mAP75")


def _timestamp() -> str:
    """UTC ISO-8601 시각을 마이크로초와 ``Z`` 접미사까지 포함해 만듭니다."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _finite_or_none(value: Any) -> float | None:
    """계산되지 않았거나 유한하지 않은 지표를 ``null``로 만듭니다.

    이 pipeline은 측정되지 않은 지표를 ``0.0``이 아니라 ``null``로 둡니다.
    진행 로그도 같은 규칙을 지키고, ``NaN``/``inf``는 브라우저 ``JSON.parse``가
    읽지 못하므로 여기서 ``null``로 바꿉니다.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


class ProgressEmitter:
    """평가를 방해하지 않고 진행 상황 한 건을 한 줄로 기록합니다."""

    def __init__(self, stream: Any | None = None) -> None:
        self._stream = stream
        # 앞으로 process가 늘어도 중복 줄이 나오지 않도록 emitter를 만든
        # process에서만 출력합니다.
        self._pid = os.getpid()
        # stage별 마지막 출력 지점입니다: {stage: (done, monotonic 시각)}
        self._last_predict: dict[str, tuple[int, float]] = {}

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
                **fields,
                "ts": _timestamp(),
            }
            line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
        except Exception:
            # 소비자가 평가를 중단하면 pipe가 닫히고 다음 write가
            # BrokenPipeError(Windows에서는 맨 OSError)를 냅니다. 진행 로그
            # 실패가 평가나 exit code를 바꾸면 안 됩니다.
            return

    def predict_progress(self, stage: str, done: int, total: int) -> None:
        """추론 진행을 알립니다. 출력량 제한을 통과한 것만 나갑니다."""

        try:
            if not self._should_emit(stage, done, total):
                return
        except Exception:
            return
        self.emit("predict_progress", stage=stage, done=done, total=total)

    def metrics_computed(self, metrics: Mapping[str, Any] | None) -> None:
        """주요 지표를 알립니다. 계산되지 않은 지표는 ``null``로 나갑니다."""

        try:
            fields = {
                name: _finite_or_none(metrics.get(name)) for name in METRIC_FIELDS
            }
        except Exception:
            return
        self.emit("metrics_computed", **fields)

    def _should_emit(self, stage: str, done: int, total: int) -> bool:
        """제한 규칙을 적용하고, 내보내기로 정했으면 기준점을 갱신합니다."""

        now = time.monotonic()
        last = self._last_predict.get(stage)
        # stage의 첫 줄은 화면이 전체 개수를 알아야 진행률을 그릴 수 있으므로
        # 바로 내보내고, 마지막 이미지는 언제나 내보냅니다.
        if last is None or done >= total:
            self._last_predict[stage] = (done, now)
            return True
        last_done, last_time = last
        enough_progress = done - last_done >= total * PREDICT_PROGRESS_MIN_RATIO
        enough_time = now - last_time >= PREDICT_PROGRESS_MIN_SECONDS
        if enough_progress or enough_time:
            self._last_predict[stage] = (done, now)
            return True
        return False


__all__ = ["SCHEMA", "ProgressEmitter"]
