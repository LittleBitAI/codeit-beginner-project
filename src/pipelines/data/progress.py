"""``data.progress/1`` 진행 로그를 stderr에 JSON Lines로 내보냅니다.

준비가 지금 무엇을 하고 있는지 알리기 위한 부가 출력일 뿐이며, 산출물이나
``run(config)`` 반환값에 어떤 영향도 주지 않습니다. 그래서 emitter는 어떤
예외도 밖으로 내보내지 않고, stdout에는 아무것도 쓰지 않습니다.

``src/pipelines/train/progress.py``의 ``train.progress/1`` emitter와 같은
구조이며, 계약 정본은 web의 진행 로그 문서입니다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any


SCHEMA = "data.progress/1"

# 읽기 event를 항목마다 내보내면 원본 1,842개가 그대로 pipe에 쏟아집니다.
# 전체의 2% 이상 진행했거나 마지막 event로부터 1초 이상 지났을 때만 내보냅니다.
READ_PROGRESS_MIN_RATIO = 0.02
READ_PROGRESS_MIN_SECONDS = 1.0


def _timestamp() -> str:
    """UTC ISO-8601 시각을 마이크로초와 ``Z`` 접미사까지 포함해 만듭니다."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ProgressEmitter:
    """준비를 방해하지 않고 진행 상황 한 건을 한 줄로 기록합니다."""

    def __init__(self, stream: Any | None = None) -> None:
        self._stream = stream
        # 읽기는 thread로 돌지만, 앞으로 process가 늘어도 중복 줄이 나오지
        # 않도록 emitter를 만든 process에서만 출력합니다.
        self._pid = os.getpid()
        # stage별 마지막 출력 지점입니다: {stage: (done, monotonic 시각)}
        self._last_read: dict[str, tuple[int, float]] = {}

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
            # 소비자가 준비를 중단하면 pipe가 닫히고 다음 write가
            # BrokenPipeError(Windows에서는 맨 OSError)를 냅니다. 진행 로그
            # 실패가 준비나 exit code를 바꾸면 안 됩니다.
            return

    def read_progress(self, stage: str, done: int, total: int) -> None:
        """읽기 진행을 알립니다. 출력량 제한을 통과한 것만 나갑니다."""

        try:
            if not self._should_emit(stage, done, total):
                return
        except Exception:
            return
        self.emit("read_progress", stage=stage, done=done, total=total)

    def _should_emit(self, stage: str, done: int, total: int) -> bool:
        """제한 규칙을 적용하고, 내보내기로 정했으면 기준점을 갱신합니다."""

        now = time.monotonic()
        last = self._last_read.get(stage)
        # stage의 첫 줄은 화면이 전체 개수를 알아야 진행률을 그릴 수 있으므로
        # 바로 내보내고, 마지막 항목은 언제나 내보냅니다.
        if last is None or done >= total:
            self._last_read[stage] = (done, now)
            return True
        last_done, last_time = last
        enough_progress = done - last_done >= total * READ_PROGRESS_MIN_RATIO
        enough_time = now - last_time >= READ_PROGRESS_MIN_SECONDS
        if enough_progress or enough_time:
            self._last_read[stage] = (done, now)
            return True
        return False
