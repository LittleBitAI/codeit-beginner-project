"""융합 실행을 한 번에 하나만 돌립니다.

`evaluation.py`의 `EvaluationRunner`를 그대로 쓸 수 없습니다. 그쪽은 **학습 기록 하나**
(`JobRecord`)에 매여 있는데, 융합에는 그런 기록이 없습니다 — 여러 실행의 예측을 합치는
것이라 어느 학습의 결과도 아닙니다.

대신 `run_evaluation()`은 config만 받으므로 그대로 씁니다. 여기서 하는 일은 상태를
들고 있는 것뿐입니다.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import JobConflictError
from .evaluation import (
    STATUS_FAILED,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    run_evaluation,
)


__all__ = ["EnsembleRunner", "get_ensemble_runner"]

#: 화면에 남길 진행 줄 수입니다. 융합은 몇 분이라 길게 쌓을 이유가 없고, 무한히
#: 모으면 오래 띄워 둔 창이 메모리를 먹습니다.
_LOG_LIMIT = 200


class EnsembleRunner:
    """한 번에 하나. 융합은 CPU로 돌지만 같은 출력 자리를 두 번 쓰면 서로를 덮습니다."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {"status": STATUS_IDLE}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(
        self,
        run_ids: Sequence[str],
        *,
        run_id: str,
        allow_copied_images: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """예측이 없는 실행은 먼저 만들고, 그다음 합칩니다.

        두 단계를 한 작업으로 묶는 것은 사람이 **한 번만** 결정하게 하려는 것입니다.
        나눠 두면 "추론이 끝났으니 이제 합치기를 눌러 주세요"를 기다려야 하고, 그
        사이에 왜 눌러야 하는지 잊습니다.
        """

        from . import ensemble

        # **추론을 걸기 전에 거절할 것은 먼저 거절합니다.** 검증을 뒤로 미루면 예측
        # 없는 후보 하나만 보내도 GPU가 9분을 돌고 나서야 "둘 이상 필요"로 실패합니다.
        ensemble.check_selection(
            run_ids,
            run_id=run_id,
            allow_copied_images=bool(allow_copied_images),
            overwrite=bool(overwrite),
        )
        pending = ensemble.pending_runs(run_ids)
        with self._lock:
            if self._state.get("status") == STATUS_RUNNING:
                raise JobConflictError(
                    f"이미 도는 중입니다: {self._state.get('run_id')}"
                )
            self._state = {
                "status": STATUS_RUNNING,
                "run_id": run_id,
                "run_ids": list(run_ids),
                "stage": "harvest" if pending else "fuse",
                "pending": [item["run_id"] for item in pending],
                "logs": [],
            }
            state = dict(self._state)

        thread = threading.Thread(
            target=self._run_all,
            args=(list(run_ids), run_id, bool(allow_copied_images), bool(overwrite), pending),
            daemon=True,
        )
        thread.start()
        return state

    def _run_all(
        self,
        run_ids: list[str],
        run_id: str,
        allow_copied_images: bool,
        overwrite: bool,
        pending: list[Mapping[str, Any]],
    ) -> None:
        from . import ensemble

        for index, candidate in enumerate(pending, start=1):
            name = str(candidate.get("run_id"))
            with self._lock:
                self._state["stage"] = "harvest"
                self._state["harvesting"] = name
                self._state["harvest_progress"] = [index, len(pending)]
            self._append_log(f"[{index}/{len(pending)}] {name} test 예측을 만드는 중")
            try:
                result = run_evaluation(ensemble.build_harvest_config(candidate), self._append_log)
            except Exception as error:  # noqa: BLE001 - thread는 예외를 밖으로 못 보냅니다.
                self._fail(run_id, f"{name} 예측을 만들지 못했습니다({type(error).__name__}).")
                return
            if not result.get("ok"):
                self._fail(run_id, f"{name} 예측을 만들지 못했습니다: {result.get('message')}")
                return

        with self._lock:
            self._state["stage"] = "fuse"
            self._state.pop("harvesting", None)
        try:
            config = ensemble.build_fusion_config(
                run_ids,
                run_id=run_id,
                allow_copied_images=allow_copied_images,
                overwrite=overwrite,
            )
        except Exception as error:  # noqa: BLE001
            self._fail(run_id, f"융합 설정을 만들지 못했습니다({type(error).__name__}).")
            return
        self._run(config, run_id)

    def _fail(self, run_id: str, message: str) -> None:
        with self._lock:
            logs = self._state.get("logs") if isinstance(self._state.get("logs"), list) else []
            self._state = {
                "status": STATUS_FAILED,
                "run_id": run_id,
                "message": message,
                "logs": list(logs),
            }

    def _append_log(self, line: str) -> None:
        with self._lock:
            logs = self._state.get("logs")
            if isinstance(logs, list):
                logs.append(line)
                del logs[:-_LOG_LIMIT]

    def _run(self, config: dict[str, Any], run_id: str) -> None:
        try:
            result = run_evaluation(config, self._append_log)
        except Exception as error:  # noqa: BLE001 - thread는 예외를 밖으로 못 보냅니다.
            self._fail(run_id, f"융합에 실패했습니다({type(error).__name__}).")
            return

        artifacts = result.get("artifacts") if isinstance(result, Mapping) else None
        summary = result.get("summary") if isinstance(result, Mapping) else None
        with self._lock:
            logs = self._state.get("logs") if isinstance(self._state.get("logs"), list) else []
            self._state = {
                "status": STATUS_SUCCEEDED if result.get("ok") else STATUS_FAILED,
                "run_id": run_id,
                "message": result.get("message"),
                "artifacts": artifacts if isinstance(artifacts, Mapping) else {},
                "summary": summary if isinstance(summary, Mapping) else {},
                "logs": list(logs),
            }


_RUNNER: EnsembleRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_ensemble_runner() -> EnsembleRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = EnsembleRunner()
        return _RUNNER
