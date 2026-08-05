"""데이터 준비를 백그라운드로 한 번에 하나만 실행합니다.

원본 전체를 읽어야 해서 몇 분이 걸릴 수 있습니다. HTTP 요청 하나를 그동안 붙잡고
있으면 browser나 proxy가 먼저 끊어 버리므로, 시작만 시키고 상태를 따로 물어보게
합니다. 학습 job과는 별개 process라 동시에 돌아도 서로 방해하지 않습니다.
"""

from __future__ import annotations

import threading
from typing import Any

from . import datasets
from .errors import JobConflictError, WebValidationError
from .masking import sanitize_line


__all__ = ["PreparationRunner", "get_preparation_runner"]

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


class PreparationRunner:
    """준비 실행 한 건의 상태를 들고 있습니다."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {"status": STATUS_IDLE}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self, request: dict[str, Any]) -> dict[str, Any]:
        """준비를 시작합니다. 이미 돌고 있으면 거부합니다."""

        # config를 먼저 만들어, 잘못된 요청이면 thread를 띄우기 전에 거부합니다.
        backend = request.get("backend") or "auto"
        config = datasets.build_prepare_config(
            request["split_ratio"],
            seed=request.get("seed", 42),
            overwrite=request.get("overwrite", False),
            backend=backend,
            raw_prefix=request.get("raw_prefix"),
            processed_root=request.get("processed_root"),
        )

        with self._lock:
            if self._state.get("status") == STATUS_RUNNING:
                raise JobConflictError(
                    "이미 데이터 준비가 실행 중입니다. 끝난 뒤 다시 시도해 주세요."
                )
            self._state = {
                "status": STATUS_RUNNING,
                "split_ratio": request["split_ratio"],
                "seed": config["data"]["seed"],
                "overwrite": config["data"]["overwrite"],
                "backend": config["storage"]["backend"],
                "started_at": datasets._now_text(),
                "finished_at": None,
                "message": "원본을 읽어 artifact를 만들고 있습니다.",
                "supported": True,
                "artifacts": {},
                "summary": {},
                "selected": False,
            }

        threading.Thread(
            target=self._run, args=(config,), name="data-preparation", daemon=True
        ).start()
        return self.status()

    def _run(self, config: dict[str, Any]) -> None:
        try:
            result = datasets.prepare_dataset(config)
        except Exception as error:  # 준비 실패가 서버를 죽이면 안 됩니다.
            with self._lock:
                self._state.update(
                    status=STATUS_FAILED,
                    finished_at=datasets._now_text(),
                    # type만 남기면 원인을 찾을 수 없습니다. 경로와 credential은
                    # sanitize_line이 가리므로 내용까지 담습니다.
                    message=sanitize_line(
                        f"데이터 준비 중 예기치 못한 오류가 났습니다: "
                        f"{type(error).__name__}: {error}"
                    ),
                )
            return

        selected = False
        if result["ok"]:
            # 준비가 끝나면 그 결과를 곧바로 현재 데이터셋으로 고릅니다.
            try:
                datasets.save_prepared_selection(result["artifacts"], result["summary"])
                selected = True
            except (WebValidationError, OSError):
                selected = False

        with self._lock:
            self._state.update(
                status=STATUS_SUCCEEDED if result["ok"] else STATUS_FAILED,
                finished_at=datasets._now_text(),
                message=result["message"],
                supported=result.get("supported", True),
                exit_code=result.get("exit_code"),
                artifacts=dict(result["artifacts"]),
                summary=dict(result["summary"]),
                selected=selected,
            )


_RUNNER: PreparationRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_preparation_runner() -> PreparationRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = PreparationRunner()
        return _RUNNER
