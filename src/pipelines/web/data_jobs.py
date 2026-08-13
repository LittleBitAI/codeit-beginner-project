"""데이터 준비를 백그라운드로 한 번에 하나만 실행합니다.

원본 전체를 읽어야 해서 몇 분이 걸릴 수 있습니다. HTTP 요청 하나를 그동안 붙잡고
있으면 browser나 proxy가 먼저 끊어 버리므로, 시작만 시키고 상태를 따로 물어보게
합니다. 학습 job과는 별개 process라 동시에 돌아도 서로 방해하지 않습니다.
"""

from __future__ import annotations

import threading
from typing import Any

from . import datasets
from .data_progress import DataProgressState, consume_line, snapshot
from .errors import FieldError, JobConflictError, WebValidationError
from .masking import sanitize_line


__all__ = ["EdaRunner", "PreparationRunner", "get_eda_runner", "get_preparation_runner"]

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

        return self._launch(
            config,
            busy="이미 데이터 준비가 실행 중입니다. 끝난 뒤 다시 시도해 주세요.",
            name="data-preparation",
            state={
                "split_ratio": request["split_ratio"],
                "seed": config["data"]["seed"],
                "overwrite": config["data"]["overwrite"],
                "backend": config["storage"]["backend"],
                "message": "원본을 읽어 artifact를 만들고 있습니다.",
                "selected": False,
            },
        )

    def _launch(
        self, config: dict[str, Any], *, busy: str, name: str, state: dict[str, Any]
    ) -> dict[str, Any]:
        """실행 하나를 띄웁니다. 같은 runner에서 둘이 동시에 돌지 않습니다."""

        progress_state = DataProgressState()
        with self._lock:
            if self._state.get("status") == STATUS_RUNNING:
                raise JobConflictError(busy)
            self._state = {
                "status": STATUS_RUNNING,
                "started_at": datasets._now_text(),
                "finished_at": None,
                "supported": True,
                "artifacts": {},
                "summary": {},
                # 진행 로그가 오기 전에는 진행률을 지어내지 않습니다.
                # ``available: False``인 채로 시작합니다.
                "progress": snapshot(progress_state),
                **state,
            }

        threading.Thread(
            target=self._run, args=(config, progress_state), name=name, daemon=True
        ).start()
        return self.status()

    def _consume(self, progress_state: DataProgressState, line: str) -> None:
        """준비 subprocess의 stderr 한 줄을 진행 상태에 반영합니다.

        ``consume_line``은 어떤 입력에도 예외를 던지지 않으므로, 이 경로가 준비를
        실패시키는 일은 없습니다.
        """

        consume_line(progress_state, line)
        with self._lock:
            self._state["progress"] = snapshot(progress_state)

    def _run(self, config: dict[str, Any], progress_state: DataProgressState) -> None:
        try:
            result = datasets.prepare_dataset(
                config, on_progress_line=lambda line: self._consume(progress_state, line)
            )
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

        with self._lock:
            self._state.update(
                status=STATUS_SUCCEEDED if result["ok"] else STATUS_FAILED,
                finished_at=datasets._now_text(),
                message=result["message"],
                supported=result.get("supported", True),
                exit_code=result.get("exit_code"),
                artifacts=dict(result["artifacts"]),
                summary=dict(result["summary"]),
                **self._on_finished(result),
            )

    def _on_finished(self, result: dict[str, Any]) -> dict[str, Any]:
        """성공한 준비 결과를 곧바로 현재 데이터셋으로 고릅니다."""

        if not result["ok"]:
            return {"selected": False}
        try:
            datasets.save_prepared_selection(result["artifacts"], result["summary"])
        except (WebValidationError, OSError):
            return {"selected": False}
        return {"selected": True}


class EdaRunner(PreparationRunner):
    """EDA 실행 한 건의 상태를 들고 있습니다.

    준비와 같은 stage(``--only data``)를 쓰지만 하는 일이 달라 상태를 따로 둡니다.
    한 runner를 같이 쓰면 화면 한쪽의 진행률이 다른 쪽 것으로 바뀝니다.
    """

    def start(self, request: dict[str, Any]) -> dict[str, Any]:
        selection = datasets.load_selection()
        if not selection or not selection.get("data"):
            raise WebValidationError(
                [FieldError("dataset", "먼저 전처리 dataset을 고르세요.")]
            )
        config = datasets.build_eda_config(
            dict(selection["data"]),
            image_sample=request.get("image_sample", 200),
            overwrite=request.get("overwrite", False),
        )
        return self._launch(
            config,
            busy="이미 EDA가 실행 중입니다. 끝난 뒤 다시 시도해 주세요.",
            name="data-eda",
            state={
                "directory": selection.get("directory"),
                "image_sample": config["data"]["eda_image_sample"],
                "overwrite": config["data"]["overwrite"],
                "message": "dataset을 읽어 리포트를 만들고 있습니다.",
                "report": None,
            },
        )

    def _on_finished(self, result: dict[str, Any]) -> dict[str, Any]:
        """리포트 URI가 나왔으면 내용까지 읽어 화면에 바로 넘깁니다."""

        uri = result["artifacts"].get("eda_report_uri") if result["ok"] else None
        return {"report": datasets.read_eda_report(uri) if uri else None}


_RUNNER: PreparationRunner | None = None
_EDA_RUNNER: EdaRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_preparation_runner() -> PreparationRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = PreparationRunner()
        return _RUNNER


def get_eda_runner() -> EdaRunner:
    global _EDA_RUNNER
    with _RUNNER_LOCK:
        if _EDA_RUNNER is None:
            _EDA_RUNNER = EdaRunner()
        return _EDA_RUNNER
