"""끝난 학습이 남긴 epoch checkpoint를 훑어 **제일 잘 맞히는 epoch**을 고릅니다.

지금까지 best epoch은 validation loss가 정했습니다. 그 loss가 정말 상자를 잘 맞히는
epoch을 고르는지는 아무도 재 보지 않았고, 제안서 011은 로컬 지표와 Kaggle 순위가
뒤집히는 것을 이미 관측했습니다. 그래서 이 화면은 **재 보고 고릅니다.**

    후보 checkpoint마다 표본 평가 → 설정에서 고른 지표로 순위 → 이긴 하나만 전수 평가
    → 제출 CSV → registry 등록

후보가 20개면 전수 평가 20번은 GPU로도 한 시간이 넘습니다. 순위만 가리는 자리에는
표본이면 충분하고(제안서 019), 최종 점수와 제출은 이긴 하나에만 전수로 합니다.

**한 번에 하나만 돕니다.** 시작할 때 도는 학습이나 평가가 있으면 거절합니다.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from . import settings as web_settings
from . import team_sync
from .errors import FieldError, JobConflictError, WebValidationError
from .evaluation import (
    STATUS_FAILED,
    STATUS_IDLE,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    build_evaluate_config,
    build_registry_config,
    resolve_device,
    run_evaluation,
    run_registry,
)
from .jobs.model import JobRecord, utc_now_text
from .masking import sanitize_line


__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "EpochSweepRunner",
    "METRIC_WEIGHTS",
    "build_candidate_config",
    "epoch_candidates",
    "get_epoch_sweep_runner",
    "next_attempt",
    "score_candidates",
    "winner_record",
]


#: 표본 크기 기본값입니다. 후보 하나가 GPU로 몇십 초에 끝나야 20개를 훑을 수 있고,
#: 알약 118종이 한 장씩은 들어갈 만한 크기입니다. 화면에서 바꿀 수 있습니다.
DEFAULT_SAMPLE_SIZE = 300

#: 1순위 지표에 가장 큰 몫을 줍니다. 고르는 순서가 곧 가중치입니다.
METRIC_WEIGHTS = (3, 2, 1)

#: train이 `epochs/epoch_012.pt`로 남긴 이름입니다. 번호를 알아야 후보에 이름을 붙이고
#: 이긴 실행을 `<원래이름>-e12`로 만들 수 있습니다. train이 이 규칙을 바꾸면 후보가
#: 하나도 안 잡히므로, 조용히 틀리는 대신 "읽을 수 있는 후보가 없다"로 멈춥니다.
_EPOCH_FILE = re.compile(r"epoch_(\d+)\.pt$")


def epoch_candidates(record: JobRecord) -> list[dict[str, Any]]:
    """이 학습이 남긴 epoch checkpoint를 번호 순으로 돌려줍니다."""

    uris = record.artifacts.get("epoch_checkpoint_uris")
    if not isinstance(uris, list):
        return []
    candidates: list[dict[str, Any]] = []
    for uri in uris:
        matched = _EPOCH_FILE.search(str(uri))
        if matched is None:
            continue
        candidates.append({"epoch": int(matched.group(1)), "checkpoint_uri": str(uri)})
    return sorted(candidates, key=lambda candidate: candidate["epoch"])


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalize(values: list[float | None]) -> list[float]:
    """후보들 **사이에서** 0~1로 폅니다.

    지표마다 사는 동네가 다릅니다. mAP는 0.9대에서 0.01씩 움직이는데 recall50은
    0.98대에서 0.001씩 움직이면, 원래 값을 그대로 더할 경우 가중치와 상관없이 변동이
    큰 지표 하나가 순위를 혼자 정합니다. 후보들 사이에서 펴면 1순위 지표가 실제로
    가장 크게 작용하고, 압도적으로 좋은 후보는 그만큼 이깁니다.

    재지 못한 값(`None`)은 0입니다. 모두 같으면 그 지표는 후보를 가르지 못하므로
    전부 1을 주어 순위에서 빠집니다.
    """

    known = [value for value in values if value is not None]
    if not known:
        return [0.0] * len(values)
    low, high = min(known), max(known)
    if high == low:
        return [1.0 if value is not None else 0.0 for value in values]
    return [
        0.0 if value is None else (value - low) / (high - low) for value in values
    ]


def score_candidates(
    candidates: list[dict[str, Any]], metric_names: list[str]
) -> list[dict[str, Any]]:
    """후보마다 0~1 점수를 매기고 높은 순으로 돌려줍니다.

    점수가 같으면 **앞선 epoch**이 이깁니다. 더 적게 학습하고 같은 값을 낸 쪽이고,
    같은 입력에 같은 답이 나와야 두 번 훑어도 같은 것을 고릅니다.
    """

    if not candidates:
        return []
    columns = {
        name: _normalize([_number(candidate["metrics"].get(name)) for candidate in candidates])
        for name in metric_names
    }
    weights = METRIC_WEIGHTS[: len(metric_names)]
    total_weight = sum(weights) or 1
    scored = []
    for index, candidate in enumerate(candidates):
        normalized = {name: columns[name][index] for name in metric_names}
        score = sum(
            weight * normalized[name] for weight, name in zip(weights, metric_names)
        ) / total_weight
        scored.append({**candidate, "score": score, "normalized": normalized})
    return sorted(scored, key=lambda candidate: (-candidate["score"], candidate["epoch"]))


def _record_for(
    record: JobRecord, *, run_id: str, checkpoint_uri: str, with_test: bool
) -> JobRecord:
    """평가 config를 만들 때만 쓰는 사본입니다. 저장하지 않습니다.

    출력 경로와 제출 위치는 `build_evaluate_config`가 `artifacts["run_id"]`에서
    만듭니다. 그 규칙을 여기에 옮겨 적으면 두 벌이 되므로, 이름을 바꾼 사본을 만들어
    같은 함수에 넘깁니다.
    """

    data_inputs = dict(record.data_inputs)
    if not with_test:
        # 후보마다 test 842장을 추론하면 훑기가 몇 시간이 됩니다. 제출은 이긴
        # 하나에만 만듭니다.
        data_inputs.pop("test_manifest_uri", None)
    return dataclasses.replace(
        record,
        artifacts={
            **record.artifacts,
            "run_id": run_id,
            "best_checkpoint_uri": checkpoint_uri,
        },
        data_inputs=data_inputs,
    )


def next_attempt(record: JobRecord) -> int:
    """이 학습을 몇 번째로 훑는지입니다. 처음이면 1입니다.

    **이름이 매번 같으면 두 번째 훑기가 통째로 실패합니다.** evaluate는 이미 있는
    artifact를 덮어쓰지 않으므로, 표본 크기를 바꿔 다시 재려 해도 후보마다 "이미
    있습니다"로 끝납니다. 번호를 붙여 지난 측정을 지우지 않고 나란히 남깁니다.
    """

    previous = record.epoch_sweep.get("attempt") if record.epoch_sweep else None
    return previous + 1 if isinstance(previous, int) and previous >= 1 else 1


def _attempt_suffix(attempt: int) -> str:
    """두 번째부터 `.2`가 붙습니다. 이어 학습 이름과 같은 규칙입니다."""

    return "" if attempt <= 1 else f".{attempt}"


def build_candidate_config(
    record: JobRecord,
    candidate: dict[str, Any],
    *,
    sample_size: int,
    device: str | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    """후보 하나를 표본으로 재는 config입니다."""

    config = build_evaluate_config(
        _record_for(
            record,
            run_id=(
                f"{record.run_id}-e{candidate['epoch']}-sample{_attempt_suffix(attempt)}"
            ),
            checkpoint_uri=candidate["checkpoint_uri"],
            with_test=False,
        ),
        device=device,
    )
    config["evaluate"]["validation_sample_size"] = sample_size
    return config


def winner_record(
    record: JobRecord, candidate: dict[str, Any], *, attempt: int = 1
) -> JobRecord:
    """이긴 후보를 별개의 실행으로 보는 사본입니다. `<원래이름>-e12`가 됩니다."""

    return _record_for(
        record,
        run_id=f"{record.run_id}-e{candidate['epoch']}{_attempt_suffix(attempt)}",
        checkpoint_uri=candidate["checkpoint_uri"],
        with_test=True,
    )


class EpochSweepRunner:
    """훑기를 한 번에 하나만 실행합니다."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {"status": STATUS_IDLE, "job_id": None}

    @contextmanager
    def locked(self) -> Iterator[None]:
        """"도는 학습이 있는가"를 보는 것과 시작하는 것을 한 문 안에서 합니다.

        갈라 두면 두 요청이 각자 "지금 도는 것은 없다"를 보고 둘 다 출발합니다.
        재진입 가능한 lock이라 이 블록 안에서 ``start()``를 불러도 막히지 않습니다.
        """

        with self._lock:
            yield

    @contextmanager
    def hold_for_delete(self, job_id: str) -> Iterator[None]:
        """이 학습의 기록을 지우는 동안 훑기가 끼어들지 못하게 붙잡습니다.

        훑는 중에 기록을 지우면, 훑기가 끝나면서 손에 든 stale record를 다시 저장해
        **log 없는 기록으로 되살립니다.** 평가가 같은 이유로 같은 문을 두고 있습니다.
        """

        with self.locked():
            if (
                self._state.get("status") == STATUS_RUNNING
                and self._state.get("job_id") == job_id
            ):
                raise JobConflictError("훑기가 도는 중인 학습의 기록은 지울 수 없습니다.")
            yield

    def is_running(self) -> bool:
        """지금 어느 학습이든 훑고 있는지입니다. GPU를 나눠 쓰는 쪽이 물어봅니다."""

        with self._lock:
            return self._state.get("status") == STATUS_RUNNING

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        if job_id is None or state.get("job_id") in (None, job_id):
            return state
        other: dict[str, Any] = {"status": STATUS_IDLE, "job_id": job_id}
        if state.get("status") == STATUS_RUNNING:
            other["busy_with"] = state.get("job_id")
        return other

    def status_for(self, record: JobRecord) -> dict[str, Any]:
        """메모리에 없으면 기록에 남아 있는 마지막 훑기 상태를 돌려줍니다."""

        state = self.status(record.job_id)
        if state.get("status") != STATUS_IDLE or not record.epoch_sweep:
            return state
        saved = {**dict(record.epoch_sweep), "job_id": record.job_id}
        if state.get("busy_with"):
            saved["busy_with"] = state["busy_with"]
        return saved

    def start(self, record: JobRecord, options: dict[str, Any]) -> dict[str, Any]:
        candidates = epoch_candidates(record)
        if not candidates:
            raise WebValidationError(
                [
                    FieldError(
                        "job",
                        "이 학습에는 epoch마다 남긴 checkpoint가 없습니다. 새 실험에서 "
                        "'epoch 보관 시작'을 채우고 학습해야 훑을 수 있습니다.",
                    )
                ]
            )
        metric_names = web_settings.epoch_metrics()
        if metric_names is None:
            raise WebValidationError(
                [
                    FieldError(
                        "epoch_metrics",
                        "설정 화면에서 순위를 매길 지표 3개를 먼저 고르세요.",
                    )
                ]
            )
        sample_size = options.get("sample_size", DEFAULT_SAMPLE_SIZE)
        if (
            isinstance(sample_size, bool)
            or not isinstance(sample_size, int)
            or sample_size < 1
        ):
            raise WebValidationError(
                [FieldError("sample_size", "1 이상의 정수여야 합니다.")]
            )
        device = resolve_device(options.get("device"))

        with self._lock:
            if self._state.get("status") == STATUS_RUNNING:
                raise JobConflictError(
                    "이미 훑기가 실행 중입니다. 끝난 뒤 다시 시도해 주세요."
                )
            # 같은 학습을 다시 훑으면 실행 이름에 번호가 붙습니다. 지난 측정을 덮어쓰지
            # 않으므로 표본 크기를 바꿔 다시 재는 것이 됩니다.
            attempt = next_attempt(record)
            self._state = {
                "status": STATUS_RUNNING,
                "job_id": record.job_id,
                "run_id": record.run_id,
                "attempt": attempt,
                "started_at": utc_now_text(),
                "finished_at": None,
                "message": f"후보 {len(candidates)}개를 {sample_size}장 표본으로 재고 있습니다.",
                "metrics": list(metric_names),
                "sample_size": sample_size,
                "device": device,
                "total": len(candidates),
                "done": 0,
                "candidates": [dict(candidate) for candidate in candidates],
                "winner": None,
                "artifacts": {},
                "registration": {"status": STATUS_IDLE},
            }

        # **시작할 때 기록에 남깁니다.** 끝날 때만 남기면, 도중에 server가 죽었을 때
        # 다음 훑기가 같은 번호를 다시 써서 이미 만들어진 산출물과 부딪힙니다.
        record.epoch_sweep = self.status()
        try:
            from .jobs import store

            store.save_record(record)
        except Exception:
            # `_finish`와 같은 이유입니다. 저장이 실패해도 훑기는 돕니다. 여기서
            # 예외가 나가면 thread도 없이 `running`인 채로 남습니다.
            pass

        threading.Thread(
            target=self._run,
            args=(record, candidates, list(metric_names), sample_size, device, attempt),
            name="epoch-sweep",
            daemon=True,
        ).start()
        return self.status()

    # ---------------------------------------------------------------- 실행

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _finish(self, record: JobRecord, **values: Any) -> None:
        """상태를 기록에 남기고 팀 화면에도 알립니다.

        **저장을 먼저 하고 상태를 끝으로 바꿉니다.** 순서를 뒤집으면, 끝난 것을 본
        삭제 요청이 `hold_for_delete`를 통과해 기록을 지운 직후에 이 저장이 그것을 다시
        살려 냅니다. 아직 `running`인 동안에는 삭제가 거절되므로 그 틈이 없습니다.
        평가 runner도 같은 순서로 저장합니다.
        """

        with self._lock:
            state = {**self._state, "finished_at": utc_now_text(), **values}
        record.epoch_sweep = state
        try:
            from .jobs import store

            store.save_record(record)
        except Exception:
            # 저장에 실패해도 상태는 끝으로 갑니다. 여기서 예외가 나가면 `_state`가
            # 영영 `running`에 갇혀 다음 훑기를 시작할 수 없습니다.
            pass
        with self._lock:
            self._state = state
        try:
            team_sync.get_team_sync().enqueue_update(record)
        except Exception:
            # 팀 화면에 못 알리는 것과 이 훑기의 결과는 다른 이야기입니다. 예외가
            # 나가면 `_run`의 catch-all이 이미 끝난 훑기를 실패로 다시 덮습니다.
            pass

    def _run(
        self,
        record: JobRecord,
        candidates: list[dict[str, Any]],
        metric_names: list[str],
        sample_size: int,
        device: str,
        attempt: int,
    ) -> None:
        try:
            self._run_once(
                record, candidates, metric_names, sample_size, device, attempt
            )
        except Exception as error:  # 훑기 실패가 server를 죽이면 안 됩니다.
            self._finish(
                record,
                status=STATUS_FAILED,
                message=sanitize_line(
                    f"훑는 중 예기치 못한 오류가 났습니다: {type(error).__name__}: {error}"
                ),
            )
        finally:
            # GPU를 놓았으니 이 때문에 물러났던 쪽들을 다시 깨웁니다. 깨우지 않으면
            # 대기열과 자동 평가가 표시도 없이 그대로 멈춰 있습니다.
            try:
                from .jobs import get_manager

                manager = get_manager()
                manager.wake_evaluation()
                manager._start_next()
            except Exception:
                pass

    def _run_once(
        self,
        record: JobRecord,
        candidates: list[dict[str, Any]],
        metric_names: list[str],
        sample_size: int,
        device: str,
        attempt: int,
    ) -> None:
        seen: list[dict[str, Any]] = []
        measured: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            config = build_candidate_config(
                record,
                candidate,
                sample_size=sample_size,
                device=device,
                attempt=attempt,
            )
            result = run_evaluation(config)
            entry = {**candidate}
            if result["ok"]:
                entry["metrics"] = dict(result["summary"].get("metrics") or {})
                measured.append(entry)
            else:
                # 후보 하나가 실패했다고 나머지를 버리지 않습니다. 20개를 다시 재는
                # 것보다 19개로 고르는 편이 낫고, 무엇이 빠졌는지는 화면에 남습니다.
                entry["failed"] = True
                entry["message"] = result["message"]
            seen.append(entry)
            self._update(
                done=index + 1,
                candidates=list(seen),
                message=f"후보 {index + 1}/{len(candidates)}를 쟀습니다.",
            )
        failed = [entry for entry in seen if entry.get("failed")]

        if not measured:
            self._finish(
                record,
                status=STATUS_FAILED,
                message="잰 후보가 하나도 없습니다. 후보 평가가 모두 실패했습니다.",
            )
            return

        scored = [*score_candidates(measured, metric_names), *failed]
        winner = scored[0]
        self._update(
            candidates=scored,
            winner=dict(winner),
            message=(
                f"epoch {winner['epoch']}이 이겼습니다. 전수로 다시 재고 제출을 만듭니다."
            ),
        )

        best = winner_record(record, winner, attempt=attempt)
        config = build_evaluate_config(best, device=device)
        result = run_evaluation(config)
        if not result["ok"]:
            self._finish(
                record,
                status=STATUS_FAILED,
                candidates=scored,
                message=f"이긴 epoch {winner['epoch']}의 전수 평가가 실패했습니다. {result['message']}",
            )
            return

        evaluation_state = {
            "status": STATUS_SUCCEEDED,
            "finished_at": utc_now_text(),
            "message": result["message"],
            "artifacts": dict(result["artifacts"]),
            "summary": dict(result["summary"]),
            "settings": dict(config["evaluate"]),
            "storage": dict(config["storage"]),
        }
        best.evaluation = evaluation_state
        registration = self._register(best)
        self._finish(
            record,
            status=STATUS_SUCCEEDED,
            candidates=scored,
            winner={
                **dict(winner),
                "run_id": config["evaluate"]["run_id"],
                "full_metrics": dict(result["summary"].get("metrics") or {}),
            },
            artifacts=dict(result["artifacts"]),
            registration=registration,
            message=(
                f"epoch {winner['epoch']}을 '{config['evaluate']['run_id']}' 이름으로 "
                "전수 평가했습니다."
            ),
        )

    def _register(self, best: JobRecord) -> dict[str, Any]:
        """이긴 실행을 registry에 남깁니다. 실패해도 평가 결과는 그대로입니다."""

        try:
            result = run_registry(build_registry_config(best, best.evaluation))
        except (WebValidationError, ValueError) as error:
            return {"status": STATUS_FAILED, "message": sanitize_line(str(error))}
        if not result["ok"]:
            return {"status": STATUS_FAILED, "message": result["message"]}
        index_status = result["summary"].get("index_status")
        return {
            "status": "index_failed" if index_status == "failed" else STATUS_SUCCEEDED,
            "message": result["message"],
            "artifacts": dict(result["artifacts"]),
        }


_RUNNER: EpochSweepRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_epoch_sweep_runner() -> EpochSweepRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = EpochSweepRunner()
        return _RUNNER
