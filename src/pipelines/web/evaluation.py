"""끝난 학습의 checkpoint로 evaluate pipeline을 돌립니다.

학습과 같은 방식으로 공개 CLI만 부릅니다.

    python -m src.main_pipeline --config <config> --only evaluate

evaluate는 검증 manifest와 checkpoint를 읽어 detection metric(mAP 등)과 예측을
만듭니다. 학습이 만드는 값은 loss뿐이라 mAP는 이 단계에서 처음 나옵니다.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any

from .errors import FieldError, JobConflictError, WebValidationError
from .jobs import runner
from .jobs.model import JobRecord
from .masking import sanitize_line
from .paths import config_dir, repository_root


__all__ = [
    "EvaluationRunner",
    "build_evaluate_config",
    "get_evaluation_runner",
    "run_evaluation",
]

# 이미지마다 추론을 돌리므로 학습만큼은 아니어도 오래 걸릴 수 있습니다.
EVALUATE_TIMEOUT_SECONDS = 60 * 60

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

# 이미지 한 장에 알약이 최대 4개라는 과제 정의에서 온 evaluate의 기본값입니다.
DEFAULT_MAX_DETECTIONS = 4


def _uses_s3(values: dict[str, str]) -> bool:
    return any(str(value).lower().startswith("s3://") for value in values.values())


def build_evaluate_config(
    record: JobRecord,
    *,
    device: str | None = None,
    score_threshold: float = 0.0,
    max_detections_per_image: int = DEFAULT_MAX_DETECTIONS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """끝난 학습 기록 하나에서 evaluate 실행 config를 만듭니다."""

    data_inputs = dict(record.data_inputs)
    train_artifacts = dict(record.artifacts)

    missing = [
        key
        for key in ("run_id", "best_checkpoint_uri")
        if not str(train_artifacts.get(key, "")).strip()
    ]
    if missing:
        raise WebValidationError(
            [FieldError("job", f"이 학습에는 {', '.join(missing)}이(가) 없습니다.")]
        )
    if not str(data_inputs.get("validation_manifest_uri", "")).strip():
        raise WebValidationError(
            [FieldError("job", "이 학습에는 검증 manifest 위치가 없습니다.")]
        )
    if not isinstance(score_threshold, (int, float)) or isinstance(score_threshold, bool):
        raise WebValidationError([FieldError("score_threshold", "숫자여야 합니다.")])
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise WebValidationError([FieldError("score_threshold", "0 이상 1 이하여야 합니다.")])
    if (
        isinstance(max_detections_per_image, bool)
        or not isinstance(max_detections_per_image, int)
        or max_detections_per_image < 1
    ):
        raise WebValidationError(
            [FieldError("max_detections_per_image", "1 이상의 정수여야 합니다.")]
        )

    remote = _uses_s3(data_inputs) or _uses_s3(train_artifacts)
    run_id = str(train_artifacts["run_id"])
    if remote:
        storage: dict[str, Any] = {"backend": "s3", "s3": {"prefix": ""}}
        # 기본값 artifacts/evaluate/... 는 저장소가 정한 S3 논리 prefix 밖입니다.
        # 학습 결과 옆에 두어 권한 설정과 정리 규칙을 그대로 따르게 합니다.
        output_dir = f"experiments/completed/{run_id}/evaluate"
    else:
        storage = {"backend": "local", "local": {"root": "artifacts"}}
        output_dir = f"artifacts/evaluate/{run_id}"

    settings: dict[str, Any] = {
        "run_id": run_id,
        "output_dir": output_dir,
        "score_threshold": float(score_threshold),
        "max_detections_per_image": max_detections_per_image,
        "overwrite": bool(overwrite),
    }
    if device:
        settings["device"] = device

    return {
        "project": {"name": "pill-object-detection"},
        # dummy면 evaluate가 평가를 건너뛰고 dummy 결과만 돌려줍니다.
        "execution": {"mode": "real"},
        "storage": storage,
        "inputs": {"data": data_inputs, "train": train_artifacts},
        "evaluate": settings,
    }


def _parse_result(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _unwrap_stage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    stage = value.get("evaluate")
    return dict(stage) if isinstance(stage, dict) else dict(value)


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """실제 evaluate pipeline을 공개 CLI로 부릅니다."""

    from .train_config import config_relative_path, write_runtime_config

    config_id = write_runtime_config(config)
    try:
        completed = runner.run_stage(
            config_relative_path(config_id),
            "evaluate",
            cwd=repository_root(),
            timeout=EVALUATE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": None, "artifacts": {}, "summary": {},
                "message": "평가가 시간 안에 끝나지 않았습니다."}
    except OSError as error:
        return {"ok": False, "exit_code": None, "artifacts": {}, "summary": {},
                "message": f"evaluate pipeline을 실행하지 못했습니다({type(error).__name__})."}
    finally:
        (config_dir() / f"{config_id}.json").unlink(missing_ok=True)

    result = _parse_result(completed.stdout)
    if result is None:
        lines = (completed.stderr or "").strip().splitlines()
        detail = sanitize_line(lines[-1]) if lines else ""
        return {"ok": False, "exit_code": completed.returncode, "artifacts": {}, "summary": {},
                "message": f"평가 결과를 해석하지 못했습니다. {detail}".strip()}

    return {
        "ok": completed.returncode == 0 and result.get("status") == "ok",
        "exit_code": completed.returncode,
        "artifacts": _unwrap_stage(result.get("artifacts")),
        "summary": _unwrap_stage(result.get("summary")),
        "message": sanitize_line(str(result.get("message") or "")),
    }


class EvaluationRunner:
    """평가를 한 번에 하나만 실행합니다. 어느 학습에 대한 것인지 함께 기록합니다."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {"status": STATUS_IDLE, "job_id": None}

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        if job_id is not None and state.get("job_id") not in (None, job_id):
            # 다른 학습의 결과를 그 학습의 것인 양 보여 주면 안 됩니다.
            return {"status": STATUS_IDLE, "job_id": job_id, "busy_with": state.get("job_id")}
        return state

    def start(self, record: JobRecord, options: dict[str, Any]) -> dict[str, Any]:
        from .jobs.model import utc_now_text

        config = build_evaluate_config(
            record,
            device=options.get("device"),
            score_threshold=options.get("score_threshold", 0.0),
            max_detections_per_image=options.get(
                "max_detections_per_image", DEFAULT_MAX_DETECTIONS
            ),
            overwrite=options.get("overwrite", False),
        )

        with self._lock:
            if self._state.get("status") == STATUS_RUNNING:
                raise JobConflictError(
                    "이미 평가가 실행 중입니다. 끝난 뒤 다시 시도해 주세요."
                )
            self._state = {
                "status": STATUS_RUNNING,
                "job_id": record.job_id,
                "run_id": record.run_id,
                "started_at": utc_now_text(),
                "finished_at": None,
                "message": "checkpoint로 검증 이미지를 추론하고 있습니다.",
                "artifacts": {},
                "summary": {},
                "device": config["evaluate"].get("device"),
                "score_threshold": config["evaluate"]["score_threshold"],
            }

        threading.Thread(
            target=self._run, args=(config,), name="evaluation", daemon=True
        ).start()
        return self.status()

    def _run(self, config: dict[str, Any]) -> None:
        from .jobs.model import utc_now_text

        try:
            result = run_evaluation(config)
        except Exception as error:  # 평가 실패가 서버를 죽이면 안 됩니다.
            with self._lock:
                self._state.update(
                    status=STATUS_FAILED,
                    finished_at=utc_now_text(),
                    message=sanitize_line(
                        f"평가 중 예기치 못한 오류가 났습니다: {type(error).__name__}"
                    ),
                )
            return

        with self._lock:
            self._state.update(
                status=STATUS_SUCCEEDED if result["ok"] else STATUS_FAILED,
                finished_at=utc_now_text(),
                message=result["message"],
                exit_code=result.get("exit_code"),
                artifacts=dict(result["artifacts"]),
                summary=dict(result["summary"]),
            )


_RUNNER: EvaluationRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_evaluation_runner() -> EvaluationRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = EvaluationRunner()
        return _RUNNER
