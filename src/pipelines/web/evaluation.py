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
from urllib.parse import urlsplit

from . import team_sync
from .errors import FieldError, JobConflictError, WebValidationError
from .jobs import runner
from .jobs.model import JobRecord
from .masking import sanitize_line
from .paths import config_dir, repository_root
from .train_config import normalize_data_inputs


__all__ = [
    "EvaluationRunner",
    "build_evaluate_config",
    "build_registry_config",
    "get_evaluation_runner",
    "run_evaluation",
    "run_registry",
]

# 이미지마다 추론을 돌리므로 학습만큼은 아니어도 오래 걸릴 수 있습니다.
EVALUATE_TIMEOUT_SECONDS = 60 * 60

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

# 이미지 한 장에 알약이 최대 4개라는 과제 정의에서 온 evaluate의 기본값입니다.
DEFAULT_MAX_DETECTIONS = 4


def _first_s3_bucket(*groups: dict[str, Any]) -> str | None:
    """학습 결과를 우선해 평가 산출물을 저장할 S3 bucket을 고릅니다."""

    for values in groups:
        for value in values.values():
            parsed = urlsplit(str(value))
            if parsed.scheme.lower() == "s3" and parsed.netloc:
                return parsed.netloc
    return None


def build_evaluate_config(
    record: JobRecord,
    *,
    device: str | None = None,
    score_threshold: float = 0.0,
    max_detections_per_image: int = DEFAULT_MAX_DETECTIONS,
    overwrite: bool = False,
    test_manifest_uri: str | None = None,
) -> dict[str, Any]:
    """끝난 학습 기록 하나에서 evaluate 실행 config를 만듭니다."""

    data_inputs = dict(record.data_inputs)
    if test_manifest_uri is not None:
        # 과거 학습 기록은 학습 당시 입력을 증명하므로 바꾸지 않고, 이번 평가 config에만
        # test manifest를 덧붙입니다.
        data_inputs["test_manifest_uri"] = test_manifest_uri
    data_inputs = normalize_data_inputs(data_inputs)
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

    s3_bucket = _first_s3_bucket(train_artifacts, data_inputs)
    run_id = str(train_artifacts["run_id"])
    if s3_bucket is not None:
        storage: dict[str, Any] = {
            "backend": "s3",
            "s3": {"bucket": s3_bucket, "prefix": ""},
        }
        # 상대 경로는 evaluate 저장 계층에서 local로 해석됩니다. S3 실행은 완전한
        # URI를 넘겨 학습 결과와 평가 결과가 같은 bucket에 남도록 합니다.
        output_dir = f"s3://{s3_bucket}/experiments/completed/{run_id}/evaluate"
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
    if s3_bucket is not None and data_inputs.get("test_manifest_uri"):
        settings["submission_uri"] = (
            f"s3://{s3_bucket}/submissions/{run_id}/submission.csv"
        )

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


def build_registry_config(
    record: JobRecord,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """저장된 학습·평가 결과를 Registry 입력 계약으로 연결합니다."""

    storage = evaluation.get("storage")
    settings = evaluation.get("settings")
    artifacts = evaluation.get("artifacts")
    if (
        not isinstance(storage, dict)
        or not isinstance(settings, dict)
        or not isinstance(artifacts, dict)
    ):
        raise WebValidationError(
            [FieldError("registration", "등록에 필요한 평가 결과가 남아 있지 않습니다.")]
        )
    seed = record.settings.get("seed", 42)
    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "real"},
        "storage": dict(storage),
        "train": dict(record.settings),
        "evaluate": dict(settings),
        "registry": {"seed": seed, "overwrite": False},
        "inputs": {
            "data": dict(record.data_inputs),
            "train": dict(record.artifacts),
            "evaluate": dict(artifacts),
        },
    }


def run_registry(config: dict[str, Any]) -> dict[str, Any]:
    """Registry stage 하나만 실행하고 등록 상태로 정규화합니다."""

    from .train_config import config_relative_path, write_runtime_config

    config_id: str | None = None
    try:
        config_id = write_runtime_config(config)
        completed = runner.run_stage(
            config_relative_path(config_id),
            "registry",
            cwd=repository_root(),
            timeout=60.0,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "artifacts": {},
            "summary": {},
            "message": "Registry 등록이 시간 안에 끝나지 않았습니다.",
        }
    except OSError as error:
        return {
            "ok": False,
            "artifacts": {},
            "summary": {},
            "message": f"Registry를 실행하지 못했습니다({type(error).__name__}).",
        }
    finally:
        if config_id is not None:
            (config_dir() / f"{config_id}.json").unlink(missing_ok=True)

    result = _parse_result(completed.stdout)
    if result is None:
        return {
            "ok": False,
            "artifacts": {},
            "summary": {},
            "message": "Registry 결과를 해석하지 못했습니다.",
        }
    artifacts = _unwrap_named_stage(result.get("artifacts"), "registry")
    summary = _unwrap_named_stage(result.get("summary"), "registry")
    return {
        "ok": completed.returncode == 0 and result.get("status") == "ok",
        "artifacts": artifacts,
        "summary": summary,
        "message": sanitize_line(str(result.get("message") or "")),
    }


def _unwrap_named_stage(value: Any, stage: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get(stage)
    return dict(nested) if isinstance(nested, dict) else dict(value)


class EvaluationRunner:
    """평가를 한 번에 하나만 실행합니다. 어느 학습에 대한 것인지 함께 기록합니다."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "status": STATUS_IDLE,
            "job_id": None,
            "registration": {"status": STATUS_IDLE},
        }

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        if job_id is None or state.get("job_id") in (None, job_id):
            return state
        # 다른 학습의 결과를 그 학습의 것인 양 보여 주면 안 됩니다.
        other: dict[str, Any] = {"status": STATUS_IDLE, "job_id": job_id}
        if state.get("status") == STATUS_RUNNING:
            # 한 번에 하나만 돌 수 있으므로, 실제로 돌고 있을 때만 잠급니다.
            other["busy_with"] = state.get("job_id")
        return other

    def status_for(self, record: JobRecord) -> dict[str, Any]:
        """메모리에 없으면 JobRecord에 영속화된 마지막 평가 상태를 돌려줍니다."""

        state = self.status(record.job_id)
        if state.get("status") != STATUS_IDLE or not record.evaluation:
            return state
        saved = {
            **dict(record.evaluation),
            "job_id": record.job_id,
            "registration": dict(record.registration or {"status": STATUS_IDLE}),
        }
        if state.get("busy_with"):
            saved["busy_with"] = state["busy_with"]
        return saved

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
            test_manifest_uri=options.get("test_manifest_uri"),
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
                "submission_requested": bool(
                    config["inputs"]["data"].get("test_manifest_uri")
                ),
                "started_at": utc_now_text(),
                "finished_at": None,
                "message": "checkpoint로 검증 이미지를 추론하고 있습니다.",
                "artifacts": {},
                "summary": {},
                "registration": {"status": STATUS_IDLE},
                "device": config["evaluate"].get("device"),
                "score_threshold": config["evaluate"]["score_threshold"],
            }

        threading.Thread(
            target=self._run, args=(record, config), name="evaluation", daemon=True
        ).start()
        return self.status()

    def _run(self, record: JobRecord, config: dict[str, Any]) -> None:
        from .jobs.model import utc_now_text

        try:
            result = run_evaluation(config)
        except Exception as error:  # 평가 실패가 서버를 죽이면 안 됩니다.
            evaluation_state = {
                "status": STATUS_FAILED,
                "finished_at": utc_now_text(),
                "message": sanitize_line(
                    "평가 중 예기치 못한 오류가 났습니다: "
                    f"{type(error).__name__}: {error}"
                ),
                "exit_code": None,
                "artifacts": {},
                "summary": {},
                "settings": dict(config["evaluate"]),
                "storage": dict(config["storage"]),
            }
            record.evaluation = evaluation_state
            record.registration = {"status": STATUS_IDLE}
            try:
                from .jobs import store

                store.save_record(record)
            except OSError:
                pass
            with self._lock:
                self._state.update(
                    **evaluation_state,
                    registration={"status": STATUS_IDLE},
                )
            # 로컬 상태를 먼저 맞춘 뒤 공유합니다. 공유가 실패해도 이 화면은 멀쩡합니다.
            team_sync.get_team_sync().enqueue_update(record)
            return

        evaluation_state = {
            "status": STATUS_SUCCEEDED if result["ok"] else STATUS_FAILED,
            "finished_at": utc_now_text(),
            "message": result["message"],
            "exit_code": result.get("exit_code"),
            "artifacts": dict(result["artifacts"]),
            "summary": dict(result["summary"]),
            "settings": dict(config["evaluate"]),
            "storage": dict(config["storage"]),
        }
        record.evaluation = evaluation_state
        registration = {"status": STATUS_IDLE}
        if result["ok"]:
            registration = self._register(record)
        record.registration = registration
        try:
            from .jobs import store

            store.save_record(record)
        except OSError:
            pass
        with self._lock:
            self._state.update(
                **evaluation_state,
                registration=dict(registration),
            )
        # mAP는 여기서 처음 나옵니다. 팀 화면이 채워지는 것도 이 시점입니다.
        team_sync.get_team_sync().enqueue_update(record)

    def _register(self, record: JobRecord) -> dict[str, Any]:
        """평가 성공을 유지하면서 Registry 결과를 별도 상태로 돌려줍니다."""

        try:
            result = run_registry(build_registry_config(record, record.evaluation))
        except (WebValidationError, ValueError) as error:
            return {"status": STATUS_FAILED, "message": sanitize_line(str(error))}
        if not result["ok"]:
            return {"status": STATUS_FAILED, "message": result["message"]}
        index_status = result["summary"].get("index_status")
        status = "index_failed" if index_status == "failed" else STATUS_SUCCEEDED
        return {
            "status": status,
            "message": result["message"],
            "artifacts": dict(result["artifacts"]),
            "summary": dict(result["summary"]),
        }

    def retry_registration(self, record: JobRecord) -> dict[str, Any]:
        """저장된 평가 artifact를 바꾸지 않고 Registry만 다시 실행합니다."""

        if record.registration.get("status") == STATUS_SUCCEEDED:
            return dict(record.registration)
        if record.registration.get("status") == "index_failed":
            raise JobConflictError(
                "Registry record는 이미 저장됐습니다. rebuild_index로 목록 index를 복구해 주세요."
            )
        if record.evaluation.get("status") != STATUS_SUCCEEDED:
            raise JobConflictError("성공으로 끝난 평가만 Registry에 등록할 수 있습니다.")
        registration = self._register(record)
        record.registration = registration
        from .jobs import store

        store.save_record(record)
        with self._lock:
            if self._state.get("job_id") == record.job_id:
                self._state["registration"] = dict(registration)
        team_sync.get_team_sync().enqueue_update(record)
        return dict(registration)


_RUNNER: EvaluationRunner | None = None
_RUNNER_LOCK = threading.Lock()


def get_evaluation_runner() -> EvaluationRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = EvaluationRunner()
        return _RUNNER
