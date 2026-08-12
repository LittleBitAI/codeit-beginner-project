"""학습 설정과 job 관련 route."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

from fastapi import APIRouter, Body, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.common import StorageError

from .. import train_capabilities
from ..errors import FieldError, JobConflictError, WebValidationError
from ..evaluate_metrics import read_per_class_summary
from ..evaluation import DEFAULT_MAX_DETECTIONS, get_evaluation_runner
from .. import experiments
from ..gpu import cuda_is_available
from ..jobs import get_manager
from ..jobs.model import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    TERMINAL_STATUSES,
    JobRecord,
)
from ..masking import redact
from ..train_config import (
    build_resume_config,
    data_field_specs,
    field_specs,
    read_runtime_config,
    resume_checkpoint_exists,
    validate_request,
    write_runtime_config,
)
from .schemas import ConfigRequest, StartJobRequest


router = APIRouter(prefix="/api/train", tags=["train"])

# 이 값은 ``src/pipelines/train/model.py``의 기존 기본 architecture를 그대로 옮긴 것입니다.
# train을 import하면 경계를 넘으므로 복제하되, 조용히 어긋나면 화면이 실제로 학습되는
# 모델과 다른 이름을 보여 주게 됩니다. 그래서
# ``test_web_train_contract.py::test_architecture_matches_train_source``가 train의
# source를 읽어 값이 같은지 확인합니다.
ARCHITECTURE = train_capabilities.LEGACY_ARCHITECTURE

SSE_POLL_SECONDS = 1.0
SSE_MAX_SECONDS = 60 * 60 * 12


def public_record(record: JobRecord) -> dict[str, Any]:
    """화면에 돌려주기 전에 credential처럼 보이는 값을 가립니다."""

    return redact(record.to_dict())


@router.get("/defaults")
def get_defaults() -> dict[str, Any]:
    """새 실험 form을 서버가 정의합니다. 기본값의 유일한 출처입니다."""

    cuda = cuda_is_available()
    capability = train_capabilities.current_train_capability()
    architecture = capability["model"]["default"]
    return {
        # 기존 frontend와의 호환을 위해 top-level 두 field를 유지합니다.
        "architecture": architecture,
        "architecture_note": (
            "Train capability가 없어 현재 Faster R-CNN 기본값을 사용합니다."
            if capability["source"] == "legacy_fallback"
            else "Train pipeline이 보고한 capability를 사용합니다."
        ),
        "train_capability": capability,
        "fields": field_specs(),
        "data_fields": data_field_specs(),
        "devices": [
            {"value": "cpu", "available": True, "reason": None},
            {
                "value": "cuda",
                "available": cuda,
                "reason": None if cuda else "이 컴퓨터에서 CUDA를 사용할 수 없습니다.",
            },
        ],
    }


@router.post("/validate")
def validate(payload: ConfigRequest = Body(...)) -> dict[str, Any]:
    """설정을 검사만 합니다. 아무것도 저장하지 않습니다."""

    result = validate_request(payload.as_payload())
    if result["normalized"] is not None:
        result["normalized"] = redact(result["normalized"])
    return result


@router.post("/configs", status_code=201)
def create_config(payload: ConfigRequest = Body(...)) -> dict[str, Any]:
    """검증을 통과한 설정을 저장하고 id를 돌려줍니다.

    설정을 저장하기 전에는 학습을 시작할 수 없습니다. 저장 경로는 응답에 넣지 않습니다.
    """

    checked = validate_request(payload.as_payload())
    if not checked["valid"]:
        raise WebValidationError(
            [FieldError(item["field"], item["message"]) for item in checked["errors"]]
        )

    config = checked["normalized"]
    config_id = write_runtime_config(config)
    return {
        "config_id": config_id,
        "run_id": config["train"]["run_id"],
        "config": redact(config),
        "warnings": checked["warnings"],
    }


@router.get("/jobs")
def list_jobs(status: str | None = Query(default=None)) -> dict[str, Any]:
    manager = get_manager()
    records = manager.list_jobs()
    if status:
        wanted = {item.strip() for item in status.split(",") if item.strip()}
        records = [record for record in records if record.status in wanted]
    active = manager.active_job()
    return {
        "jobs": [public_record(record) for record in records],
        "active_job_id": active.job_id if active else None,
    }


@router.get("/experiments")
def list_experiments() -> dict[str, Any]:
    """Registry에 등록된 완료 실험을 최신순으로 돌려줍니다.

    ``scope``는 이 목록에 팀원의 실험도 들어오는지입니다. Registry index는 storage
    backend를 따라가서, S3면 팀 공용이고 local이면 이 컴퓨터 것뿐입니다.
    """

    return {
        "experiments": experiments.list_registry_experiments(),
        "scope": experiments.registry_scope(),
    }


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str) -> dict[str, Any]:
    """실험 하나의 설정과 평가 결과 전체입니다.

    목록에는 지표 9개 중 5개만 있고 loss 곡선은 아예 없습니다. 여기서는 record가
    가리키는 `metrics.json`과 `training_history.json`을 직접 읽어 화면이 쓸 만큼만
    골라 돌려줍니다. 650KB짜리 파일을 그대로 흘려보내지 않습니다.
    """

    return experiments.read_registry_experiment(run_id)


class KaggleScoreRequest(BaseModel):
    """Kaggle에 실제 제출한 뒤 받은 0~1 점수입니다."""

    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    # 잘못 적은 점수를 고치는 요청에만 붙습니다. 화면이 수정 버튼을 켜야 True가 옵니다.
    overwrite: bool = False


@router.put("/experiments/{run_id}/kaggle-score")
def save_kaggle_score(
    run_id: str, payload: KaggleScoreRequest = Body(...)
) -> dict[str, Any]:
    return experiments.save_kaggle_score(run_id, payload.score, payload.overwrite)


class CompareExperimentsRequest(BaseModel):
    run_ids: list[str]


@router.post("/experiments/compare")
def compare_experiments(payload: CompareExperimentsRequest = Body(...)) -> dict[str, Any]:
    """선택한 Registry record만 읽어 학습 설정과 평가 결과를 비교합니다."""

    return experiments.compare_registry_experiments(payload.run_ids)


@router.post("/jobs", status_code=201)
def start_job(
    payload: StartJobRequest = Body(...), authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    token = _bearer_token(authorization)
    record = get_manager().start(payload.config_id, access_token=token)
    return public_record(record)


def _bearer_token(authorization: str | None) -> str | None:
    """Authorization header에서 browser login token만 꺼냅니다."""

    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    return token or None


@router.get("/queue")
def read_queue() -> dict[str, Any]:
    """아직 시작하지 않은 학습과, 대기열이 멈춰 있는지를 알려 줍니다."""

    manager = get_manager()
    return {"entries": manager.queue_entries(), "paused": manager.queue_paused()}


@router.post("/queue", status_code=201)
def add_to_queue(
    payload: StartJobRequest = Body(...), authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """설정 하나를 대기열에 넣습니다. 비어 있으면 곧바로 시작합니다.

    ``/jobs``와 같은 login token을 받습니다. 대기열이 항목을 꺼내 실제로 시작할 때
    팀 기록을 만들어야 하는데, token이 없으면 이미 로그인한 사람에게도 "먼저
    로그인해야 합니다"라고 답하며 멈춰 섭니다.
    """

    manager = get_manager()
    started = manager.enqueue(payload.config_id, access_token=_bearer_token(authorization))
    return {
        "started": public_record(started) if started is not None else None,
        "entries": manager.queue_entries(),
        "paused": manager.queue_paused(),
    }


@router.delete("/queue/{entry_id}")
def remove_queue_entry(entry_id: str) -> dict[str, Any]:
    manager = get_manager()
    manager.remove_from_queue(entry_id)  # 없으면 404
    return {"entries": manager.queue_entries(), "paused": manager.queue_paused()}


@router.post("/queue/resume", status_code=202)
def resume_queue(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """중지나 서버 재시작으로 멈춘 대기열을 다시 돌립니다.

    Login token을 함께 받습니다. 서버가 다시 뜨면 memory에만 두던 token이 사라지므로,
    브라우저가 지금 보내 주지 않으면 남아 있는 목록을 하나도 시작하지 못합니다.
    """

    manager = get_manager()
    started = manager.resume_queue(access_token=_bearer_token(authorization))
    return {
        "started": public_record(started) if started is not None else None,
        "entries": manager.queue_entries(),
        "paused": manager.queue_paused(),
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return public_record(get_manager().get(job_id))


@router.get("/jobs/{job_id}/config")
def get_job_config(job_id: str) -> dict[str, Any]:
    record = get_manager().get(job_id)
    return {"config": redact(read_runtime_config(record.config_id))}


@router.get("/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    return get_manager().logs(job_id, after=after, limit=limit)


@router.post("/jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str) -> dict[str, Any]:
    return public_record(get_manager().cancel(job_id))


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    """이 GUI가 들고 있던 학습 기록 하나를 지웁니다.

    **학습 산출물은 지우지 않습니다.** checkpoint와 학습 결과 폴더는 train이
    만든 것이고, registry에 등록된 실험과 팀에 공유된 기록도 이 화면의 것이
    아닙니다. 지우는 것은 ``artifacts/web/jobs/<job_id>/`` 하나뿐입니다.

    평가가 도는 중이면 거절합니다. 그 runner가 끝나면서 같은 record를 다시
    저장하기 때문에, 지워도 곧 되살아나 지운 것처럼 보이지 않습니다. 확인과 삭제를
    같은 잠금 안에서 해야 그 사이에 평가가 시작되는 틈이 없습니다.
    """

    manager = get_manager()
    with get_evaluation_runner().hold_for_delete(job_id):
        manager.delete(job_id)  # 없거나 실행 중이면 404 또는 409
    active = manager.active_job()
    return {
        "jobs": [public_record(record) for record in manager.list_jobs()],
        "active_job_id": active.job_id if active else None,
    }


def _event_stream(job_id: str) -> Iterator[str]:
    manager = get_manager()
    cursor = 0
    started = time.monotonic()
    while True:
        record = manager.get(job_id)
        payload = manager.logs(job_id, after=cursor, limit=200)
        cursor = payload["next"]
        frame = {
            "job": public_record(record),
            "lines": payload["lines"],
            "cursor": cursor,
        }
        yield f"data: {json.dumps(frame, ensure_ascii=False, allow_nan=False)}\n\n"
        if record.status in TERMINAL_STATUSES and payload["complete"]:
            break
        if time.monotonic() - started > SSE_MAX_SECONDS:
            break
        time.sleep(SSE_POLL_SECONDS)


class EvaluateRequest(BaseModel):
    """평가 실행 설정. 값은 evaluate pipeline이 정한 규칙을 그대로 씁니다."""

    device: str | None = Field(default=None)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_detections_per_image: int = Field(default=DEFAULT_MAX_DETECTIONS, ge=1, le=1000)
    overwrite: bool = Field(default=False)
    test_manifest_uri: str | None = Field(default=None)


@router.get("/jobs/{job_id}/evaluate")
def evaluation_status(job_id: str) -> dict[str, Any]:
    """이 학습에 대한 평가 상태입니다."""

    record = get_manager().get(job_id)  # 없는 job이면 404
    return {"evaluation": get_evaluation_runner().status_for(record)}


@router.get("/jobs/{job_id}/evaluate/per-class")
def evaluation_per_class(job_id: str) -> dict[str, Any]:
    """평가 결과의 class별 요약입니다. 없으면 `summary`가 `null`입니다.

    metrics.json은 confusion matrix까지 들어 650KB가 넘으므로 상태 polling에 얹지
    않고, 화면이 이 표를 펼칠 때만 부릅니다.
    """

    record = get_manager().get(job_id)  # 없는 job이면 404
    return {"summary": read_per_class_summary(record.evaluation)}


@router.post("/jobs/{job_id}/evaluate", status_code=202)
def start_evaluation(job_id: str, payload: EvaluateRequest = Body(...)) -> dict[str, Any]:
    """끝난 학습의 checkpoint로 evaluate pipeline을 부릅니다.

    ``python -m src.main_pipeline --config <config> --only evaluate``

    학습이 만드는 값은 loss뿐이라 mAP 같은 detection metric은 여기서 처음 나옵니다.
    이미지마다 추론을 돌리므로 시작만 시키고 상태는 따로 확인합니다.

    **record를 읽는 것부터 시작까지를 삭제와 같은 잠금 안에서 합니다.** 밖에서 읽으면
    그 사이에 DELETE가 기록을 지울 수 있고, 뒤늦게 시작한 평가가 끝나면서 손에 든
    stale record를 다시 저장해 빈 log와 함께 되살립니다.
    """

    runner = get_evaluation_runner()
    with runner.locked():
        record = get_manager().get(job_id)  # 지워졌으면 여기서 404
        if record.status != "succeeded":
            raise JobConflictError("성공으로 끝난 학습만 평가할 수 있습니다.")
        return {"evaluation": runner.start(record, payload.model_dump())}


class ResumeRequest(BaseModel):
    """이어서 학습 설정. 비워 두면 중단된 실행의 계획을 그대로 이어갑니다."""

    run_id: str | None = Field(default=None)
    epochs: int | None = Field(default=None, ge=1)


@router.post("/jobs/{job_id}/resume", status_code=201)
def resume_job(
    job_id: str,
    payload: ResumeRequest = Body(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """중단됐거나 checkpoint를 남기고 실패한 학습을 이어서 시작합니다.

    이어서 하는 실행은 **새 이름**을 받습니다. 같은 이름을 다시 쓰면 train이 남아 있는
    작업 폴더를 보고 시작을 거부하고, 결과도 섞입니다. `epochs`는 남은 수가 아니라
    전체 목표이므로 비워 두면 중단된 실행의 계획을 그대로 씁니다.
    """

    manager = get_manager()
    record = manager.get(job_id)
    if record.status not in {STATUS_INTERRUPTED, STATUS_FAILED}:
        raise JobConflictError(
            "중단된 학습이나 checkpoint를 남기고 실패한 학습만 이어갈 수 있습니다."
        )

    source_config = read_runtime_config(record.config_id)
    if record.status == STATUS_FAILED:
        try:
            checkpoint_exists = resume_checkpoint_exists(source_config)
        except StorageError as error:
            raise JobConflictError(
                "실패한 학습의 checkpoint를 확인하지 못했습니다. "
                "저장소 설정과 권한을 확인하세요."
            ) from error
        if not checkpoint_exists:
            raise JobConflictError(
                "실패한 학습의 checkpoint를 찾을 수 없어 이어서 학습할 수 없습니다."
            )

    config = build_resume_config(
        source_config,
        run_id=payload.run_id,
        epochs=payload.epochs,
    )
    config_id = write_runtime_config(config)
    started = manager.enqueue(
        config_id,
        access_token=_bearer_token(authorization),
        # 앞선 실행의 손실 곡선을 이어 그리려면 어느 job에서 왔는지 알아야 합니다.
        resumed_from_job_id=record.job_id,
    )
    return {
        "config_id": config_id,
        "run_id": config["train"]["run_id"],
        "resumed_from_job_id": record.job_id,
        "resume_from": config["train"]["resume_from"],
        "started": public_record(started) if started is not None else None,
        "entries": manager.queue_entries(),
        "paused": manager.queue_paused(),
    }


@router.post("/jobs/{job_id}/register")
def retry_registration(job_id: str) -> dict[str, Any]:
    """성공한 평가 artifact를 바꾸지 않고 Registry 등록만 다시 시도합니다."""

    record = get_manager().get(job_id)
    return {"registration": get_evaluation_runner().retry_registration(record)}


@router.get("/jobs/{job_id}/events")
def stream_job_events(job_id: str) -> StreamingResponse:
    """진행 상황을 Server-Sent Events로 흘려보냅니다."""

    get_manager().get(job_id)  # 없는 job이면 여기서 404
    return StreamingResponse(
        _event_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
