"""학습 설정과 job 관련 route."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

from fastapi import APIRouter, Body, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import train_capabilities
from ..errors import FieldError, JobConflictError, WebValidationError
from ..evaluate_metrics import read_per_class_summary
from ..evaluation import DEFAULT_MAX_DETECTIONS, get_evaluation_runner
from .. import experiments
from ..gpu import cuda_is_available
from ..jobs import get_manager
from ..jobs.model import TERMINAL_STATUSES, JobRecord
from ..masking import redact
from ..train_config import (
    data_field_specs,
    field_specs,
    read_runtime_config,
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
    """Registry에 등록된 완료 실험을 최신순으로 돌려줍니다."""

    return {"experiments": experiments.list_registry_experiments()}


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
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    record = get_manager().start(payload.config_id, access_token=token)
    return public_record(record)


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
    """

    record = get_manager().get(job_id)
    if record.status != "succeeded":
        raise JobConflictError("성공으로 끝난 학습만 평가할 수 있습니다.")
    return {"evaluation": get_evaluation_runner().start(record, payload.model_dump())}


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
