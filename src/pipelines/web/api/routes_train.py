"""학습 설정과 job 관련 route."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

from fastapi import APIRouter, Body, Query
from fastapi.responses import StreamingResponse

from ..errors import FieldError, WebValidationError
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

# 이 저장소의 train pipeline은 torchvision Faster R-CNN 하나만 지원합니다.
# design mockup에 있는 모델 family 선택은 실제로 존재하지 않으므로 만들지 않습니다.
ARCHITECTURE = "fasterrcnn_resnet50_fpn"

SSE_POLL_SECONDS = 1.0
SSE_MAX_SECONDS = 60 * 60 * 12


def public_record(record: JobRecord) -> dict[str, Any]:
    """화면에 돌려주기 전에 credential처럼 보이는 값을 가립니다."""

    return redact(record.to_dict())


@router.get("/defaults")
def get_defaults() -> dict[str, Any]:
    """새 실험 form을 서버가 정의합니다. 기본값의 유일한 출처입니다."""

    cuda = cuda_is_available()
    return {
        "architecture": ARCHITECTURE,
        "architecture_note": "이 저장소의 train pipeline은 Faster R-CNN만 지원합니다.",
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


@router.post("/jobs", status_code=201)
def start_job(payload: StartJobRequest = Body(...)) -> dict[str, Any]:
    record = get_manager().start(payload.config_id)
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


@router.get("/jobs/{job_id}/events")
def stream_job_events(job_id: str) -> StreamingResponse:
    """진행 상황을 Server-Sent Events로 흘려보냅니다."""

    get_manager().get(job_id)  # 없는 job이면 여기서 404
    return StreamingResponse(
        _event_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
