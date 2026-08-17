"""학습 설정과 job 관련 route."""

from __future__ import annotations

import json
import threading
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
from ..epoch_sweep import (
    DEFAULT_SAMPLE_SIZE,
    epoch_candidates,
    get_epoch_sweep_runner,
)
from .. import experiments
from .. import settings as web_settings
from ..gpu import cuda_is_available
from ..jobs import get_manager
from ..jobs.model import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_SUCCEEDED,
    TERMINAL_STATUSES,
    JobRecord,
)
from ..masking import redact
from ..train_config import (
    build_resume_config,
    data_field_specs,
    field_specs,
    next_resume_run_id,
    read_runtime_config,
    resume_checkpoint_exists,
    stored_run_ids,
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


RESUMABLE_STATUSES = {
    STATUS_INTERRUPTED,
    STATUS_FAILED,
    STATUS_CANCELLED,
    # 끝까지 간 학습도 이어갈 수 있습니다. best_epoch이 마지막 epoch이면 더 배울 것이
    # 남아 있다는 뜻이고, 그때 처음부터 다시 도는 것은 이미 한 학습을 두 번 하는 일입니다.
    STATUS_SUCCEEDED,
}


def _completed_epochs(record: JobRecord) -> int | None:
    """그 실행이 실제로 마친 epoch 수입니다. train이 알려 주지 않았으면 ``None``."""

    value = record.summary.get("completed_epochs")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _stopped_early_refusal(record: JobRecord) -> str | None:
    """조기 종료로 끝난 실행을 이어갈 수 없는 이유입니다. 이어갈 수 있으면 ``None``.

    조기 종료로 끝난 실행의 checkpoint에는 patience를 다 쓴 상태가 그대로 들어 있어,
    이어서 하면 train이 첫 batch 전에 거절합니다.

    **단추를 세울지 정하는 GET과 실제로 시작하는 POST가 같은 함수를 봅니다.** 한쪽에만
    두면 화면은 단추를 감추는데 직접 부른 POST는 통과해, 반드시 실패할 학습이 대기열에
    들어갑니다.
    """

    if record.status != STATUS_SUCCEEDED or not record.summary.get("stopped_early"):
        return None
    return (
        "조기 종료로 끝난 학습입니다. patience를 다 쓴 상태가 checkpoint에 함께 "
        "저장돼 이어갈 수 없습니다. 새 실험에서 patience를 올리거나 early "
        "stopping을 끄고 시작하세요."
    )


#: 이어 학습 이름을 고르고 대기열에 넣기까지를 한 번에 하나만 지나갑니다.
#:
#: 이름을 고르는 것과 줄을 세우는 것이 갈라져 있으면, 두 요청이 각자 "A.2는 아직
#: 없다"를 보고 둘 다 A.2를 만듭니다. 사람이 단추를 두 번 누르거나 두 화면에서 누르면
#: 그렇게 되고, 뒤엣것은 밤새 기다렸다 이름 충돌로 죽습니다. FastAPI는 `def` route를
#: threadpool에서 돌리므로 실제로 동시에 들어옵니다.
#:
#: **manager의 lock을 쓰면 안 됩니다.** `enqueue`가 안에서 `_start_lock` → `_lock`
#: 순서로 잡는데, 그 바깥에서 `_lock`을 먼저 쥐면 순서가 뒤집혀 교착합니다. 이 lock은
#: 여기서만 잡고 아무도 기다리지 않으므로 그 고리를 만들지 않습니다.
_RESUME_NAMING_LOCK = threading.Lock()


def _taken_run_ids(manager: Any) -> list[str]:
    """이 서버가 아는 run_id 전부입니다.

    **저장된 설정이 기준입니다.** 이름은 config를 쓰는 순간부터 붙잡히고 그 파일은
    지워지지 않으므로, 대기열에서 빠져 `JobRecord`가 되기 전 같은 어느 목록에도 없는
    순간이 생기지 않습니다. job 기록과 대기열도 함께 세는 것은 config 파일이 없는
    기록(다른 곳에서 복원된 것)까지 덮기 위해서입니다 — 합집합은 더 세는 쪽으로만
    틀리고, 그쪽은 번호 하나를 건너뛸 뿐입니다.
    """

    names = [job.run_id for job in manager.list_jobs()]
    names.extend(str(entry.get("run_id") or "") for entry in manager.queue_entries())
    names.extend(stored_run_ids())
    return names


def _plan_refusal(record: JobRecord, epochs: int | None) -> str | None:
    """총 epoch을 늘리지 않은 이어 학습을 시작 전에 거절할 이유입니다.

    ``epochs``는 남은 수가 아니라 **전체 목표**입니다. 끝까지 간 학습을 그대로 이어가면
    "이미 지난 epoch보다 크지 않다"며 train이 거절하는데, 그 답은 job과 config를 만들고
    대기열을 다시 돌린 뒤에 옵니다. 화면에서 몇 epoch 더 돌릴지 받으면 여기서 끝납니다.
    """

    if record.status != STATUS_SUCCEEDED:
        return None
    done = _completed_epochs(record)
    if done is None:
        return None
    if epochs is None:
        return f"이 학습은 epoch {done}까지 마쳤습니다. 몇 epoch까지 더 돌릴지 정해 주세요."
    if epochs <= done:
        return (
            f"총 epoch은 이미 마친 {done}보다 커야 합니다. "
            f"{done + 5}처럼 더 큰 값을 보내세요."
        )
    return None


@router.get("/jobs/{job_id}/resume")
def resume_availability(job_id: str) -> dict[str, Any]:
    """이 학습을 이어서 **시도할 수 있는지** 알려 줍니다.

    화면이 이어서 학습 단추를 세울지 정하는 데 씁니다. 화면이 완료한 epoch 수로 셈하면
    "저장됐을 가능성"만 알 뿐입니다 — checkpoint는 `checkpoint_every` 주기로 저장되고,
    이어온 실행에는 앞선 실행의 epoch까지 섞여 있습니다. 실제 저장소를 보는 쪽이
    알려 줘야 단추와 서버의 답이 같아집니다.

    `available`은 "이어갈 수 있다"가 아니라 **"눌러 볼 수 있다"**입니다. 저장소를 읽지
    못한 것과 checkpoint가 없는 것은 다릅니다. 못 읽었다고 단추를 없애면 눌러서 알아낼
    수 있는 것까지 막고, 사람은 새로고침 말고 할 것이 없습니다. 모를 때는 시도할 수 있게
    두고 이유를 함께 적습니다 — 실제로 이어갈 수 없으면 POST가 같은 말로 거절합니다.

    끝나지 않은 학습은 저장소를 보지 않고 답합니다. 목록마다 부르면 job 수만큼 S3를
    두드리게 되므로, 화면은 학습 하나를 열 때만 부릅니다.
    """

    record = get_manager().get(job_id)
    if record.status not in RESUMABLE_STATUSES:
        return {"available": False, "reason": "끝난 학습만 이어갈 수 있습니다."}
    stopped_early = _stopped_early_refusal(record)
    if stopped_early is not None:
        return {"available": False, "reason": stopped_early}
    try:
        exists = resume_checkpoint_exists(
            read_runtime_config(record.config_id), record.artifacts
        )
    except StorageError:
        return {
            "available": True,
            "reason": "checkpoint를 확인하지 못했습니다. 저장소 설정과 권한을 확인하세요.",
        }
    if not exists:
        return {
            "available": False,
            "reason": "저장된 checkpoint가 없습니다. checkpoint 주기를 채우기 전에 끝난 학습입니다.",
        }
    return {"available": True, "reason": None}


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
    """끝난 학습을 이어서 시작합니다. 중단·중지·실패한 것도, 끝까지 간 것도 같습니다.

    이어서 하는 실행은 **새 이름**을 받습니다. 같은 이름을 다시 쓰면 train이 남아 있는
    작업 폴더를 보고 시작을 거부하고, 결과도 섞입니다. `epochs`는 남은 수가 아니라
    전체 목표이므로 비워 두면 중단된 실행의 계획을 그대로 씁니다. 끝까지 간 학습은
    그 계획을 이미 채웠으므로 더 큰 값을 받아야 합니다.

    끝난 이유는 묻지 않고 **checkpoint가 있는지만** 봅니다. 사람이 중지했든, 실패했든,
    서버가 상태를 잃었든 이어갈 수 있는 조건은 같습니다. 확인을 건너뛰면 새 설정과 job을
    만들고 대기열까지 다시 돌린 뒤에야 train이 checkpoint를 읽다가 죽습니다 — 미리 말할
    수 있는 실패를 실행을 만든 뒤로 미루는 셈입니다.
    """

    manager = get_manager()
    record = manager.get(job_id)
    if record.status not in RESUMABLE_STATUSES:
        raise JobConflictError(
            "끝난 학습만 이어갈 수 있습니다. 돌고 있는 학습은 끝난 뒤에 이어가세요."
        )
    refusal = _stopped_early_refusal(record) or _plan_refusal(record, payload.epochs)
    if refusal is not None:
        raise JobConflictError(refusal)

    source_config = read_runtime_config(record.config_id)
    try:
        checkpoint_exists = resume_checkpoint_exists(source_config, record.artifacts)
    except StorageError as error:
        raise JobConflictError(
            "그 학습의 checkpoint를 확인하지 못했습니다. "
            "저장소 설정과 권한을 확인하세요."
        ) from error
    if not checkpoint_exists:
        raise JobConflictError(
            "저장된 checkpoint를 찾을 수 없어 이어서 학습할 수 없습니다. "
            "checkpoint 주기를 채우기 전에 끝난 학습입니다."
        )

    # 이름을 고르는 것과 그 이름이 대기열에 보이는 것 사이를 갈라 두면, 그 틈에 들어온
    # 두 번째 요청이 같은 이름을 고릅니다. 한 번에 하나만 지나갑니다.
    with _RESUME_NAMING_LOCK:
        taken = _taken_run_ids(manager)
        # 사람이 이름을 직접 적었으면 번호를 매기지 않습니다. 대신 **같은 목록으로**
        # 겹치는지 봅니다. 여기서 보지 않으면 자동 이름만 규칙을 지키고, 직접 적은
        # 이름은 대기열까지 갔다가 train이 첫 batch 전에 거절합니다.
        if payload.run_id is not None and payload.run_id in set(taken):
            raise JobConflictError(
                f"'{payload.run_id}'는 이미 쓰고 있는 실행 이름입니다. "
                "다른 이름을 쓰거나 이름을 비워 자동으로 짓게 하세요."
            )
        config = build_resume_config(
            source_config,
            artifacts=record.artifacts,
            # 이름은 A -> A.2 -> A.3으로 이어집니다. 이미 있는 번호를 다시 쓰면 train이
            # 시작을 거부하므로, 이 서버가 아는 이름을 모두 건네 건너뛰게 합니다.
            run_id=payload.run_id or next_resume_run_id(record.run_id, taken),
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


class EpochSweepRequest(BaseModel):
    """epoch 훑기 설정. 순위를 매길 지표는 설정 화면에서 미리 고릅니다."""

    device: str | None = Field(default=None)
    sample_size: int = Field(default=DEFAULT_SAMPLE_SIZE, ge=1, le=100000)


@router.get("/jobs/{job_id}/epoch-sweep")
def epoch_sweep_status(job_id: str) -> dict[str, Any]:
    """이 학습에 대한 훑기 상태와, 훑을 수 있는 후보 목록입니다."""

    record = get_manager().get(job_id)  # 없는 job이면 404
    return {
        "epoch_sweep": get_epoch_sweep_runner().status_for(record),
        "candidates": epoch_candidates(record),
        "metrics": web_settings.epoch_metrics(),
    }


@router.post("/jobs/{job_id}/epoch-sweep", status_code=202)
def start_epoch_sweep(
    job_id: str, payload: EpochSweepRequest = Body(...)
) -> dict[str, Any]:
    """보관해 둔 epoch checkpoint를 훑어 제일 잘 맞히는 것을 고릅니다.

    후보마다 표본 평가를 돌리고, 이긴 하나만 전수로 다시 재어 제출까지 만듭니다.
    후보가 20개면 몇십 분이 걸리므로 시작만 시키고 상태는 따로 확인합니다.

    도는 학습이나 평가가 있으면 시작하지 않습니다. 8GB 카드에서 겹치면 둘 다 out of
    memory로 잃습니다.
    """

    manager = get_manager()
    runner = get_epoch_sweep_runner()
    with runner.locked():
        record = manager.get(job_id)  # 지워졌으면 여기서 404
        if record.status != STATUS_SUCCEEDED:
            raise JobConflictError("성공으로 끝난 학습만 훑을 수 있습니다.")
        if manager.active_job() is not None:
            raise JobConflictError(
                "학습이 도는 중에는 훑을 수 없습니다. 끝난 뒤 다시 눌러 주세요."
            )
        if get_evaluation_runner().status().get("status") == "running":
            raise JobConflictError("평가가 도는 중에는 훑을 수 없습니다.")
        return {"epoch_sweep": runner.start(record, payload.model_dump())}


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
