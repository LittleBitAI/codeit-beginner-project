"""crop embedding을 학습하고 목록으로 보여 주는 route.

학습은 detector와 같은 대기열을 지납니다. 그래서 여기서 하는 일은 설정을
검사해 저장하고 그 대기열에 넣는 것뿐이며, 진행과 log와 취소는 학습 화면이
이미 쓰는 `/api/train/jobs/...`를 그대로 씁니다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Header
from pydantic import BaseModel, ConfigDict, Field

from .. import embedding
from ..jobs import get_manager
from ..masking import redact
from ..train_config import write_runtime_config


router = APIRouter(prefix="/api/embedding", tags=["embedding"])


class TrainRequest(BaseModel):
    """embedding 학습 하나를 거는 요청입니다.

    값을 검사하는 곳은 `embedding.build_config` 하나입니다. 여기서 형식을 다시
    적으면 두 곳이 갈립니다 — 화면이 보는 규칙과 실제로 걸리는 규칙이 달라집니다.
    """

    model_config = ConfigDict(extra="forbid")

    crop_bank_uri: str = Field(min_length=1, max_length=1024)
    class_map_uri: str = Field(min_length=1, max_length=1024)
    run_id: str | None = Field(default=None, max_length=128)
    backbone: str | None = Field(default=None, max_length=64)
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    pretrained: bool | None = None
    seed: int | None = None
    device: str | None = Field(default=None, max_length=16)

    def as_payload(self) -> dict[str, Any]:
        return {key: value for key, value in self.model_dump().items() if value is not None}


@router.get("/defaults")
def get_defaults() -> dict[str, Any]:
    """폼이 그릴 backbone 목록과 기본값입니다."""

    return embedding.defaults()


@router.get("/runs")
def list_runs() -> dict[str, Any]:
    """이 서버가 학습한 embedding 목록입니다. 재순위에서 고를 후보이기도 합니다."""

    return {"runs": embedding.list_runs()}


@router.post("/jobs", status_code=201)
def start(
    payload: TrainRequest = Body(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """embedding 학습을 대기열에 넣습니다. 비어 있으면 곧바로 시작합니다."""

    config = embedding.build_config(payload.as_payload())
    config_id = write_runtime_config(config)
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip() or None
    started = get_manager().enqueue(config_id, access_token=token)
    return {
        "config_id": config_id,
        "run_id": config["train"]["run_id"],
        "config": redact(config),
        "started": started.to_dict() if started is not None else None,
    }
