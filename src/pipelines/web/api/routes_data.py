"""전처리 데이터셋 선택 route.

새 실험마다 artifact 경로 4개를 손으로 넣는 대신, 전처리 결과 폴더를 한 번 고르면
그 안에서 4개를 찾아 기억합니다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from .. import datasets


router = APIRouter(prefix="/api/data", tags=["data"])


class DirectoryRequest(BaseModel):
    """전처리 결과가 들어 있는 저장소 기준 상대 폴더 경로."""

    directory: str = Field(min_length=1, max_length=512)


@router.post("/inspect")
def inspect(payload: DirectoryRequest = Body(...)) -> dict[str, Any]:
    """폴더를 살펴보기만 합니다. 고르지도, 저장하지도 않습니다."""

    return datasets.inspect_directory(payload.directory)


@router.get("/source")
def get_source() -> dict[str, Any]:
    """지금 고른 전처리 데이터셋입니다. 고른 적이 없으면 ``source``가 ``null``입니다."""

    return {"source": datasets.load_selection()}


@router.post("/source")
def set_source(payload: DirectoryRequest = Body(...)) -> dict[str, Any]:
    """전처리 데이터셋을 고릅니다. 4개를 모두 찾은 경우에만 저장됩니다."""

    return {"source": datasets.save_selection(payload.directory)}


@router.delete("/source")
def clear_source() -> dict[str, Any]:
    datasets.clear_selection()
    return {"source": None}


@router.post("/verify")
def verify(payload: DirectoryRequest = Body(...)) -> dict[str, Any]:
    """실제 data pipeline을 공개 CLI로 불러 계약이 성립하는지 확인합니다.

    ``python -m src.main_pipeline --config <config> --only data``

    data pipeline은 파일을 만들지 않습니다. 넘긴 URI 4개가 다음 pipeline으로 넘어갈 수
    있는지 검증해서 그대로 돌려줄 뿐입니다. 그래서 이 검사는 학습 전에 data → train
    연결이 성립하는지를 실제 pipeline 경로로 확인하는 용도입니다.
    """

    inspected = datasets.inspect_directory(payload.directory)
    if not inspected["complete"]:
        return {
            "inspected": inspected,
            "verification": {
                "ok": False,
                "exit_code": None,
                "message": "artifact 4개를 모두 찾은 폴더만 검증할 수 있습니다.",
                "artifacts": {},
                "summary": {},
            },
        }
    return {
        "inspected": inspected,
        "verification": datasets.verify_with_pipeline(inspected["data"]),
    }
