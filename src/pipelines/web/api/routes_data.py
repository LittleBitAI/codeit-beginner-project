"""전처리 데이터셋 선택 route.

새 실험마다 artifact 경로를 손으로 넣는 대신, 전처리 결과 폴더를 한 번 고르면
필수 4개와 선택 test manifest를 찾아 기억합니다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from .. import datasets
from ..data_jobs import get_eda_runner, get_preparation_runner
from ..errors import FieldError, WebValidationError
from ..train_config import normalize_data_inputs


router = APIRouter(prefix="/api/data", tags=["data"])


class DirectoryRequest(BaseModel):
    """전처리 결과가 들어 있는 저장소 기준 상대 폴더 경로."""

    directory: str = Field(min_length=1, max_length=512)


@router.get("/datasets")
def list_datasets() -> dict[str, Any]:
    """전처리 결과 폴더 목록입니다. 골라 쓰라고 보여 주기만 합니다.

    경로를 손으로 붙여넣는 것 말고는 고를 방법이 없어서, 어떤 판이 있는지 알려면
    저장소를 직접 뒤져야 했습니다.
    """

    return datasets.list_processed_datasets()


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
    """전처리 데이터셋을 고릅니다. 필수 4개를 모두 찾은 경우에만 저장됩니다."""

    return {"source": datasets.save_selection(payload.directory)}


@router.delete("/source")
def clear_source() -> dict[str, Any]:
    datasets.clear_selection()
    return {"source": None}


class PrepareRequest(BaseModel):
    """원본에서 artifact를 만들 때 쓰는 설정."""

    split_ratio: str = Field(description='"8:2" 또는 "9:1"')
    seed: int = Field(default=42, ge=0, lt=2**32)
    overwrite: bool = Field(default=False)
    backend: str = Field(default="auto", description='"auto", "local", "s3"')
    raw_prefix: str | None = Field(default=None, max_length=512)
    processed_root: str | None = Field(default=None, max_length=512)
    # 임베딩 학습과 재순위가 읽을 참조 crop을 함께 만들지입니다. 판마다 따로 만들어지고
    # 나중에 덧붙일 수 없어서, 준비할 때가 유일한 기회입니다.
    crop_bank: bool = Field(default=False)


@router.get("/prepare")
def prepare_status() -> dict[str, Any]:
    """지금 돌고 있거나 마지막으로 끝난 준비 실행의 상태입니다."""

    return {
        "split_ratios": list(datasets.SPLIT_RATIOS),
        "backends": list(datasets.STORAGE_BACKENDS),
        "storage": datasets.storage_environment(),
        "preparation": get_preparation_runner().status(),
    }


@router.post("/prepare", status_code=202)
def start_prepare(payload: PrepareRequest = Body(...)) -> dict[str, Any]:
    """원본에서 필수 4개와 선택 test manifest를 만들도록 data pipeline을 부릅니다.

    원본을 다 읽어야 해서 오래 걸릴 수 있으므로 시작만 시키고 바로 응답합니다.
    상태는 ``GET /api/data/prepare``로 확인합니다. 성공하면 그 결과가 곧바로 현재
    전처리 데이터셋으로 선택됩니다.
    """

    runner = get_preparation_runner()
    return {
        "split_ratios": list(datasets.SPLIT_RATIOS),
        "backends": list(datasets.STORAGE_BACKENDS),
        "storage": datasets.storage_environment(),
        "preparation": runner.start(payload.model_dump()),
    }


class EdaRequest(BaseModel):
    """EDA를 돌릴 때 쓰는 설정. 대상은 지금 고른 전처리 dataset입니다."""

    image_sample: int = Field(default=200, ge=1, le=5000)
    overwrite: bool = Field(default=False)


@router.get("/eda")
def eda_status() -> dict[str, Any]:
    """지금 돌고 있거나 마지막으로 끝난 EDA의 상태와 리포트입니다."""

    return {"eda": get_eda_runner().status()}


@router.post("/eda", status_code=202)
def start_eda(payload: EdaRequest = Body(...)) -> dict[str, Any]:
    """고른 dataset을 model 없이 뜯어보도록 data pipeline을 부릅니다.

    이미지를 전부 열어야 해서 오래 걸리므로 시작만 시키고 바로 응답합니다. 상태와
    결과는 ``GET /api/data/eda``로 확인합니다.
    """

    return {"eda": get_eda_runner().start(payload.model_dump())}


class VerifyRequest(BaseModel):
    """검증할 대상. artifact URI를 직접 주거나, 위치를 줘서 찾게 할 수 있습니다."""

    data: dict[str, str] | None = Field(default=None, description="artifact URI 4개")
    directory: str | None = Field(default=None, max_length=512)


@router.post("/verify")
def verify(payload: VerifyRequest = Body(...)) -> dict[str, Any]:
    """실제 data pipeline을 공개 CLI로 불러 계약이 성립하는지 확인합니다.

    ``python -m src.main_pipeline --config <config> --only data``

    data pipeline은 파일을 만들지 않습니다. 넘긴 필수 URI와 선택 test URI가 다음 pipeline으로 넘어갈 수
    있는지 검증해서 그대로 돌려줄 뿐입니다. 그래서 이 검사는 학습 전에 data → train
    연결이 성립하는지를 실제 pipeline 경로로 확인하는 용도입니다.

    이미 artifact URI를 알고 있으면 그대로 검증합니다. 위치를 다시 훑지 않는 이유는,
    준비로 만들어진 산출물이 S3에 있을 수 있어 폴더 조회가 성립하지 않기 때문입니다.
    """

    if payload.data:
        data_inputs = normalize_data_inputs(payload.data)
        return {"inspected": None, "verification": datasets.verify_with_pipeline(data_inputs)}

    if not payload.directory:
        raise WebValidationError(
            [FieldError("data", "검증할 artifact URI나 위치가 필요합니다.")]
        )

    inspected = datasets.inspect_directory(payload.directory)
    if not inspected["complete"]:
        return {
            "inspected": inspected,
            "verification": {
                "ok": False,
                "supported": True,
                "exit_code": None,
                "message": "artifact 4개를 모두 찾은 위치만 검증할 수 있습니다.",
                "artifacts": {},
                "summary": {},
            },
        }
    return {
        "inspected": inspected,
        "verification": datasets.verify_with_pipeline(inspected["data"]),
    }
