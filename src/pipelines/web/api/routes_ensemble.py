"""여러 실행의 test 예측을 합치는 route.

합칠 값어치가 있는지 **합치기 전에** 알려 주는 것이 이 route의 목적입니다. 이득을
확인하는 방법이 Kaggle 제출뿐이라, 잘못 고르면 하루치 제출이 사라집니다.

진단은 막지 않고 알려 주기만 합니다. 예측이 틀릴 때가 있고, 막아 버리면 반증할 길까지
막히기 때문입니다.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from .. import ensemble
from ..embedding import MAX_RERANK_MODELS
from ..ensemble_jobs import get_ensemble_runner


router = APIRouter(prefix="/api/ensemble", tags=["ensemble"])


class SelectionRequest(BaseModel):
    """합칠 실행 이름들."""

    run_ids: list[str] = Field(min_length=1, max_length=32)


class StartRequest(SelectionRequest):
    """합쳐서 제출을 만들 때 쓰는 요청."""

    run_id: str = Field(min_length=1, max_length=128)
    # 무엇을 앙상블하는지입니다. `model`은 실행 여럿의 예측을 합치고, `embedding`은
    # 실행 하나의 점수를 embedding 여럿으로 다시 매깁니다. 둘은 GPU를 쓰는 양도
    # 결과의 뜻도 달라서, 고른 개수로 눈치껏 가르지 않고 화면이 말하게 합니다.
    mode: Literal["model", "embedding"] = "model"
    # 사진이 같은데 위치만 다른 것을 사람이 확인했을 때만 켭니다. 진단이 시험지
    # 경고를 냈을 때 화면이 이 값을 물어봅니다.
    allow_copied_images: bool = False
    overwrite: bool = False
    # 합친 뒤 점수를 다시 매기는 데 쓸 embedding 학습들입니다. 비우면 지금까지처럼
    # detector 점수를 그대로 냅니다. 상한은 `embedding.py`가 정합니다 — 여기에
    # 숫자를 다시 적으면 한쪽만 고쳐졌을 때 형식 검사와 실제 검사가 갈립니다.
    embedding_run_ids: list[str] = Field(default_factory=list, max_length=MAX_RERANK_MODELS)


@router.get("/candidates")
def list_candidates() -> dict[str, Any]:
    """합칠 수 있는 실행 목록입니다. 점수가 높은 것부터 옵니다."""

    return {"candidates": ensemble.list_candidates()}


@router.post("/diagnose")
def diagnose(request: SelectionRequest = Body(...)) -> dict[str, Any]:
    """고른 조합을 합치기 전에 알 수 있는 것을 전부 잽니다.

    예측 파일을 읽어야 해서 처음 한 번은 수 초 걸립니다. 재 본 쌍은 저장해 두고 다시
    씁니다 — 후보를 하나씩 바꿔 볼 때 같은 쌍을 계속 다시 재지 않으려는 것입니다.
    """

    return ensemble.diagnose(request.run_ids)


@router.post("/jobs", status_code=201)
def start(request: StartRequest = Body(...)) -> dict[str, Any]:
    """고른 실행을 합쳐 제출 CSV를 만듭니다.

    예측이 아직 없는 실행은 **먼저 만듭니다.** 체크포인트만 있으면 후보가 되므로,
    한 번도 test 추론을 안 돌린 학습도 여기서 바로 고를 수 있습니다. 그 단계만
    GPU를 쓰고, 합치는 것 자체는 CPU로 몇 분입니다.

    `mode="embedding"`이면 합치지 않습니다. 실행 하나의 test 추론을 다시 돌리고 고른
    embedding들로 점수만 다시 매깁니다 — 융합 없이 재순위만 한 제출을 재현하는 길입니다.
    """

    if request.mode == "embedding":
        return get_ensemble_runner().start_rerank(
            request.run_ids,
            run_id=request.run_id,
            embedding_run_ids=request.embedding_run_ids,
            overwrite=request.overwrite,
        )
    return get_ensemble_runner().start(
        request.run_ids,
        run_id=request.run_id,
        allow_copied_images=request.allow_copied_images,
        overwrite=request.overwrite,
        embedding_run_ids=request.embedding_run_ids,
    )


@router.get("/jobs")
def status() -> dict[str, Any]:
    """지금 도는 융합의 상태입니다. 한 번에 하나만 돕니다."""

    return get_ensemble_runner().status()
