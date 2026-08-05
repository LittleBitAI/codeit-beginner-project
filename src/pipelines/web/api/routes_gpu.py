"""로컬 GPU 상태 route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..gpu import probe


router = APIRouter(prefix="/api/gpu", tags=["gpu"])


@router.get("/status")
def gpu_status() -> dict[str, Any]:
    """GPU 상태를 돌려줍니다. 조회에 실패해도 200으로 응답합니다.

    실패를 오류로 만들면 화면이 통째로 깨집니다. 대신 어떤 이유로 못 가져왔는지를
    담아 보내고, 화면은 0으로 채운 게이지 대신 "가져올 수 없습니다"를 표시합니다.
    """

    return probe()
