"""사람이 한 번 정해 두고 계속 쓰는 값의 route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .. import settings as web_settings
from ..jobs import get_manager
from .schemas import SettingsBody


router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def read_settings() -> dict[str, Any]:
    return web_settings.read_settings()


@router.put("")
def save_settings(body: SettingsBody) -> dict[str, Any]:
    # `exclude_unset`이라야 **보내지 않은 값**과 **비우려고 보낸 null**이 갈립니다.
    # `exclude_none`은 둘을 똑같이 지워서, 고른 지표를 해제할 방법이 없어집니다.
    saved = web_settings.write_settings(body.model_dump(exclude_unset=True))
    # 직렬에서 병렬로 바꾸면 기다리던 평가가 곧바로 돌 수 있습니다. 다음 학습이
    # 끝날 때까지 기다리게 두면 바꾼 것이 아무 일도 하지 않은 것처럼 보입니다.
    get_manager().wake_evaluation()
    return saved
