"""팀 실시간 동기화의 공개 설정 route."""

from __future__ import annotations

from fastapi import APIRouter

from ..team_sync import get_team_sync


router = APIRouter(prefix="/api/team", tags=["team"])


@router.get("/config")
def team_config() -> dict[str, object]:
    """Frontend 초기화에 필요한 비밀이 아닌 값만 돌려줍니다."""

    return get_team_sync().config.public_dict()
