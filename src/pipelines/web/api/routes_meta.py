"""서버 상태 확인 route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter


router = APIRouter(prefix="/api", tags=["meta"])

API_VERSION = "1"


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": API_VERSION}
