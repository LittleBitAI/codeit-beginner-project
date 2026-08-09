"""Web pipeline 내부에서 쓰는 오류 type."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


__all__ = [
    "FieldError",
    "JobConflictError",
    "JobNotFoundError",
    "TeamSyncAuthError",
    "TeamSyncError",
    "WebError",
    "WebPathError",
    "WebStateError",
    "WebValidationError",
]


class WebError(RuntimeError):
    """Web pipeline의 모든 공개 오류가 상속하는 기본 type."""


class WebPathError(WebError):
    """저장소를 벗어나거나 허용되지 않는 경로가 들어왔을 때 발생합니다."""


class WebStateError(WebError):
    """런타임 사이에 상태를 옮기는 설정이 잘못됐을 때 발생합니다."""


class FieldError:
    """설정 화면의 특정 입력 칸에 붙일 오류 하나."""

    __slots__ = ("field", "message")

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldError):
            return NotImplemented
        return self.field == other.field and self.message == other.message

    def __hash__(self) -> int:
        return hash((self.field, self.message))

    def __repr__(self) -> str:
        return f"FieldError(field={self.field!r}, message={self.message!r})"


class WebValidationError(WebError):
    """설정 검증 실패. 문제를 한 번에 모두 모아 보고합니다."""

    def __init__(self, errors: Sequence[FieldError]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(f"{item.field}: {item.message}" for item in self.errors))

    def as_list(self) -> list[dict[str, str]]:
        return [item.as_dict() for item in self.errors]


class JobConflictError(WebError):
    """이미 실행 중인 학습이 있거나, 상태 전이가 허용되지 않을 때 발생합니다."""


class JobNotFoundError(WebError):
    """요청한 job 또는 config를 찾지 못했을 때 발생합니다."""


class TeamSyncError(WebError):
    """팀 동기화 설정, 연결 또는 원격 API 호출에 실패했습니다."""


class TeamSyncAuthError(TeamSyncError):
    """팀 동기화에 필요한 사용자 인증이 없거나 만료됐습니다."""


def collect(errors: list[FieldError], field: str, message: str) -> None:
    """검증 도중 발견한 문제를 누적합니다."""

    errors.append(FieldError(field, message))


def raise_if_any(errors: Sequence[FieldError]) -> None:
    """누적된 문제가 하나라도 있으면 한꺼번에 보고합니다."""

    if errors:
        raise WebValidationError(errors)


def error_payload(error: WebError) -> dict[str, Any]:
    """HTTP 응답에 넣을 오류 표현을 만듭니다."""

    if isinstance(error, WebValidationError):
        return {"message": "설정을 확인해 주세요.", "errors": error.as_list()}
    return {"message": str(error), "errors": []}
