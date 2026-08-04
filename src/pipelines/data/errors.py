"""Data pipeline 준비 경로가 사용하는 공개 예외입니다."""

from __future__ import annotations


__all__ = ["DatasetPreparationError"]


class DatasetPreparationError(RuntimeError):
    """사용자가 config나 원본 data를 고쳐야 하는 준비 실패입니다.

    이 예외의 message는 그대로 `run()` 결과의 `message`가 되므로, credential이나
    개인 컴퓨터 절대경로가 들어가지 않도록 file 이름 같은 안전한 값만 담습니다.
    """
