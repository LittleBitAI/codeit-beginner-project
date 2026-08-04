"""Evaluate pipeline이 사용하는 오류 유형입니다."""

from __future__ import annotations


class EvaluateError(RuntimeError):
    """Evaluate pipeline 실행이 실패했을 때 사용하는 기본 예외입니다."""


class ConfigurationError(EvaluateError):
    """config["evaluate"] 설정이 없거나 잘못된 경우입니다."""


class InputArtifactError(EvaluateError):
    """입력 artifact가 없거나 schema가 잘못되었거나 손상된 경우입니다."""


class PredictionError(EvaluateError):
    """Checkpoint 로드 또는 추론이 실패한 경우입니다."""


class ArtifactWriteError(EvaluateError):
    """산출물 저장이 실패했거나 기존 artifact를 덮어써야 하는 경우입니다."""


__all__ = [
    "ArtifactWriteError",
    "ConfigurationError",
    "EvaluateError",
    "InputArtifactError",
    "PredictionError",
]
