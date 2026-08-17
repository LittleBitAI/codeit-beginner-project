"""요청 본문 model.

값의 타입 검증은 일부러 pydantic에 맡기지 않습니다. pydantic은 ``"3"``을 ``3``으로
바꾸는 등 값을 관대하게 변환하는데, 그러면 train과 똑같이 거부해야 할 값이 통과해
버립니다. 그래서 안쪽 값은 ``dict[str, Any]``로 그대로 받고 ``train_config``의 검증
미러가 유일한 판단 기준이 되게 합니다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


__all__ = ["ConfigRequest", "StartJobRequest"]


class ConfigRequest(BaseModel):
    """새 실험 화면이 보내는 설정 초안."""

    train: dict[str, Any] | None = Field(default=None, description="train 설정 후보")
    inputs: dict[str, Any] | None = Field(default=None, description="data artifact 입력")
    data: dict[str, Any] | None = Field(default=None, description="inputs.data 축약형")

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"train": self.train}
        if self.inputs is not None:
            payload["inputs"] = self.inputs
        if self.data is not None:
            payload["data"] = self.data
        return payload


class StartJobRequest(BaseModel):
    """저장해 둔 설정으로 학습을 시작하는 요청."""

    config_id: str = Field(min_length=32, max_length=32)


class SettingsBody(BaseModel):
    """설정 화면이 보내는 값."""

    evaluation_mode: Literal["parallel", "serial"] | None = Field(
        default=None, description="평가를 학습과 함께 돌릴지(parallel), 끝난 뒤에 돌릴지(serial)"
    )
    # 이름과 개수 검증은 `settings.py`가 합니다. 여기서 Literal로 다시 적으면 고를 수
    # 있는 지표 목록이 두 곳에 생깁니다.
    epoch_metrics: list[str] | None = Field(
        default=None, description="epoch 훑기가 순위를 매길 지표 3개(1순위부터)"
    )
