"""Pipeline `run(config)` 반환값의 공통 계약 정의와 검증."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


#: 모든 pipeline이 반환해야 하는 필수 key와 값 타입입니다.
#: 계약에 없는 key는 허용하지 않습니다.
RETURN_SCHEMA: dict[str, type] = {
    "status": str,
    "artifacts": Mapping,
    "summary": Mapping,
    "message": str,
}

#: `RETURN_SCHEMA`의 key 집합입니다.
REQUIRED_RETURN_KEYS = frozenset(RETURN_SCHEMA)

#: 오류 메시지에 사용할 사람이 읽기 쉬운 타입 이름입니다.
_TYPE_NAMES: dict[type, str] = {
    str: "문자열(str)",
    Mapping: "object(dict)",
}


def _type_name(value_type: type) -> str:
    return _TYPE_NAMES.get(value_type, value_type.__name__)


def _actual_type_name(value: Any) -> str:
    return type(value).__name__


class PipelineContractError(ValueError):
    """Pipeline 반환값이 공통 계약을 위반한 경우입니다.

    기존 호출부와의 호환을 위해 `ValueError`를 상속합니다.
    """


def validate_pipeline_result(result: Any, *, pipeline_name: str) -> Mapping[str, Any]:
    """`run(config)` 반환값이 공통 계약을 지키는지 확인하고 그대로 돌려줍니다.

    누락 key, 계약에 없는 key, 타입 불일치를 모두 모아 한 번에 알립니다.
    문제가 없으면 전달받은 `result`를 그대로 반환합니다.
    """

    if not isinstance(result, Mapping):
        raise PipelineContractError(
            f"{pipeline_name} pipeline은 공통 계약에 따라 object(dict)를 반환해야 하지만 "
            f"{_actual_type_name(result)}을(를) 반환했습니다."
        )

    problems: list[str] = []

    missing_keys = sorted(REQUIRED_RETURN_KEYS - set(result))
    if missing_keys:
        problems.append(f"필수 key 누락: {', '.join(missing_keys)}")

    unexpected_keys = sorted(set(result) - REQUIRED_RETURN_KEYS)
    if unexpected_keys:
        problems.append(f"공통 계약에 없는 key: {', '.join(unexpected_keys)}")

    for key in sorted(REQUIRED_RETURN_KEYS & set(result)):
        expected_type = RETURN_SCHEMA[key]
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, expected_type):
            problems.append(
                f"'{key}' 타입 불일치: {_type_name(expected_type)}이(가) 필요하지만 "
                f"{_actual_type_name(value)}을(를) 받음"
            )

    if problems:
        detail = "; ".join(problems)
        raise PipelineContractError(
            f"{pipeline_name} pipeline의 반환값이 공통 계약과 다릅니다. {detail}"
        )

    return result
