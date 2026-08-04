import json
from collections import OrderedDict
from types import MappingProxyType

import pytest

from src.common.contract import (
    JSON_SERIALIZABLE_KEYS,
    REQUIRED_RETURN_KEYS,
    RETURN_SCHEMA,
    PipelineContractError,
    validate_pipeline_result,
)


def valid_result(**overrides):
    result = {
        "status": "ok",
        "artifacts": {},
        "summary": {},
        "message": "",
    }
    result.update(overrides)
    return result


def test_schema_matches_documented_keys():
    assert set(RETURN_SCHEMA) == REQUIRED_RETURN_KEYS
    assert REQUIRED_RETURN_KEYS == {"status", "artifacts", "summary", "message"}


def test_valid_result_passes_and_is_returned_unchanged():
    result = valid_result(summary={"pipeline": "data"})

    assert validate_pipeline_result(result, pipeline_name="data") is result


def test_status_outside_allowed_values_is_rejected():
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(valid_result(status="pending"), pipeline_name="train")

    assert "'status'는 'ok' 또는 'error'여야 합니다." in str(error.value)


def test_missing_key_names_the_key():
    result = valid_result()
    del result["message"]

    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(result, pipeline_name="data")

    assert "필수 key 누락: message" in str(error.value)
    assert "data pipeline" in str(error.value)


def test_all_missing_keys_are_reported_together():
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result({"status": "ok"}, pipeline_name="train")

    message = str(error.value)
    assert "필수 key 누락: artifacts, message, summary" in message


def test_unexpected_key_is_rejected():
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(valid_result(extra=1), pipeline_name="evaluate")

    assert "공통 계약에 없는 key: extra" in str(error.value)


def test_wrong_type_names_expected_and_actual():
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(valid_result(artifacts=[]), pipeline_name="data")

    message = str(error.value)
    assert "'artifacts' 타입 불일치" in message
    assert "JSON 직렬화 가능한 dict" in message
    assert "list" in message


def test_bool_is_not_accepted_as_str_or_dict():
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(valid_result(status=True), pipeline_name="data")

    assert "'status' 타입 불일치" in str(error.value)


def test_missing_and_type_problems_are_combined():
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(
            {"status": 1, "artifacts": {}, "summary": {}},
            pipeline_name="web",
        )

    message = str(error.value)
    assert "필수 key 누락: message" in message
    assert "'status' 타입 불일치" in message


def test_non_mapping_result_is_rejected():
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(["status"], pipeline_name="registry")

    assert "object(dict)를 반환해야" in str(error.value)
    assert "list" in str(error.value)


def test_top_level_mapping_proxy_type_is_rejected():
    """최상위 반환값은 읽기 전용 Mapping이 아니라 실제 dict여야 합니다."""

    result = MappingProxyType(valid_result())

    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(result, pipeline_name="registry")

    message = str(error.value)
    assert "object(dict)를 반환해야" in message
    assert "mappingproxy" in message


def test_contract_error_is_a_value_error():
    assert issubclass(PipelineContractError, ValueError)


# --- 리뷰 지적 회귀: 일반 Mapping이 아니라 JSON 직렬화 가능한 dict만 허용 ---


def test_schema_requires_plain_dict_for_artifacts_and_summary():
    assert RETURN_SCHEMA["artifacts"] is dict
    assert RETURN_SCHEMA["summary"] is dict
    assert JSON_SERIALIZABLE_KEYS == {"artifacts", "summary"}


@pytest.mark.parametrize("key", ["artifacts", "summary"])
def test_mapping_proxy_type_is_rejected(key):
    """회귀: MappingProxyType은 Mapping이지만 json.dumps()에서 실패합니다."""
    proxy = MappingProxyType({"a": 1})
    with pytest.raises(TypeError):  # 전제 확인: 실제로 직렬화가 불가능합니다.
        json.dumps(proxy)

    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(valid_result(**{key: proxy}), pipeline_name="data")

    message = str(error.value)
    assert f"'{key}' 타입 불일치" in message
    assert "JSON 직렬화 가능한 dict" in message
    assert "mappingproxy" in message


@pytest.mark.parametrize("key", ["artifacts", "summary"])
def test_non_serializable_value_inside_dict_is_rejected(key):
    """dict이지만 안에 JSON으로 못 바꾸는 값이 있으면 거부합니다."""
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(
            valid_result(**{key: {"bad": {1, 2}}}), pipeline_name="train"
        )

    assert f"'{key}'을(를) JSON으로 직렬화할 수 없습니다" in str(error.value)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nan_and_infinity_are_rejected(bad_value):
    """NaN과 Infinity는 표준 JSON이 아니므로 거부합니다."""
    with pytest.raises(PipelineContractError) as error:
        validate_pipeline_result(
            valid_result(summary={"score": bad_value}), pipeline_name="evaluate"
        )

    assert "'summary'을(를) JSON으로 직렬화할 수 없습니다" in str(error.value)


def test_dict_subclass_is_accepted():
    """OrderedDict처럼 dict를 상속하고 직렬화 가능한 값은 통과합니다."""
    result = valid_result(artifacts=OrderedDict({"uri": "artifacts/a.json"}))

    assert validate_pipeline_result(result, pipeline_name="data") is result


def test_nested_serializable_values_are_accepted():
    result = valid_result(
        artifacts={"train_manifest_uri": "artifacts/train.jsonl", "count": 10},
        summary={"classes": ["a", "b"], "ratio": 0.8, "nested": {"ok": True}},
    )

    assert validate_pipeline_result(result, pipeline_name="data") is result
