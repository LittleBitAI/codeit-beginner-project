import pytest

from src.common.contract import (
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
    assert "object(dict)" in message
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


def test_contract_error_is_a_value_error():
    assert issubclass(PipelineContractError, ValueError)
