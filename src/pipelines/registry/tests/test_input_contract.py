"""이전 pipeline artifact의 필수 key와 타입 검증 테스트.

`README.md`에 문서화한 계약이 실제 코드와 일치하는지, 그리고 key 누락과 타입
불일치가 각각 명확한 예외로 이어지는지 확인합니다.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.pipelines.registry.record import (
    REQUIRED_ARTIFACT_KEYS,
    CorruptedArtifactError,
    InvalidSchemaError,
    MissingInputError,
    RegistryError,
    resolve_run_id,
    validate_inputs,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
README_PATH = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture
def valid_inputs() -> dict:
    return json.loads((FIXTURE_DIR / "inputs_s3.json").read_text(encoding="utf-8"))


# --- 계약 문서와 코드의 일치 ------------------------------------------------


def test_required_keys_match_the_artifact_contract():
    assert REQUIRED_ARTIFACT_KEYS == {
        "data": (
            "train_manifest_uri",
            "validation_manifest_uri",
            "class_map_uri",
            "dataset_summary_uri",
        ),
        "train": (
            "run_id",
            "best_checkpoint_uri",
            "last_checkpoint_uri",
            "training_history_uri",
        ),
        "evaluate": (
            "run_id",
            "metrics_uri",
            "predictions_uri",
        ),
    }


def test_readme_documents_every_required_key():
    readme = README_PATH.read_text(encoding="utf-8")
    for pipeline, required_keys in REQUIRED_ARTIFACT_KEYS.items():
        assert f'config["inputs"]["{pipeline}"]' in readme
        for key in required_keys:
            assert f"`{key}`" in readme


def test_all_registry_exceptions_share_one_base_class():
    for error_type in (MissingInputError, InvalidSchemaError, CorruptedArtifactError):
        assert issubclass(error_type, RegistryError)


# --- 성공 경로 -------------------------------------------------------------


def test_valid_inputs_pass_validation(valid_inputs):
    validated = validate_inputs(valid_inputs)

    assert set(validated) == set(REQUIRED_ARTIFACT_KEYS)
    assert validated["train"]["run_id"] == "exp-0002"


def test_validation_does_not_mutate_inputs(valid_inputs):
    before = copy.deepcopy(valid_inputs)

    validate_inputs(valid_inputs)

    assert valid_inputs == before


def test_surrounding_whitespace_is_trimmed(valid_inputs):
    valid_inputs["evaluate"]["metrics_uri"] = "  s3://example-bucket/m.json  "

    validated = validate_inputs(valid_inputs)

    assert validated["evaluate"]["metrics_uri"] == "s3://example-bucket/m.json"


def test_unknown_extra_keys_are_ignored(valid_inputs):
    valid_inputs["data"]["experimental_uri"] = "s3://example-bucket/extra.json"

    validated = validate_inputs(valid_inputs)

    assert "experimental_uri" not in validated["data"]


# --- key 누락 --------------------------------------------------------------


@pytest.mark.parametrize("pipeline", sorted(REQUIRED_ARTIFACT_KEYS))
def test_missing_pipeline_raises_missing_input_error(valid_inputs, pipeline):
    del valid_inputs[pipeline]

    with pytest.raises(MissingInputError) as error:
        validate_inputs(valid_inputs)

    assert pipeline in str(error.value)


@pytest.mark.parametrize(
    ("pipeline", "key"),
    [
        (pipeline, key)
        for pipeline, keys in REQUIRED_ARTIFACT_KEYS.items()
        for key in keys
    ],
)
def test_missing_key_raises_missing_input_error(valid_inputs, pipeline, key):
    del valid_inputs[pipeline][key]

    with pytest.raises(MissingInputError) as error:
        validate_inputs(valid_inputs)

    assert key in str(error.value)


# --- 타입 불일치 -----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        (123, "int"),
        (1.5, "float"),
        (True, "bool"),
        (None, "NoneType"),
        (["s3://example-bucket/a.json"], "list"),
        ({"uri": "s3://example-bucket/a.json"}, "dict"),
    ],
)
def test_wrong_value_type_raises_invalid_schema_error(valid_inputs, value, type_name):
    valid_inputs["data"]["class_map_uri"] = value

    with pytest.raises(InvalidSchemaError) as error:
        validate_inputs(valid_inputs)

    message = str(error.value)
    assert "class_map_uri" in message
    assert "str이어야 하는데" in message
    assert type_name in message


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_value_raises_invalid_schema_error(valid_inputs, value):
    valid_inputs["train"]["best_checkpoint_uri"] = value

    with pytest.raises(InvalidSchemaError, match="비어 있지 않은 문자열"):
        validate_inputs(valid_inputs)


@pytest.mark.parametrize("value", [[], "artifacts", 7])
def test_non_object_pipeline_entry_raises_invalid_schema_error(valid_inputs, value):
    valid_inputs["evaluate"] = value

    with pytest.raises((InvalidSchemaError, MissingInputError)):
        validate_inputs(valid_inputs)


def test_non_object_inputs_raises_invalid_schema_error():
    with pytest.raises(InvalidSchemaError, match="object여야 합니다"):
        validate_inputs(["data", "train", "evaluate"])


# --- run_id 일관성 ---------------------------------------------------------


def test_run_id_mismatch_raises_invalid_schema_error(valid_inputs):
    valid_inputs["evaluate"]["run_id"] = "exp-9999"
    validated = validate_inputs(valid_inputs)

    with pytest.raises(InvalidSchemaError, match="run_id가 서로 다릅니다"):
        resolve_run_id(validated)


def test_blank_configured_run_id_raises_invalid_schema_error(valid_inputs):
    validated = validate_inputs(valid_inputs)

    with pytest.raises(InvalidSchemaError, match="비어 있지 않은 문자열"):
        resolve_run_id(validated, configured_run_id="   ")
