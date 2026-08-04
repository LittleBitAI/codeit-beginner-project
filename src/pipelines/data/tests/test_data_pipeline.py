from copy import deepcopy

import pytest

from src.common import validate_pipeline_result
from src.pipelines.data import run


ARTIFACTS = {
    "train_manifest_uri": "artifacts/data/train.json",
    "validation_manifest_uri": "artifacts/data/validation.json",
    "class_map_uri": "artifacts/data/class_map.json",
    "dataset_summary_uri": "artifacts/data/summary.json",
}


def integration_config() -> dict:
    return {
        "execution": {"mode": "local"},
        "inputs": {"data": dict(ARTIFACTS)},
    }


def test_dummy_mode_keeps_previous_result():
    result = run(
        {
            "execution": {"mode": "dummy"},
            "inputs": {"data": {"train_manifest_uri": None}},
        }
    )

    assert result == {
        "status": "ok",
        "artifacts": {},
        "summary": {"pipeline": "data", "mode": "dummy"},
        "message": "data pipeline dummy 실행 완료",
    }


def test_non_dummy_mode_returns_provided_artifacts_without_mutating_inputs():
    config = integration_config()
    config["inputs"]["data"]["extra"] = "ignored"
    before = deepcopy(config)

    result = run(config)

    assert result["status"] == "ok"
    assert result["artifacts"] == ARTIFACTS
    assert result["artifacts"] is not config["inputs"]["data"]
    assert config == before
    assert validate_pipeline_result(result, pipeline_name="data") is result


def test_artifact_uri_is_returned_exactly_as_provided():
    config = integration_config()
    config["inputs"]["data"]["train_manifest_uri"] = "  s3://bucket/train.json  "

    result = run(config)

    assert result["artifacts"]["train_manifest_uri"] == "  s3://bucket/train.json  "


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"inputs": None},
        {"inputs": {}},
        {"inputs": {"data": None}},
        {"inputs": {"data": []}},
        {
            "inputs": {
                "data": {
                    key: value
                    for key, value in ARTIFACTS.items()
                    if key != "train_manifest_uri"
                }
            }
        },
        {
            "inputs": {
                "data": {**ARTIFACTS, "validation_manifest_uri": "   "}
            }
        },
        {"inputs": {"data": {**ARTIFACTS, "class_map_uri": 123}}},
    ],
)
def test_invalid_non_dummy_inputs_share_one_four_key_error_result(config):
    result = run(config)

    assert result == run({"inputs": {"data": {}}})
    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert result["summary"]["required_artifact_keys"] == list(ARTIFACTS)
    assert list(ARTIFACTS) == [
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
        "dataset_summary_uri",
    ]
    assert validate_pipeline_result(result, pipeline_name="data") is result
