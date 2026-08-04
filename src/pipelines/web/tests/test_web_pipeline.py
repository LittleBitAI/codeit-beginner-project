import ast
import inspect
from unittest.mock import Mock

import pytest

import src.common.experiment_registry as experiment_registry
from src.common import ExperimentRegistryError, validate_pipeline_result
from src.common.storage import ObjectNotFoundError
from src.pipelines import web


def test_missing_web_config_keeps_dummy_compatibility():
    result = web.run({})

    assert web.__all__ == ["run"]
    assert result == {
        "status": "ok",
        "artifacts": {},
        "summary": {"pipeline": "web", "mode": "dummy"},
        "message": "web pipeline dummy 실행 완료",
    }


def test_web_calls_public_experiment_facade(monkeypatch):
    calls = []
    config = {
        "web": {
            "experiment_record_uri": "s3://bucket/registry/exp-1/record.json",
            "expected_run_id": "exp-1",
        }
    }

    def read_record(uri, received_config, *, expected_run_id=None):
        calls.append((uri, received_config, expected_run_id))
        return {"run_id": "exp-1", "schema_version": "1.0"}

    monkeypatch.setattr(web, "read_experiment_record", read_record)

    result = web.run(config)

    assert result["status"] == "ok"
    assert result["artifacts"] == {}
    assert result["summary"] == {
        "pipeline": "web",
        "mode": "experiment_registry",
        "run_id": "exp-1",
        "schema_version": "1.0",
        "experiment_record_uri": "s3://bucket/registry/exp-1/record.json",
    }
    assert calls == [
        ("s3://bucket/registry/exp-1/record.json", config, "exp-1")
    ]
    assert validate_pipeline_result(result, pipeline_name="web") is result


def test_web_returns_contract_error_when_facade_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise ExperimentRegistryError("record schema 오류")

    monkeypatch.setattr(web, "read_experiment_record", fail)

    result = web.run({"web": {"experiment_record_uri": "registry/record.json"}})

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "record schema 오류" in result["message"]
    assert validate_pipeline_result(result, pipeline_name="web") is result


def test_web_message_hides_record_uri_and_storage_error_detail(monkeypatch):
    uri = "s3://example-bucket/record.json?credential=SENSITIVE_URI_VALUE"
    storage = Mock()
    storage.read_json.side_effect = ObjectNotFoundError(
        "token=SENSITIVE_STORAGE_VALUE"
    )
    monkeypatch.setattr(experiment_registry, "create_storage", lambda config: storage)

    result = web.run({"web": {"experiment_record_uri": uri}})

    assert result["status"] == "error"
    assert "ObjectNotFoundError" in result["message"]
    assert uri not in result["message"]
    assert "SENSITIVE_URI_VALUE" not in result["message"]
    assert "SENSITIVE_STORAGE_VALUE" not in result["message"]


@pytest.mark.parametrize("uri", (None, "", "   ", 123))
def test_web_rejects_invalid_configured_record_uri(uri):
    result = web.run({"web": {"experiment_record_uri": uri}})

    assert result["status"] == "error"
    assert "experiment_record_uri" in result["message"]


def test_web_source_has_no_direct_artifact_reader_or_registry_pipeline_import():
    tree = ast.parse(inspect.getsource(web))
    forbidden_calls = {"Path", "open", "read_text", "create_storage", "read_json"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                imported.name != "src.pipelines.registry" for imported in node.names
            )
        if isinstance(node, ast.ImportFrom):
            assert node.module != "src.pipelines.registry"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls


@pytest.mark.parametrize("web_config", ([], "registry/record.json", 1))
def test_web_rejects_non_object_settings(web_config):
    result = web.run({"web": web_config})

    assert result["status"] == "error"
    assert "config['web']는 object" in result["message"]
