from unittest.mock import Mock

import pytest

import src.common.experiment_registry as experiment_registry
from src.common import ExperimentRegistryError, read_experiment_record
from src.common.storage import ObjectNotFoundError
from src.pipelines import registry


def registry_inputs() -> dict:
    return {
        "data": {
            "train_manifest_uri": "s3://example-bucket/datasets/train.json",
            "validation_manifest_uri": "s3://example-bucket/datasets/validation.json",
            "class_map_uri": "s3://example-bucket/datasets/classes.json",
            "dataset_summary_uri": "s3://example-bucket/datasets/summary.json",
        },
        "train": {
            "run_id": "exp-lookup",
            "best_checkpoint_uri": "s3://example-bucket/train/best.pt",
            "last_checkpoint_uri": "s3://example-bucket/train/last.pt",
            "training_history_uri": "s3://example-bucket/train/history.json",
        },
        "evaluate": {
            "run_id": "exp-lookup",
            "metrics_uri": "s3://example-bucket/evaluate/metrics.json",
            "predictions_uri": "s3://example-bucket/evaluate/predictions.json",
        },
    }


def test_reads_registry_local_result_uri_without_duplicating_storage_root(tmp_path):
    config = {
        "storage": {
            "backend": "local",
            "local": {"root": str(tmp_path / "artifacts")},
        },
        "registry": {
            "repo_root": str(tmp_path),
            "verify_artifacts": False,
            "created_at": "2026-08-04T00:00:00+00:00",
        },
        "inputs": registry_inputs(),
    }
    registry_result = registry.run(config)

    assert registry_result["status"] == "ok"
    record_uri = registry_result["artifacts"]["experiment_record_uri"]
    assert record_uri == "artifacts/registry/exp-lookup/experiment_record.json"

    record = read_experiment_record(
        record_uri,
        config,
        expected_run_id="exp-lookup",
    )

    assert record["run_id"] == "exp-lookup"
    assert record["schema_version"] == "1.0"


def test_reads_local_result_when_repo_root_equals_registry_named_storage_root(tmp_path):
    repo_and_storage_root = tmp_path / "registry"
    config = {
        "storage": {
            "backend": "local",
            "local": {"root": str(repo_and_storage_root)},
        },
        "registry": {
            "repo_root": str(repo_and_storage_root),
            "verify_artifacts": False,
            "created_at": "2026-08-04T00:00:00+00:00",
        },
        "inputs": registry_inputs(),
    }
    registry_result = registry.run(config)

    assert registry_result["status"] == "ok"
    record_uri = registry_result["artifacts"]["experiment_record_uri"]
    assert record_uri == "registry/exp-lookup/experiment_record.json"

    record = read_experiment_record(
        record_uri,
        config,
        expected_run_id="exp-lookup",
    )

    assert record["run_id"] == "exp-lookup"


def test_s3_uri_is_passed_to_storage_unchanged(monkeypatch):
    uri = "s3://example-bucket/registry/exp-lookup/experiment_record.json"
    storage = Mock()
    storage.read_json.return_value = {"run_id": "exp-lookup", "schema_version": "1.0"}
    monkeypatch.setattr(experiment_registry, "create_storage", lambda config: storage)

    record = read_experiment_record(uri, {}, expected_run_id="exp-lookup")

    assert record["run_id"] == "exp-lookup"
    storage.read_json.assert_called_once_with(uri)


@pytest.mark.parametrize(
    "record",
    (
        [],
        {},
        {"run_id": ""},
        {"run_id": "   "},
        {"run_id": 123},
    ),
)
def test_rejects_invalid_record_schema(monkeypatch, record):
    storage = Mock()
    storage.read_json.return_value = record
    monkeypatch.setattr(experiment_registry, "create_storage", lambda config: storage)

    with pytest.raises(ExperimentRegistryError, match="experiment record"):
        read_experiment_record("registry/record.json", {})


def test_rejects_expected_run_id_mismatch(monkeypatch):
    storage = Mock()
    storage.read_json.return_value = {"run_id": "actual"}
    monkeypatch.setattr(experiment_registry, "create_storage", lambda config: storage)

    with pytest.raises(ExperimentRegistryError, match="expected=expected, actual=actual"):
        read_experiment_record(
            "registry/record.json",
            {},
            expected_run_id="expected",
        )


def test_wraps_storage_error_with_public_error(monkeypatch):
    uri = "registry/record.json?credential=SENSITIVE_URI_VALUE"
    storage = Mock()
    storage.read_json.side_effect = ObjectNotFoundError(
        "token=SENSITIVE_STORAGE_VALUE"
    )
    monkeypatch.setattr(experiment_registry, "create_storage", lambda config: storage)

    with pytest.raises(ExperimentRegistryError, match="ObjectNotFoundError") as error:
        read_experiment_record(uri, {})

    message = str(error.value)
    assert uri not in message
    assert "SENSITIVE_URI_VALUE" not in message
    assert "SENSITIVE_STORAGE_VALUE" not in message
    assert isinstance(error.value.__cause__, ObjectNotFoundError)
