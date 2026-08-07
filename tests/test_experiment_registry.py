from unittest.mock import Mock

import pytest

import src.common.experiment_registry as experiment_registry
from src.common import (
    ExperimentRegistryError,
    compare_experiment_summaries,
    list_experiment_summaries,
    read_experiment_record,
    search_experiment_summaries,
)
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
    assert record["schema_version"] == "1.2"


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


# --- 목록·검색·비교 -------------------------------------------------------
#
# Exact-URI 조회와 달리 index prefix만 읽는 별개 경로입니다.


def local_config(tmp_path, **registry_options) -> dict:
    registry_config = {
        "repo_root": str(tmp_path),
        "verify_artifacts": False,
    }
    registry_config.update(registry_options)
    return {
        "storage": {"backend": "local", "local": {"root": str(tmp_path / "artifacts")}},
        "registry": registry_config,
        "inputs": registry_inputs(),
    }


def register(tmp_path, run_id: str, created_at: str) -> dict:
    """실험 하나를 실제 registry로 등록해 index까지 남깁니다."""

    config = local_config(tmp_path, run_id=run_id, created_at=created_at)
    result = registry.run(config)
    assert result["status"] == "ok"
    return config


def test_lists_registered_experiments_newest_first(tmp_path):
    register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    config = register(tmp_path, "exp-b", "2026-08-05T00:00:00+00:00")

    summaries = list_experiment_summaries(config)

    assert [summary["run_id"] for summary in summaries] == ["exp-b", "exp-a"]
    assert summaries[0]["summary_version"] == "2"


def test_search_filters_by_run_id_and_submission(tmp_path):
    register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    config = register(tmp_path, "other-b", "2026-08-05T00:00:00+00:00")

    assert [
        summary["run_id"]
        for summary in search_experiment_summaries(config, run_id_contains="exp-")
    ] == ["exp-a"]
    # 이 fixture에는 submission artifact가 없습니다.
    assert search_experiment_summaries(config, has_submission=True) == []
    assert len(search_experiment_summaries(config, has_submission=False)) == 2


def test_compare_reports_differing_fields_and_missing_runs(tmp_path):
    register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    config = register(tmp_path, "exp-b", "2026-08-05T00:00:00+00:00")

    comparison = compare_experiment_summaries(["exp-a", "exp-b", "없는-실험"], config)

    assert comparison["run_ids"] == ["exp-a", "exp-b"]
    assert comparison["missing"] == ["없는-실험"]
    assert comparison["fields"]["created_at"]["differs"] is True
    assert comparison["fields"]["schema_version"]["differs"] is False


def test_one_unreadable_index_entry_does_not_break_the_listing(tmp_path):
    register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    config = register(tmp_path, "exp-b", "2026-08-05T00:00:00+00:00")
    (tmp_path / "artifacts/registry/index/exp-a.json").write_text(
        "{망가진 JSON", encoding="utf-8", newline="\n"
    )

    summaries = list_experiment_summaries(config)

    assert [summary["run_id"] for summary in summaries] == ["exp-b"]


def test_listing_does_not_read_records(tmp_path):
    """목록은 index만 봅니다. record를 훑어 fallback하지 않습니다."""

    config = register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    (tmp_path / "artifacts/registry/index/exp-a.json").unlink()

    assert list_experiment_summaries(config) == []


def test_exact_uri_read_still_refuses_to_search(tmp_path):
    """계약 회귀: exact-URI 조회는 목록 기능이 생겨도 listing을 하지 않습니다."""

    config = register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")

    with pytest.raises(ExperimentRegistryError):
        read_experiment_record("artifacts/registry/없는-실험/experiment_record.json", config)


def test_index_prefix_escaping_the_storage_root_is_rejected(tmp_path):
    """안전 장치: 저장소 root 밖은 어떤 설정으로도 읽지 않습니다."""

    config = local_config(tmp_path, index_prefix="../../바깥")

    with pytest.raises(ExperimentRegistryError) as error:
        list_experiment_summaries(config)

    assert "바깥" not in str(error.value)
