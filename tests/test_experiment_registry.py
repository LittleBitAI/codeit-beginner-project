import json
from unittest.mock import Mock

import pytest

import src.common.experiment_registry as experiment_registry
from src.common import (
    ExperimentRegistryError,
    compare_experiment_summaries,
    list_experiment_summaries,
    read_experiment_record,
    read_experiment_summary,
    search_experiment_summaries,
)
from src.common.storage import LocalStorage, ObjectNotFoundError
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


def test_reads_one_summary_by_name_without_listing_the_index(tmp_path, monkeypatch):
    """이름을 아는 조회는 index 전체를 훑지 않습니다.

    Registry는 index를 ``<prefix>/<run_id>.json`` 한 파일로 남기므로, 이름을 이미
    아는 조회에 목록이 필요하지 않습니다. 목록은 등록된 실험 수만큼 storage를
    왕복하기 때문에, 상세·비교 화면이 그 길을 쓰면 기록이 쌓일수록 함께 느려집니다.
    """

    register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    config = register(tmp_path, "exp-b", "2026-08-05T00:00:00+00:00")

    def refuse(self, prefix=""):
        raise AssertionError("이름을 아는 조회는 목록을 읽으면 안 됩니다.")

    monkeypatch.setattr(LocalStorage, "list", refuse)

    assert read_experiment_summary("exp-a", config)["run_id"] == "exp-a"


def test_summary_that_was_never_registered_is_absent_not_an_error(tmp_path):
    """없는 것과 못 읽은 것을 구분합니다. 없으면 ``None``입니다."""

    config = register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")

    assert read_experiment_summary("없는-실험", config) is None


def test_broken_index_entry_is_an_error_not_an_absent_experiment(tmp_path):
    """읽지 못한 것을 "없다"로 답하면 화면이 멀쩡한 실험을 사라졌다고 말합니다.

    목록은 항목 하나가 깨지면 그것만 건너뛰지만(실험 하나 때문에 화면 전체가 비면
    더 곤란합니다), 이름을 대고 그 하나를 물었을 때는 답이 달라야 합니다.
    """

    config = register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    (tmp_path / "artifacts/registry/index/exp-a.json").write_text(
        "{망가진 JSON", encoding="utf-8", newline="\n"
    )

    with pytest.raises(ExperimentRegistryError):
        read_experiment_summary("exp-a", config)


def test_summary_of_an_index_entry_that_names_another_experiment_is_an_error(tmp_path):
    """index 파일이 다른 실험을 가리키면 그 실험을 대신 돌려주면 안 됩니다.

    index는 이름으로 찾는 파일이라, 내용이 어긋나면 화면이 A를 물었는데 B의 결과를
    A라고 보여 주게 됩니다. 값이 아니라 이름이 어긋난 것이므로 조용히 넘어가면
    사람은 끝까지 알아챌 수 없습니다.
    """

    config = register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    path = tmp_path / "artifacts/registry/index/exp-a.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["run_id"] = "exp-b"
    path.write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ExperimentRegistryError):
        read_experiment_summary("exp-a", config)


@pytest.mark.parametrize("run_id", ["\x00", "줄\n바꿈"])
def test_summary_read_refuses_a_name_the_file_system_cannot_take(
    tmp_path, monkeypatch, run_id
):
    """제어문자가 든 이름은 storage에 넘기기 전에 거부합니다.

    넘기면 backend가 아니라 그 아래 OS가 죽습니다. NUL이 든 이름은 pathlib이
    ValueError를 던지는데, 그것은 이 module의 오류가 아니라서 호출자(web GUI)가
    404 대신 500으로 흘려보냅니다. 목록을 훑던 예전 경로는 같은 이름에 "없음"으로
    답했으므로, 막지 않으면 이 함수가 그 자리에 회귀를 들여놓는 셈입니다.
    """

    config = register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")

    def refuse(_config):
        raise AssertionError("이런 이름은 storage까지 가면 안 됩니다.")

    monkeypatch.setattr(experiment_registry, "create_storage", refuse)

    with pytest.raises(ExperimentRegistryError):
        read_experiment_summary(run_id, config)


def test_summary_reads_back_every_name_the_registry_can_write(tmp_path):
    """registry가 저장할 수 있는 이름이면 이 조회도 읽어 내야 합니다.

    registry는 설정으로 지정한 run_id를 검증 없이 그대로 씁니다. ``폴더/이름``은
    ``registry/index/폴더/이름.json``에 저장되고 목록 경로는 그것을 찾아냅니다.
    읽는 쪽만 경로 구분자를 거부하면 등록은 됐는데 조회는 안 되는 실험이 생깁니다.
    """

    config = register(tmp_path, "폴더/이름", "2026-08-01T00:00:00+00:00")

    assert read_experiment_summary("폴더/이름", config)["run_id"] == "폴더/이름"


@pytest.mark.parametrize("run_id", ["../바깥", "../../../../바깥"])
def test_summary_read_refuses_a_name_that_points_outside_the_index(tmp_path, run_id):
    """안전 장치: index 밖을 가리키는 이름은 읽지 않고, 그 이름을 되풀이하지도 않습니다.

    storage backend에 기대면 부족합니다. LocalStorage가 지키는 것은 storage **root**
    라서, `..` 한 단계는 index prefix를 벗어나면서도 root 안에 남습니다. 목록은 그런
    파일을 세지 않으므로(prefix로 거릅니다) 읽는 쪽만 읽어 주면 두 경로가 서로 다른
    실험 집합을 보게 됩니다. 아래에 심어 둔 미끼는 run_id까지 맞아서, 막지 않으면
    마지막 대조도 통과해 그대로 반환됩니다.
    """

    config = register(tmp_path, "exp-a", "2026-08-01T00:00:00+00:00")
    bait = tmp_path / "artifacts/registry/바깥.json"
    bait.write_text(
        json.dumps({"run_id": "../바깥", "summary_version": "2"}, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ExperimentRegistryError) as error:
        read_experiment_summary(run_id, config)

    assert "바깥" not in str(error.value)
