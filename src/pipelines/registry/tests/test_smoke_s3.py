"""Registry S3 smoke test 명령의 테스트.

실제 AWS에 접속하지 않고, in-memory로 대체한 S3 storage로 명령의 흐름만
검증합니다. 실제 bucket 왕복은 credential이 있는 환경에서
`python -m src.pipelines.registry.smoke_s3 --config <aws config>`로 직접
실행해야 합니다.
"""

from __future__ import annotations

import copy
import json
from unittest.mock import Mock, patch

import pytest

from src.common import LocalStorage, StorageError
from src.common.storage import S3Storage
from src.pipelines import registry
from src.pipelines.registry import smoke_s3
from src.pipelines.registry.record import REQUIRED_ARTIFACT_KEYS


BUCKET = "test-bucket"


def make_fake_s3_storage() -> tuple[S3Storage, dict]:
    """실제 network 호출 없이 동작하는 S3Storage 대역을 만듭니다."""

    storage = S3Storage(BUCKET, client=Mock())
    objects: dict[str, object] = {}

    def write_json(destination, value, *, overwrite=False):
        uri = f"s3://{BUCKET}/{destination}"
        if uri in objects and not overwrite:
            raise StorageError(f"S3 object가 이미 있습니다: {uri}")
        objects[uri] = copy.deepcopy(value)
        return uri

    def read_json(source):
        if source not in objects:
            raise StorageError(f"S3 object가 없습니다: {source}")
        return copy.deepcopy(objects[source])

    storage.write_json = Mock(side_effect=write_json)
    storage.read_json = Mock(side_effect=read_json)
    storage.exists = Mock(side_effect=lambda uri: uri in objects)
    storage.list = Mock(
        side_effect=lambda prefix="": sorted(
            uri for uri in objects if uri.startswith(f"s3://{BUCKET}/{prefix}")
        )
    )
    return storage, objects


def run_with_fake_storage(storage, config=None, **kwargs) -> dict:
    """smoke_s3와 registry가 같은 대역 storage를 쓰도록 묶어서 실행합니다."""

    with (
        patch.object(smoke_s3, "load_config", return_value=config or {}),
        patch.object(smoke_s3, "create_storage", return_value=storage),
        patch.object(registry, "create_storage", return_value=storage),
    ):
        return smoke_s3.run_smoke_test("configs/env.aws.json", **kwargs)


# --- 성공 경로 -------------------------------------------------------------


def test_smoke_test_registers_and_reads_back_record():
    storage, objects = make_fake_s3_storage()

    result = run_with_fake_storage(storage, smoke_id="fixed-id")

    assert result["status"] == "ok"
    assert result["run_id"] == "smoke-fixed-id"
    expected_uri = (
        f"s3://{BUCKET}/registry/smoke-tests/fixed-id/experiment_record.json"
    )
    assert result["experiment_record_uri"] == expected_uri
    # experiment record 하나와 index sidecar 하나가 전용 prefix 아래에 생깁니다.
    assert result["listed_count"] == 2
    assert result["artifacts_skipped_remote"] == 9

    stored = objects[expected_uri]
    assert stored["run_id"] == "smoke-fixed-id"
    assert stored["schema_version"] == "1.3"
    storage.read_json.assert_called_once_with(expected_uri)
    storage.list.assert_called_once_with(smoke_s3.SMOKE_PREFIX)


def test_smoke_test_uses_dedicated_prefix_and_does_not_delete():
    storage, objects = make_fake_s3_storage()

    result = run_with_fake_storage(storage, smoke_id="fixed-id")

    assert smoke_s3.SMOKE_PREFIX == "registry/smoke-tests/"
    assert result["experiment_record_uri"].startswith(
        f"s3://{BUCKET}/{smoke_s3.SMOKE_PREFIX}"
    )
    # 저장소 공통 정책: smoke test object를 자동 삭제하지 않습니다.
    assert not hasattr(storage, "delete")
    storage.client.delete_object.assert_not_called()
    assert "자동 삭제하지 않았습니다" in result["cleanup"]
    # smoke test는 지우지 않으므로, index를 포함해 만든 object가 모두 전용 prefix
    # 안에 있어야 실제 실험 목록이 오염되지 않습니다.
    assert objects
    for uri in objects:
        assert uri.startswith(f"s3://{BUCKET}/{smoke_s3.SMOKE_PREFIX}")


def test_smoke_inputs_follow_the_artifact_contract():
    assert set(smoke_s3.SMOKE_INPUTS) == set(REQUIRED_ARTIFACT_KEYS)
    for pipeline, required_keys in REQUIRED_ARTIFACT_KEYS.items():
        assert set(smoke_s3.SMOKE_INPUTS[pipeline]) == set(required_keys)
        for value in smoke_s3.SMOKE_INPUTS[pipeline].values():
            assert isinstance(value, str) and value.strip()


def test_build_smoke_config_does_not_mutate_original_config():
    config = {"storage": {"backend": "s3", "s3": {"bucket": BUCKET}}}
    before = copy.deepcopy(config)

    smoke_config = smoke_s3.build_smoke_config(
        config,
        run_id="smoke-1",
        record_uri="registry/smoke-tests/1/record.json",
        index_prefix="registry/smoke-tests/1/index",
    )

    assert config == before
    assert smoke_config["registry"]["verify_artifacts"] is False
    assert smoke_config["inputs"] == smoke_s3.SMOKE_INPUTS


def test_smoke_config_is_json_serializable():
    """CLI가 결과를 JSON으로 출력할 수 있어야 합니다."""

    storage, _ = make_fake_s3_storage()

    result = run_with_fake_storage(storage, smoke_id="fixed-id")

    assert json.loads(json.dumps(result, ensure_ascii=False)) == result


# --- 실패 경로 -------------------------------------------------------------


def test_smoke_test_requires_s3_backend(tmp_path):
    local_storage = LocalStorage(tmp_path)

    with (
        patch.object(smoke_s3, "load_config", return_value={}),
        patch.object(smoke_s3, "create_storage", return_value=local_storage),
    ):
        with pytest.raises(StorageError, match="s3 storage backend"):
            smoke_s3.run_smoke_test("configs/env.local.json")


def test_smoke_test_detects_record_mismatch():
    storage, _ = make_fake_s3_storage()
    storage.read_json = Mock(return_value={"run_id": "다른-run", "schema_version": "1.2"})

    with pytest.raises(StorageError, match="run_id가 다릅니다"):
        run_with_fake_storage(storage, smoke_id="fixed-id")


def test_smoke_test_detects_missing_object_in_listing():
    storage, _ = make_fake_s3_storage()
    storage.list = Mock(return_value=[])

    with pytest.raises(StorageError, match="prefix listing에 없습니다"):
        run_with_fake_storage(storage, smoke_id="fixed-id")


def test_smoke_test_reports_registry_failure():
    storage, _ = make_fake_s3_storage()
    storage.write_json = Mock(side_effect=StorageError("이미 있습니다"))

    with pytest.raises(smoke_s3.RegistryError, match="registry pipeline이 실패했습니다"):
        run_with_fake_storage(storage, smoke_id="fixed-id")


def test_cli_returns_error_exit_code_without_aws(capsys):
    """config 파일이 없으면 stack trace 대신 종료 코드 1로 끝납니다."""

    exit_code = smoke_s3.main(["--config", "존재하지-않는-config.json"])

    assert exit_code == 1
    assert "registry S3 smoke test 실패" in capsys.readouterr().err
