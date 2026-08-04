"""Registry pipeline 테스트.

이전 pipeline을 실행하지 않고, 계약 형식의 fixture만으로 성공과 실패를
모두 확인합니다. 작은 local 환경과 CPU에서 그대로 동작합니다.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.pipelines import registry


FIXTURE_DIR = Path(__file__).parent / "fixtures"
RETURN_KEYS = {"status", "artifacts", "summary", "message"}


def load_fixture(name: str) -> dict:
    """계약 형식의 artifact fixture를 읽습니다."""

    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def materialize_local_artifacts(inputs: dict, repo_root: Path) -> None:
    """fixture가 가리키는 local artifact file을 임시 저장소 안에 만듭니다."""

    for pipeline, artifacts in inputs.items():
        for key, uri in artifacts.items():
            if not key.endswith("_uri"):
                continue
            path = repo_root / uri
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"pipeline": pipeline, "artifact": key}, ensure_ascii=False),
                encoding="utf-8",
                newline="\n",
            )


def make_config(repo_root: Path, inputs: dict, **registry_options) -> dict:
    """저장소 밖을 건드리지 않는 local storage config를 만듭니다."""

    registry_config = {"repo_root": str(repo_root)}
    registry_config.update(registry_options)
    return {
        "project": {"name": "pill-object-detection"},
        "seed": 42,
        "storage": {
            "backend": "local",
            "local": {"root": str(repo_root / "artifacts")},
        },
        "registry": registry_config,
        "inputs": inputs,
    }


@pytest.fixture
def local_run(tmp_path: Path):
    """local artifact가 준비된 성공 시나리오 config를 돌려줍니다."""

    inputs = load_fixture("inputs_local.json")
    materialize_local_artifacts(inputs, tmp_path)
    return tmp_path, inputs


# --- 성공 경로 -------------------------------------------------------------


def test_dummy_run_passes_when_no_inputs():
    result = registry.run({})

    assert result["status"] == "ok"
    assert result["artifacts"] == {}
    assert result["summary"]["mode"] == "dummy"


def test_dummy_run_passes_when_previous_artifacts_are_empty():
    """main_pipeline의 dummy 실행처럼 빈 artifact만 오는 경우입니다."""

    result = registry.run({"inputs": {"data": {}, "train": {}, "evaluate": {}}})

    assert result["status"] == "ok"
    assert result["summary"]["mode"] == "dummy"


def test_run_returns_exact_contract_keys(local_run):
    repo_root, inputs = local_run

    result = registry.run(make_config(repo_root, inputs))

    assert set(result) == RETURN_KEYS
    assert result["status"] == "ok"
    assert set(result["artifacts"]) == {"run_id", "experiment_record_uri"}
    assert isinstance(result["artifacts"]["run_id"], str)
    assert isinstance(result["artifacts"]["experiment_record_uri"], str)


def test_creates_experiment_record_with_checksums(local_run):
    repo_root, inputs = local_run

    result = registry.run(make_config(repo_root, inputs))

    assert result["status"] == "ok"
    record_uri = result["artifacts"]["experiment_record_uri"]
    assert record_uri == "artifacts/registry/exp-0001/experiment_record.json"

    record = json.loads((repo_root / record_uri).read_text(encoding="utf-8"))
    assert record["schema_version"] == "1.0"
    assert record["run_id"] == "exp-0001"
    assert record["seed"] == 42
    assert set(record["pipelines"]) == {"data", "train", "evaluate"}

    metrics = record["pipelines"]["evaluate"]["metrics_uri"]
    expected = hashlib.sha256(
        (repo_root / inputs["evaluate"]["metrics_uri"]).read_bytes()
    ).hexdigest()
    assert metrics["sha256"] == expected
    assert metrics["location"] == "local"
    assert metrics["verified"] is True
    assert record["verification"]["artifacts_hashed"] == 9
    assert record["pipelines"]["train"]["run_id"] == "exp-0001"


def test_record_uri_is_repository_relative(local_run):
    repo_root, inputs = local_run

    result = registry.run(make_config(repo_root, inputs))

    assert not Path(result["artifacts"]["experiment_record_uri"]).is_absolute()


def test_inputs_are_not_modified(local_run):
    repo_root, inputs = local_run
    config = make_config(repo_root, inputs)
    before = copy.deepcopy(config["inputs"])

    registry.run(config)

    assert config["inputs"] == before


def test_remote_artifacts_are_recorded_without_aws_access(tmp_path: Path):
    inputs = load_fixture("inputs_s3.json")

    result = registry.run(make_config(tmp_path, inputs))

    assert result["status"] == "ok"
    record = json.loads(
        (tmp_path / result["artifacts"]["experiment_record_uri"]).read_text(
            encoding="utf-8"
        )
    )
    entry = record["pipelines"]["data"]["class_map_uri"]
    assert entry["location"] == "s3"
    assert entry["sha256"] is None
    assert entry["verified"] is False
    assert record["verification"]["artifacts_skipped_remote"] == 9


def test_secret_like_config_values_are_redacted(local_run):
    repo_root, inputs = local_run
    config = make_config(repo_root, inputs)
    # 실제 credential이 아니라 마스킹 동작만 확인하기 위한 가짜 값입니다.
    config["aws"] = {"access_key_id": "가짜-값", "region": "ap-northeast-2"}

    result = registry.run(config)

    record = json.loads(
        (repo_root / result["artifacts"]["experiment_record_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert record["config_snapshot"]["aws"]["access_key_id"] == "***"
    assert record["config_snapshot"]["aws"]["region"] == "ap-northeast-2"
    assert "inputs" not in record["config_snapshot"]


def test_configured_run_id_overrides_pipeline_run_id(local_run):
    repo_root, inputs = local_run

    result = registry.run(make_config(repo_root, inputs, run_id="manual-run"))

    assert result["artifacts"]["run_id"] == "manual-run"


def test_overwrite_flag_allows_rewriting_existing_record(local_run):
    repo_root, inputs = local_run

    first = registry.run(make_config(repo_root, inputs))
    second = registry.run(make_config(repo_root, inputs, overwrite=True))

    assert first["status"] == "ok"
    assert second["status"] == "ok"


# --- 실패 경로 -------------------------------------------------------------


def test_missing_pipeline_artifact_returns_error(local_run):
    repo_root, inputs = local_run
    broken = copy.deepcopy(inputs)
    del broken["evaluate"]

    result = registry.run(make_config(repo_root, broken))

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "evaluate" in result["message"]


def test_missing_artifact_key_returns_error(local_run):
    repo_root, inputs = local_run
    broken = copy.deepcopy(inputs)
    del broken["train"]["training_history_uri"]

    result = registry.run(make_config(repo_root, broken))

    assert result["status"] == "error"
    assert "training_history_uri" in result["message"]


def test_wrong_artifact_value_type_returns_error(local_run):
    repo_root, inputs = local_run
    broken = copy.deepcopy(inputs)
    broken["data"]["class_map_uri"] = 123

    result = registry.run(make_config(repo_root, broken))

    assert result["status"] == "error"
    assert "class_map_uri" in result["message"]


def test_missing_local_artifact_file_returns_error(local_run):
    repo_root, inputs = local_run
    (repo_root / inputs["evaluate"]["predictions_uri"]).unlink()

    result = registry.run(make_config(repo_root, inputs))

    assert result["status"] == "error"
    assert "predictions" in result["message"]
    assert result["artifacts"] == {}


def test_run_id_mismatch_returns_error(local_run):
    repo_root, inputs = local_run
    broken = copy.deepcopy(inputs)
    broken["evaluate"]["run_id"] = "exp-9999"

    result = registry.run(make_config(repo_root, broken))

    assert result["status"] == "error"
    assert "run_id" in result["message"]


def test_absolute_local_artifact_uri_returns_error(local_run):
    repo_root, inputs = local_run
    broken = copy.deepcopy(inputs)
    broken["data"]["class_map_uri"] = str(repo_root / "artifacts/data/class_map.json")

    result = registry.run(make_config(repo_root, broken))

    assert result["status"] == "error"
    assert "상대 경로" in result["message"]


def test_artifact_outside_repository_returns_error(local_run):
    repo_root, inputs = local_run
    broken = copy.deepcopy(inputs)
    broken["data"]["class_map_uri"] = "../outside/class_map.json"

    result = registry.run(make_config(repo_root, broken))

    assert result["status"] == "error"
    assert "저장소 밖" in result["message"]


@pytest.mark.parametrize(
    "uri",
    [
        "/etc/passwd",
        "C:/Windows/model.pt",
        "C:artifacts/model.pt",
        "../outside/class_map.json",
        "artifacts/../../outside/class_map.json",
        "https://example.com/class_map.json",
    ],
)
def test_bad_local_uri_is_rejected_even_when_verification_is_off(local_run, uri):
    """verify_artifacts=false로 URI 계약을 우회할 수 없어야 합니다."""

    repo_root, inputs = local_run
    broken = copy.deepcopy(inputs)
    broken["data"]["class_map_uri"] = uri

    result = registry.run(make_config(repo_root, broken, verify_artifacts=False))

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert not list((repo_root / "artifacts" / "registry").rglob("*.json"))


def test_verification_off_skips_only_existence_and_hash(local_run):
    """존재 확인과 해시만 생략하고, 나머지 동작은 그대로여야 합니다."""

    repo_root, inputs = local_run
    (repo_root / inputs["evaluate"]["predictions_uri"]).unlink()

    result = registry.run(make_config(repo_root, inputs, verify_artifacts=False))

    assert result["status"] == "ok"
    assert result["summary"]["artifacts_hashed"] == 0
    assert result["summary"]["artifacts_checked"] == 9

    record = json.loads(
        (repo_root / result["artifacts"]["experiment_record_uri"]).read_text(
            encoding="utf-8"
        )
    )
    entry = record["pipelines"]["evaluate"]["predictions_uri"]
    assert entry["location"] == "local"
    assert entry["sha256"] is None
    assert entry["verified"] is False
    assert "URI schema는 검증했습니다" in entry["note"]


def test_existing_record_is_not_overwritten_by_default(local_run):
    repo_root, inputs = local_run

    first = registry.run(make_config(repo_root, inputs))
    original = (repo_root / first["artifacts"]["experiment_record_uri"]).read_text(
        encoding="utf-8"
    )
    second = registry.run(make_config(repo_root, inputs))

    assert second["status"] == "error"
    assert second["artifacts"] == {}
    assert (repo_root / first["artifacts"]["experiment_record_uri"]).read_text(
        encoding="utf-8"
    ) == original


def test_invalid_registry_config_type_returns_error():
    result = registry.run({"registry": [], "inputs": load_fixture("inputs_s3.json")})

    assert result["status"] == "error"
    assert "object여야 합니다" in result["message"]


def test_invalid_seed_returns_error(local_run):
    repo_root, inputs = local_run
    config = make_config(repo_root, inputs, seed="빠른값")

    result = registry.run(config)

    assert result["status"] == "error"
    assert "seed" in result["message"]
