"""Experiment summary 발행과 index sidecar 테스트.

실행이 끝난 실험을 재시작 뒤에도 찾을 수 있도록, registry가 run별 index 파일을
남기는지 확인합니다. index는 record에서 다시 만들 수 있는 cache이므로 index 저장이
실패해도 실행 자체는 성공해야 합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import StorageError
from src.pipelines import registry
from src.pipelines.registry import rebuild_index as rebuild_module


FIXTURE_DIR = Path(__file__).parent / "fixtures"

VALID_SUBMISSION_CSV = (
    "annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score\n"
    "1,10,3,1.0,2.0,3.0,4.0,0.9\n"
)

METRICS_DOCUMENT = {
    "run_id": "exp-0001",
    "metrics": {
        "mAP": 0.31,
        "mAP50": 0.55,
        "mAP75": 0.33,
        "precision50": 0.61,
        "recall50": 0.48,
    },
}


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def materialize(inputs: dict, repo_root: Path) -> None:
    """계약 형식의 local artifact file을 임시 저장소 안에 만듭니다."""

    for pipeline, artifacts in inputs.items():
        for key, uri in artifacts.items():
            if not key.endswith("_uri"):
                continue
            path = repo_root / uri
            path.parent.mkdir(parents=True, exist_ok=True)
            if key == "submission_uri":
                content = VALID_SUBMISSION_CSV
            elif key == "metrics_uri":
                content = json.dumps(METRICS_DOCUMENT, ensure_ascii=False)
            else:
                content = json.dumps(
                    {"pipeline": pipeline, "artifact": key}, ensure_ascii=False
                )
            path.write_text(content, encoding="utf-8", newline="\n")


def make_config(repo_root: Path, inputs: dict, **registry_options) -> dict:
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
    inputs = load_fixture("inputs_local.json")
    materialize(inputs, tmp_path)
    return tmp_path, inputs


# --- summary 발행 ----------------------------------------------------------


def test_run_writes_an_index_entry_next_to_the_record(local_run):
    repo_root, inputs = local_run

    result = registry.run(make_config(repo_root, inputs))

    assert result["status"] == "ok"
    assert result["summary"]["index_status"] == "written"

    summary_uri = result["artifacts"]["experiment_summary_uri"]
    assert summary_uri == "artifacts/registry/index/exp-0001.json"

    summary = json.loads((repo_root / summary_uri).read_text(encoding="utf-8"))
    assert summary["summary_version"] == "1"
    assert summary["run_id"] == "exp-0001"
    assert summary["seed"] == 42
    assert summary["schema_version"] == "1.2"
    assert summary["experiment_record_uri"] == (
        result["artifacts"]["experiment_record_uri"]
    )
    assert summary["metrics_source"] == "metrics_file"
    assert summary["metrics"]["mAP"] == pytest.approx(0.31)
    assert summary["metrics"]["mAP50"] == pytest.approx(0.55)
    assert summary["submission_check"]["checked"] is False

    # 선언된 artifact key는 없더라도 null로 자리를 채워 소비자가 분기하지 않게 합니다.
    assert summary["artifacts"]["metrics_uri"] == inputs["evaluate"]["metrics_uri"]
    assert summary["artifacts"]["submission_uri"] is None
    assert summary["artifacts"]["test_manifest_uri"] is None


def test_unreadable_metrics_file_does_not_fail_the_run(tmp_path: Path):
    inputs = load_fixture("inputs_local.json")
    materialize(inputs, tmp_path)
    (tmp_path / inputs["evaluate"]["metrics_uri"]).write_text(
        "{망가진 JSON", encoding="utf-8", newline="\n"
    )

    result = registry.run(make_config(tmp_path, inputs))

    assert result["status"] == "ok"
    summary = json.loads(
        (tmp_path / result["artifacts"]["experiment_summary_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["metrics_source"] == "unavailable"
    assert summary["metrics"]["mAP"] is None


def test_remote_metrics_are_not_fetched(tmp_path: Path):
    """원격 artifact는 AWS 접근 없이 참조만 기록한다는 정책을 지킵니다."""

    inputs = load_fixture("inputs_s3.json")

    result = registry.run(make_config(tmp_path, inputs))

    summary = json.loads(
        (tmp_path / result["artifacts"]["experiment_summary_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["metrics_source"] == "unavailable"


def read_summary(repo_root: Path, result: dict) -> dict:
    """실행 결과가 가리키는 index summary 문서를 읽습니다."""

    return json.loads(
        (repo_root / result["artifacts"]["experiment_summary_uri"]).read_text(
            encoding="utf-8"
        )
    )


def test_training_block_is_filled_from_the_config_snapshot(local_run):
    repo_root, inputs = local_run
    config = make_config(repo_root, inputs)
    config["train"] = {
        "architecture": "retinanet_resnet50_fpn_v2",
        "pretrained": True,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "device": "cuda",
        "epochs": 50,
        "batch_size": 4,
        "num_workers": 0,
        # 경로와 seed는 summary에 옮기지 않습니다.
        "output_dir": "artifacts/train",
        "seed": 7,
    }

    summary = read_summary(repo_root, registry.run(config))

    assert summary["summary_version"] == "1"
    assert summary["training_source"] == "config_snapshot"
    assert summary["training"] == {
        "architecture": "retinanet_resnet50_fpn_v2",
        "pretrained": True,
        "optimizer": "AdamW",
        "learning_rate": pytest.approx(0.0001),
        # optimizer 종류에 따라 record에 한쪽만 있으므로 없는 쪽은 null입니다.
        "momentum": None,
        "weight_decay": pytest.approx(0.01),
        "beta1": pytest.approx(0.9),
        "beta2": pytest.approx(0.999),
        "epsilon": pytest.approx(1e-8),
        "device": "cuda",
        "epochs": 50,
        "batch_size": 4,
        "num_workers": 0,
    }


def test_record_without_a_train_section_stays_successful(local_run):
    """train 설정이 없던 옛 record도 실패시키지 않고 null로 둡니다."""

    repo_root, inputs = local_run

    result = registry.run(make_config(repo_root, inputs))

    assert result["status"] == "ok"
    summary = read_summary(repo_root, result)
    assert summary["training_source"] == "unavailable"
    assert set(summary["training"].values()) == {None}


def test_wrongly_typed_training_values_become_null(local_run):
    """기본값으로 채우지 않고 null로 둡니다. 기록에 없는 것과 기본값은 다릅니다."""

    repo_root, inputs = local_run
    config = make_config(repo_root, inputs)
    config["train"] = {
        "architecture": 123,
        "pretrained": "true",
        "epochs": "50",
        "batch_size": True,
        "learning_rate": "0.0001",
        "device": "cuda",
    }

    result = registry.run(config)

    assert result["status"] == "ok"
    summary = read_summary(repo_root, result)
    # train 섹션 자체는 있으므로 출처는 config_snapshot입니다.
    assert summary["training_source"] == "config_snapshot"
    assert summary["training"]["device"] == "cuda"
    for key in ("architecture", "pretrained", "epochs", "batch_size", "learning_rate"):
        assert summary["training"][key] is None


# --- index는 cache, record가 진실 -----------------------------------------


def test_index_write_failure_keeps_the_run_successful(local_run, monkeypatch):
    repo_root, inputs = local_run
    real_create_storage = registry.create_storage

    def create_failing_storage(config):
        storage = real_create_storage(config)
        original_write_json = storage.write_json

        def write_json(destination, value, *, overwrite=False):
            if "index" in str(destination):
                raise StorageError("index 저장 실패를 흉내 냅니다.")
            return original_write_json(destination, value, overwrite=overwrite)

        storage.write_json = write_json
        return storage

    monkeypatch.setattr(registry, "create_storage", create_failing_storage)

    result = registry.run(make_config(repo_root, inputs))

    assert result["status"] == "ok"
    assert result["summary"]["index_status"] == "failed"
    assert "experiment_summary_uri" not in result["artifacts"]
    # record는 진실이므로 그대로 남아 있어야 합니다.
    record_path = repo_root / result["artifacts"]["experiment_record_uri"]
    assert record_path.is_file()


def test_existing_index_is_not_overwritten_by_default(local_run):
    repo_root, inputs = local_run
    registry.run(make_config(repo_root, inputs))
    index_path = repo_root / "artifacts/registry/index/exp-0001.json"
    before = index_path.read_bytes()

    second = registry.run(make_config(repo_root, inputs))

    assert second["status"] == "error"
    assert index_path.read_bytes() == before


# --- rebuild CLI -----------------------------------------------------------


def test_rebuild_creates_only_missing_index_entries(local_run):
    repo_root, inputs = local_run
    config = make_config(repo_root, inputs)
    registry.run(config)

    index_path = repo_root / "artifacts/registry/index/exp-0001.json"
    index_path.unlink()

    report = rebuild_module.rebuild_index(config)

    assert report["written"] == 1
    assert report["skipped"] == 0
    assert index_path.is_file()

    # 이미 있는 index는 다시 쓰지 않습니다.
    again = rebuild_module.rebuild_index(config)
    assert again["written"] == 0
    assert again["skipped"] == 1


def test_rebuild_cli_reports_success(local_run, tmp_path: Path, capsys):
    repo_root, inputs = local_run
    config = make_config(repo_root, inputs)
    registry.run(config)
    (repo_root / "artifacts/registry/index/exp-0001.json").unlink()

    config_path = tmp_path / "rebuild_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    exit_code = rebuild_module.main(["--config", str(config_path)])

    assert exit_code == 0
    assert (repo_root / "artifacts/registry/index/exp-0001.json").is_file()
