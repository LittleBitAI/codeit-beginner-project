"""Experiment summary 발행과 index sidecar 테스트.

실행이 끝난 실험을 재시작 뒤에도 찾을 수 있도록, registry가 run별 index 파일을
남기는지 확인합니다. index는 record에서 다시 만들 수 있는 cache이므로 index 저장이
실패해도 실행 자체는 성공해야 합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import LocalStorage, ObjectNotFoundError, StorageError, train_contract
from src.pipelines import registry
from src.pipelines.registry import rebuild_index as rebuild_module
from src.pipelines.registry import summary as summary_module


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
    # Evaluate가 어디가 약한 class인지 이미 간추려 둔 자리입니다. registry는 세지
    # 않고 그대로 옮기기만 합니다.
    "analysis": {
        "per_class_summary": {
            "min_truth_count": 5,
            "top_n": 10,
            "counts": {"weak": 1, "sparse": 0, "unmeasured": 0},
            "weak": [{"category_id": 16548, "name": "가바토파정 100mg", "ap": 0.12}],
            "sparse": [],
        }
    },
}

# Train이 training_history.json에 쓰는 모양 그대로입니다. epoch 2가 validation
# loss가 가장 낮고, 마지막 epoch은 3입니다.
TRAINING_HISTORY_DOCUMENT = [
    {"epoch": 1, "train_loss": 0.90, "validation_loss": 0.80},
    {"epoch": 2, "train_loss": 0.50, "validation_loss": 0.40},
    {"epoch": 3, "train_loss": 0.30, "validation_loss": 0.60},
]


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def artifact_content(pipeline: str, key: str) -> str:
    """계약 형식의 artifact file 내용을 만듭니다."""

    if key == "submission_uri":
        return VALID_SUBMISSION_CSV
    if key == "metrics_uri":
        return json.dumps(METRICS_DOCUMENT, ensure_ascii=False)
    if key == "training_history_uri":
        return json.dumps(TRAINING_HISTORY_DOCUMENT, ensure_ascii=False)
    return json.dumps({"pipeline": pipeline, "artifact": key}, ensure_ascii=False)


def materialize(inputs: dict, repo_root: Path) -> None:
    """계약 형식의 local artifact file을 임시 저장소 안에 만듭니다."""

    for pipeline, artifacts in inputs.items():
        for key, uri in artifacts.items():
            if not key.endswith("_uri"):
                continue
            path = repo_root / uri
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                artifact_content(pipeline, key), encoding="utf-8", newline="\n"
            )


class FakeS3Storage:
    """실제 AWS를 부르지 않고 s3:// 문서를 돌려주는 가짜 storage입니다.

    읽기만 흉내 내고 쓰기는 local storage에 맡깁니다. registry가 boto3를 직접
    쓰지 않고 `src/common/storage.py`의 interface만 쓰는지 확인하는 용도입니다.
    """

    def __init__(self, documents: dict, local: LocalStorage) -> None:
        self.documents = documents
        self._local = local
        self.read_locations: list[str] = []

    def read_json(self, source):
        location = str(source)
        self.read_locations.append(location)
        if location not in self.documents:
            raise ObjectNotFoundError(f"S3 object가 없습니다: {location}")
        value = self.documents[location]
        if isinstance(value, StorageError):
            raise value
        return value

    def write_json(self, destination, value, *, overwrite=False):
        return self._local.write_json(destination, value, overwrite=overwrite)

    def exists(self, location) -> bool:
        return self._local.exists(location)

    def list(self, prefix=""):
        return self._local.list(prefix)


def use_fake_s3(monkeypatch, documents: dict, repo_root: Path) -> FakeS3Storage:
    """registry가 만드는 storage를 가짜 S3 storage로 바꿉니다."""

    storage = FakeS3Storage(documents, LocalStorage(repo_root / "artifacts"))
    monkeypatch.setattr(registry, "create_storage", lambda config: storage)
    return storage


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
    assert summary["summary_version"] == "4"
    assert summary["run_id"] == "exp-0001"
    assert summary["seed"] == 42
    assert summary["schema_version"] == "1.3"
    assert summary["experiment_record_uri"] == (
        result["artifacts"]["experiment_record_uri"]
    )
    assert summary["metrics_source"] == "metrics_file"
    assert summary["metrics"]["mAP"] == pytest.approx(0.31)
    assert summary["metrics"]["mAP50"] == pytest.approx(0.55)
    # evaluate가 간추린 약한 class를 그대로 옮깁니다. 여기서 다시 세면 화면과
    # evaluate의 판정이 갈립니다.
    assert summary["per_class_summary"]["counts"]["weak"] == 1
    assert summary["per_class_summary"]["weak"][0]["category_id"] == 16548
    assert summary["submission_check"]["checked"] is False

    # 선언된 artifact key는 없더라도 null로 자리를 채워 소비자가 분기하지 않게 합니다.
    assert summary["artifacts"]["metrics_uri"] == inputs["evaluate"]["metrics_uri"]
    assert summary["artifacts"]["submission_uri"] is None
    assert summary["artifacts"]["test_manifest_uri"] is None
    assert summary["artifacts"]["test_predictions_uri"] is None


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


def read_summary(repo_root: Path, result: dict) -> dict:
    """실행 결과가 가리키는 index summary 문서를 읽습니다."""

    return json.loads(
        (repo_root / result["artifacts"]["experiment_summary_uri"]).read_text(
            encoding="utf-8"
        )
    )


# --- 지표와 loss 읽기 ------------------------------------------------------


def test_local_losses_are_read_from_the_training_history(local_run):
    """로컬 경로는 지금까지처럼 저장소 상대 경로로 직접 읽습니다."""

    repo_root, inputs = local_run

    summary = read_summary(repo_root, registry.run(make_config(repo_root, inputs)))

    assert summary["losses_source"] == "training_history"
    assert summary["losses"] == {
        "best_epoch": 2,
        "best_validation_loss": pytest.approx(0.40),
        "final_train_loss": pytest.approx(0.30),
        "final_validation_loss": pytest.approx(0.60),
    }


def test_remote_metrics_and_losses_are_read_through_storage(tmp_path: Path, monkeypatch):
    """팀이 전원 S3를 쓰므로 s3:// artifact도 storage를 거쳐 읽습니다."""

    inputs = load_fixture("inputs_s3.json")
    metrics_uri = inputs["evaluate"]["metrics_uri"]
    history_uri = inputs["train"]["training_history_uri"]
    storage = use_fake_s3(
        monkeypatch,
        {metrics_uri: METRICS_DOCUMENT, history_uri: TRAINING_HISTORY_DOCUMENT},
        tmp_path,
    )

    result = registry.run(make_config(tmp_path, inputs))

    assert result["status"] == "ok"
    summary = read_summary(tmp_path, result)
    assert summary["metrics_source"] == "metrics_file"
    assert summary["metrics"]["mAP50"] == pytest.approx(0.55)
    assert summary["losses_source"] == "training_history"
    assert summary["losses"]["best_epoch"] == 2
    assert summary["losses"]["final_validation_loss"] == pytest.approx(0.60)
    assert storage.read_locations == [metrics_uri, history_uri]


def test_remote_read_failure_leaves_values_null_without_failing_the_run(
    tmp_path: Path, monkeypatch
):
    """권한이 없거나 파일이 없어도 등록은 성공하고 값만 비어 있습니다."""

    inputs = load_fixture("inputs_s3.json")
    use_fake_s3(
        monkeypatch,
        {inputs["evaluate"]["metrics_uri"]: StorageError("접근 권한 거부를 흉내 냅니다.")},
        tmp_path,
    )

    result = registry.run(make_config(tmp_path, inputs))

    assert result["status"] == "ok"
    summary = read_summary(tmp_path, result)
    assert summary["metrics_source"] == "unavailable"
    assert summary["metrics"]["mAP"] is None
    # training_history는 아예 없는 경우입니다.
    assert summary["losses_source"] == "unavailable"
    assert set(summary["losses"].values()) == {None}


def test_wrongly_typed_loss_values_become_null(local_run):
    """epoch별 값의 타입이 어긋나면 그 값만 null이 됩니다."""

    repo_root, inputs = local_run
    (repo_root / inputs["train"]["training_history_uri"]).write_text(
        json.dumps([{"epoch": "1", "train_loss": "0.5", "validation_loss": 0.4}]),
        encoding="utf-8",
        newline="\n",
    )

    summary = read_summary(repo_root, registry.run(make_config(repo_root, inputs)))

    assert summary["losses_source"] == "training_history"
    assert summary["losses"]["best_epoch"] is None
    assert summary["losses"]["final_train_loss"] is None
    assert summary["losses"]["best_validation_loss"] == pytest.approx(0.4)


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
        "precision": "amp",
        "checkpoint_every": 2,
        "gradient_accumulation_steps": 8,
        "input_size": 640,
        "augmentation": {"preset": "pill_geometric"},
        "lr_scheduler": {"name": "cosine", "warmup_steps": 500},
        "early_stopping": {"patience": 4, "min_delta": 0.0},
        # 이어서 학습한 실행인지는 설정이라 옮깁니다.
        "resume_from": "artifacts/experiments/exp-0000/last_checkpoint.pt",
        # 결과를 어디에 두는지와 seed는 옮기지 않습니다.
        "output_dir": "artifacts/train",
        "seed": 7,
    }

    summary = read_summary(repo_root, registry.run(config))

    assert summary["summary_version"] == "4"
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
        "precision": "amp",
        "checkpoint_every": 2,
        "gradient_accumulation_steps": 8,
        "input_size": 640,
        # 중첩 설정은 모양 그대로 옮깁니다. 평평하게 펴면 화면이 그 값으로 새 실험을
        # 다시 채울 때 원래 모양을 되살릴 수 없습니다.
        "augmentation": {"preset": "pill_geometric"},
        "lr_scheduler": {"name": "cosine", "warmup_steps": 500},
        "early_stopping": {"patience": 4, "min_delta": 0.0},
        "resume_from": "artifacts/experiments/exp-0000/last_checkpoint.pt",
        "seed": 7,
    }


def test_every_contract_setting_is_summarized_or_deliberately_left_out():
    """계약이 정한 학습 설정은 summary에 담기거나, 왜 빼는지 여기 적혀 있어야 합니다.

    `TRAINING_KEYS`를 손으로 적는 한 계약에 설정이 늘 때마다 조용히 빠집니다. 실제로
    `resume_from`이 그렇게 빠져 있었고, 이어서 학습한 실행이 처음부터 학습한 실행과
    구별되지 않았습니다. 양쪽을 다 대조하므로 계약에서 사라진 이름도 함께 잡힙니다.
    """

    summarized = {key for key, _ in summary_module.TRAINING_KEYS}
    # 담지 않는 셋입니다. 결과를 어디에 두는지일 뿐 무엇을 학습했는지가 아닙니다.
    # seed는 summary 최상위에도 있지만 그쪽은 registry 자신의 seed라 따로 담습니다.
    left_out = {"run_id", "output_dir", "output_prefix"}

    assert summarized | left_out == set(train_contract.SETTING_KEYS)
    assert not (summarized & left_out)


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
