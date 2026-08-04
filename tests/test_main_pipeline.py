import copy
import json
from types import SimpleNamespace

from src import main_pipeline


REQUIRED_ARTIFACTS = {
    "data": {
        "train_manifest_uri": "artifacts/data/train.json",
        "validation_manifest_uri": "artifacts/data/validation.json",
        "class_map_uri": "artifacts/data/classes.json",
        "dataset_summary_uri": "artifacts/data/summary.json",
    },
    "train": {
        "run_id": "exp-0001",
        "best_checkpoint_uri": "artifacts/train/best.pt",
        "last_checkpoint_uri": "artifacts/train/last.pt",
        "training_history_uri": "artifacts/train/history.json",
    },
    "evaluate": {
        "run_id": "exp-0001",
        "metrics_uri": "artifacts/evaluate/metrics.json",
        "predictions_uri": "artifacts/evaluate/predictions.json",
    },
    "registry": {
        "run_id": "exp-0001",
        "experiment_record_uri": "artifacts/registry/record.json",
    },
}


def result_for(stage, *, status="ok", artifacts=None, message=""):
    return {
        "status": status,
        "artifacts": REQUIRED_ARTIFACTS[stage] if artifacts is None else artifacts,
        "summary": {"pipeline": stage},
        "message": message,
    }


def test_full_dummy_pipeline_completes_without_web():
    result = main_pipeline.run({"execution": {"mode": "dummy"}})

    assert result["status"] == "ok"
    assert list(result["summary"]) == ["data", "train", "evaluate", "registry"]
    assert "web" not in result["summary"]
    assert set(result) == {"status", "artifacts", "summary", "message"}


def test_full_pipeline_passes_only_previous_artifacts(monkeypatch):
    calls = []
    initial_data = {"source_uri": "artifacts/import/source.json"}
    config = {
        "inputs": {
            "data": initial_data,
            "train": {"must_not_leak": "old-checkpoint.pt"},
        },
        "seed": 7,
    }
    original = copy.deepcopy(config)

    def stage(stage_name):
        def execute(stage_config):
            calls.append((stage_name, copy.deepcopy(stage_config["inputs"])))
            assert stage_config["seed"] == 7
            return result_for(stage_name)

        return SimpleNamespace(run=execute)

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        tuple((name, stage(name)) for name in REQUIRED_ARTIFACTS),
    )

    result = main_pipeline.run(config)

    assert result["status"] == "ok"
    assert calls == [
        ("data", {"data": initial_data}),
        ("train", {"data": REQUIRED_ARTIFACTS["data"]}),
        (
            "evaluate",
            {
                "data": REQUIRED_ARTIFACTS["data"],
                "train": REQUIRED_ARTIFACTS["train"],
            },
        ),
        (
            "registry",
            {
                "data": REQUIRED_ARTIFACTS["data"],
                "train": REQUIRED_ARTIFACTS["train"],
                "evaluate": REQUIRED_ARTIFACTS["evaluate"],
            },
        ),
    ]
    assert config == original


def test_only_executes_one_pipeline_with_configured_inputs(monkeypatch):
    calls = []
    configured_inputs = {"data": REQUIRED_ARTIFACTS["data"]}

    def train(stage_config):
        calls.append(stage_config["inputs"])
        return result_for("train")

    monkeypatch.setattr(
        main_pipeline, "_STAGES", (("train", SimpleNamespace(run=train)),)
    )

    result = main_pipeline.run({"inputs": configured_inputs}, only="train")

    assert result["status"] == "ok"
    assert list(result["summary"]) == ["train"]
    assert calls == [configured_inputs]


def test_dry_run_reports_order_without_executing(monkeypatch):
    def must_not_run(config):
        raise AssertionError("dry-run에서 pipeline이 실행되었습니다.")

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (
            ("data", SimpleNamespace(run=must_not_run)),
            ("train", SimpleNamespace(run=must_not_run)),
        ),
    )

    result = main_pipeline.run({}, only="train", dry_run=True)

    assert result["status"] == "ok"
    assert result["summary"] == {"dry_run": True, "stages": ["train"]}
    assert result["artifacts"] == {}


def test_pipeline_stops_on_reported_failure(monkeypatch):
    calls = []

    def succeed(config):
        calls.append("data")
        return result_for("data")

    def fail(config):
        calls.append("train")
        return result_for(
            "train", status="error", artifacts={}, message="의도한 실패"
        )

    def must_not_run(config):
        calls.append("evaluate")
        raise AssertionError("실패 이후 pipeline이 실행되었습니다.")

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (
            ("data", SimpleNamespace(run=succeed)),
            ("train", SimpleNamespace(run=fail)),
            ("evaluate", SimpleNamespace(run=must_not_run)),
        ),
    )

    result = main_pipeline.run({})

    assert result["status"] == "error"
    assert result["message"] == "train: 의도한 실패"
    assert calls == ["data", "train"]


def test_pipeline_reports_raised_failure_and_stops(monkeypatch):
    def explode(config):
        raise RuntimeError("모델을 열 수 없음")

    def must_not_run(config):
        raise AssertionError("실패 이후 pipeline이 실행되었습니다.")

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (
            ("train", SimpleNamespace(run=explode)),
            ("evaluate", SimpleNamespace(run=must_not_run)),
        ),
    )

    result = main_pipeline.run({"inputs": {}}, only="train")

    assert result["status"] == "error"
    assert result["message"] == "train: RuntimeError: 모델을 열 수 없음"


def test_pipeline_reports_invalid_return_contract(monkeypatch):
    def invalid(config):
        return {
            "status": "ok",
            "artifacts": {},
            "summary": {},
        }

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (("data", SimpleNamespace(run=invalid)),),
    )

    result = main_pipeline.run({})

    assert result["status"] == "error"
    assert result["message"].startswith("data: PipelineContractError:")
    assert "필수 key 누락: message" in result["message"]


def test_pipeline_reports_wrong_return_contract_type(monkeypatch):
    def invalid(config):
        return {
            "status": "ok",
            "artifacts": [],
            "summary": {},
            "message": "",
        }

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (("data", SimpleNamespace(run=invalid)),),
    )

    result = main_pipeline.run({})

    assert result["status"] == "error"
    assert "'artifacts' 타입 불일치" in result["message"]


def test_pipeline_rejects_missing_required_artifact(monkeypatch):
    artifacts = dict(REQUIRED_ARTIFACTS["data"])
    artifacts.pop("class_map_uri")

    def missing(config):
        return result_for("data", artifacts=artifacts)

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (("data", SimpleNamespace(run=missing)),),
    )

    result = main_pipeline.run({})

    assert result["status"] == "error"
    assert result["message"] == "data: 필수 artifact 누락: class_map_uri"


def test_pipeline_rejects_invalid_required_artifact_value(monkeypatch):
    artifacts = dict(REQUIRED_ARTIFACTS["evaluate"])
    artifacts["metrics_uri"] = ""

    def invalid(config):
        return result_for("evaluate", artifacts=artifacts)

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (("evaluate", SimpleNamespace(run=invalid)),),
    )

    result = main_pipeline.run({"inputs": {}}, only="evaluate")

    assert result["status"] == "error"
    assert result["message"] == (
        "evaluate: 비어 있지 않은 문자열이어야 하는 artifact: metrics_uri"
    )


def test_cli_config_and_dry_run(capsys, tmp_path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text('{"seed": 3}\n', encoding="utf-8", newline="\n")

    exit_code = main_pipeline.main(
        ["--config", str(config_path), "--only", "evaluate", "--dry-run"]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["summary"] == {"dry_run": True, "stages": ["evaluate"]}


def test_cli_returns_nonzero_for_pipeline_failure(monkeypatch, capsys):
    def fail(config):
        return result_for("data", status="error", artifacts={}, message="입력 없음")

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (("data", SimpleNamespace(run=fail)),),
    )

    exit_code = main_pipeline.main([])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["message"] == "data: 입력 없음"


def test_cli_returns_nonzero_for_invalid_config(capsys, tmp_path):
    config_path = tmp_path / "broken.json"
    config_path.write_text("not-json\n", encoding="utf-8", newline="\n")

    exit_code = main_pipeline.main(["--config", str(config_path), "--dry-run"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["status"] == "error"
    assert output["message"].startswith("config: JSONDecodeError:")
