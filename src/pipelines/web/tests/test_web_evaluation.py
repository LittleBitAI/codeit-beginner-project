"""끝난 학습의 checkpoint로 evaluate pipeline을 부르는 경로.

실제 추론은 하지 않고 subprocess를 patch합니다.
"""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from src.pipelines.web import evaluation
from src.pipelines.web.errors import JobConflictError, WebValidationError
from src.pipelines.web.evaluation import EvaluationRunner
from src.pipelines.web.jobs.model import JobRecord
from src.pipelines.web.jobs import store
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


def make_record(
    *,
    remote: bool = False,
    artifacts: dict | None = None,
    with_test_manifest: bool = False,
) -> JobRecord:
    base = "s3://bucket/datasets/p/" if remote else "artifacts/datasets/p/"
    checkpoint_base = "s3://bucket/experiments/completed/run-1/" if remote else "artifacts/x/"
    data_inputs = {key: base + f"{key}.json" for key in DATA_ARTIFACT_KEYS}
    if with_test_manifest:
        data_inputs["test_manifest_uri"] = base + "test_manifest.json"
    return JobRecord(
        job_id="a" * 32,
        config_id="b" * 32,
        run_id="run-1",
        status="succeeded",
        data_inputs=data_inputs,
        artifacts=artifacts
        if artifacts is not None
        else {
            "run_id": "run-1",
            "best_checkpoint_uri": checkpoint_base + "best_checkpoint.pt",
            "last_checkpoint_uri": checkpoint_base + "last_checkpoint.pt",
            "training_history_uri": checkpoint_base + "training_history.json",
        },
    )


def completed(ok: bool = True, returncode: int = 0, *, with_submission: bool = False):
    evaluate_artifacts = {
        "run_id": "run-1",
        "metrics_uri": "artifacts/evaluate/run-1/metrics.json",
        "predictions_uri": "artifacts/evaluate/run-1/predictions.json",
    }
    if with_submission:
        evaluate_artifacts["submission_uri"] = "submissions/run-1/submission.csv"
    return subprocess.CompletedProcess(
        [],
        returncode,
        json.dumps(
            {
                "status": "ok" if ok else "error",
                "artifacts": {
                    "evaluate": evaluate_artifacts
                },
                "summary": {
                    "evaluate": {
                        "pipeline": "evaluate",
                        "image_count": 46,
                        "prediction_count": 120,
                        "evaluated_class_count": 56,
                        "iou_thresholds": [0.75, 0.8, 0.85, 0.9, 0.95]
                        if with_submission
                        else [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                        "metrics": {
                            "mAP": 0.3123,
                            "mAP50": 0.5512,
                            "mAP75": 0.2811,
                            "precision50": 0.61,
                            "recall50": 0.48,
                        },
                    }
                },
                "message": "evaluate pipeline 실행 완료",
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
    )


def registry_completed(ok: bool = True, *, index_status: str = "written"):
    return subprocess.CompletedProcess(
        [],
        0 if ok else 1,
        json.dumps(
            {
                "status": "ok" if ok else "error",
                "artifacts": {
                    "registry": {
                        "run_id": "run-1",
                        "experiment_record_uri": "artifacts/registry/run-1/experiment_record.json",
                        "experiment_summary_uri": "artifacts/registry/index/run-1.json",
                    }
                },
                "summary": {"registry": {"index_status": index_status}},
                "message": "등록 완료" if ok else "등록 실패",
            },
            ensure_ascii=False,
        ),
        "",
    )


# --- config 만들기 ------------------------------------------------------------


def test_config_carries_the_finished_training_as_input():
    config = evaluation.build_evaluate_config(make_record())

    assert config["execution"] == {"mode": "real"}
    assert config["inputs"]["train"]["best_checkpoint_uri"].endswith("best_checkpoint.pt")
    assert set(config["inputs"]["data"]) == set(DATA_ARTIFACT_KEYS)
    assert config["evaluate"]["run_id"] == "run-1"


def test_local_run_writes_next_to_other_local_artifacts():
    config = evaluation.build_evaluate_config(make_record(remote=False))

    assert config["storage"]["backend"] == "local"
    assert config["evaluate"]["output_dir"] == "artifacts/evaluate/run-1"


def test_remote_run_writes_inside_an_allowed_s3_prefix():
    """S3 실행의 출력 위치는 local 상대 경로가 아니라 완전한 S3 URI입니다."""

    config = evaluation.build_evaluate_config(make_record(remote=True))

    assert config["storage"]["backend"] == "s3"
    assert config["storage"]["s3"]["bucket"] == "bucket"
    assert config["evaluate"]["output_dir"] == (
        "s3://bucket/experiments/completed/run-1/evaluate"
    )


def test_defaults_follow_the_evaluate_contract():
    config = evaluation.build_evaluate_config(make_record())["evaluate"]

    # 이미지 한 장에 알약 최대 4개라는 과제 정의에서 온 값입니다.
    assert config["max_detections_per_image"] == 4
    assert config["score_threshold"] == 0.0
    assert config["overwrite"] is False


def test_test_manifest_is_forwarded_for_submission_generation():
    config = evaluation.build_evaluate_config(make_record(with_test_manifest=True))

    assert config["inputs"]["data"]["test_manifest_uri"].endswith("test_manifest.json")


def test_test_manifest_can_be_attached_to_an_existing_training():
    record = make_record(remote=True)

    config = evaluation.build_evaluate_config(
        record,
        test_manifest_uri="s3://bucket/datasets/test/test_manifest.json",
    )

    assert config["inputs"]["data"]["test_manifest_uri"] == (
        "s3://bucket/datasets/test/test_manifest.json"
    )
    assert config["evaluate"]["submission_uri"] == (
        "s3://bucket/submissions/run-1/submission.csv"
    )
    # 완료된 학습 기록은 증거이므로 평가 입력을 붙이더라도 바꾸지 않습니다.
    assert "test_manifest_uri" not in record.data_inputs


def test_registry_config_connects_saved_train_and_evaluate_artifacts():
    record = make_record()
    record.settings.update(
        {
            "architecture": "retinanet_resnet50_fpn_v2",
            "optimizer": "AdamW",
            "seed": 17,
        }
    )
    evaluation_state = {
        "storage": {"backend": "local", "local": {"root": "artifacts"}},
        "settings": {"run_id": "run-1", "output_dir": "artifacts/evaluate/run-1"},
        "artifacts": {
            "run_id": "run-1",
            "metrics_uri": "artifacts/evaluate/run-1/metrics.json",
            "predictions_uri": "artifacts/evaluate/run-1/predictions.json",
        },
    }

    config = evaluation.build_registry_config(record, evaluation_state)

    assert config["train"]["architecture"] == "retinanet_resnet50_fpn_v2"
    assert config["train"]["optimizer"] == "AdamW"
    assert config["registry"] == {"seed": 17, "overwrite": False}
    assert config["inputs"]["data"] == record.data_inputs
    assert config["inputs"]["train"] == record.artifacts
    assert config["inputs"]["evaluate"] == evaluation_state["artifacts"]


@pytest.mark.parametrize(
    "bad",
    ("../test_manifest.json", "https://example.com/test_manifest.json", "s3://bucket"),
)
def test_attached_test_manifest_uses_the_data_uri_safety_rules(bad):
    with pytest.raises(WebValidationError) as error:
        evaluation.build_evaluate_config(make_record(), test_manifest_uri=bad)

    assert error.value.errors[0].field == "inputs.data.test_manifest_uri"


@pytest.mark.parametrize("missing", ("run_id", "best_checkpoint_uri"))
def test_training_without_a_checkpoint_cannot_be_evaluated(missing):
    artifacts = {
        "run_id": "run-1",
        "best_checkpoint_uri": "artifacts/x/best_checkpoint.pt",
    }
    artifacts.pop(missing)

    with pytest.raises(WebValidationError):
        evaluation.build_evaluate_config(make_record(artifacts=artifacts))


@pytest.mark.parametrize("bad", (-0.1, 1.1, "0.5", True))
def test_bad_score_threshold_is_rejected(bad):
    with pytest.raises(WebValidationError) as error:
        evaluation.build_evaluate_config(make_record(), score_threshold=bad)

    assert error.value.errors[0].field == "score_threshold"


@pytest.mark.parametrize("bad", (0, -1, 1.5, True, "4"))
def test_bad_detection_limit_is_rejected(bad):
    with pytest.raises(WebValidationError):
        evaluation.build_evaluate_config(make_record(), max_detections_per_image=bad)


# --- 실행 ---------------------------------------------------------------------


def test_run_calls_only_the_evaluate_stage(isolated_repo, monkeypatch):
    captured = {}

    def fake_run_stage(path, stage, *, cwd, timeout):
        captured["stage"] = stage
        return completed()

    monkeypatch.setattr(evaluation.runner, "run_stage", fake_run_stage)

    result = evaluation.run_evaluation(evaluation.build_evaluate_config(make_record()))

    assert captured["stage"] == "evaluate"
    assert result["ok"] is True
    # stage 이름으로 감싼 한 겹을 벗겨야 합니다.
    assert result["artifacts"]["metrics_uri"].endswith("metrics.json")
    assert result["summary"]["metrics"]["mAP50"] == 0.5512


def test_run_removes_the_temporary_config(isolated_repo, monkeypatch):
    monkeypatch.setattr(evaluation.runner, "run_stage", lambda *a, **k: completed())

    evaluation.run_evaluation(evaluation.build_evaluate_config(make_record()))

    assert list((isolated_repo / "artifacts" / "web" / "configs").glob("*.json")) == []


def test_run_registry_calls_only_registry_and_removes_temporary_config(
    isolated_repo, monkeypatch
):
    stages = []
    monkeypatch.setattr(
        evaluation.runner,
        "run_stage",
        lambda path, stage, **kwargs: (
            stages.append(stage),
            registry_completed(),
        )[1],
    )
    record = make_record()
    evaluate_config = evaluation.build_evaluate_config(record)
    registry_config = evaluation.build_registry_config(
        record,
        {
            "storage": evaluate_config["storage"],
            "settings": evaluate_config["evaluate"],
            "artifacts": {
                "run_id": "run-1",
                "metrics_uri": "artifacts/evaluate/run-1/metrics.json",
                "predictions_uri": "artifacts/evaluate/run-1/predictions.json",
            },
        },
    )

    result = evaluation.run_registry(registry_config)

    assert stages == ["registry"]
    assert result["ok"] is True
    assert result["summary"]["index_status"] == "written"
    assert list((isolated_repo / "artifacts" / "web" / "configs").glob("*.json")) == []


def test_failure_is_reported(isolated_repo, monkeypatch):
    monkeypatch.setattr(
        evaluation.runner, "run_stage", lambda *a, **k: completed(ok=False, returncode=1)
    )

    result = evaluation.run_evaluation(evaluation.build_evaluate_config(make_record()))

    assert result["ok"] is False
    assert result["exit_code"] == 1


def test_timeout_is_reported(isolated_repo, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(evaluation.runner, "run_stage", timeout)

    result = evaluation.run_evaluation(evaluation.build_evaluate_config(make_record()))

    assert result["ok"] is False
    assert "시간 안에 끝나지 않았습니다" in result["message"]


def test_unparsable_output_does_not_leak_paths(isolated_repo, monkeypatch):
    from src.pipelines.web.paths import REPOSITORY_ROOT

    monkeypatch.setattr(
        evaluation.runner,
        "run_stage",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "깨짐", f"실패 {REPOSITORY_ROOT}/x"),
    )

    result = evaluation.run_evaluation(evaluation.build_evaluate_config(make_record()))

    assert result["ok"] is False
    assert str(REPOSITORY_ROOT) not in result["message"]


# --- 한 번에 하나 -------------------------------------------------------------


@pytest.fixture
def runner_slot(isolated_repo, monkeypatch):
    fresh = EvaluationRunner()
    monkeypatch.setattr(evaluation, "_RUNNER", fresh)
    yield fresh
    deadline = time.monotonic() + 10
    while fresh.status().get("status") == "running" and time.monotonic() < deadline:
        time.sleep(0.02)


def wait_done(slot: EvaluationRunner, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = slot.status()
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("평가가 시간 안에 끝나지 않았습니다.")


def test_successful_evaluation_records_metrics(runner_slot, monkeypatch):
    monkeypatch.setattr(
        evaluation.runner,
        "run_stage",
        lambda path, stage, **kwargs: completed() if stage == "evaluate" else registry_completed(),
    )

    runner_slot.start(make_record(), {})
    state = wait_done(runner_slot)

    assert state["status"] == "succeeded"
    assert state["summary"]["metrics"]["mAP"] == 0.3123
    assert state["artifacts"]["predictions_uri"].endswith("predictions.json")
    assert state["registration"]["status"] == "succeeded"


def test_registry_failure_keeps_evaluation_success_and_can_retry(
    runner_slot, monkeypatch
):
    calls = []

    def first_run(path, stage, **kwargs):
        calls.append(stage)
        return completed() if stage == "evaluate" else registry_completed(ok=False)

    monkeypatch.setattr(evaluation.runner, "run_stage", first_run)
    record = make_record()

    runner_slot.start(record, {})
    state = wait_done(runner_slot)

    assert state["status"] == "succeeded"
    assert state["registration"]["status"] == "failed"
    persisted = store.load_record(record.job_id)
    assert persisted.evaluation["artifacts"]["metrics_uri"].endswith("metrics.json")
    assert persisted.registration["status"] == "failed"

    monkeypatch.setattr(
        evaluation.runner,
        "run_stage",
        lambda path, stage, **kwargs: (calls.append(stage), registry_completed())[1],
    )
    retried = runner_slot.retry_registration(persisted)

    assert retried["status"] == "succeeded"
    assert calls == ["evaluate", "registry", "registry"]


def test_index_failure_requires_rebuild_instead_of_registration_retry(
    runner_slot, monkeypatch
):
    record = make_record()
    record.evaluation = {
        "status": "succeeded",
        "artifacts": {},
        "settings": {},
        "storage": {},
    }
    record.registration = {"status": "index_failed"}
    called = []
    monkeypatch.setattr(evaluation, "run_registry", lambda config: called.append(config))

    with pytest.raises(JobConflictError):
        runner_slot.retry_registration(record)

    assert called == []


def test_registration_retry_requires_a_successful_saved_evaluation(runner_slot):
    with pytest.raises(JobConflictError):
        runner_slot.retry_registration(make_record())


def test_failed_evaluation_never_calls_registry(runner_slot, monkeypatch):
    calls = []

    def fake(path, stage, **kwargs):
        calls.append(stage)
        return completed(ok=False, returncode=1)

    monkeypatch.setattr(evaluation.runner, "run_stage", fake)
    runner_slot.start(make_record(), {})

    state = wait_done(runner_slot)

    assert state["status"] == "failed"
    assert state["registration"]["status"] == "idle"
    assert calls == ["evaluate"]


def test_unexpected_evaluation_failure_is_persisted(runner_slot, monkeypatch):
    def fail(config):
        raise RuntimeError("temporary path failed")

    monkeypatch.setattr(evaluation, "run_evaluation", fail)
    record = make_record()

    runner_slot.start(record, {})
    state = wait_done(runner_slot)

    persisted = store.load_record(record.job_id)
    assert state["status"] == "failed"
    assert persisted.evaluation["status"] == "failed"
    assert persisted.registration["status"] == "idle"


def test_submission_request_and_artifact_are_visible_in_status(runner_slot, monkeypatch):
    monkeypatch.setattr(
        evaluation.runner,
        "run_stage",
        lambda *a, **k: completed(with_submission=True),
    )

    started = runner_slot.start(make_record(with_test_manifest=True), {})
    state = wait_done(runner_slot)

    assert started["submission_requested"] is True
    assert state["submission_requested"] is True
    assert state["artifacts"]["submission_uri"].endswith("submission.csv")
    assert state["summary"]["iou_thresholds"] == [0.75, 0.8, 0.85, 0.9, 0.95]


def test_runner_marks_a_manifest_attached_at_evaluation_time(runner_slot, monkeypatch):
    captured = {}

    def fake_run(config):
        captured.update(config)
        result = completed(with_submission=True)
        return {
            "ok": True,
            "exit_code": result.returncode,
            "artifacts": {"submission_uri": "submissions/run-1/submission.csv"},
            "summary": {"iou_thresholds": [0.75, 0.8, 0.85, 0.9, 0.95]},
            "message": "완료",
        }

    monkeypatch.setattr(evaluation, "run_evaluation", fake_run)

    started = runner_slot.start(
        make_record(),
        {"test_manifest_uri": "artifacts/data/test_manifest.json"},
    )
    state = wait_done(runner_slot)

    assert started["submission_requested"] is True
    assert captured["inputs"]["data"]["test_manifest_uri"] == (
        "artifacts/data/test_manifest.json"
    )
    assert state["artifacts"]["submission_uri"].endswith("submission.csv")


def test_second_evaluation_is_rejected_while_running(runner_slot, monkeypatch):
    import threading

    release = threading.Event()
    monkeypatch.setattr(
        evaluation.runner, "run_stage", lambda *a, **k: (release.wait(5), completed())[1]
    )
    runner_slot.start(make_record(), {})

    with pytest.raises(JobConflictError):
        runner_slot.start(make_record(), {})

    release.set()
    wait_done(runner_slot)


def test_status_does_not_show_another_jobs_result(runner_slot, monkeypatch):
    """다른 학습의 평가 결과를 이 학습의 것인 양 보여 주면 안 됩니다."""

    monkeypatch.setattr(evaluation.runner, "run_stage", lambda *a, **k: completed())
    runner_slot.start(make_record(), {})
    wait_done(runner_slot)

    other = runner_slot.status("c" * 32)

    assert other["status"] == "idle"
    assert other["busy_with"] == "a" * 32
    assert runner_slot.status("a" * 32)["status"] == "succeeded"


def test_saved_evaluation_keeps_other_running_job_as_busy_hint(runner_slot):
    record = make_record()
    record.evaluation = {"status": "succeeded", "message": "이전 평가 완료"}
    runner_slot._state = {
        "status": "running",
        "job_id": "c" * 32,
        "registration": {"status": "idle"},
    }

    state = runner_slot.status_for(record)

    assert state["status"] == "succeeded"
    assert state["busy_with"] == "c" * 32


def test_bad_request_does_not_start_a_thread(runner_slot, monkeypatch):
    called = []
    monkeypatch.setattr(evaluation, "run_evaluation", lambda config: called.append(config))

    with pytest.raises(WebValidationError):
        runner_slot.start(make_record(), {"score_threshold": 5})

    assert called == []
    assert runner_slot.status()["status"] == "idle"
