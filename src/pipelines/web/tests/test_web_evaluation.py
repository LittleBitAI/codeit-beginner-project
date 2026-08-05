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
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


def make_record(*, remote: bool = False, artifacts: dict | None = None) -> JobRecord:
    base = "s3://bucket/datasets/p/" if remote else "artifacts/datasets/p/"
    checkpoint_base = "s3://bucket/experiments/completed/run-1/" if remote else "artifacts/x/"
    return JobRecord(
        job_id="a" * 32,
        config_id="b" * 32,
        run_id="run-1",
        status="succeeded",
        data_inputs={key: base + f"{key}.json" for key in DATA_ARTIFACT_KEYS},
        artifacts=artifacts
        if artifacts is not None
        else {
            "run_id": "run-1",
            "best_checkpoint_uri": checkpoint_base + "best_checkpoint.pt",
            "last_checkpoint_uri": checkpoint_base + "last_checkpoint.pt",
            "training_history_uri": checkpoint_base + "training_history.json",
        },
    )


def completed(ok: bool = True, returncode: int = 0):
    return subprocess.CompletedProcess(
        [],
        returncode,
        json.dumps(
            {
                "status": "ok" if ok else "error",
                "artifacts": {
                    "evaluate": {
                        "run_id": "run-1",
                        "metrics_uri": "artifacts/evaluate/run-1/metrics.json",
                        "predictions_uri": "artifacts/evaluate/run-1/predictions.json",
                    }
                },
                "summary": {
                    "evaluate": {
                        "pipeline": "evaluate",
                        "image_count": 46,
                        "prediction_count": 120,
                        "evaluated_class_count": 56,
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
    """기본값 artifacts/evaluate/... 는 저장소가 정한 S3 논리 prefix 밖입니다."""

    config = evaluation.build_evaluate_config(make_record(remote=True))

    assert config["storage"]["backend"] == "s3"
    assert config["evaluate"]["output_dir"] == "experiments/completed/run-1/evaluate"


def test_defaults_follow_the_evaluate_contract():
    config = evaluation.build_evaluate_config(make_record())["evaluate"]

    # 이미지 한 장에 알약 최대 4개라는 과제 정의에서 온 값입니다.
    assert config["max_detections_per_image"] == 4
    assert config["score_threshold"] == 0.0
    assert config["overwrite"] is False


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
    monkeypatch.setattr(evaluation.runner, "run_stage", lambda *a, **k: completed())

    runner_slot.start(make_record(), {})
    state = wait_done(runner_slot)

    assert state["status"] == "succeeded"
    assert state["summary"]["metrics"]["mAP"] == 0.3123
    assert state["artifacts"]["predictions_uri"].endswith("predictions.json")


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


def test_bad_request_does_not_start_a_thread(runner_slot, monkeypatch):
    called = []
    monkeypatch.setattr(evaluation, "run_evaluation", lambda config: called.append(config))

    with pytest.raises(WebValidationError):
        runner_slot.start(make_record(), {"score_threshold": 5})

    assert called == []
    assert runner_slot.status()["status"] == "idle"
