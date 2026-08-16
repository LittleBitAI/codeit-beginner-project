"""epoch 훑기.

validation loss가 고른 best epoch이 정말 제일 잘 맞히는 epoch인지 재 보는 화면입니다
(제안서 011·018·019). 후보마다 표본 평가를 돌리고, 이긴 하나만 전수로 다시 재어
제출까지 만듭니다.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.pipelines.web import epoch_sweep
from src.pipelines.web.epoch_sweep import epoch_candidates, score_candidates
from src.pipelines.web.jobs.model import JobRecord


def _finished_sweep() -> dict:
    """훑기가 끝날 때까지 기다렸다 마지막 상태를 돌려줍니다.

    **thread가 실제로 끝날 때까지** 기다립니다. 상태만 보고 넘어가면 뒷정리를 하는
    background thread가 test 밖에서 살아남아, monkeypatch가 되돌린 진짜 환경 변수로
    팀 동기화 singleton을 만들어 버립니다. 그러면 뒤이어 도는 다른 test가 "로그인이
    필요합니다"로 깨집니다.
    """

    runner = epoch_sweep.get_epoch_sweep_runner()
    deadline = time.monotonic() + 10
    while runner.status().get("status") == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
    for thread in threading.enumerate():
        if thread.name == "epoch-sweep":
            thread.join(timeout=10)
    return runner.status()


def _record(**overrides) -> JobRecord:
    record = JobRecord(job_id="0123456789abcdef0123456789abcdef", config_id="cfg-1", run_id="retina-run")
    record.status = "succeeded"
    record.artifacts = {
        "run_id": "retina-run",
        "best_checkpoint_uri": "artifacts/experiments/completed/retina-run/best_checkpoint.pt",
        "epoch_checkpoint_uris": [
            "artifacts/experiments/completed/retina-run/epochs/epoch_015.pt",
            "artifacts/experiments/completed/retina-run/epochs/epoch_016.pt",
        ],
    }
    record.data_inputs = {
        "train_manifest_uri": "artifacts/processed/v5/train.json",
        "validation_manifest_uri": "artifacts/processed/v5/validation.json",
        "class_map_uri": "artifacts/processed/v5/class_map.json",
        "dataset_summary_uri": "artifacts/processed/v5/dataset_summary.json",
        "test_manifest_uri": "artifacts/processed/v5/test.json",
    }
    record.settings = {"seed": 42}
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


# --- 후보 고르기 ------------------------------------------------------------


def test_candidates_come_from_what_train_left_behind():
    assert [candidate["epoch"] for candidate in epoch_candidates(_record())] == [15, 16]


# --- 순위 -------------------------------------------------------------------


def _candidate(epoch: int, **metrics) -> dict:
    return {"epoch": epoch, "checkpoint_uri": f"e{epoch}.pt", "metrics": metrics}


def test_the_first_choice_metric_outweighs_the_second():
    """1순위에 가장 큰 몫을 줍니다. 고르는 순서가 곧 가중치입니다.

    3순위 지표가 같아 순위를 가르지 못하므로, 1순위(3)와 2순위(2)만 남습니다.
    """

    candidates = [
        _candidate(10, mAP=0.70, mAP50=0.80, recall50=0.90),
        _candidate(11, mAP=0.60, mAP50=0.90, recall50=0.90),
    ]

    ranked = score_candidates(candidates, ["mAP", "mAP50", "recall50"])

    assert ranked[0]["epoch"] == 10


def test_a_metric_that_lives_in_another_range_does_not_take_over():
    """원래 값을 그대로 더하면 변동이 큰 지표 하나가 순위를 혼자 정합니다.

    여기서 mAP는 0.1이나 벌어지고 recall50은 0.001만 벌어집니다. 정규화하지 않으면
    1순위(recall50)가 아니라 mAP가 이깁니다.
    """

    candidates = [
        _candidate(10, mAP=0.50, recall50=0.981),
        _candidate(11, mAP=0.60, recall50=0.980),
    ]

    ranked = score_candidates(candidates, ["recall50", "mAP"])

    assert ranked[0]["epoch"] == 10


def test_an_unmeasured_metric_counts_as_the_worst():
    candidates = [_candidate(10, mAP=0.5, mAP50=None), _candidate(11, mAP=0.5, mAP50=0.9)]

    ranked = score_candidates(candidates, ["mAP50", "mAP"])

    assert ranked[0]["epoch"] == 11


def test_the_earlier_epoch_wins_a_tie():
    """같은 값이면 덜 학습한 쪽이고, 두 번 훑어도 같은 답이 나와야 합니다."""

    candidates = [_candidate(12, mAP=0.5), _candidate(10, mAP=0.5)]

    assert score_candidates(candidates, ["mAP"])[0]["epoch"] == 10


# --- 실행 config ------------------------------------------------------------


def test_a_candidate_is_measured_on_a_sample_and_makes_no_submission(monkeypatch):
    """후보 20개마다 test 842장을 추론하면 훑기가 몇 시간이 됩니다."""

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "team-bucket")
    config = epoch_sweep.build_candidate_config(
        _record(),
        {"epoch": 15, "checkpoint_uri": "epochs/epoch_015.pt"},
        sample_size=300,
        device="cpu",
    )

    assert config["evaluate"]["validation_sample_size"] == 300
    assert config["evaluate"]["run_id"] == "retina-run-e15-sample"
    assert "test_manifest_uri" not in config["inputs"]["data"]
    assert "submission_uri" not in config["evaluate"]
    # 후보를 재는 것은 그 checkpoint이지 원래 실행의 best가 아닙니다.
    assert config["inputs"]["train"]["best_checkpoint_uri"] == "epochs/epoch_015.pt"


def test_the_winner_is_a_separate_run_that_keeps_the_original_settings():
    """A의 12번째 epoch은 A가 아니라 `A-e12`입니다.

    같은 이름을 쓰면 원래 실행의 평가 결과와 제출을 덮어쓰게 됩니다.
    """

    best = epoch_sweep.winner_record(
        _record(), {"epoch": 12, "checkpoint_uri": "epochs/epoch_012.pt"}
    )

    assert best.artifacts["run_id"] == "retina-run-e12"
    assert best.artifacts["best_checkpoint_uri"] == "epochs/epoch_012.pt"
    # 제출을 만들려면 test manifest가 그대로 있어야 합니다.
    assert best.data_inputs["test_manifest_uri"].endswith("test.json")
    assert best.settings == _record().settings


# --- route ------------------------------------------------------------------


def test_sweeping_needs_metrics_chosen_in_the_settings_sheet(client, manager):
    record = _record()
    manager._records[record.job_id] = record

    response = client.post("/api/train/jobs/0123456789abcdef0123456789abcdef/epoch-sweep", json={})

    assert response.status_code == 400
    assert "지표" in response.text


def test_sweeping_needs_archived_epochs(client, manager):
    client.put(
        "/api/settings",
        json={"evaluation_mode": "serial", "epoch_metrics": ["mAP", "mAP50", "recall50"]},
    )
    record = _record()
    record.artifacts.pop("epoch_checkpoint_uris")
    manager._records[record.job_id] = record

    response = client.post("/api/train/jobs/0123456789abcdef0123456789abcdef/epoch-sweep", json={})

    assert response.status_code == 400
    assert "checkpoint" in response.text


def test_a_sweep_measures_every_candidate_then_evaluates_the_winner_in_full(
    client, manager, monkeypatch
):
    """훑기 한 번이 끝까지 도는 길입니다.

    후보마다 표본 평가 → 순위 → 이긴 하나만 전수 평가 → 등록. 실제 evaluate는
    subprocess라 여기서는 부른 config만 받아 적습니다.
    """

    client.put(
        "/api/settings",
        json={"evaluation_mode": "serial", "epoch_metrics": ["mAP", "mAP50", "recall50"]},
    )
    record = _record()
    manager._records[record.job_id] = record
    calls: list[dict] = []

    def fake_run_evaluation(config, on_progress_line=None):  # noqa: ARG001
        calls.append(config)
        epoch = 16 if "e16" in config["evaluate"]["run_id"] else 15
        return {
            "ok": True,
            "exit_code": 0,
            "artifacts": {"metrics_uri": "m.json", "submission_uri": "s.csv"},
            "summary": {"metrics": {"mAP": 0.1 * epoch, "mAP50": 0.9, "recall50": 0.9}},
            "message": "완료",
        }

    monkeypatch.setattr(epoch_sweep, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        epoch_sweep, "run_registry", lambda config: {"ok": True, "artifacts": {}, "summary": {}, "message": "등록"}
    )

    response = client.post("/api/train/jobs/0123456789abcdef0123456789abcdef/epoch-sweep", json={"sample_size": 5})

    assert response.status_code == 202, response.text
    state = _finished_sweep()
    assert state["status"] == "succeeded", state.get("message")
    # 후보 둘을 표본으로 재고, 이긴 하나만 전수로 다시 쟀습니다.
    assert [config["evaluate"]["run_id"] for config in calls] == [
        "retina-run-e15-sample",
        "retina-run-e16-sample",
        "retina-run-e16",
    ]
    assert calls[-1]["evaluate"].get("validation_sample_size") is None
    assert state["winner"]["epoch"] == 16
    assert state["winner"]["run_id"] == "retina-run-e16"
    assert state["registration"]["status"] == "succeeded"
    # 서버를 다시 띄워도 남아 있어야 합니다. 몇십 분을 다시 쓰게 됩니다.
    assert manager.get("0123456789abcdef0123456789abcdef").epoch_sweep["winner"]["epoch"] == 16


def test_one_failed_candidate_does_not_throw_away_the_others(client, manager, monkeypatch):
    """20개를 다시 재는 것보다 19개로 고르는 편이 낫습니다."""

    client.put(
        "/api/settings",
        json={"evaluation_mode": "serial", "epoch_metrics": ["mAP", "mAP50", "recall50"]},
    )
    manager._records["0123456789abcdef0123456789abcdef"] = _record()

    def fake_run_evaluation(config, on_progress_line=None):  # noqa: ARG001
        if config["evaluate"]["run_id"] == "retina-run-e15-sample":
            return {"ok": False, "exit_code": 1, "artifacts": {}, "summary": {}, "message": "터졌습니다"}
        return {
            "ok": True,
            "exit_code": 0,
            "artifacts": {},
            "summary": {"metrics": {"mAP": 0.5, "mAP50": 0.5, "recall50": 0.5}},
            "message": "완료",
        }

    monkeypatch.setattr(epoch_sweep, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        epoch_sweep, "run_registry", lambda config: {"ok": True, "artifacts": {}, "summary": {}, "message": "등록"}
    )

    client.post("/api/train/jobs/0123456789abcdef0123456789abcdef/epoch-sweep", json={"sample_size": 5})

    state = _finished_sweep()
    assert state["status"] == "succeeded", state.get("message")
    assert state["winner"]["epoch"] == 16
    # 무엇이 빠졌는지는 화면에 남습니다.
    failed = [entry for entry in state["candidates"] if entry.get("failed")]
    assert [entry["epoch"] for entry in failed] == [15]


def test_a_sweep_refuses_while_a_training_run_is_going(
    client, manager, monkeypatch, valid_payload, fake_process_factory
):
    """8GB 카드에서 학습과 겹치면 둘 다 out of memory로 잃습니다."""

    from src.pipelines.web.jobs import runner as job_runner

    client.put(
        "/api/settings",
        json={"evaluation_mode": "parallel", "epoch_metrics": ["mAP", "mAP50", "recall50"]},
    )
    manager._records["0123456789abcdef0123456789abcdef"] = _record()
    created = client.post("/api/train/configs", json=valid_payload).json()
    monkeypatch.setattr(job_runner, "spawn", lambda *a, **k: fake_process_factory())
    manager.start(created["config_id"])

    response = client.post("/api/train/jobs/0123456789abcdef0123456789abcdef/epoch-sweep", json={})

    assert response.status_code == 409
    assert "학습이 도는 중" in response.text


@pytest.fixture(autouse=True)
def fresh_runner(monkeypatch):
    """훑기 runner는 하나뿐이라 test끼리 상태가 새지 않게 매번 새로 만듭니다."""

    monkeypatch.setattr(epoch_sweep, "_RUNNER", None)
    yield
    monkeypatch.setattr(epoch_sweep, "_RUNNER", None)
