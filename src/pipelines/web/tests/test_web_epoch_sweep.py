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


def test_a_record_being_swept_cannot_be_deleted(client, manager, monkeypatch):
    """훑는 중에 기록을 지우면 훑기가 끝나면서 그것을 다시 저장합니다.

    log 없는 기록으로 되살아나므로, 지운 사람은 지워지지 않았다고 봅니다. 평가가 같은
    이유로 같은 문을 두고 있습니다.
    """

    record = _record()
    manager._records[record.job_id] = record
    runner = epoch_sweep.get_epoch_sweep_runner()
    runner._state = {"status": "running", "job_id": record.job_id}

    response = client.delete(f"/api/train/jobs/{record.job_id}")

    assert response.status_code == 409
    assert "훑기" in response.text
    assert manager.get(record.job_id) is record


def test_evaluation_does_not_start_while_a_sweep_is_running(client, manager, monkeypatch):
    """훑기도 같은 GPU로 추론합니다. 한 방향만 막으면 반대편이 그대로 들어옵니다."""

    record = _record()
    manager._records[record.job_id] = record
    epoch_sweep.get_epoch_sweep_runner()._state = {
        "status": "running",
        "job_id": "다른-학습",
    }

    response = client.post(f"/api/train/jobs/{record.job_id}/evaluate", json={})

    assert response.status_code == 409
    assert "훑기" in response.text


def test_the_automatic_evaluation_also_stands_back_while_a_sweep_runs(client, manager):
    """사람이 누른 평가만 막으면 밤새 도는 자동 평가가 그대로 겹칩니다."""

    from src.pipelines.web import evaluation
    from src.pipelines.web.errors import JobConflictError

    record = _record()
    manager._records[record.job_id] = record
    manager._evaluation_pending.append(record.job_id)
    epoch_sweep.get_epoch_sweep_runner()._state = {"status": "running", "job_id": "다른-학습"}

    more = manager._start_one_evaluation(
        evaluation.get_evaluation_runner(), JobConflictError, serial=True
    )

    assert more is False
    # 물러났을 뿐이라 줄은 그대로 남아 있어야 합니다. 훑기가 끝나면서 다시 깨웁니다.
    assert manager._evaluation_pending == [record.job_id]


def test_a_second_sweep_of_the_same_run_takes_new_names(client, manager, monkeypatch):
    """이름이 매번 같으면 두 번째 훑기가 통째로 실패합니다.

    evaluate는 이미 있는 artifact를 덮어쓰지 않으므로, 표본 크기를 바꿔 다시 재려 해도
    후보마다 "이미 있습니다"로 끝나고 그 실패가 지난 성공 상태까지 덮습니다.
    """

    client.put(
        "/api/settings",
        json={"evaluation_mode": "serial", "epoch_metrics": ["mAP", "mAP50", "recall50"]},
    )
    record = _record()
    record.epoch_sweep = {"status": "succeeded", "attempt": 1}
    manager._records[record.job_id] = record
    calls: list[str] = []

    def fake_run_evaluation(config, on_progress_line=None):  # noqa: ARG001
        calls.append(config["evaluate"]["run_id"])
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

    client.post(f"/api/train/jobs/{record.job_id}/epoch-sweep", json={"sample_size": 5})
    state = _finished_sweep()

    assert state["status"] == "succeeded", state.get("message")
    assert state["attempt"] == 2
    assert calls == [
        "retina-run-e15-sample.2",
        "retina-run-e16-sample.2",
        "retina-run-e15.2",
    ]


class _RecordingLock:
    """어느 문을 먼저 잡았는지 적어 두는 lock 껍데기입니다."""

    def __init__(self, name: str, order: list[str], lock) -> None:
        self._name = name
        self._order = order
        self._lock = lock

    def __enter__(self):
        self._order.append(self._name)
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


def test_both_runners_are_taken_in_the_same_order(client, manager, monkeypatch):
    """평가 문과 훑기 문을 서로 반대 순서로 잡으면 두 요청이 서로를 기다립니다.

    한쪽이 `evaluation -> sweep`, 다른 쪽이 `sweep -> evaluation`이면 그 둘이 동시에
    들어온 순간 어느 쪽도 돌아오지 않습니다. 실제 교착을 재현하는 test는 실패할 때
    영영 멈추므로, **어느 문을 먼저 잡는지**를 봅니다. 두 경로 모두 평가가 먼저여야
    고리가 생기지 않습니다.
    """

    from src.pipelines.web import evaluation

    record = _record()
    manager._records[record.job_id] = record
    order: list[str] = []
    evaluation_runner = evaluation.get_evaluation_runner()
    sweep_runner = epoch_sweep.get_epoch_sweep_runner()
    monkeypatch.setattr(
        evaluation_runner, "_lock", _RecordingLock("evaluation", order, evaluation_runner._lock)
    )
    monkeypatch.setattr(
        sweep_runner, "_lock", _RecordingLock("sweep", order, sweep_runner._lock)
    )

    # 지표를 고르지 않아 훑기는 거절되지만, 거절 전에 이미 두 문을 지납니다.
    client.post(f"/api/train/jobs/{record.job_id}/epoch-sweep", json={})
    sweep_route = list(order)
    order.clear()
    client.post(f"/api/train/jobs/{record.job_id}/evaluate", json={})
    evaluate_route = list(order)

    assert sweep_route[0] == "evaluation", f"훑기가 훑기 문을 먼저 잡았습니다: {sweep_route}"
    assert evaluate_route[0] == "evaluation", f"평가 순서가 바뀌었습니다: {evaluate_route}"


def test_a_sweep_that_just_finished_does_not_revive_a_deleted_record(
    client, manager, monkeypatch
):
    """저장보다 상태를 먼저 끝으로 바꾸면, 그 틈에 지운 기록을 저장이 되살립니다.

    저장이 일어나는 그 순간 삭제를 시도해 봅니다. 아직 `running`이어야 삭제가 409로
    거절되고, 그래야 저장이 지워진 기록을 되살리는 일이 없습니다.
    """

    from src.pipelines.web.jobs import store

    record = _record()
    manager._records[record.job_id] = record
    runner = epoch_sweep.get_epoch_sweep_runner()
    runner._state = {"status": "running", "job_id": record.job_id}
    refusals: list[int] = []

    def watching_save(saved_record):
        refusals.append(client.delete(f"/api/train/jobs/{record.job_id}").status_code)

    monkeypatch.setattr(store, "save_record", watching_save)

    runner._finish(record, status="succeeded", message="끝")

    assert refusals == [409], "저장하는 동안 삭제가 통과했습니다"
    assert runner.status()["status"] == "succeeded"


@pytest.mark.parametrize("broken", ["save", "share"])
def test_neither_saving_nor_sharing_can_turn_a_finished_sweep_into_a_failure(
    manager, monkeypatch, broken
):
    """전수 평가와 등록이 끝난 뒤에 남은 것은 저장과 팀 화면 알림뿐입니다.

    둘 중 무엇이 실패해도 이미 잰 결과는 그대로여야 합니다. 예외가 `_finish` 밖으로
    나가면 `_run`의 catch-all이 이미 끝난 훑기를 실패로 다시 덮거나, 상태가 영영
    `running`에 갇혀 다음 훑기를 시작할 수 없게 됩니다.
    """

    from src.pipelines.web.jobs import store

    def angry(*args, **kwargs):
        raise RuntimeError("터졌습니다")

    if broken == "save":
        monkeypatch.setattr(store, "save_record", angry)
    else:
        monkeypatch.setattr(epoch_sweep.team_sync, "get_team_sync", angry)

    record = _record()
    manager._records[record.job_id] = record
    runner = epoch_sweep.get_epoch_sweep_runner()
    runner._state = {"status": "running", "job_id": record.job_id}
    monkeypatch.setattr(
        runner,
        "_run_once",
        lambda *args: runner._finish(record, status="succeeded", message="끝"),
    )

    runner._run(record, [], [], 5, "cpu", 1)

    assert runner.status()["status"] == "succeeded"


def test_the_attempt_number_is_written_before_the_sweep_runs(
    client, manager, monkeypatch
):
    """끝날 때만 남기면 도중에 server가 죽었을 때 같은 이름을 다시 씁니다."""

    client.put(
        "/api/settings",
        json={"evaluation_mode": "serial", "epoch_metrics": ["mAP", "mAP50", "recall50"]},
    )
    record = _record()
    manager._records[record.job_id] = record
    started = threading.Event()
    release = threading.Event()

    def slow_run_evaluation(config, on_progress_line=None):  # noqa: ARG001
        started.set()
        release.wait(timeout=5)
        return {"ok": False, "exit_code": 1, "artifacts": {}, "summary": {}, "message": "중단"}

    monkeypatch.setattr(epoch_sweep, "run_evaluation", slow_run_evaluation)

    client.post(f"/api/train/jobs/{record.job_id}/epoch-sweep", json={"sample_size": 5})
    assert started.wait(timeout=5), "훑기가 첫 후보에 닿지 못했습니다"

    # 아직 도는 중입니다. 이 시점의 기록에 이미 번호가 있어야 합니다.
    assert manager.get(record.job_id).epoch_sweep["attempt"] == 1
    assert manager.get(record.job_id).epoch_sweep["status"] == "running"
    release.set()
    _finished_sweep()


def test_a_sweep_left_running_by_a_restart_reads_as_interrupted(manager, monkeypatch):
    """훑기는 thread로만 돕니다. 다시 뜬 server에는 그 thread가 없습니다."""

    from src.pipelines.web.jobs import store

    record = _record()
    record.epoch_sweep = {"status": "running", "job_id": record.job_id, "attempt": 2}
    store.save_record(record)
    manager._loaded = False
    manager._records = {}

    manager.load()

    revived = manager.get(record.job_id).epoch_sweep
    assert revived["status"] == "interrupted"
    # 번호는 남아 있어야 다음 훑기가 그 뒤부터 셉니다.
    assert revived["attempt"] == 2


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
