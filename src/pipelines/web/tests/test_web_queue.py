"""학습 대기열: 저장한 설정 여러 개를 직렬로 돌립니다.

용도는 "돌려 놓고 자러 가기"입니다. 그래서 자는 동안 사람이 개입하지 않아도 되는
쪽으로 규칙을 정했습니다. 실제 학습은 하지 않고 subprocess를 patch합니다.
"""

from __future__ import annotations

import time

import pytest

from src.pipelines.web.errors import JobNotFoundError, TeamSyncAuthError
from src.pipelines.web.jobs import runner
from src.pipelines.web.jobs.manager import JobManager
from src.pipelines.web.train_config import (
    build_runtime_config,
    normalize_data_inputs,
    normalize_train_settings,
    write_runtime_config,
)


TRAIN_STDOUT = (
    '{"status": "ok", "artifacts": {"train": {"run_id": "r"}}, '
    '"summary": {"train": {}}, "message": "완료"}'
)


@pytest.fixture
def config_ids(isolated_repo, data_inputs):
    """서로 다른 run_id를 가진 설정 세 개를 저장해 둡니다."""

    made = []
    for name in ("first", "second", "third"):
        config = build_runtime_config(
            normalize_train_settings({"run_id": name, "epochs": 1}),
            normalize_data_inputs(data_inputs),
        )
        made.append(write_runtime_config(config))
    return made


def wait_until(condition, timeout: float = 10.0, what: str = "조건"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"{what}이(가) {timeout}초 안에 이뤄지지 않았습니다.")


def finished_run_ids(manager: JobManager) -> list[str]:
    return [
        record.run_id
        for record in sorted(manager.list_jobs(), key=lambda item: item.created_at)
        if not record.is_active()
    ]


# --- 직렬 실행 --------------------------------------------------------------


def test_first_entry_starts_at_once_and_the_rest_wait(
    manager, config_ids, monkeypatch, fake_process_factory
):
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_STDOUT)
    )

    for config_id in config_ids:
        manager.enqueue(config_id)

    wait_until(
        lambda: len(finished_run_ids(manager)) == 3, what="세 학습이 모두 끝나는 것"
    )
    # 넣은 순서대로, 한 번에 하나씩입니다.
    assert finished_run_ids(manager) == ["first", "second", "third"]
    assert manager.queue_entries() == []


def test_a_failed_training_does_not_strand_the_rest(
    manager, config_ids, monkeypatch, fake_process_factory
):
    """자는 동안 하나가 실패했다고 나머지가 안 돌면 밤을 통째로 버립니다."""

    spawns = {"count": 0}

    def spawn(*args, **kwargs):
        spawns["count"] += 1
        if spawns["count"] == 1:
            return fake_process_factory(stdout="깨진 출력", exit_code=1)
        return fake_process_factory(stdout=TRAIN_STDOUT)

    monkeypatch.setattr(runner, "spawn", spawn)

    for config_id in config_ids:
        manager.enqueue(config_id)

    wait_until(lambda: len(finished_run_ids(manager)) == 3, what="세 학습 종료")
    statuses = {
        record.run_id: record.status for record in manager.list_jobs()
    }
    assert statuses["first"] == "failed"
    assert statuses["second"] == "succeeded"
    assert statuses["third"] == "succeeded"


def test_a_start_failure_pauses_and_keeps_the_entry_and_login_token(
    manager, config_ids, monkeypatch, fake_process_factory
):
    """팀 기록 시작이 실패해도 성공 응답 뒤에서 학습 요청을 버리면 안 됩니다."""

    from src.pipelines.web import team_sync

    class RejectingSync:
        def create_run(self, **_kwargs):
            raise TeamSyncAuthError("로그인 token이 거절됐습니다.")

    monkeypatch.setattr(team_sync, "get_team_sync", lambda: RejectingSync())

    with pytest.raises(TeamSyncAuthError, match="거절"):
        manager.enqueue(config_ids[0], access_token="browser-token")

    assert manager.queue_paused() is True
    assert [item["run_id"] for item in manager.queue_entries()] == ["first"]

    received_tokens = []

    class AcceptingSync:
        def create_run(self, *, access_token, **_kwargs):
            received_tokens.append(access_token)
            return None

        def enqueue_update(self, _record):
            return None

        def enqueue_log(self, _record, _entry):
            return None

    monkeypatch.setattr(team_sync, "get_team_sync", lambda: AcceptingSync())
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_STDOUT)
    )

    started = manager.resume_queue()

    assert started is not None
    assert received_tokens == ["browser-token"]
    assert manager.queue_entries() == []


def test_resuming_replaces_a_token_that_was_already_refused(
    manager, config_ids, monkeypatch, fake_process_factory
):
    """다시 돌릴 때는 **받은 token으로 덮어씁니다.** 남겨 두면 영원히 못 고칩니다.

    Cognito access token의 기본 수명은 한 시간입니다(실제 user pool client에
    ``AccessTokenValidity``가 없어 기본값이 적용됩니다). 그래서 앞 학습이 한 시간을
    넘기면, 대기열이 다음 항목을 꺼낼 때 그 항목이 쥔 token은 이미 만료돼 있습니다.
    AppSync는 401로 거절하고 대기열은 그 자리에서 멈추면서 만료된 token을 도로
    항목에 붙여 둡니다.

    이때 사람이 새 token으로 다시 돌리기를 눌렀는데 비어 있을 때만 채우면, 항목에는
    이미 만료된 token이 있으므로 새 token이 무시되고 같은 401이 반복됩니다. 서버를
    다시 띄우기 전에는 밤새 걸어 둔 목록을 하나도 살릴 수 없습니다.
    """

    from src.pipelines.web import team_sync

    class ExpiringSync:
        """만료된 token만 거절합니다. 실제 AppSync의 401을 대신합니다."""

        def create_run(self, *, access_token, **_kwargs):
            if access_token != "fresh-token":
                raise TeamSyncAuthError("로그인이 만료됐습니다.")
            received_tokens.append(access_token)
            return None

        def enqueue_update(self, _record):
            return None

        def enqueue_log(self, _record, _entry):
            return None

    received_tokens: list[str] = []
    monkeypatch.setattr(team_sync, "get_team_sync", lambda: ExpiringSync())

    # 한 시간 전에 받은 token으로 걸어 둡니다. 꺼내는 순간 이미 만료됐습니다.
    with pytest.raises(TeamSyncAuthError):
        manager.enqueue(config_ids[0], access_token="expired-token")

    assert manager.queue_paused() is True
    assert [item["run_id"] for item in manager.queue_entries()] == ["first"]

    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_STDOUT)
    )

    started = manager.resume_queue(access_token="fresh-token")

    assert started is not None
    assert received_tokens == ["fresh-token"]
    assert manager.queue_entries() == []


# --- 중지 ------------------------------------------------------------------


def test_cancelling_the_running_job_holds_the_queue(
    manager, config_ids, monkeypatch, fake_process_factory
):
    """중지를 눌렀는데 다음 학습이 곧바로 뜨면 멈춘 것이 아닙니다.

    자연스럽게 끝났을 때만 다음으로 넘어가고, 사람이 중지하면 대기열은 멈춰 섭니다.
    """

    process = fake_process_factory(stdout="", exit_code=1, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    monkeypatch.setattr(runner, "terminate_tree", lambda proc: proc.release(1))

    started = manager.enqueue(config_ids[0])
    manager.enqueue(config_ids[1])
    wait_until(lambda: manager._process is not None, what="process 시작")
    manager.cancel(started.job_id)

    wait_until(lambda: manager.get(started.job_id).status == "cancelled", what="취소 반영")
    assert manager.queue_entries()[0]["run_id"] == "second"  # 여전히 기다립니다
    assert manager.queue_paused() is True


def test_the_queue_is_already_held_when_stop_returns(
    manager, config_ids, monkeypatch, fake_process_factory
):
    """중지 요청이 돌아온 순간 대기열은 이미 멈춰 있어야 합니다.

    학습 process는 종료 신호를 받고도 곧바로 죽지 않습니다. GPU 메모리를 정리하는
    동안 살아 있고, 그래도 안 죽으면 10초 뒤에야 강제로 끝냅니다. 그 시간을 job
    thread의 정리 구간이 끝날 때까지로 보고 대기열을 그때 멈추면, 그 사이에 들어온
    시작 요청이 다음 학습을 띄웁니다 — 사람은 방금 중지를 눌렀는데도.
    """

    process = fake_process_factory(stdout="", exit_code=1, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    # 신호는 보냈지만 이 process는 아직 죽지 않습니다. job thread는 계속 기다립니다.
    monkeypatch.setattr(runner, "terminate_tree", lambda proc: None)

    started = manager.enqueue(config_ids[0])
    manager.enqueue(config_ids[1])
    wait_until(lambda: manager._process is not None, what="process 시작")

    manager.cancel(started.job_id)

    assert manager.queue_paused() is True
    process.release(1)
    wait_until(lambda: manager.get(started.job_id).status == "cancelled", what="취소 반영")


def test_resuming_while_the_stopped_job_winds_down_still_starts_the_next(
    manager, config_ids, monkeypatch, fake_process_factory
):
    """중지 직후 누른 다시 돌리기는 자리가 비는 대로 다음을 시작해야 합니다.

    중지한 process가 정리되는 동안 그 학습이 아직 자리를 쥐고 있어, 그때 들어온
    요청은 빈손으로 돌아갑니다. 자리가 빈 뒤에도 아무도 다시 밀어 주지 않으면
    대기열은 사람이 분명히 다시 눌렀는데도 그대로 서 있습니다.
    """

    stopped = fake_process_factory(stdout="", exit_code=1, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: stopped)
    monkeypatch.setattr(runner, "terminate_tree", lambda proc: None)

    started = manager.enqueue(config_ids[0])
    manager.enqueue(config_ids[1])
    wait_until(lambda: manager._process is not None, what="process 시작")
    manager.cancel(started.job_id)

    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_STDOUT)
    )
    # 중지한 학습이 아직 죽지 않았습니다. 자리가 없어 이 요청은 아무것도 시작하지 못합니다.
    assert manager.resume_queue() is None

    stopped.release(1)

    wait_until(lambda: "second" in finished_run_ids(manager), what="다음 학습 실행")


def test_a_held_queue_starts_again_when_asked(
    manager, config_ids, monkeypatch, fake_process_factory
):
    process = fake_process_factory(stdout="", exit_code=1, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    monkeypatch.setattr(runner, "terminate_tree", lambda proc: proc.release(1))

    started = manager.enqueue(config_ids[0])
    manager.enqueue(config_ids[1])
    wait_until(lambda: manager._process is not None, what="process 시작")
    manager.cancel(started.job_id)
    wait_until(lambda: manager.queue_paused(), what="대기열 멈춤")

    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_STDOUT)
    )
    manager.resume_queue()

    wait_until(lambda: "second" in finished_run_ids(manager), what="다음 학습 실행")
    assert manager.queue_paused() is False


# --- 대기열 편집 ------------------------------------------------------------


def test_waiting_entries_can_be_removed(manager, config_ids, monkeypatch, fake_process_factory):
    process = fake_process_factory(stdout="", exit_code=0, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)

    manager.enqueue(config_ids[0])
    manager.enqueue(config_ids[1])
    manager.enqueue(config_ids[2])
    entries = manager.queue_entries()

    manager.remove_from_queue(entries[0]["entry_id"])

    assert [item["run_id"] for item in manager.queue_entries()] == ["third"]
    process.release(0)


def test_removing_something_that_is_not_waiting_says_so(manager, config_ids):
    with pytest.raises(JobNotFoundError):
        manager.remove_from_queue("없는-항목")


def test_an_unknown_config_is_refused_before_it_reaches_the_queue(manager, isolated_repo):
    """대기열에 넣을 때 걸러야 자는 동안 조용히 건너뛰는 항목이 생기지 않습니다."""

    with pytest.raises(Exception):
        manager.enqueue("0" * 32)

    assert manager.queue_entries() == []


# --- 서버가 다시 떴을 때 ----------------------------------------------------


def test_a_restart_keeps_the_queue_but_does_not_start_it(
    manager, config_ids, monkeypatch, fake_process_factory
):
    """자는 동안 서버가 다시 떴다고 대기열이 사라지면 아침에 아무것도 없습니다.

    다만 서버가 뜨자마자 GPU 학습이 저절로 시작되면 그것도 곤란하므로, 목록은
    남기고 시작은 사람이 시킵니다.
    """

    process = fake_process_factory(stdout="", exit_code=0, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    manager.enqueue(config_ids[0])
    manager.enqueue(config_ids[1])
    # 첫 학습이 아직 도는 동안 확인합니다. 끝나 버리면 두 번째가 이미 시작됩니다.
    wait_until(lambda: manager._process is not None, what="process 시작")

    fresh = JobManager()
    fresh.load()

    assert [item["run_id"] for item in fresh.queue_entries()] == ["second"]
    assert fresh.queue_paused() is True
    process.release(0)
