"""학습 job 실행·중지·기록.

실제 GPU 학습은 하지 않습니다. subprocess는 ``jobs/runner.py``의 named helper를
patch해서 대신합니다(``tools/git_pr.py``의 ``capture``/``execute``와 같은 방식).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.pipelines.web.errors import JobConflictError, JobNotFoundError
from src.pipelines.web.jobs import runner, store
from src.pipelines.web.jobs.manager import JobManager
from src.pipelines.web.paths import REPOSITORY_ROOT
from src.pipelines.web.train_config import (
    build_runtime_config,
    normalize_data_inputs,
    normalize_train_settings,
    write_runtime_config,
)


TRAIN_SUCCESS_STDOUT = json.dumps(
    {
        "status": "ok",
        "artifacts": {
            "train": {
                "run_id": "test-run",
                "best_checkpoint_uri": "artifacts/experiments/completed/test-run/best_checkpoint.pt",
                "last_checkpoint_uri": "artifacts/experiments/completed/test-run/last_checkpoint.pt",
                "training_history_uri": "artifacts/experiments/completed/test-run/training_history.json",
            }
        },
        "summary": {"train": {"architecture": "fasterrcnn_resnet50_fpn", "epochs": 2, "best_epoch": 2}},
        "message": "pipeline 실행을 완료했습니다.",
    },
    ensure_ascii=False,
    indent=2,
)

TRAIN_FAILURE_STDOUT = json.dumps(
    {
        "status": "error",
        "artifacts": {},
        "summary": {},
        "message": "train: training failed: JSON artifact does not exist: artifacts/x.json",
    },
    ensure_ascii=False,
    indent=2,
)


@pytest.fixture
def config_id(isolated_repo, data_inputs):
    config = build_runtime_config(
        normalize_train_settings({"run_id": "test-run", "epochs": 2}),
        normalize_data_inputs(data_inputs),
    )
    return write_runtime_config(config)


def wait_for_finish(manager: JobManager, job_id: str, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if not record.is_active():
            return record
        time.sleep(0.02)
    raise AssertionError(f"job이 {timeout}초 안에 끝나지 않았습니다: {manager.get(job_id).status}")


def wait_for_process(manager: JobManager, timeout: float = 15.0):
    """``_process``가 붙을 때까지 기다립니다.

    ``start()``는 thread를 띄우고 곧바로 돌아옵니다. spawn이 끝난 것과 그 결과가
    ``_process``에 담긴 것은 다른 순간이라, 기다리지 않고 만지면 ``None``을 봅니다.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process = manager._process
        if process is not None:
            return process
        time.sleep(0.02)
    raise AssertionError(f"process가 {timeout}초 안에 뜨지 않았습니다.")


def wait_for_spawn(manager: JobManager, timeout: float = 5.0) -> None:
    """process가 실제로 떠서 ``_process``가 채워질 때까지 기다립니다."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager._process is not None:
            return
        time.sleep(0.01)
    raise AssertionError("process가 시간 안에 시작되지 않았습니다.")


# --- argv -------------------------------------------------------------------


def test_build_argv_is_exact_list():
    """저장소에서 학습을 시작하는 유일하게 허용된 명령입니다."""

    assert runner.build_argv("artifacts/web/configs/abc.json") == [
        sys.executable,
        "-m",
        "src.main_pipeline",
        "--config",
        "artifacts/web/configs/abc.json",
        "--only",
        "train",
    ]


@pytest.mark.parametrize(
    "hostile",
    (
        "run; rm -rf /",
        "run && calc.exe",
        "$(whoami)",
        "`id`",
        "../../etc/passwd",
        "a" * 200,
        "--only=evaluate",
    ),
)
def test_build_argv_ignores_user_strings(hostile, isolated_repo, data_inputs):
    """사용자 입력은 argv에 절대 닿지 않습니다. config 경로는 서버가 만든 uuid뿐입니다."""

    settings = normalize_train_settings({})
    settings["output_dir"] = hostile  # 검증을 우회해 직접 넣어 봅니다
    config = build_runtime_config(settings, normalize_data_inputs(data_inputs))
    identifier = write_runtime_config(config)

    argv = runner.build_argv(f"artifacts/web/configs/{identifier}.json")

    assert hostile not in " ".join(argv)
    assert argv[:4] == [sys.executable, "-m", "src.main_pipeline", "--config"]
    assert argv[4] == f"artifacts/web/configs/{identifier}.json"
    assert argv[5:] == ["--only", "train"]


@pytest.mark.parametrize("stage", ("train", "data", "evaluate", "registry"))
def test_build_argv_supports_every_stage_the_gui_can_run(stage):
    """GUI가 부를 수 있는 stage는 모두 argv를 만들 수 있어야 합니다.

    evaluate 기능을 만들고 이 목록에 추가하지 않아 실행이 통째로 실패한 적이 있습니다.
    """

    assert runner.build_argv("artifacts/web/configs/abc.json", stage)[-2:] == ["--only", stage]


@pytest.mark.parametrize("stage", ("web", "", "train; rm -rf /", None, "EVALUATE"))
def test_build_argv_rejects_stages_outside_the_allowlist(stage):
    """--only에 들어갈 수 있는 값을 못 박아 둡니다."""

    with pytest.raises(ValueError):
        runner.build_argv("artifacts/web/configs/abc.json", stage)


def test_run_stage_never_uses_shell(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner.run_stage("artifacts/web/configs/abc.json", "data", cwd=tmp_path, timeout=5)

    assert "shell" not in captured["kwargs"]
    assert captured["args"][-2:] == ["--only", "data"]
    assert captured["kwargs"]["timeout"] == 5


def test_spawn_never_uses_shell(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise RuntimeError("stop here")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError):
        runner.spawn(["python", "-m", "x"], cwd=tmp_path, env={"A": "B"})

    assert "shell" not in captured["kwargs"]
    assert isinstance(captured["args"], list)
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert captured["kwargs"]["stdout"] is subprocess.PIPE
    assert captured["kwargs"]["stderr"] is subprocess.PIPE
    assert captured["kwargs"]["encoding"] == "utf-8"


def test_child_environment_forces_utf8_without_mutating_process_env():
    before = dict(os.environ)

    environment = runner.child_environment()

    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["PYTHONUTF8"] == "1"
    assert dict(os.environ) == before


# --- 성공과 실패 ------------------------------------------------------------


def test_successful_job_records_artifacts_and_summary(
    manager, config_id, monkeypatch, fake_process_factory
):
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT)
    )

    started = manager.start(config_id)
    record = wait_for_finish(manager, started.job_id)

    assert record.status == "succeeded"
    assert record.exit_code == 0
    # main_pipeline이 stage 이름으로 한 겹 감싼 것을 벗겨야 합니다.
    assert record.artifacts["run_id"] == "test-run"
    assert record.artifacts["best_checkpoint_uri"].endswith("best_checkpoint.pt")
    assert record.summary["architecture"] == "fasterrcnn_resnet50_fpn"


def test_failed_job_records_error_message(manager, config_id, monkeypatch, fake_process_factory):
    monkeypatch.setattr(
        runner,
        "spawn",
        lambda *a, **k: fake_process_factory(stdout=TRAIN_FAILURE_STDOUT, exit_code=1),
    )

    started = manager.start(config_id)
    record = wait_for_finish(manager, started.job_id)

    assert record.status == "failed"
    assert record.exit_code == 1
    assert "training failed" in record.message


def test_unparsable_stdout_is_failed_and_logged(
    manager, config_id, monkeypatch, fake_process_factory
):
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout="완전히 깨진 출력", exit_code=1)
    )

    started = manager.start(config_id)
    record = wait_for_finish(manager, started.job_id)

    assert record.status == "failed"
    assert record.message == "학습 결과 JSON을 해석하지 못했습니다."
    logs = store.read_logs(started.job_id)
    assert any("깨진 출력" in line["text"] for line in logs["lines"])


def test_spawn_failure_is_reported_without_leaking_paths(manager, config_id, monkeypatch):
    def explode(*args, **kwargs):
        raise FileNotFoundError(str(REPOSITORY_ROOT / "python.exe"))

    monkeypatch.setattr(runner, "spawn", explode)

    started = manager.start(config_id)
    record = wait_for_finish(manager, started.job_id)

    assert record.status == "failed"
    assert "FileNotFoundError" in record.message
    assert str(REPOSITORY_ROOT) not in record.message


# --- 동시 실행 --------------------------------------------------------------


def test_duplicate_start_is_rejected(manager, config_id, monkeypatch, fake_process_factory):
    calls = []

    def counted_spawn(*args, **kwargs):
        calls.append(args)
        return fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT, block_until_signalled=True)

    monkeypatch.setattr(runner, "spawn", counted_spawn)
    first = manager.start(config_id)

    with pytest.raises(JobConflictError) as error:
        manager.start(config_id)

    assert "한 번에 하나만 실행할 수 있습니다" in str(error.value)
    assert len(calls) == 1  # 두 번째는 process를 띄우지도 않습니다

    wait_for_process(manager).release()
    wait_for_finish(manager, first.job_id)


def test_start_allowed_after_previous_job_finished(
    manager, config_id, monkeypatch, fake_process_factory
):
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT)
    )

    first = manager.start(config_id)
    wait_for_finish(manager, first.job_id)
    second = manager.start(config_id)
    wait_for_finish(manager, second.job_id)

    assert first.job_id != second.job_id
    assert len(manager.list_jobs()) == 2


# --- 취소 -------------------------------------------------------------------


def test_cancel_transitions_to_cancelled_not_failed(
    manager, config_id, monkeypatch, fake_process_factory
):
    process = fake_process_factory(stdout="", exit_code=1, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    monkeypatch.setattr(runner, "terminate_tree", lambda proc: proc.release(1))

    started = manager.start(config_id)
    wait_for_spawn(manager)
    manager.cancel(started.job_id)
    record = wait_for_finish(manager, started.job_id)

    # 취소로 인한 비정상 종료 코드를 실패로 오해하면 안 됩니다.
    assert record.status == "cancelled"
    assert record.exit_code == 1
    assert "중지" in record.message
    assert record.orphan_note is not None


def test_cancel_before_spawn_does_not_start_the_process(manager, config_id, monkeypatch):
    """취소가 spawn보다 먼저 도착하면 process를 아예 띄우지 않아야 합니다."""

    spawned = []
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: spawned.append(a))

    # job thread가 spawn에 이르기 전에 취소가 도착하도록 build_argv에서 붙잡아 둡니다.
    release = __import__("threading").Event()
    original = runner.build_argv
    monkeypatch.setattr(runner, "build_argv", lambda path: (release.wait(5), original(path))[1])

    started = manager.start(config_id)
    manager.cancel(started.job_id)
    release.set()
    record = wait_for_finish(manager, started.job_id)

    assert record.status == "cancelled"
    assert spawned == []
    assert "시작하기 전에 취소" in record.message


def test_cancel_escalates_to_kill_after_grace(monkeypatch, fake_process_factory):
    manager = JobManager()
    process = fake_process_factory(block_until_signalled=True)
    killed = []
    monkeypatch.setattr(runner, "kill_tree", lambda proc: killed.append(proc))
    monkeypatch.setattr(runner, "TERMINATE_GRACE_SECONDS", 0.05)

    manager._escalate(process)

    assert killed == [process]


def test_cancel_unknown_job_raises_not_found(manager):
    with pytest.raises(JobNotFoundError):
        manager.cancel("0" * 32)


def test_cancel_finished_job_is_conflict(manager, config_id, monkeypatch, fake_process_factory):
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT)
    )
    started = manager.start(config_id)
    wait_for_finish(manager, started.job_id)

    with pytest.raises(JobConflictError):
        manager.cancel(started.job_id)


def test_cancel_on_windows_uses_taskkill_argv(monkeypatch, fake_process_factory):
    monkeypatch.setattr(runner, "IS_WINDOWS", True)
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    process = fake_process_factory()

    runner.terminate_tree(process)

    assert captured["args"] == ["taskkill", "/F", "/T", "/PID", "4242"]
    assert "shell" not in captured["kwargs"]


def test_cancel_on_posix_uses_process_group_signal(monkeypatch, fake_process_factory):
    monkeypatch.setattr(runner, "IS_WINDOWS", False)
    sent = []
    monkeypatch.setattr(runner, "signal_group", lambda pid, number: sent.append((pid, number)))
    process = fake_process_factory()

    runner.terminate_tree(process)
    runner.kill_tree(process)

    assert sent[0] == (4242, signal.SIGTERM)
    assert sent[1][1] == getattr(signal, "SIGKILL", signal.SIGTERM)


# --- 교착 회피 --------------------------------------------------------------


def test_reader_threads_do_not_deadlock_on_large_output(manager, config_id, monkeypatch):
    """양쪽 pipe를 동시에 읽지 않으면 OS buffer가 차면서 교착합니다.

    실제 subprocess를 써야만 증명되는 성질이라 여기서만 진짜 process를 띄웁니다.
    학습은 하지 않습니다.
    """

    script = (
        "import sys\n"
        "for i in range(4000):\n"
        "    sys.stdout.write('x' * 80 + '\\n')\n"
        "    sys.stderr.write('y' * 80 + '\\n')\n"
    )
    monkeypatch.setattr(runner, "build_argv", lambda path: [sys.executable, "-c", script])

    started = manager.start(config_id)
    record = wait_for_finish(manager, started.job_id, timeout=90)

    assert record.exit_code == 0
    logs = store.read_logs(started.job_id, limit=5000)
    stderr_lines = [line for line in logs["lines"] if line["stream"] == "stderr"]
    assert len(stderr_lines) == 4000


# --- 진행 상태 갱신 ----------------------------------------------------------


def progress_line(event: str, **fields) -> str:
    return json.dumps({"schema": "train.progress/1", "event": event, **fields})


def wait_for_step(manager: JobManager, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        step = manager.get(job_id).progress.get("step")
        if step is not None:
            return step
        time.sleep(0.01)
    raise AssertionError("batch 진행이 시간 안에 기록에 반영되지 않았습니다.")


def test_batch_progress_reaches_the_record_even_though_it_writes_no_log(
    manager, config_id, monkeypatch, fake_process_factory
):
    """batch 진행은 log 줄을 만들지 않습니다. 그래서 화면 갱신이 멈추면 안 됩니다.

    학습이 아직 도는 동안 확인합니다. 끝난 뒤에는 어차피 한 번 더 갱신되므로,
    끝난 기록만 보면 흐르는 동안 갱신되지 않는 문제를 잡지 못합니다.
    """

    stderr = "".join(
        line + "\n"
        for line in [
            progress_line("epoch_started", epoch=1, epochs=2),
            progress_line("step_progress", epoch=1, epochs=2, phase="train", step=7, total_steps=20),
        ]
    )
    process = fake_process_factory(stderr=stderr, block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)

    started = manager.start(config_id)
    step = wait_for_step(manager, started.job_id)
    process.release(0)
    wait_for_finish(manager, started.job_id)

    assert step == {"phase": "train", "step": 7, "total_steps": 20, "percent": 35.0}
    # 5초에 한 번씩 나오는 event가 log를 채우면 경고와 오류가 묻힙니다.
    logs = store.read_logs(started.job_id)
    assert not any("step_progress" in line["text"] for line in logs["lines"])


# --- 저장과 복구 ------------------------------------------------------------


def test_records_are_persisted_and_reloaded(
    manager, config_id, monkeypatch, fake_process_factory, isolated_repo
):
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT)
    )
    started = manager.start(config_id)
    wait_for_finish(manager, started.job_id)

    reloaded = JobManager()
    reloaded.load()

    assert [record.job_id for record in reloaded.list_jobs()] == [started.job_id]
    assert reloaded.get(started.job_id).status == "succeeded"


def test_stale_running_record_becomes_interrupted_on_load(
    manager, config_id, monkeypatch, fake_process_factory
):
    """서버가 죽고 학습 process도 사라졌을 때입니다."""

    process = fake_process_factory(block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    monkeypatch.setattr(runner, "process_alive", lambda _pid: False)
    started = manager.start(config_id)

    reloaded = JobManager()
    reloaded.load()

    reclaimed = reloaded.get(started.job_id)
    assert reclaimed.status == "interrupted"
    assert "잃었습니다" in (reclaimed.message or "")
    process.release()
    wait_for_finish(manager, started.job_id)


def test_load_says_so_when_the_training_process_outlived_the_server(
    manager, config_id, monkeypatch, fake_process_factory
):
    """POSIX에서 학습은 별도 session으로 떠서 서버가 죽어도 같이 죽지 않습니다.

    그때 "상태를 잃었습니다"만 보여 주면 학습이 끝난 줄 알고 새로 시작하거나
    process를 죽입니다. checkpoint는 학습이 다 끝난 뒤 한 번에 저장되므로
    그렇게 죽이면 그때까지 학습한 것이 전부 사라집니다.
    """

    process = fake_process_factory(block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    monkeypatch.setattr(runner, "process_alive", lambda pid: pid == process.pid)
    started = manager.start(config_id)
    # process가 떴다는 것은 PID까지 기록에 남았다는 뜻입니다.
    wait_for_spawn(manager)

    reloaded = JobManager()
    reloaded.load()

    reclaimed = reloaded.get(started.job_id)
    assert reclaimed.status == "interrupted"
    assert str(process.pid) in (reclaimed.message or "")
    assert "아직" in (reclaimed.message or "")
    process.release()
    wait_for_finish(manager, started.job_id)


def test_job_record_file_is_utf8_lf(manager, config_id, monkeypatch, fake_process_factory, isolated_repo):
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT)
    )
    started = manager.start(config_id)
    wait_for_finish(manager, started.job_id)

    raw = (isolated_repo / "artifacts" / "web" / "jobs" / started.job_id / "record.json").read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw


def test_corrupt_record_is_skipped(manager, isolated_repo):
    directory = isolated_repo / "artifacts" / "web" / "jobs" / ("a" * 32)
    directory.mkdir(parents=True)
    (directory / "record.json").write_text("{broken", encoding="utf-8")

    assert store.load_all_records() == []


@pytest.mark.parametrize("bad", ("../../etc/passwd", "..%2f..%2fx", "zz", "A" * 32, ""))
def test_store_rejects_bad_job_id(bad):
    with pytest.raises(JobNotFoundError):
        store.job_directory(bad)


def test_logs_are_masked_before_reaching_disk(
    manager, config_id, monkeypatch, fake_process_factory
):
    process = fake_process_factory(
        stdout=TRAIN_SUCCESS_STDOUT, stderr="token=AKIAIOSFODNN7EXAMPLE 로 실패\n"
    )
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)

    started = manager.start(config_id)
    wait_for_finish(manager, started.job_id)

    logs = store.read_logs(started.job_id)
    assert all("AKIAIOSFODNN7EXAMPLE" not in line["text"] for line in logs["lines"])


def test_logs_paginate_with_cursor(manager, config_id, monkeypatch, fake_process_factory):
    stderr = "".join(f"line {index}\n" for index in range(30))
    monkeypatch.setattr(
        runner,
        "spawn",
        lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT, stderr=stderr),
    )
    started = manager.start(config_id)
    wait_for_finish(manager, started.job_id)

    first = store.read_logs(started.job_id, after=0, limit=10)
    second = store.read_logs(started.job_id, after=first["next"], limit=10)

    assert len(first["lines"]) == 10
    assert first["complete"] is False
    assert second["lines"][0]["seq"] == first["lines"][-1]["seq"] + 1


# --- 기록 삭제 ---------------------------------------------------------------


def test_delete_removes_only_this_gui_record(
    manager, config_id, monkeypatch, fake_process_factory, isolated_repo
):
    """지우는 것은 ``artifacts/web/jobs/<job_id>/`` 하나뿐입니다.

    checkpoint와 학습 결과 폴더는 train이 만든 산출물이라 이 화면이 지우지 않습니다.
    설정 파일도 남깁니다. 대기열이나 이어서 학습이 아직 그것을 가리킬 수 있습니다.
    """

    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT)
    )
    started = manager.start(config_id)
    wait_for_finish(manager, started.job_id)

    output = isolated_repo / "artifacts" / "experiments" / "completed" / "test-run"
    output.mkdir(parents=True, exist_ok=True)
    (output / "best_checkpoint.pt").write_text("weights", encoding="utf-8")
    config_path = isolated_repo / "artifacts" / "web" / "configs" / f"{config_id}.json"

    manager.delete(started.job_id)

    assert not store.job_directory(started.job_id).exists()
    assert [record.job_id for record in manager.list_jobs()] == []
    assert (output / "best_checkpoint.pt").read_text(encoding="utf-8") == "weights"
    assert config_path.exists()


def test_delete_refuses_while_the_job_is_running(
    manager, config_id, monkeypatch, fake_process_factory
):
    """돌고 있는 학습의 기록을 지우면 그 학습을 관리할 방법이 사라집니다."""

    process = fake_process_factory(block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    started = manager.start(config_id)
    wait_for_spawn(manager)

    with pytest.raises(JobConflictError):
        manager.delete(started.job_id)

    process.release()
    wait_for_finish(manager, started.job_id)
    assert store.job_directory(started.job_id).exists()


def test_delete_rejects_ids_that_are_not_job_ids(manager):
    """경로 조작은 디스크에 닿기 전에 막습니다."""

    with pytest.raises(JobNotFoundError):
        manager.delete("../../../etc/passwd")


def test_delete_endpoint_reports_conflict_and_removes_from_listing(
    client, valid_payload, monkeypatch, fake_process_factory
):
    created = client.post("/api/train/configs", json=valid_payload).json()
    process = fake_process_factory(block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    started = client.post("/api/train/jobs", json={"config_id": created["config_id"]}).json()

    assert client.delete(f"/api/train/jobs/{started['job_id']}").status_code == 409

    process.release()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if client.get(f"/api/train/jobs/{started['job_id']}").json()["status"] != "running":
            break
        time.sleep(0.02)

    assert client.delete(f"/api/train/jobs/{started['job_id']}").status_code == 200
    assert client.get("/api/train/jobs").json()["jobs"] == []
    assert client.get(f"/api/train/jobs/{started['job_id']}").status_code == 404


def _link_directory(link: Path, target: Path) -> bool:
    """``link``가 ``target``을 가리키는 directory 연결을 만듭니다. 못 만들면 False입니다.

    Windows에서는 junction을 먼저 씁니다. ``shutil.rmtree``는 맨 위가 symlink면
    거부하지만 junction은 따라 들어가 그 안을 지우기 때문에, 실제로 위험한 쪽이
    junction입니다. symlink만으로 시험하면 고치기 전에도 test가 통과합니다.
    """

    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        if result.returncode == 0 and link.exists():
            return True
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


def test_delete_does_not_follow_a_link_out_of_the_repository(manager, isolated_repo):
    """jobs 루트가 저장소 밖을 가리키면 그 안을 지우지 않습니다.

    이름은 32자리 16진수만 통과하므로 이름으로는 빠져나갈 수 없습니다. ``rmtree``도
    맨 위가 link면 거부합니다. 그런데 ``artifacts/web/jobs`` **자체**가 link면 그
    아래 job directory는 진짜 directory라서 아무것도 막지 못하고, 저장소 밖이
    통째로 지워집니다. Windows에서 junction은 권한 없이도 만들 수 있어 실제로
    일어날 수 있는 배치입니다.
    """

    outside = isolated_repo.parent / "outside-target"
    job_id = "a" * 32
    (outside / job_id).mkdir(parents=True, exist_ok=True)
    (outside / job_id / "record.json").write_text("남의 기록", encoding="utf-8")

    web_state = isolated_repo / "artifacts" / "web"
    web_state.mkdir(parents=True, exist_ok=True)
    if not _link_directory(web_state / "jobs", outside):
        pytest.skip("이 환경에서는 directory 연결을 만들 수 없습니다.")

    with pytest.raises(JobNotFoundError):
        store.delete_record(job_id)

    assert (outside / job_id / "record.json").exists()


def test_delete_holds_the_evaluation_lock_so_a_start_cannot_slip_in(evaluation_runner):
    """평가 확인과 삭제 사이에 평가가 시작되면 지운 기록이 되살아납니다.

    evaluator는 끝나면서 자기가 들고 있던 record를 다시 저장합니다. 그때 log는 이미
    지워졌으므로 성공 응답 뒤에 반쪽짜리 기록만 남습니다. 그래서 확인과 삭제가 같은
    잠금 안에서 일어나야 합니다.
    """

    job_id = "b" * 32
    acquired: list[bool] = []

    with evaluation_runner.hold_for_delete(job_id):
        # 같은 thread에서는 RLock이 다시 잡히므로 다른 thread에서 확인합니다.
        def probe() -> None:
            acquired.append(evaluation_runner._lock.acquire(blocking=False))
            if acquired[-1]:
                evaluation_runner._lock.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(5)

    assert acquired == [False]


def test_delete_guard_refuses_while_that_job_is_being_evaluated(evaluation_runner):
    evaluation_runner._state = {"status": "running", "job_id": "c" * 32}

    with pytest.raises(JobConflictError):
        with evaluation_runner.hold_for_delete("c" * 32):
            pass

    # 다른 학습의 평가가 도는 것은 이 기록을 지우는 데 상관이 없습니다.
    with evaluation_runner.hold_for_delete("d" * 32):
        pass


def test_delete_does_not_follow_a_link_to_another_components_output(manager, isolated_repo):
    """jobs 루트가 저장소 **안**의 다른 곳을 가리켜도 거부합니다.

    저장소 밖만 막으면 부족합니다. jobs 루트가 ``artifacts/experiments/completed``
    같은 train 산출물을 가리키면, 그 아래 32자리 이름의 directory는 형식 검사도
    통과하고 저장소 안이기도 해서 checkpoint가 통째로 지워집니다. 다른 component가
    만든 산출물은 이 화면이 지우지 않습니다.
    """

    job_id = "e" * 32
    train_output = isolated_repo / "artifacts" / "experiments" / "completed"
    (train_output / job_id).mkdir(parents=True, exist_ok=True)
    (train_output / job_id / "best_checkpoint.pt").write_text("가중치", encoding="utf-8")

    web_state = isolated_repo / "artifacts" / "web"
    web_state.mkdir(parents=True, exist_ok=True)
    if not _link_directory(web_state / "jobs", train_output):
        pytest.skip("이 환경에서는 directory 연결을 만들 수 없습니다.")

    with pytest.raises(JobNotFoundError):
        store.delete_record(job_id)

    assert (train_output / job_id / "best_checkpoint.pt").exists()


def test_evaluation_cannot_start_for_a_record_that_was_deleted(
    client, valid_payload, monkeypatch, fake_process_factory
):
    """삭제가 먼저 끝났으면 평가는 시작하지 않습니다.

    평가 POST가 record를 읽고 config를 만드는 동안 lock을 쥐지 않으면, 그 사이에
    DELETE가 idle 상태를 보고 기록을 지웁니다. 뒤늦게 시작한 평가는 끝나면서 손에 든
    stale record를 다시 저장해 빈 log와 함께 기록을 되살립니다. 그래서 record를 읽는
    것부터 시작까지가 삭제와 같은 잠금 안에 있어야 합니다.
    """

    created = client.post("/api/train/configs", json=valid_payload).json()
    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout=TRAIN_SUCCESS_STDOUT)
    )
    started = client.post("/api/train/jobs", json={"config_id": created["config_id"]}).json()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if client.get(f"/api/train/jobs/{started['job_id']}").json()["status"] == "succeeded":
            break
        time.sleep(0.02)

    assert client.delete(f"/api/train/jobs/{started['job_id']}").status_code == 200

    assert client.post(f"/api/train/jobs/{started['job_id']}/evaluate", json={}).status_code == 404
    # 되살아나지 않았는지 확인합니다.
    assert client.get("/api/train/jobs").json()["jobs"] == []
