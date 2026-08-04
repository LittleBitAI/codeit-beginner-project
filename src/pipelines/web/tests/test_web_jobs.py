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
import time

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


def test_build_argv_supports_the_data_stage():
    assert runner.build_argv("artifacts/web/configs/abc.json", "data")[-2:] == ["--only", "data"]


@pytest.mark.parametrize("stage", ("evaluate", "registry", "web", "", "train; rm -rf /", None))
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

    manager._process.release()
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
    """서버가 죽으면 running 기록만 남고 OS process는 사라집니다."""

    process = fake_process_factory(block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    started = manager.start(config_id)

    reloaded = JobManager()
    reloaded.load()

    assert reloaded.get(started.job_id).status == "interrupted"
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
