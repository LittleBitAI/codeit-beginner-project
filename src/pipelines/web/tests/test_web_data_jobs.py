"""데이터 준비를 백그라운드로 한 번에 하나만 실행하는지 확인합니다."""

from __future__ import annotations

import time

import pytest

from src.pipelines.web import datasets
from src.pipelines.web.data_jobs import PreparationRunner
from src.pipelines.web.errors import JobConflictError, WebValidationError
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


def prepared(ok: bool = True, supported: bool = True) -> dict:
    return {
        "ok": ok,
        "supported": supported,
        "exit_code": 0 if ok else 1,
        "artifacts": {key: f"artifacts/p/{key}.json" for key in DATA_ARTIFACT_KEYS} if ok else {},
        "summary": {"mode": "prepare", "split_ratio": "8:2", "train_images": 8},
        "message": "준비 완료" if ok else "실패",
    }


def wait_until_done(runner: PreparationRunner, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = runner.status()
        if state["status"] not in ("running",):
            return state
        time.sleep(0.02)
    raise AssertionError("준비가 시간 안에 끝나지 않았습니다.")


@pytest.fixture
def runner(preparation_runner):
    return preparation_runner


def test_idle_before_anything_runs(runner):
    assert runner.status() == {"status": "idle"}


def test_successful_preparation_is_selected_automatically(runner, monkeypatch):
    monkeypatch.setattr(datasets, "prepare_dataset", lambda config: prepared())

    runner.start({"split_ratio": "8:2"})
    state = wait_until_done(runner)

    assert state["status"] == "succeeded"
    assert state["selected"] is True
    assert state["summary"]["train_images"] == 8
    # 끝나자마자 현재 데이터셋이 되어야 합니다.
    selection = datasets.load_selection()
    assert selection is not None
    assert selection["origin"] == "prepared"
    assert set(selection["data"]) == set(DATA_ARTIFACT_KEYS)


def test_successful_preparation_keeps_test_manifest_for_later_submission(runner, monkeypatch):
    result = prepared()
    result["artifacts"]["test_manifest_uri"] = "artifacts/p/test_manifest.json"
    result["summary"].update(test_manifest_images=842, test_images_used=0)
    monkeypatch.setattr(datasets, "prepare_dataset", lambda config: result)

    runner.start({"split_ratio": "8:2"})
    state = wait_until_done(runner)

    assert state["selected"] is True
    selection = datasets.load_selection()
    assert selection["data"]["test_manifest_uri"] == "artifacts/p/test_manifest.json"
    assert selection["preparation"]["test_manifest_images"] == 842


def test_failed_preparation_does_not_change_the_selection(runner, monkeypatch):
    monkeypatch.setattr(datasets, "prepare_dataset", lambda config: prepared(ok=False))

    runner.start({"split_ratio": "9:1"})
    state = wait_until_done(runner)

    assert state["status"] == "failed"
    assert state["selected"] is False
    assert datasets.load_selection() is None


def test_unsupported_pipeline_is_reported_honestly(runner, monkeypatch):
    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config: prepared(ok=False, supported=False)
    )

    runner.start({"split_ratio": "8:2"})
    state = wait_until_done(runner)

    assert state["status"] == "failed"
    assert state["supported"] is False


def test_second_start_is_rejected_while_running(runner, monkeypatch):
    import threading

    release = threading.Event()

    def slow(config):
        release.wait(5)
        return prepared()

    monkeypatch.setattr(datasets, "prepare_dataset", slow)
    runner.start({"split_ratio": "8:2"})

    with pytest.raises(JobConflictError) as error:
        runner.start({"split_ratio": "9:1"})

    assert "이미 데이터 준비가 실행 중입니다" in str(error.value)
    release.set()
    wait_until_done(runner)


def test_bad_request_is_rejected_before_starting_a_thread(runner, monkeypatch):
    called = []
    monkeypatch.setattr(datasets, "prepare_dataset", lambda config: called.append(config))

    with pytest.raises(WebValidationError):
        runner.start({"split_ratio": "7:3"})

    assert called == []
    assert runner.status() == {"status": "idle"}


def test_unexpected_error_does_not_leave_it_running(runner, monkeypatch):
    def explode(config):
        raise RuntimeError("예기치 못한 오류")

    monkeypatch.setattr(datasets, "prepare_dataset", explode)

    runner.start({"split_ratio": "8:2"})
    state = wait_until_done(runner)

    assert state["status"] == "failed"
    assert "RuntimeError" in state["message"]


def test_start_is_allowed_again_after_finishing(runner, monkeypatch):
    monkeypatch.setattr(datasets, "prepare_dataset", lambda config: prepared())

    runner.start({"split_ratio": "8:2"})
    wait_until_done(runner)
    runner.start({"split_ratio": "9:1"})
    state = wait_until_done(runner)

    assert state["status"] == "succeeded"
    assert state["split_ratio"] == "9:1"
