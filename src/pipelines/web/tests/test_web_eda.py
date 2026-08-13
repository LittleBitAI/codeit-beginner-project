"""EDA를 화면에서 시작하고 결과를 받아 오는 부분입니다.

data pipeline은 subprocess로만 부르므로, 여기서는 그 호출이 만드는 config와
runner가 들고 있는 상태만 확인합니다.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from src.pipelines.web import datasets
from src.pipelines.web.data_jobs import EdaRunner, PreparationRunner
from src.pipelines.web.errors import JobConflictError, WebValidationError
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


REPORT = {"schema_version": "1.0", "classes": {"class_count": 118}}


def eda_result(ok: bool = True) -> dict:
    return {
        "ok": ok,
        "supported": True,
        "exit_code": 0 if ok else 1,
        "artifacts": (
            {
                **{key: f"artifacts/p/{key}.json" for key in DATA_ARTIFACT_KEYS},
                "eda_report_uri": "artifacts/p/eda/report.json",
            }
            if ok
            else {}
        ),
        "summary": {"mode": "eda", "class_count": 118},
        "message": "EDA 완료" if ok else "실패",
    }


def wait_until_done(runner: EdaRunner, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = runner.status()
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("EDA가 시간 안에 끝나지 않았습니다.")


@pytest.fixture
def selected(monkeypatch):
    """전처리 dataset을 하나 고른 상태로 둡니다."""

    data = {key: f"artifacts/p/{key}.json" for key in DATA_ARTIFACT_KEYS}
    monkeypatch.setattr(
        datasets,
        "load_selection",
        lambda: {"directory": "artifacts/p", "selected_at": "2026-01-01T00:00:00Z", "data": data},
    )
    return data


# --- config ---------------------------------------------------------------


def test_the_config_asks_data_to_do_eda_and_nothing_else(selected):
    """준비와 같은 stage를 쓰므로, 준비를 켜지 않았다는 것이 중요합니다."""

    config = datasets.build_eda_config(selected, image_sample=25)

    assert config["data"] == {"eda": True, "eda_image_sample": 25, "overwrite": False}
    assert "prepare" not in config["data"]
    assert config["inputs"]["data"] == selected
    assert config["execution"]["mode"] == "real"


@pytest.mark.parametrize("bad", (0, -1, "40", True))
def test_an_unusable_image_sample_is_rejected_before_the_subprocess(selected, bad):
    with pytest.raises(WebValidationError) as error:
        datasets.build_eda_config(selected, image_sample=bad)

    assert error.value.as_list()[0]["field"] == "image_sample"


# --- 실행 -----------------------------------------------------------------


def test_a_finished_run_carries_the_report_itself(monkeypatch, selected):
    """URI만 주면 화면이 그 파일을 또 받아 와야 합니다."""

    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config, on_progress_line=None, mode="prepare": eda_result()
    )
    monkeypatch.setattr(datasets, "read_eda_report", lambda uri: dict(REPORT))
    runner = EdaRunner()

    runner.start({"image_sample": 10})
    state = wait_until_done(runner)

    assert state["status"] == "succeeded"
    assert state["report"] == REPORT
    assert state["artifacts"]["eda_report_uri"].endswith("eda/report.json")


def test_a_successful_eda_is_not_mistaken_for_an_unsupported_pipeline(monkeypatch, selected):
    """준비와 같은 함수를 쓰므로, 그 함수가 "prepare"만 정상으로 보면 안 됩니다.

    이걸 놓치면 모든 정상 EDA 실행이 "기능 미지원" 실패로 뒤집힙니다.
    """

    captured: dict[str, Any] = {}

    def fake_run_process(process, on_progress_line, timeout):
        return (json.dumps({"status": "ok", "artifacts": {}, "summary": {"mode": "eda"}, "message": "완료"}), "", 0)

    monkeypatch.setattr(datasets.runner, "spawn", lambda *a, **k: object())
    monkeypatch.setattr(datasets, "_run_prepare_process", fake_run_process)
    monkeypatch.setattr(datasets, "_now_text", lambda: "2026-01-01T00:00:00Z")
    captured["result"] = datasets.prepare_dataset({"data": {"eda": True}}, mode="eda")

    assert captured["result"]["ok"] is True
    assert captured["result"]["supported"] is True


def test_a_report_from_another_dataset_is_not_shown_as_this_one(monkeypatch, selected):
    """dataset을 바꾸면 앞선 결과가 새 dataset의 숫자처럼 보이면 안 됩니다."""

    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config, on_progress_line=None, mode="prepare": eda_result()
    )
    monkeypatch.setattr(datasets, "read_eda_report", lambda uri: dict(REPORT))
    runner = EdaRunner()
    runner.start({})
    assert wait_until_done(runner)["report"] == REPORT

    monkeypatch.setattr(
        datasets, "load_selection", lambda: {"directory": "artifacts/other", "data": {}}
    )

    state = runner.status()

    assert state["stale"] is True
    assert state["report"] is None


def test_a_report_from_before_the_dataset_was_rebuilt_is_not_shown(monkeypatch, selected):
    """같은 폴더에 원본을 다시 준비하면 안의 내용이 바뀝니다.

    폴더 이름만 대조하면 옛 숫자가 최신 결과처럼 남습니다.
    """

    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config, on_progress_line=None, mode="prepare": eda_result()
    )
    monkeypatch.setattr(datasets, "read_eda_report", lambda uri: dict(REPORT))
    runner = EdaRunner()
    runner.start({})
    wait_until_done(runner)

    monkeypatch.setattr(
        datasets,
        "load_selection",
        lambda: {"directory": "artifacts/p", "selected_at": "2026-02-02T00:00:00Z", "data": {}},
    )

    state = runner.status()

    assert state["stale"] is True
    assert state["report"] is None


def test_prepare_and_eda_do_not_run_at_the_same_time(monkeypatch, selected):
    """둘 다 같은 전처리 폴더를 건드립니다. 겹치면 읽는 중에 파일이 바뀝니다."""

    release = threading.Event()
    monkeypatch.setattr(
        datasets,
        "prepare_dataset",
        lambda config, on_progress_line=None, mode="prepare": (release.wait(5), eda_result())[1],
    )
    monkeypatch.setattr(
        datasets,
        "build_prepare_config",
        lambda *a, **k: {"data": {"seed": 42, "overwrite": False}, "storage": {"backend": "local"}},
    )
    eda = EdaRunner()
    preparation = PreparationRunner()
    eda.start({})

    try:
        with pytest.raises(JobConflictError):
            preparation.start({"split_ratio": "8:2"})
    finally:
        release.set()
    wait_until_done(eda)


def test_a_failed_run_carries_no_report(monkeypatch, selected):
    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config, on_progress_line=None, mode="prepare": eda_result(ok=False)
    )
    runner = EdaRunner()

    runner.start({})
    state = wait_until_done(runner)

    assert state["status"] == "failed"
    assert state["report"] is None


def test_eda_without_a_selected_dataset_is_refused_before_starting(monkeypatch):
    monkeypatch.setattr(datasets, "load_selection", lambda: None)
    runner = EdaRunner()

    with pytest.raises(WebValidationError) as error:
        runner.start({})

    assert error.value.as_list()[0]["field"] == "dataset"
    assert runner.status()["status"] == "idle"


def test_a_second_run_is_refused_while_one_is_going(monkeypatch, selected):
    release = threading.Event()
    monkeypatch.setattr(
        datasets,
        "prepare_dataset",
        lambda config, on_progress_line=None, mode="prepare": (release.wait(5), eda_result())[1],
    )
    runner = EdaRunner()
    runner.start({})

    try:
        with pytest.raises(JobConflictError):
            runner.start({})
    finally:
        release.set()
    wait_until_done(runner)


def test_a_local_report_is_read_through_the_repository_root(isolated_repo):
    """local artifact URI는 저장소 root 기준입니다.

    storage root를 `artifacts`로 두면 `artifacts/artifacts/…`를 찾고, 저장소 안이지만
    `artifacts/` 밖에 있는 전처리 폴더는 아예 읽지 못합니다.
    """

    from src.pipelines.web.paths import repository_root

    target = repository_root() / "datasets" / "processed" / "v9" / "eda"
    target.mkdir(parents=True)
    (target / "report.json").write_text(json.dumps(REPORT), encoding="utf-8")

    assert datasets.read_eda_report("datasets/processed/v9/eda/report.json") == REPORT
    assert datasets.read_eda_report("datasets/processed/v9/eda/missing.json") is None
    # 저장소 밖은 읽지 않습니다.
    assert datasets.read_eda_report("../outside.json") is None


# --- route ----------------------------------------------------------------


def test_the_eda_route_starts_idle_and_starts_a_run(client, monkeypatch, selected):
    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config, on_progress_line=None, mode="prepare": eda_result()
    )
    monkeypatch.setattr(datasets, "read_eda_report", lambda uri: dict(REPORT))

    assert client.get("/api/data/eda").json()["eda"]["status"] == "idle"

    response = client.post("/api/data/eda", json={"image_sample": 10})

    assert response.status_code == 202
    eda = response.json()["eda"]
    assert eda["image_sample"] == 10
    # 진행 줄을 아직 못 봤으므로 진행률을 지어내지 않습니다.
    assert eda["progress"]["available"] is False


def test_the_route_refuses_an_image_sample_outside_the_allowed_range(client):
    assert client.post("/api/data/eda", json={"image_sample": 0}).status_code == 422
