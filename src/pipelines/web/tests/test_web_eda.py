"""EDA를 화면에서 시작하고 결과를 받아 오는 부분입니다.

data pipeline은 subprocess로만 부르므로, 여기서는 그 호출이 만드는 config와
runner가 들고 있는 상태만 확인합니다.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.pipelines.web import datasets
from src.pipelines.web.data_jobs import EdaRunner
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
        datasets, "load_selection", lambda: {"directory": "artifacts/p", "data": data}
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
        datasets, "prepare_dataset", lambda config, on_progress_line=None: eda_result()
    )
    monkeypatch.setattr(datasets, "read_eda_report", lambda uri: dict(REPORT))
    runner = EdaRunner()

    runner.start({"image_sample": 10})
    state = wait_until_done(runner)

    assert state["status"] == "succeeded"
    assert state["report"] == REPORT
    assert state["artifacts"]["eda_report_uri"].endswith("eda/report.json")


def test_a_failed_run_carries_no_report(monkeypatch, selected):
    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config, on_progress_line=None: eda_result(ok=False)
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
        lambda config, on_progress_line=None: (release.wait(5), eda_result())[1],
    )
    runner = EdaRunner()
    runner.start({})

    try:
        with pytest.raises(JobConflictError):
            runner.start({})
    finally:
        release.set()
    wait_until_done(runner)


# --- route ----------------------------------------------------------------


def test_the_eda_route_starts_idle_and_starts_a_run(client, monkeypatch, selected):
    monkeypatch.setattr(
        datasets, "prepare_dataset", lambda config, on_progress_line=None: eda_result()
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
