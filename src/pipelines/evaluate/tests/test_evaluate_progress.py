"""`evaluate.progress/1` 진행 로그가 계약대로 stderr에만 나가는지 확인합니다.

emitter test는 실제 stderr 대신 주입한 stream을 써서, throttle 규칙과 예외
안전(닫힌 pipe에 써도 조용히 넘어가는지)을 직접 확인합니다.

이 pipeline에서 회귀 위험이 가장 큰 지점은 stdout 오염입니다. `COCOeval`이
stdout에 쓰기 때문에 그 호출을 `redirect_stdout`으로 감싸 두었고 web이 그
subprocess 로그를 파싱하므로, 진행 로그가 stdout에 한 글자도 더하면 안 됩니다.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from conftest import write_json
from src.pipelines.evaluate import pipeline
from src.pipelines.evaluate import progress as progress_module
from src.pipelines.evaluate.pipeline import run
from src.pipelines.evaluate.progress import SCHEMA, ProgressEmitter


# --- 도우미 ---------------------------------------------------------------


def progress_events(captured_stderr: str) -> list[dict[str, Any]]:
    """JSON이 아닌 줄은 건너뛰고 `evaluate.progress/1` event만 모읍니다."""

    events = []
    for line in captured_stderr.splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == SCHEMA:
            events.append(payload)
    return events


class FakeClock:
    """`time.monotonic()`을 test가 직접 움직이게 하는 대역입니다."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def emitter_with_clock(monkeypatch) -> tuple[ProgressEmitter, io.StringIO, FakeClock]:
    clock = FakeClock()
    monkeypatch.setattr(progress_module, "time", clock)
    stream = io.StringIO()
    return ProgressEmitter(stream), stream, clock


def predict_stages(stream: io.StringIO) -> list[tuple[int, int]]:
    return [(event["done"], event["total"]) for event in progress_events(stream.getvalue())]


def add_test_manifest(base_config: dict, repository_root: Path) -> None:
    write_json(
        repository_root / "data/test/instances.json",
        {
            "images": [
                {"id": 20, "file_name": "0020.jpg", "width": 100, "height": 100},
                {"id": 10, "file_name": "0010.jpg", "width": 100, "height": 100},
            ],
            "annotations": [],
            "categories": [{"id": 3, "name": "pill-a"}, {"id": 7, "name": "pill-b"}],
        },
    )
    base_config["inputs"]["data"]["test_manifest_uri"] = "data/test/instances.json"


# --- 평가 실행이 내보내는 event -------------------------------------------


def test_run_emits_the_contract_events_on_stderr_and_leaves_stdout_empty(
    base_config: dict, repository_root: Path, capsys
):
    result = run(base_config)
    captured = capsys.readouterr()

    assert result["status"] == "ok", result["message"]
    # 계약 2번: stdout에는 아무것도 더하지 않습니다. COCOeval 로그를 web이
    # 파싱하므로 이 pipeline에서 특히 중요합니다.
    assert captured.out == ""

    events = progress_events(captured.err)
    names = [event["event"] for event in events]
    assert names == ["evaluate_started", "metrics_computed", "evaluate_completed"]
    for event in events:
        datetime.strptime(event["ts"], "%Y-%m-%dT%H:%M:%S.%fZ")

    started = events[0]
    assert started == {
        "schema": SCHEMA,
        "event": "evaluate_started",
        "run_id": "evaluate-0001",
        "device": "cpu",
        "validation_images": 2,
        # test manifest가 없으면 0입니다.
        "test_images": 0,
        "ts": started["ts"],
    }
    metrics = result["summary"]["metrics"]
    assert events[1] == {
        "schema": SCHEMA,
        "event": "metrics_computed",
        "mAP": metrics["mAP"],
        "mAP50": metrics["mAP50"],
        "mAP75": metrics["mAP75"],
        "ts": events[1]["ts"],
    }
    assert events[2] == {
        "schema": SCHEMA,
        "event": "evaluate_completed",
        "validation_images": 2,
        "test_images": 0,
        "ts": events[2]["ts"],
    }


def test_competition_run_reports_both_inference_stages_and_the_submission(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    add_test_manifest(base_config, repository_root)
    base_config["evaluate"].pop("predictions_input_uri")

    def fake_predict_groups(
        store, record_groups, *, checkpoint_uri, device, seed, on_progress=None
    ):
        groups = []
        for index, records in enumerate(record_groups):
            for order, record in enumerate(records):
                if on_progress is not None:
                    on_progress(index, order + 1, len(records))
            groups.append(
                [
                    {
                        "image_id": record["image_id"],
                        "image_key": record["image_key"],
                        "category_id": 1 if index == 0 else 3,
                        "bbox": [10.0, 10.0, 20.0, 20.0],
                        "score": 0.9,
                    }
                    for record in records
                ]
            )
        return groups

    monkeypatch.setattr(pipeline, "predict_record_groups_with_checkpoint", fake_predict_groups)

    result = run(base_config)
    captured = capsys.readouterr()

    assert result["status"] == "ok", result["message"]
    assert captured.out == ""

    events = progress_events(captured.err)
    assert events[0]["event"] == "evaluate_started"
    assert events[0]["test_images"] == 2
    # 가장 오래 걸리는 구간을 구분하기 위해 stage로 validation과 test를 나눕니다.
    assert [
        (event["stage"], event["done"], event["total"])
        for event in events
        if event["event"] == "predict_progress"
    ] == [
        ("validation", 1, 2),
        ("validation", 2, 2),
        ("test", 1, 2),
        ("test", 2, 2),
    ]
    written = [event for event in events if event["event"] == "submission_written"]
    submission_rows = (
        (repository_root / result["artifacts"]["submission_uri"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert [event["rows"] for event in written] == [len(submission_rows) - 1]
    assert events[-1] == {
        "schema": SCHEMA,
        "event": "evaluate_completed",
        "validation_images": 2,
        "test_images": 2,
        "ts": events[-1]["ts"],
    }


def test_progress_stream_stays_silent_for_the_dummy_execution(capsys):
    from src.pipelines.evaluate import run as public_run

    result = public_run({"execution": {"mode": "dummy"}})
    captured = capsys.readouterr()

    assert result["status"] == "ok"
    assert captured.out == ""
    assert progress_events(captured.err) == []


def test_failed_run_keeps_stdout_empty(base_config: dict, capsys):
    base_config["evaluate"]["predictions_input_uri"] = "data/val/missing.json"

    result = run(base_config)
    captured = capsys.readouterr()

    assert result["status"] == "error"
    assert captured.out == ""


# --- 출력량 제한 -----------------------------------------------------------


def test_predict_progress_skips_lines_below_two_percent_within_one_second(monkeypatch):
    emitter, stream, clock = emitter_with_clock(monkeypatch)

    # 첫 줄은 전체 개수를 알리기 위해 바로 나갑니다.
    emitter.predict_progress("test", 1, 1000)
    # 2%(20장)에 못 미치고 1초도 지나지 않은 줄은 버립니다.
    for done in range(2, 21):
        clock.now += 0.01
        emitter.predict_progress("test", done, 1000)
    # 21번째에서 마지막 event 대비 20장을 채웁니다.
    emitter.predict_progress("test", 21, 1000)

    assert predict_stages(stream) == [(1, 1000), (21, 1000)]


def test_predict_progress_emits_after_one_second_without_enough_progress(monkeypatch):
    emitter, stream, clock = emitter_with_clock(monkeypatch)

    emitter.predict_progress("test", 1, 1000)
    clock.now += 0.5
    emitter.predict_progress("test", 2, 1000)
    clock.now += 0.6
    emitter.predict_progress("test", 3, 1000)

    assert predict_stages(stream) == [(1, 1000), (3, 1000)]


def test_predict_progress_always_emits_the_last_image(monkeypatch):
    emitter, stream, clock = emitter_with_clock(monkeypatch)

    emitter.predict_progress("test", 1, 1000)
    clock.now += 0.01
    emitter.predict_progress("test", 2, 1000)
    clock.now += 0.01
    emitter.predict_progress("test", 1000, 1000)

    assert predict_stages(stream) == [(1, 1000), (1000, 1000)]


# --- 지표는 계산되지 않으면 null ------------------------------------------


def test_metrics_computed_keeps_uncomputed_and_non_finite_metrics_null():
    stream = io.StringIO()
    emitter = ProgressEmitter(stream)

    emitter.metrics_computed({"mAP": None, "mAP50": float("nan"), "mAP75": 0.5})

    assert progress_events(stream.getvalue()) == [
        {
            "schema": SCHEMA,
            "event": "metrics_computed",
            "mAP": None,
            "mAP50": None,
            "mAP75": 0.5,
            "ts": progress_events(stream.getvalue())[0]["ts"],
        }
    ]


# --- 예외 안전과 process 경계 ---------------------------------------------


def test_progress_emitter_does_not_raise_when_the_reader_closed_the_pipe():
    class ClosedPipe:
        def write(self, line: str) -> None:
            raise BrokenPipeError("reader is gone")

        def flush(self) -> None:
            raise BrokenPipeError("reader is gone")

    emitter = ProgressEmitter(ClosedPipe())

    assert emitter.emit("submission_written", rows=3) is None
    assert emitter.predict_progress("test", 1, 10) is None
    assert emitter.predict_progress("test", 10, 10) is None
    assert emitter.metrics_computed({"mAP": 0.5}) is None


def test_predict_progress_does_not_raise_on_unusable_counts():
    stream = io.StringIO()
    emitter = ProgressEmitter(stream)

    assert emitter.predict_progress("test", 1, None) is None
    assert emitter.predict_progress("test", "1", "10") is None
    assert emitter.metrics_computed(None) is None


def test_progress_emitter_stays_quiet_outside_the_creating_process(monkeypatch):
    stream = io.StringIO()
    emitter = ProgressEmitter(stream)
    monkeypatch.setattr(progress_module.os, "getpid", lambda: -1)

    emitter.emit("submission_written", rows=3)
    emitter.predict_progress("test", 1, 10)

    assert stream.getvalue() == ""
