"""`data.progress/1` 진행 로그가 계약대로 stderr에만 나가는지 확인합니다.

emitter test는 실제 stderr 대신 주입한 stream을 써서, throttle 규칙과 예외
안전(닫힌 pipe에 써도 조용히 넘어가는지)을 직접 확인합니다.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

from test_dataset_preparation import prepare, prepare_config

from src.pipelines.data import progress as progress_module
from src.pipelines.data.progress import SCHEMA, ProgressEmitter


# --- 도우미 ---------------------------------------------------------------


def progress_events(captured_stderr: str) -> list[dict[str, Any]]:
    """JSON이 아닌 줄은 건너뛰고 `data.progress/1` event만 모읍니다."""

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


def read_stages(stream: io.StringIO) -> list[tuple[int, int]]:
    return [
        (event["done"], event["total"]) for event in progress_events(stream.getvalue())
    ]


# --- 준비 실행이 내보내는 event -------------------------------------------


def test_prepare_emits_the_contract_events_on_stderr_and_leaves_stdout_empty(capsys):
    result, _ = prepare(prepare_config("8:2"))
    captured = capsys.readouterr()

    assert result["status"] == "ok", result["message"]
    # 계약 2번: stdout에는 아무것도 더하지 않습니다.
    assert captured.out == ""

    events = progress_events(captured.err)
    names = [event["event"] for event in events]
    assert names[0] == "prepare_started"
    assert names[1] == "sources_listed"
    assert names[-1] == "prepare_completed"
    assert [
        event["step"] for event in events if event["event"] == "step_started"
    ] == ["split", "manifests", "publish"]
    for event in events:
        datetime.strptime(event["ts"], "%Y-%m-%dT%H:%M:%S.%fZ")

    started, listed = events[0], events[1]
    assert started == {
        "schema": SCHEMA,
        "event": "prepare_started",
        "raw_prefix": "datasets/pill_detection/raw/v1/",
        "split_ratio": "8:2",
        "seed": 42,
        "split_method": "group",
        "ts": started["ts"],
    }
    assert listed == {
        "schema": SCHEMA,
        "event": "sources_listed",
        "train_images": 40,
        "annotations": 40,
        "test_images": 5,
        "ts": listed["ts"],
    }
    assert events[-1] == {
        "schema": SCHEMA,
        "event": "prepare_completed",
        "train_images": result["summary"]["train_images"],
        "validation_images": result["summary"]["validation_images"],
        "category_count": result["summary"]["category_count"],
        "ts": events[-1]["ts"],
    }

    stages = {
        event["stage"] for event in events if event["event"] == "read_progress"
    }
    assert stages == {"annotations", "test_images"}
    for stage, total in (("annotations", 40), ("test_images", 5)):
        reads = [
            event
            for event in events
            if event["event"] == "read_progress" and event["stage"] == stage
        ]
        assert {event["total"] for event in reads} == {total}
        # 마지막 항목은 반드시 나옵니다.
        assert reads[-1]["done"] == total
        assert [event["done"] for event in reads] == sorted(
            event["done"] for event in reads
        )


def test_progress_stream_stays_silent_for_the_dummy_execution(capsys):
    from src.pipelines.data import run

    result = run({"execution": {"mode": "dummy"}})
    captured = capsys.readouterr()

    assert result["status"] == "ok"
    assert captured.out == ""
    assert progress_events(captured.err) == []


# --- throttle 규칙 ---------------------------------------------------------


def test_read_progress_skips_lines_below_two_percent_within_one_second(monkeypatch):
    emitter, stream, clock = emitter_with_clock(monkeypatch)

    # 첫 줄은 전체 개수를 알리기 위해 바로 나갑니다.
    emitter.read_progress("annotations", 1, 1000)
    # 2%(20개)에 못 미치고 1초도 지나지 않은 줄은 버립니다.
    for done in range(2, 21):
        clock.now += 0.01
        emitter.read_progress("annotations", done, 1000)
    # 21번째에서 마지막 event 대비 20개를 채웁니다.
    emitter.read_progress("annotations", 21, 1000)

    assert read_stages(stream) == [(1, 1000), (21, 1000)]


def test_read_progress_emits_after_one_second_without_enough_progress(monkeypatch):
    emitter, stream, clock = emitter_with_clock(monkeypatch)

    emitter.read_progress("annotations", 1, 1000)
    clock.now += 0.5
    emitter.read_progress("annotations", 2, 1000)
    clock.now += 0.6
    emitter.read_progress("annotations", 3, 1000)

    assert read_stages(stream) == [(1, 1000), (3, 1000)]


def test_read_progress_always_emits_the_last_item(monkeypatch):
    emitter, stream, clock = emitter_with_clock(monkeypatch)

    emitter.read_progress("test_images", 1, 1000)
    clock.now += 0.01
    emitter.read_progress("test_images", 2, 1000)
    clock.now += 0.01
    emitter.read_progress("test_images", 1000, 1000)

    assert read_stages(stream) == [(1, 1000), (1000, 1000)]


def test_read_progress_counts_each_stage_separately(monkeypatch):
    emitter, stream, clock = emitter_with_clock(monkeypatch)

    emitter.read_progress("annotations", 1, 1000)
    clock.now += 0.01
    # 다른 stage의 첫 줄은 앞 stage의 진행량에 눌리지 않습니다.
    emitter.read_progress("test_images", 1, 1000)

    events = progress_events(stream.getvalue())
    assert [event["stage"] for event in events] == ["annotations", "test_images"]


# --- 예외 안전과 process 경계 ---------------------------------------------


def test_progress_emitter_does_not_raise_when_the_reader_closed_the_pipe():
    class ClosedPipe:
        def write(self, line: str) -> None:
            raise BrokenPipeError("reader is gone")

        def flush(self) -> None:
            raise BrokenPipeError("reader is gone")

    emitter = ProgressEmitter(ClosedPipe())

    assert emitter.emit("step_started", step="split") is None
    assert emitter.read_progress("annotations", 1, 10) is None
    assert emitter.read_progress("annotations", 10, 10) is None


def test_read_progress_does_not_raise_on_unusable_counts():
    stream = io.StringIO()
    emitter = ProgressEmitter(stream)

    assert emitter.read_progress("annotations", 1, None) is None
    assert emitter.read_progress("annotations", "1", "10") is None


def test_progress_emitter_stays_quiet_outside_the_creating_process(monkeypatch):
    stream = io.StringIO()
    emitter = ProgressEmitter(stream)
    monkeypatch.setattr(progress_module.os, "getpid", lambda: -1)

    emitter.emit("step_started", step="split")
    emitter.read_progress("annotations", 1, 10)

    assert stream.getvalue() == ""
