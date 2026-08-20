"""data.progress/1 스트림 파서.

data pipeline을 import하지 않고, 계약(`docs/data_progress_contract.md`)에 적힌 줄
형식만 가지고 확인합니다. data 쪽 구현이 아직 없어도 합성한 줄로 계약을 고정합니다.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.web.data_progress import DataProgressState, consume_line, snapshot


def line(event: str, **fields) -> str:
    return json.dumps({"schema": "data.progress/1", "event": event, **fields})


def feed(*lines: str) -> DataProgressState:
    state = DataProgressState()
    for item in lines:
        consume_line(state, item)
    return state


# --- 정상 흐름 --------------------------------------------------------------


def test_prepare_started_populates_settings():
    state = feed(
        line(
            "prepare_started",
            raw_prefix="raw/",
            split_ratio="8:2",
            seed=42,
            split_method="image",
        )
    )

    result = snapshot(state)

    assert result["available"] is True
    assert result["stage"] == "listing"
    assert result["stage_label"] == "원본 목록 확인"
    assert result["raw_prefix"] == "raw/"
    assert result["split_ratio"] == "8:2"
    assert result["seed"] == 42
    assert result["split_method"] == "image"


def test_sources_listed_gives_the_totals_the_screen_needs():
    state = feed(line("sources_listed", train_images=1842, annotations=1842, test_images=842))

    result = snapshot(state)

    assert result["sources"] == {
        "train_images": 1842,
        "annotations": 1842,
        "test_images": 842,
    }
    assert result["stage"] == "listing"


def test_read_progress_reports_done_total_and_percent():
    state = feed(
        line("read_progress", stage="annotations", done=400, total=1842),
    )

    result = snapshot(state)

    assert result["stage"] == "annotations"
    assert result["stage_label"] == "annotation 읽는 중"
    assert result["read"] == {
        "stage": "annotations",
        "done": 400,
        "total": 1842,
        "percent": pytest.approx(21.7),
    }


def test_crop_bank_is_a_read_stage_too():
    """은행 자르기는 ``step_started``와 ``read_progress``를 둘 다 냅니다.

    한쪽만 알면 준비에서 가장 오래 걸리는 단계가 통째로 "모르는 단계"로 세어져,
    막대는 멈춘 채 malformed 줄만 쌓입니다.
    """

    state = feed(line("read_progress", stage="crop_bank", done=1180, total=4720))

    result = snapshot(state)

    assert result["read"]["stage"] == "crop_bank"
    assert result["read"]["percent"] == pytest.approx(25.0)
    assert state.malformed_lines == 0


def test_second_read_stage_replaces_the_first():
    state = feed(
        line("read_progress", stage="annotations", done=1842, total=1842),
        line("read_progress", stage="test_images", done=100, total=842),
    )

    result = snapshot(state)

    assert result["stage"] == "test_images"
    assert result["stage_label"] == "test 이미지 읽는 중"
    assert result["read"]["done"] == 100
    assert result["read"]["total"] == 842


@pytest.mark.parametrize(
    ("step", "label"),
    (
        ("split", "나누는 중"),
        ("manifests", "manifest 만드는 중"),
        ("crop_bank", "참조 crop 자르는 중"),
        ("publish", "올리는 중"),
    ),
)
def test_step_started_moves_the_stage(step, label):
    state = feed(
        line("read_progress", stage="annotations", done=1842, total=1842),
        line("step_started", step=step),
    )

    result = snapshot(state)

    assert result["stage"] == step
    assert result["stage_label"] == label
    # 읽기가 끝난 뒤에는 막대를 더 이상 그리지 않습니다.
    assert result["read"] is None


def test_prepare_completed_carries_the_final_numbers():
    state = feed(
        line("prepare_completed", train_images=1473, validation_images=369, category_count=73)
    )

    result = snapshot(state)

    assert result["stage"] == "completed"
    assert result["stage_label"] == "준비 완료"
    assert result["completed"] == {
        "train_images": 1473,
        "validation_images": 369,
        "category_count": 73,
    }


def test_read_progress_produces_a_readable_log_line():
    state = DataProgressState()

    entry = consume_line(state, line("read_progress", stage="annotations", done=400, total=1842))

    assert "annotation" in entry["text"]
    assert "400" in entry["text"] and "1842" in entry["text"]
    assert entry["level"] == "info"


# --- 열화 표: 이상한 입력 ----------------------------------------------------


def test_blank_lines_are_skipped():
    state = DataProgressState()

    assert consume_line(state, "   \n") is None
    assert consume_line(state, "") is None


@pytest.mark.parametrize(
    "raw",
    (
        "boto3 경고 한 줄",  # `{`로 시작하지 않음
        "{not json",  # JSON 파싱 실패
        "[]",
        '"just a string"',
        '{"schema": 5}',  # schema 타입이 틀림
        '{"no_schema": true}',  # schema 없음
        '{"schema": "other.thing/1", "event": "read_progress"}',  # 남의 schema
        '{"schema": "data.progress/2", "event": "read_progress"}',  # 모르는 major
        '{"schema": "data.progress/1", "event": "chunk_uploaded"}',  # 모르는 event
    ),
)
def test_foreign_or_malformed_lines_become_raw_logs(raw):
    state = DataProgressState()

    entry = consume_line(state, raw)

    assert entry is not None  # 버리지 않습니다
    assert state.saw_progress is False  # 상태도 바꾸지 않습니다
    assert snapshot(state)["available"] is False


def test_unknown_event_does_not_break_the_events_around_it():
    state = feed(
        line("sources_listed", train_images=10, annotations=10, test_images=4),
        line("chunk_uploaded", key="a/b.json"),  # 계약에 아직 없는 event
        line("read_progress", stage="annotations", done=5, total=10),
    )

    result = snapshot(state)

    assert result["sources"]["annotations"] == 10
    assert result["read"]["done"] == 5


@pytest.mark.parametrize("bad", ("400", 3.5, True, None))
def test_bad_field_type_drops_only_that_field(bad):
    """``done``만 망가져도 ``total``과 ``stage``는 살아남습니다."""

    state = feed(line("read_progress", stage="annotations", done=bad, total=1842))

    result = snapshot(state)

    assert result["available"] is True
    assert result["stage"] == "annotations"
    assert result["read"]["total"] == 1842
    assert result["read"]["done"] is None
    assert result["read"]["percent"] is None
    assert state.malformed_lines == 1


def test_bad_read_stage_keeps_the_counts():
    state = feed(line("read_progress", stage="images", done=5, total=10))

    result = snapshot(state)

    assert result["read"]["done"] == 5
    assert result["read"]["stage"] is None
    assert state.malformed_lines == 1


def test_unknown_step_name_is_not_invented():
    state = feed(line("step_started", step="polish"))

    result = snapshot(state)

    assert result["stage"] is None
    assert result["stage_label"] is None
    assert state.malformed_lines == 1


def test_done_greater_than_total_is_kept_as_observed_and_percent_is_capped():
    state = feed(line("read_progress", stage="annotations", done=2000, total=1842))

    result = snapshot(state)

    assert result["read"]["done"] == 2000  # 관측한 값을 고치지 않습니다
    assert result["read"]["percent"] == 100.0  # 막대는 넘치지 않습니다


def test_out_of_order_reads_are_last_write_wins():
    """단조 증가를 가정하지 않습니다."""

    state = feed(
        line("read_progress", stage="annotations", done=900, total=1842),
        line("read_progress", stage="annotations", done=400, total=1842),
    )

    assert snapshot(state)["read"]["done"] == 400


def test_zero_total_does_not_divide_by_zero():
    state = feed(line("read_progress", stage="annotations", done=0, total=0))

    assert snapshot(state)["read"]["percent"] is None


# --- 진행 줄이 아예 없을 때 --------------------------------------------------


def test_progress_absent_reports_unavailable_without_inventing_values():
    state = feed("원본을 읽는 중입니다", "또 한 줄")

    result = snapshot(state)

    assert result["available"] is False
    assert result["reason"] == "data_pipeline_no_progress_stream"
    assert result["stage"] is None
    assert result["read"] is None
    assert result["sources"] is None
    assert result["eta_seconds"] is None


# --- 남은 시간은 관측한 속도로만 --------------------------------------------


def stamped(event: str, ts: str, **fields) -> str:
    return json.dumps({"schema": "data.progress/1", "event": event, "ts": ts, **fields})


def test_eta_needs_two_observations():
    one = feed(stamped("read_progress", "2026-08-07T03:35:00Z", stage="annotations", done=100, total=1000))

    assert snapshot(one)["eta_seconds"] is None


def test_eta_uses_the_measured_reading_speed():
    state = feed(
        stamped("read_progress", "2026-08-07T03:35:00Z", stage="annotations", done=100, total=1000),
        stamped("read_progress", "2026-08-07T03:35:10Z", stage="annotations", done=300, total=1000),
    )

    # 10초에 200장이면 남은 700장은 35초입니다.
    assert snapshot(state)["eta_seconds"] == 35.0


def test_eta_is_not_estimated_when_time_did_not_move():
    state = feed(
        stamped("read_progress", "2026-08-07T03:35:00Z", stage="annotations", done=100, total=1000),
        stamped("read_progress", "2026-08-07T03:35:00Z", stage="annotations", done=300, total=1000),
    )

    assert snapshot(state)["eta_seconds"] is None


def test_eta_resets_when_the_read_stage_changes():
    state = feed(
        stamped("read_progress", "2026-08-07T03:35:00Z", stage="annotations", done=100, total=1000),
        stamped("read_progress", "2026-08-07T03:35:10Z", stage="annotations", done=300, total=1000),
        stamped("read_progress", "2026-08-07T03:36:00Z", stage="test_images", done=10, total=842),
    )

    assert snapshot(state)["eta_seconds"] is None


@pytest.mark.parametrize("bad_ts", ("어제", 5, None, "2026-13-45T99:99:99Z"))
def test_unparsable_timestamps_never_raise_and_never_estimate(bad_ts):
    state = feed(
        stamped("read_progress", "2026-08-07T03:35:00Z", stage="annotations", done=100, total=1000),
        json.dumps(
            {
                "schema": "data.progress/1",
                "event": "read_progress",
                "ts": bad_ts,
                "stage": "annotations",
                "done": 300,
                "total": 1000,
            }
        ),
    )

    result = snapshot(state)

    assert result["read"]["done"] == 300
    assert result["eta_seconds"] is None


# --- 화면으로 그대로 넘어갈 수 있어야 합니다 ---------------------------------


def test_log_text_is_masked():
    state = DataProgressState()

    entry = consume_line(state, "token=AKIAIOSFODNN7EXAMPLE 로 실패")

    assert "AKIAIOSFODNN7EXAMPLE" not in entry["text"]


@pytest.mark.parametrize("weird", (None, 5, b"bytes", [], {}))
def test_consume_line_never_raises_on_non_text(weird):
    assert consume_line(DataProgressState(), weird) is None


# --- data emitter가 실제로 내보내는 형식 -------------------------------------


def emitted(event: str, **fields) -> str:
    """계약서의 예시 줄과 같은 순서·형식입니다.

    key 순서는 schema, event, 개별 필드, ts입니다.
    """

    payload = {"schema": "data.progress/1", "event": event}
    payload.update(fields)
    payload["ts"] = "2026-08-07T03:35:20.123456Z"
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_parses_a_full_preparation_in_the_emitters_own_format():
    state = DataProgressState()
    lines = [
        emitted("prepare_started", raw_prefix="raw/", split_ratio="8:2", seed=42, split_method="image"),
        emitted("sources_listed", train_images=1842, annotations=1842, test_images=842),
        emitted("read_progress", stage="annotations", done=1842, total=1842),
        "botocore 경고가 섞여 들어옵니다",
        emitted("read_progress", stage="test_images", done=842, total=842),
        emitted("step_started", step="split"),
        emitted("step_started", step="manifests"),
        emitted("step_started", step="publish"),
        emitted("prepare_completed", train_images=1473, validation_images=369, category_count=73),
    ]
    for item in lines:
        consume_line(state, item)

    result = snapshot(state)

    assert result["available"] is True
    assert result["stage"] == "completed"
    assert result["sources"]["train_images"] == 1842
    assert result["completed"]["category_count"] == 73
    json.dumps(result, allow_nan=False, ensure_ascii=False)
