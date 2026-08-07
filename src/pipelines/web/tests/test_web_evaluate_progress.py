"""evaluate.progress/1 스트림 파서.

evaluate pipeline을 import하지 않고, 계약(`docs/evaluate_progress_contract.md`)에
적힌 줄 형식만 가지고 확인합니다. evaluate 쪽 구현이 아직 없어도 합성한 줄로
계약을 고정합니다.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.web.evaluate_progress import (
    EvaluateProgressState,
    consume_line,
    snapshot,
)


def line(event: str, **fields) -> str:
    return json.dumps({"schema": "evaluate.progress/1", "event": event, **fields})


def feed(*lines: str) -> EvaluateProgressState:
    state = EvaluateProgressState()
    for item in lines:
        consume_line(state, item)
    return state


# --- 정상 흐름 --------------------------------------------------------------


def test_evaluate_started_carries_the_totals_the_screen_needs():
    state = feed(
        line(
            "evaluate_started",
            run_id="run-1",
            device="cuda",
            validation_images=46,
            test_images=842,
        )
    )

    result = snapshot(state)

    assert result["available"] is True
    assert result["stage"] == "started"
    assert result["run_id"] == "run-1"
    assert result["device"] == "cuda"
    # 842장이라는 사실을 화면이 미리 알아야 오래 걸리는 것이 정상임을 알립니다.
    assert result["images"] == {"validation_images": 46, "test_images": 842}


def test_test_images_zero_is_reported_as_zero_not_as_missing():
    """test manifest 없이 도는 실행은 ``test_images``가 0입니다."""

    state = feed(line("evaluate_started", validation_images=46, test_images=0))

    assert snapshot(state)["images"] == {"validation_images": 46, "test_images": 0}


@pytest.mark.parametrize(
    ("stage", "label"),
    (("validation", "validation 추론 중"), ("test", "test 추론 중")),
)
def test_predict_progress_reports_done_total_and_percent(stage, label):
    state = feed(line("predict_progress", stage=stage, done=421, total=842))

    result = snapshot(state)

    assert result["stage"] == stage
    assert result["stage_label"] == label
    assert result["predict"] == {
        "stage": stage,
        "done": 421,
        "total": 842,
        "percent": pytest.approx(50.0),
    }


def test_test_inference_replaces_the_validation_bar():
    state = feed(
        line("predict_progress", stage="validation", done=46, total=46),
        line("predict_progress", stage="test", done=100, total=842),
    )

    result = snapshot(state)

    assert result["stage"] == "test"
    assert result["predict"]["done"] == 100
    assert result["predict"]["total"] == 842


def test_metrics_computed_moves_the_stage_and_keeps_the_numbers():
    state = feed(
        line("predict_progress", stage="test", done=842, total=842),
        line("metrics_computed", mAP=0.3123, mAP50=0.5512, mAP75=0.2811),
    )

    result = snapshot(state)

    assert result["stage"] == "metrics"
    assert result["stage_label"] == "지표 계산 중"
    assert result["metrics"] == {"mAP": 0.3123, "mAP50": 0.5512, "mAP75": 0.2811}
    # 추론이 끝났으므로 막대를 더 이상 그리지 않습니다.
    assert result["predict"] is None


def test_submission_written_moves_the_stage_and_carries_the_row_count():
    state = feed(line("submission_written", rows=1024))

    result = snapshot(state)

    assert result["stage"] == "submission"
    assert result["stage_label"] == "submission 쓰는 중"
    assert result["submission_rows"] == 1024


def test_evaluate_completed_carries_the_final_numbers():
    state = feed(line("evaluate_completed", validation_images=46, test_images=842))

    result = snapshot(state)

    assert result["stage"] == "completed"
    assert result["stage_label"] == "평가 완료"
    assert result["completed"] == {"validation_images": 46, "test_images": 842}
    assert result["predict"] is None


def test_predict_progress_produces_a_readable_log_line():
    state = EvaluateProgressState()

    entry = consume_line(state, line("predict_progress", stage="test", done=421, total=842))

    assert "test" in entry["text"]
    assert "421" in entry["text"] and "842" in entry["text"]
    assert entry["level"] == "info"


# --- 열화 표: 이상한 입력 ----------------------------------------------------


def test_blank_lines_are_skipped():
    state = EvaluateProgressState()

    assert consume_line(state, "   \n") is None
    assert consume_line(state, "") is None


@pytest.mark.parametrize(
    "raw",
    (
        " Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.312",  # COCOeval 로그
        "{not json",  # JSON 파싱 실패
        '{"no_schema": true}',  # schema 없음 또는 타입이 틀림
        '{"schema": "other.thing/1", "event": "predict_progress"}',  # 남의 schema
        '{"schema": "evaluate.progress/2", "event": "predict_progress"}',  # 모르는 major
        '{"schema": "evaluate.progress/1", "event": "batch_done"}',  # 모르는 event
    ),
)
def test_foreign_or_malformed_lines_become_raw_logs(raw):
    state = EvaluateProgressState()

    entry = consume_line(state, raw)

    assert entry is not None  # 버리지 않습니다
    assert state.saw_progress is False  # 상태도 바꾸지 않습니다
    assert snapshot(state)["available"] is False


def test_unknown_event_does_not_break_the_events_around_it():
    state = feed(
        line("evaluate_started", validation_images=46, test_images=842),
        line("batch_done", index=3),  # 계약에 아직 없는 event
        line("predict_progress", stage="test", done=5, total=842),
    )

    result = snapshot(state)

    assert result["images"]["test_images"] == 842
    assert result["predict"]["done"] == 5


@pytest.mark.parametrize("bad", ("400", 3.5, True, None))
def test_bad_field_type_drops_only_that_field(bad):
    """``done``만 망가져도 ``total``과 ``stage``는 살아남습니다."""

    state = feed(line("predict_progress", stage="test", done=bad, total=842))

    result = snapshot(state)

    assert result["available"] is True
    assert result["stage"] == "test"
    assert result["predict"]["total"] == 842
    assert result["predict"]["done"] is None
    assert result["predict"]["percent"] is None
    assert state.malformed_lines == 1


def test_bad_predict_stage_keeps_the_counts():
    state = feed(line("predict_progress", stage="검증", done=5, total=10))

    result = snapshot(state)

    assert result["predict"]["done"] == 5
    assert result["predict"]["stage"] is None
    assert state.malformed_lines == 1


def test_done_greater_than_total_is_kept_as_observed_and_percent_is_capped():
    state = feed(line("predict_progress", stage="test", done=900, total=842))

    result = snapshot(state)

    assert result["predict"]["done"] == 900  # 관측한 값을 고치지 않습니다
    assert result["predict"]["percent"] == 100.0  # 막대는 넘치지 않습니다


def test_out_of_order_predictions_are_last_write_wins():
    """단조 증가를 가정하지 않습니다."""

    state = feed(
        line("predict_progress", stage="test", done=900, total=842),
        line("predict_progress", stage="test", done=400, total=842),
    )

    assert snapshot(state)["predict"]["done"] == 400


def test_zero_total_does_not_divide_by_zero():
    state = feed(line("predict_progress", stage="validation", done=0, total=0))

    assert snapshot(state)["predict"]["percent"] is None


# --- 지표는 지어내지 않습니다 ------------------------------------------------


def test_uncomputed_metrics_stay_null_and_are_never_zero():
    state = feed(line("metrics_computed", mAP=0.3123, mAP50=None, mAP75=None))

    result = snapshot(state)

    assert result["metrics"] == {"mAP": 0.3123, "mAP50": None, "mAP75": None}


@pytest.mark.parametrize("bad", ("NaN", "Infinity", "-Infinity"))
def test_non_finite_metrics_become_none_so_the_browser_can_parse_them(bad):
    """browser의 ``JSON.parse``는 맨 ``NaN``에서 실패합니다."""

    state = feed(f'{{"schema": "evaluate.progress/1", "event": "metrics_computed", "mAP": {bad}}}')

    result = snapshot(state)

    assert result["metrics"]["mAP"] is None
    json.dumps(result, allow_nan=False)


def test_bad_metric_type_drops_only_that_metric():
    state = feed(line("metrics_computed", mAP="0.3", mAP50=0.5512))

    result = snapshot(state)

    assert result["metrics"]["mAP"] is None
    assert result["metrics"]["mAP50"] == 0.5512
    assert state.malformed_lines == 1


# --- 진행 줄이 아예 없을 때 --------------------------------------------------


def test_progress_absent_reports_unavailable_without_inventing_values():
    state = feed("Loading and preparing results...", "DONE (t=0.03s).")

    result = snapshot(state)

    assert result["available"] is False
    assert result["reason"] == "evaluate_pipeline_no_progress_stream"
    assert result["stage"] is None
    assert result["predict"] is None
    assert result["images"] is None
    assert result["metrics"] is None
    assert result["eta_seconds"] is None


# --- 남은 시간은 관측한 속도로만 --------------------------------------------


def stamped(event: str, ts: str, **fields) -> str:
    return json.dumps({"schema": "evaluate.progress/1", "event": event, "ts": ts, **fields})


def test_eta_needs_two_observations():
    one = feed(stamped("predict_progress", "2026-08-07T03:35:00Z", stage="test", done=100, total=1000))

    assert snapshot(one)["eta_seconds"] is None


def test_eta_uses_the_measured_inference_speed():
    state = feed(
        stamped("predict_progress", "2026-08-07T03:35:00Z", stage="test", done=100, total=1000),
        stamped("predict_progress", "2026-08-07T03:35:10Z", stage="test", done=300, total=1000),
    )

    # 10초에 200장이면 남은 700장은 35초입니다.
    assert snapshot(state)["eta_seconds"] == 35.0


def test_eta_is_not_estimated_when_time_did_not_move():
    state = feed(
        stamped("predict_progress", "2026-08-07T03:35:00Z", stage="test", done=100, total=1000),
        stamped("predict_progress", "2026-08-07T03:35:00Z", stage="test", done=300, total=1000),
    )

    assert snapshot(state)["eta_seconds"] is None


def test_eta_resets_when_the_predict_stage_changes():
    """validation의 속도로 test의 남은 시간을 재면 거짓말이 됩니다."""

    state = feed(
        stamped("predict_progress", "2026-08-07T03:35:00Z", stage="validation", done=10, total=46),
        stamped("predict_progress", "2026-08-07T03:35:10Z", stage="validation", done=46, total=46),
        stamped("predict_progress", "2026-08-07T03:36:00Z", stage="test", done=10, total=842),
    )

    assert snapshot(state)["eta_seconds"] is None


@pytest.mark.parametrize("bad_ts", ("어제", 5, None, "2026-13-45T99:99:99Z"))
def test_unparsable_timestamps_never_raise_and_never_estimate(bad_ts):
    state = feed(
        stamped("predict_progress", "2026-08-07T03:35:00Z", stage="test", done=100, total=1000),
        json.dumps(
            {
                "schema": "evaluate.progress/1",
                "event": "predict_progress",
                "ts": bad_ts,
                "stage": "test",
                "done": 300,
                "total": 1000,
            }
        ),
    )

    result = snapshot(state)

    assert result["predict"]["done"] == 300
    assert result["eta_seconds"] is None


# --- 화면으로 그대로 넘어갈 수 있어야 합니다 ---------------------------------


def test_log_text_is_masked():
    state = EvaluateProgressState()

    entry = consume_line(state, "token=AKIAIOSFODNN7EXAMPLE 로 실패")

    assert "AKIAIOSFODNN7EXAMPLE" not in entry["text"]


@pytest.mark.parametrize("weird", (None, 5, b"bytes", [], {}))
def test_consume_line_never_raises_on_non_text(weird):
    assert consume_line(EvaluateProgressState(), weird) is None


# --- evaluate emitter가 실제로 내보내는 형식 ---------------------------------


def emitted(event: str, **fields) -> str:
    """계약서의 예시 줄과 같은 순서·형식입니다.

    key 순서는 schema, event, 개별 필드, ts입니다.
    """

    payload = {"schema": "evaluate.progress/1", "event": event}
    payload.update(fields)
    payload["ts"] = "2026-08-07T03:35:20.123456Z"
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_parses_a_full_evaluation_in_the_emitters_own_format():
    state = EvaluateProgressState()
    lines = [
        emitted("evaluate_started", run_id="run-1", device="cuda", validation_images=46, test_images=842),
        emitted("predict_progress", stage="validation", done=46, total=46),
        " Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.312",  # COCOeval이 섞여 들어옵니다
        emitted("predict_progress", stage="test", done=842, total=842),
        emitted("metrics_computed", mAP=0.3123, mAP50=0.5512, mAP75=0.2811),
        emitted("submission_written", rows=1024),
        emitted("evaluate_completed", validation_images=46, test_images=842),
    ]
    for item in lines:
        consume_line(state, item)

    result = snapshot(state)

    assert result["available"] is True
    assert result["stage"] == "completed"
    assert result["images"]["test_images"] == 842
    assert result["metrics"]["mAP50"] == 0.5512
    assert result["submission_rows"] == 1024
    assert result["completed"]["validation_images"] == 46
    json.dumps(result, allow_nan=False, ensure_ascii=False)
