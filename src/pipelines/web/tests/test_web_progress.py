"""train.progress/1 스트림 파서.

train pipeline을 import하지 않고, 계약에 적힌 줄 형식만 가지고 확인합니다.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.web.progress import ProgressState, consume_line, snapshot


def line(event: str, **fields) -> str:
    return json.dumps({"schema": "train.progress/1", "event": event, **fields})


def feed(*lines: str) -> ProgressState:
    state = ProgressState()
    for item in lines:
        consume_line(state, item)
    return state


# --- 정상 흐름 --------------------------------------------------------------


def test_parses_run_started_populates_totals():
    state = feed(
        line(
            "run_started",
            run_id="web-1",
            architecture="fasterrcnn_resnet50_fpn",
            device="cuda",
            epochs=50,
            train_images=3200,
            validation_images=800,
            class_count=12,
        )
    )

    result = snapshot(state)

    assert result["available"] is True
    assert result["total_epochs"] == 50
    assert result["train_images"] == 3200
    assert result["class_count"] == 12
    assert result["architecture"] == "fasterrcnn_resnet50_fpn"


def test_parses_epoch_started():
    state = feed(line("epoch_started", run_id="web-1", epoch=1, epochs=10))

    assert snapshot(state)["current_epoch"] == 1


def test_parses_epoch_completed():
    state = feed(
        line(
            "epoch_completed",
            run_id="web-1",
            epoch=1,
            epochs=10,
            train_loss=0.4312,
            validation_loss=0.5109,
            epoch_seconds=42.1,
            is_best=True,
        )
    )

    result = snapshot(state)

    assert result["completed_epochs"] == 1
    assert result["epochs"][0]["train_loss"] == 0.4312
    assert result["best"] == {"epoch": 1, "validation_loss": 0.5109}
    assert result["percent"] == 10.0


def test_epoch_completed_produces_readable_log_line():
    state = ProgressState()

    entry = consume_line(
        state,
        line("epoch_completed", epoch=3, epochs=10, train_loss=0.4, validation_loss=0.5, is_best=True),
    )

    assert "epoch 3/10 완료" in entry["text"]
    assert "최고 기록" in entry["text"]
    assert entry["level"] == "info"


# --- 학습 완료 --------------------------------------------------------------


def test_training_completed_finishes_progress_even_when_stopped_early():
    state = feed(
        line("run_started", epochs=50),
        line("epoch_completed", epoch=1, epochs=50, validation_loss=0.9, epoch_seconds=10.0),
        line("epoch_completed", epoch=2, epochs=50, validation_loss=0.8, epoch_seconds=10.0),
        line(
            "training_completed",
            planned_epochs=50,
            completed_epochs=2,
            stopped_early=True,
            best_epoch=2,
            best_validation_loss=0.8,
        ),
    )

    result = snapshot(state)

    assert result["finished"] is True
    assert result["stopped_early"] is True
    assert result["total_epochs"] == 50  # 계획은 그대로 두고
    assert result["completed_epochs"] == 2  # 실제로 돈 횟수를 따로 알려 줍니다
    assert result["percent"] == 100.0  # 2/50이 아니라 끝난 학습입니다
    assert result["eta_seconds"] == 0.0


def test_training_completed_without_early_stop_reports_it_as_not_stopped():
    state = feed(
        line("run_started", epochs=3),
        line("epoch_completed", epoch=1, epochs=3, validation_loss=0.5, epoch_seconds=1.0),
        line("training_completed", planned_epochs=3, completed_epochs=3, stopped_early=False),
    )

    assert snapshot(state)["stopped_early"] is False


def test_training_completed_produces_readable_log_line():
    state = ProgressState()

    entry = consume_line(
        state,
        line(
            "training_completed",
            planned_epochs=50,
            completed_epochs=12,
            stopped_early=True,
            best_epoch=7,
            best_validation_loss=0.41,
        ),
    )

    assert "12/50" in entry["text"]
    assert "조기 종료" in entry["text"]
    assert entry["level"] == "info"


def test_run_without_a_completion_event_is_not_reported_as_finished():
    """취소된 학습과 이 event 이전의 옛 실행이 오늘과 똑같이 읽혀야 합니다."""

    state = feed(
        line("run_started", epochs=3),
        line("epoch_completed", epoch=1, epochs=3, validation_loss=0.5),
    )

    result = snapshot(state)

    assert result["finished"] is False
    assert result["stopped_early"] is None  # 모르는 것을 지어내지 않습니다
    assert result["percent"] == pytest.approx(33.3)
    assert result["epochs"][0]["train_loss_components"] is None


# --- 이상한 입력 ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    (
        "{not json",
        '{"schema": "train.progress/1"',
        "[]",
        '"just a string"',
        '{"schema": 5}',
        '{"no_schema": true}',
        '{"schema": "other.thing/1", "event": "epoch_completed"}',
    ),
)
def test_malformed_or_foreign_lines_become_raw_logs(raw):
    state = ProgressState()

    entry = consume_line(state, raw)

    assert entry is not None  # 버리지 않습니다
    assert state.saw_progress is False  # 상태도 바꾸지 않습니다
    assert snapshot(state)["available"] is False


def test_unknown_schema_major_version_is_ignored():
    state = feed(line("epoch_completed", epoch=1).replace("progress/1", "progress/2"))

    assert snapshot(state)["available"] is False


def test_unknown_event_does_not_break_state():
    state = feed(
        line("run_started", epochs=5),
        line("batch_progress", epoch=1, batch=10),  # 아직 없는 event
        line("epoch_completed", epoch=1, epochs=5, validation_loss=0.5, epoch_seconds=1.0),
    )

    assert snapshot(state)["completed_epochs"] == 1


def test_interleaved_torch_warning_is_kept_as_log():
    state = ProgressState()

    entry = consume_line(state, "UserWarning: TypedStorage is deprecated")

    assert entry["level"] == "warn"
    assert state.saw_progress is False


def test_error_lines_are_marked():
    state = ProgressState()

    assert consume_line(state, "RuntimeError: CUDA out of memory")["level"] == "error"


def test_blank_lines_are_skipped():
    state = ProgressState()

    assert consume_line(state, "   \n") is None
    assert consume_line(state, "") is None


def test_out_of_order_and_duplicate_epochs_are_sorted_and_deduplicated():
    state = feed(
        line("epoch_completed", epoch=3, epochs=5, validation_loss=0.3),
        line("epoch_completed", epoch=1, epochs=5, validation_loss=0.9),
        line("epoch_completed", epoch=3, epochs=5, validation_loss=0.2),  # 같은 epoch 재전송
    )

    result = snapshot(state)

    assert [item["epoch"] for item in result["epochs"]] == [1, 3]
    assert result["epochs"][1]["validation_loss"] == 0.2  # 나중 값이 이깁니다
    assert result["best"]["epoch"] == 3


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_non_finite_loss_is_dropped_not_serialized_as_nan(bad):
    """json.loads는 NaN을 받아들이지만 브라우저 JSON.parse는 거부합니다."""

    state = ProgressState()
    consume_line(state, json.dumps({"schema": "train.progress/1", "event": "epoch_completed",
                                    "epoch": 1, "epochs": 2, "train_loss": bad}))

    result = snapshot(state)

    assert result["epochs"][0]["train_loss"] is None
    json.dumps(result, allow_nan=False)  # 여기서 터지면 안 됩니다


@pytest.mark.parametrize(
    "components",
    (
        "mapping이 아님",
        5,
        ["classification", 0.1],
        {"classification": "0.7"},
        {"classification": float("nan")},
        {"classification": True},
        {},
    ),
)
def test_unusable_loss_components_are_dropped_without_losing_the_epoch(components):
    state = ProgressState()

    consume_line(
        state,
        json.dumps(
            {
                "schema": "train.progress/1",
                "event": "epoch_completed",
                "epoch": 1,
                "epochs": 2,
                "train_loss": 0.5,
                "train_loss_components": components,
            }
        ),
    )

    record = snapshot(state)["epochs"][0]

    assert record["train_loss"] == 0.5  # 나머지 event는 살아남습니다
    assert record["train_loss_components"] is None
    json.dumps(snapshot(state), allow_nan=False)


def test_partly_unusable_loss_components_keep_the_usable_names():
    state = feed(
        line(
            "epoch_completed",
            epoch=1,
            epochs=2,
            train_loss_components={"classification": 0.72, "bbox_regression": "x"},
        )
    )

    assert snapshot(state)["epochs"][0]["train_loss_components"] == {"classification": 0.72}


@pytest.mark.parametrize("field", ("planned_epochs", "completed_epochs", "stopped_early"))
def test_training_completed_with_a_broken_field_still_finishes(field):
    state = feed(
        line("run_started", epochs=4),
        line("epoch_completed", epoch=1, epochs=4, validation_loss=0.5),
        line(
            "training_completed",
            **{"planned_epochs": 4, "completed_epochs": 1, "stopped_early": True, field: "이상한 값"},
        ),
    )

    result = snapshot(state)

    assert result["finished"] is True  # 한 필드가 깨져도 끝난 사실은 압니다
    assert result["percent"] == 100.0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad_epoch", ("3", 0, -1, True, None, 3.5))
def test_invalid_epoch_number_is_counted_not_crashing(bad_epoch):
    state = ProgressState()

    consume_line(state, line("epoch_completed", epoch=bad_epoch, epochs=10))

    assert state.malformed_lines == 1
    assert snapshot(state)["completed_epochs"] == 0


# --- 진행 로그가 아예 없을 때 ------------------------------------------------


def test_progress_absent_reports_unavailable_without_inventing_values():
    state = feed("보통 로그 한 줄", "또 한 줄")

    result = snapshot(state)

    assert result["available"] is False
    assert result["reason"] == "train_pipeline_no_progress_stream"
    assert result["eta_seconds"] is None
    assert result["current_epoch"] is None
    assert result["epochs"] == []
    assert "percent" not in result or result.get("percent") is None


def test_eta_requires_two_measured_epochs():
    one = feed(
        line("run_started", epochs=10),
        line("epoch_completed", epoch=1, epochs=10, epoch_seconds=10.0),
    )
    assert snapshot(one)["eta_seconds"] is None

    two = feed(
        line("run_started", epochs=10),
        line("epoch_completed", epoch=1, epochs=10, epoch_seconds=10.0),
        line("epoch_completed", epoch=2, epochs=10, epoch_seconds=20.0),
    )
    assert snapshot(two)["eta_seconds"] == 120.0  # 평균 15초 x 남은 8 epoch


def test_snapshot_is_json_serializable_without_nan():
    state = feed(
        line("run_started", epochs=3),
        line("epoch_completed", epoch=1, epochs=3, train_loss=0.5, validation_loss=0.6, epoch_seconds=1.0),
    )

    json.dumps(snapshot(state), allow_nan=False, ensure_ascii=False)


# --- 진행률 표시 줄 접기 -----------------------------------------------------


def test_download_percentages_are_collapsed():
    """터미널에선 한 줄이지만 pipe로 받으면 갱신마다 새 줄이 됩니다.

    실제로 모델 가중치 한 번 내려받는 데 598줄 중 590줄이 퍼센트였습니다.
    """

    state = ProgressState()
    kept = [consume_line(state, f"{value / 10:.1f}%") for value in range(2, 1001, 2)]
    kept = [entry for entry in kept if entry is not None]

    assert len(kept) <= 8  # 500줄이 몇 줄로 줄어듭니다
    assert kept[0]["text"] == "0.2%"  # 시작은 남깁니다
    assert kept[-1]["text"] == "100.0%"  # 끝도 남깁니다
    assert state.suppressed_lines > 400
    assert snapshot(state)["suppressed_lines"] == state.suppressed_lines


def test_new_download_restarts_the_collapsing():
    state = ProgressState()
    for value in (10, 50, 100):
        consume_line(state, f"{value}%")

    # 두 번째 내려받기가 시작되면 다시 남깁니다.
    entry = consume_line(state, "1%")

    assert entry is not None
    assert entry["text"] == "1%"


def test_ordinary_lines_are_never_collapsed():
    state = ProgressState()
    consume_line(state, "10%")

    warning = consume_line(state, "UserWarning: 조심하세요")
    after = consume_line(state, "12%")

    assert warning is not None
    # 퍼센트가 아닌 줄이 끼면 접기 상태가 초기화되어 다음 퍼센트도 남습니다.
    assert after is not None


@pytest.mark.parametrize("text", ("100%", " 100.0 % ", "0%"))
def test_percentage_shapes_are_recognised(text):
    state = ProgressState()

    assert consume_line(state, text) is not None
    assert state.last_percent is not None


@pytest.mark.parametrize("text", ("100%%", "진행 50%", "50% 완료", "abc%"))
def test_lines_that_only_contain_a_percentage_elsewhere_are_kept(text):
    state = ProgressState()

    assert consume_line(state, text)["text"] == text.strip()
    assert state.last_percent is None


def test_log_text_is_masked():
    state = ProgressState()

    entry = consume_line(state, "token=AKIAIOSFODNN7EXAMPLE 로 실패")

    assert "AKIAIOSFODNN7EXAMPLE" not in entry["text"]


# --- train emitter가 실제로 내보내는 형식 ------------------------------------


def emitted(event: str, **fields) -> str:
    """``src/pipelines/train/progress.py``의 ``ProgressEmitter.emit``과 같은 순서·형식.

    key 순서는 schema, event, run_id, 개별 필드, ts입니다.
    """

    payload = {"schema": "train.progress/1", "event": event, "run_id": "web-1"}
    payload.update(fields)
    payload["ts"] = "2026-08-05T01:22:33.123456Z"
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_parses_a_full_session_in_the_emitters_own_format():
    """train이 실제로 내보내는 줄을 그대로 넣어 계약을 고정합니다."""

    state = ProgressState()
    lines = [
        emitted(
            "run_started",
            architecture="fasterrcnn_resnet50_fpn",
            device="cpu",
            epochs=3,
            train_images=8,
            validation_images=4,
            class_count=2,
        ),
        emitted("epoch_started", epoch=1, epochs=3),
        emitted(
            "epoch_completed",
            epoch=1,
            epochs=3,
            train_loss=1.2345,
            validation_loss=1.1111,
            best_validation_loss=1.1111,
            best_epoch=1,
            is_best=True,
            epoch_seconds=2.5,
        ),
        "UserWarning: TypedStorage is deprecated",  # torch 경고가 섞여 들어옵니다
        emitted("epoch_started", epoch=2, epochs=3),
        emitted(
            "epoch_completed",
            epoch=2,
            epochs=3,
            train_loss=0.9,
            validation_loss=1.5,
            best_validation_loss=1.1111,
            best_epoch=1,
            is_best=False,
            epoch_seconds=2.5,
        ),
    ]
    for line in lines:
        consume_line(state, line)

    result = snapshot(state)

    assert result["available"] is True
    assert result["architecture"] == "fasterrcnn_resnet50_fpn"
    assert result["total_epochs"] == 3
    assert result["train_images"] == 8
    assert result["completed_epochs"] == 2
    assert result["current_epoch"] == 2
    assert result["percent"] == pytest.approx(66.7)
    assert result["best"] == {"epoch": 1, "validation_loss": 1.1111}
    assert result["eta_seconds"] == 2.5  # 평균 2.5초 x 남은 1 epoch
    json.dumps(result, allow_nan=False, ensure_ascii=False)


def test_emitter_null_loss_is_handled():
    """emitter는 NaN/inf를 null로 바꿔 내보냅니다."""

    state = ProgressState()
    consume_line(state, emitted("epoch_completed", epoch=1, epochs=2, train_loss=None))

    assert snapshot(state)["epochs"][0]["train_loss"] is None


def test_parses_an_early_stopped_session_in_the_emitters_own_format():
    """조기 종료된 실행을 train이 실제로 내보내는 줄 그대로 넣어 고정합니다."""

    state = ProgressState()
    lines = [
        emitted(
            "run_started",
            architecture="retinanet_resnet50_fpn",
            device="cpu",
            epochs=50,
            train_images=8,
            validation_images=4,
            class_count=2,
        ),
        emitted("epoch_started", epoch=1, epochs=50),
        emitted(
            "epoch_completed",
            epoch=1,
            epochs=50,
            train_loss=1.25,
            validation_loss=1.38,
            train_loss_components={"classification": 0.72, "bbox_regression": 0.53},
            validation_loss_components={"classification": 0.79, "bbox_regression": 0.59},
            best_validation_loss=1.38,
            best_epoch=1,
            is_best=True,
            epoch_seconds=2.5,
        ),
        emitted("epoch_started", epoch=2, epochs=50),
        emitted(
            "epoch_completed",
            epoch=2,
            epochs=50,
            train_loss=1.1,
            validation_loss=1.4,
            train_loss_components={"classification": 0.6, "bbox_regression": 0.5},
            validation_loss_components={"classification": 0.82, "bbox_regression": 0.58},
            best_validation_loss=1.38,
            best_epoch=1,
            is_best=False,
            epoch_seconds=2.5,
        ),
        emitted(
            "training_completed",
            planned_epochs=50,
            completed_epochs=2,
            stopped_early=True,
            best_epoch=1,
            best_validation_loss=1.38,
        ),
    ]
    for item in lines:
        consume_line(state, item)

    result = snapshot(state)

    assert result["finished"] is True
    assert result["stopped_early"] is True
    assert result["total_epochs"] == 50
    assert result["completed_epochs"] == 2
    assert result["percent"] == 100.0
    assert result["eta_seconds"] == 0.0
    assert result["best"] == {"epoch": 1, "validation_loss": 1.38}
    assert result["epochs"][1]["validation_loss_components"] == {
        "classification": 0.82,
        "bbox_regression": 0.58,
    }
    json.dumps(result, allow_nan=False, ensure_ascii=False)
