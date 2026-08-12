"""이어서 학습(train.resume_from)을 화면에서 시작하는 부분입니다.

`contracts/proposals/010-web-train-resume-mirror.md`가 요청한 내용입니다. train을
import할 수 없으므로 검증 규칙은 여기에 복제하고, 이어서 할 checkpoint 경로도 train이
정한 규칙을 그대로 옮겨 만듭니다.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from src.pipelines.web import train_config
from src.pipelines.web.errors import TeamSyncAuthError, WebValidationError
from src.pipelines.web.train_config import (
    build_resume_config,
    field_specs,
    normalize_train_settings,
    read_runtime_config,
    resume_checkpoint_exists,
    resume_checkpoint_uri,
    check_run_id_collision,
    run_id_working_path,
)


def _settings(**overrides) -> dict:
    settings = {"run_id": "web-run", "output_dir": "artifacts/experiments/completed"}
    settings.update(overrides)
    return settings


# --- train 복제본 -----------------------------------------------------------


def test_checkpoint_every_matches_the_train_default():
    """train이 _integer로 받는 기본값과 같아야 합니다.

    test_web_train_contract.py가 train source에서 뽑은 값과 이 값을 대조합니다.
    """

    assert normalize_train_settings({})["checkpoint_every"] == 1


def test_checkpoint_every_appears_on_the_new_experiment_form():
    names = [spec["name"] for spec in field_specs()]

    assert "checkpoint_every" in names


def test_a_run_that_starts_from_scratch_sends_no_resume_from():
    """train은 key가 없으면 처음부터 학습합니다. 빈 값을 실어 보내면 안 됩니다."""

    assert "resume_from" not in normalize_train_settings({})


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        17,
        "../outside.pt",
        "https://example.com/checkpoint.pt",
    ],
)
def test_unusable_resume_from_values_are_rejected(value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"resume_from": value})

    assert any(item["field"] == "train.resume_from" for item in error.value.as_list())


@pytest.mark.parametrize(
    "value",
    [
        "artifacts/experiments/completed/.web-run.partial/last_checkpoint.pt",
        "s3://bucket/experiments/completed/web-run/running/last_checkpoint.pt",
    ],
)
def test_a_usable_resume_from_reaches_train_unchanged(value):
    assert normalize_train_settings({"resume_from": value})["resume_from"] == value


# --- 이름 충돌 --------------------------------------------------------------


def test_run_id_collision_also_looks_at_an_interrupted_working_directory(isolated_repo):
    """train은 작업 폴더가 비어 있지 않으면 시작을 거부합니다.

    화면이 그 자리를 보지 않으면 train과 다른 답을 주게 됩니다.
    """

    settings = _settings()
    working = run_id_working_path(settings)
    working.mkdir(parents=True)
    (working / "last_checkpoint.pt").write_bytes(b"leftover")

    with pytest.raises(WebValidationError) as error:
        check_run_id_collision(settings, {"train_manifest_uri": "artifacts/a.json"})

    assert "이어서" in error.value.as_list()[0]["message"]


def test_an_empty_working_directory_does_not_block_a_run(isolated_repo):
    settings = _settings()
    run_id_working_path(settings).mkdir(parents=True)

    check_run_id_collision(settings, {"train_manifest_uri": "artifacts/a.json"})


# --- 이어서 할 checkpoint 경로 ----------------------------------------------


def _runtime_config(backend: str) -> dict:
    storage = (
        {"backend": "s3", "s3": {"prefix": ""}}
        if backend == "s3"
        else {"backend": "local", "local": {"root": "artifacts"}}
    )
    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "real"},
        "storage": storage,
        "train": {
            "run_id": "web-run",
            "epochs": 50,
            "output_dir": "artifacts/experiments/completed",
            "output_prefix": "experiments/completed",
        },
        "inputs": {"data": {"train_manifest_uri": "artifacts/a.json"}},
    }


def test_local_runs_resume_from_the_working_directory():
    uri = resume_checkpoint_uri(_runtime_config("local"))

    assert uri == (
        "artifacts/experiments/completed/.web-run.partial/last_checkpoint.pt"
    )


def test_s3_runs_resume_from_the_running_prefix(monkeypatch):
    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "team-bucket")

    uri = resume_checkpoint_uri(_runtime_config("s3"))

    assert uri == (
        "s3://team-bucket/experiments/completed/web-run/running/last_checkpoint.pt"
    )


def test_s3_runs_resume_from_the_common_storage_prefix(monkeypatch):
    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "team-bucket")
    monkeypatch.setenv("PILL_STORAGE_S3_PREFIX", "/team-space/")

    uri = resume_checkpoint_uri(_runtime_config("s3"))

    assert uri == (
        "s3://team-bucket/team-space/experiments/completed/"
        "web-run/running/last_checkpoint.pt"
    )


def test_s3_resume_checks_the_complete_checkpoint_uri(monkeypatch):
    checked = []

    class FakeStorage:
        def exists(self, location):
            checked.append(location)
            return True

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "team-bucket")
    monkeypatch.setattr(train_config, "create_storage", lambda _config: FakeStorage())

    assert resume_checkpoint_exists(_runtime_config("s3")) is True
    assert checked == [
        "s3://team-bucket/experiments/completed/web-run/running/last_checkpoint.pt"
    ]


def test_s3_runs_say_what_is_missing_without_a_bucket():
    with pytest.raises(WebValidationError) as error:
        resume_checkpoint_uri(_runtime_config("s3"))

    assert "PILL_STORAGE_S3_BUCKET" in error.value.as_list()[0]["message"]


# --- 이어서 학습 config -----------------------------------------------------


def test_resume_config_keeps_the_whole_plan_and_takes_a_new_name():
    resumed = build_resume_config(_runtime_config("local"))

    train = resumed["train"]
    assert train["run_id"] != "web-run"
    # epochs는 남은 수가 아니라 전체 목표입니다. 그대로 둡니다.
    assert train["epochs"] == 50
    assert train["resume_from"] == (
        "artifacts/experiments/completed/.web-run.partial/last_checkpoint.pt"
    )
    assert resumed["inputs"] == _runtime_config("local")["inputs"]


def test_resume_config_shortens_a_long_original_name_without_losing_the_suffix(
    monkeypatch,
):
    monkeypatch.setattr(
        train_config,
        "_utc_now",
        lambda: datetime(2026, 8, 10, 9, 8, 7, 654321, tzinfo=timezone.utc),
    )
    source = _runtime_config("local")
    source["train"]["run_id"] = "a" * 128

    resumed = build_resume_config(source)

    name = resumed["train"]["run_id"]
    assert len(name) == 128
    assert name.endswith("-resume-20260810T090807654321Z")
    assert train_config.RUN_ID_PATTERN.fullmatch(name)


def test_resume_carries_every_setting_it_does_not_deliberately_change():
    """이어서 하는 실행은 원 실행과 같은 설정으로 돌아야 합니다.

    개별 항목을 하나씩 확인하면 **나중에 생기는 설정**이 조용히 빠집니다. 빠진 값은
    train의 기본값으로 대체되므로 오류가 나지 않고, 이어서 한 실행만 다른 설정으로
    학습됩니다. 그래서 바꾸기로 한 세 가지 말고는 전부 그대로인지 봅니다.
    """

    source = _runtime_config("local")
    source["train"].update(
        {"gradient_accumulation_steps": 4, "architecture": "retinanet_resnet50_fpn_v2"}
    )

    resumed = build_resume_config(source)

    changed = {"run_id", "resume_from"}
    assert set(resumed["train"]) - set(source["train"]) == {"resume_from"}
    for name, value in source["train"].items():
        if name in changed:
            continue
        assert resumed["train"][name] == value, f"train.{name}이(가) 이어지지 않았습니다."


def test_resume_config_can_extend_the_plan():
    resumed = build_resume_config(_runtime_config("local"), epochs=80)

    assert resumed["train"]["epochs"] == 80


def test_resume_config_does_not_touch_the_source():
    source = _runtime_config("local")

    build_resume_config(source)

    assert source["train"]["run_id"] == "web-run"
    assert "resume_from" not in source["train"]


def test_resume_config_rejects_a_bad_run_id():
    with pytest.raises(WebValidationError):
        build_resume_config(_runtime_config("local"), run_id="이름 with spaces")


# --- route ------------------------------------------------------------------


def _interrupted_job(client, manager, monkeypatch, fake_process_factory, data_inputs):
    from src.pipelines.web.jobs import runner

    created = client.post(
        "/api/train/configs",
        json={
            "train": {"run_id": "web-run", "epochs": 50},
            "inputs": {"data": data_inputs},
        },
    )
    assert created.status_code == 201, created.text
    config_id = created.json()["config_id"]
    process = fake_process_factory()
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    started = manager.start(config_id)
    deadline = time.monotonic() + 10
    while manager._active_job_id is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert manager._active_job_id is None, "test 학습이 종료되지 않았습니다."
    record = manager.get(started.job_id)
    record.status = "interrupted"
    return record


def test_resume_route_queues_a_new_run_from_an_interrupted_job(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["run_id"].startswith("web-run-resume-")
    assert body["resumed_from_job_id"] == record.job_id


def test_resume_route_queues_a_failed_run_when_its_checkpoint_exists(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "failed"
    config = read_runtime_config(record.config_id)
    checkpoint = run_id_working_path(config["train"]) / "last_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"resumable")

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 201, response.text
    assert response.json()["resumed_from_job_id"] == record.job_id


def test_resume_route_refuses_a_failed_run_without_a_checkpoint(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "failed"

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 409
    assert "checkpoint" in response.text


def test_resume_route_passes_the_login_token_to_the_started_run(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    from src.pipelines.web import team_sync

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    received_tokens = []

    class TokenCheckingSync:
        def create_run(self, *, access_token, **_kwargs):
            received_tokens.append(access_token)
            return None

        def enqueue_update(self, _record):
            return None

        def enqueue_log(self, _record, _entry):
            return None

    monkeypatch.setattr(team_sync, "get_team_sync", lambda: TokenCheckingSync())

    response = client.post(
        f"/api/train/jobs/{record.job_id}/resume",
        json={},
        headers={"Authorization": "Bearer user-token"},
    )

    assert response.status_code == 201, response.text
    assert received_tokens == ["user-token"]


def test_resume_route_reports_team_sync_auth_failure_and_keeps_the_request(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    from src.pipelines.web import team_sync

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )

    class RejectingSync:
        def create_run(self, **_kwargs):
            raise TeamSyncAuthError("로그인 token이 거절됐습니다.")

    monkeypatch.setattr(team_sync, "get_team_sync", lambda: RejectingSync())

    response = client.post(
        f"/api/train/jobs/{record.job_id}/resume",
        json={},
        headers={"Authorization": "Bearer rejected-token"},
    )

    assert response.status_code == 401, response.text
    assert "거절" in response.text
    assert manager.queue_paused() is True
    assert len(manager.queue_entries()) == 1
    assert manager.queue_entries()[0]["run_id"] != record.run_id


def test_resumed_run_keeps_the_loss_curve_of_the_run_it_continues(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """앞선 epoch는 checkpoint 안에만 있고 진행 log로는 오지 않습니다.

    채우지 않으면 11 epoch부터 이어서 한 실행의 손실 그래프가 11에서 시작해, 그전
    곡선이 사라진 것처럼 보입니다.
    """

    from src.pipelines.web.jobs import runner

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.progress = {
        "available": True,
        "epochs": [
            {"epoch": number, "train_loss": 0.5, "validation_loss": 0.6}
            for number in range(1, 11)
        ],
    }
    line = json.dumps(
        {
            "schema": "train.progress/1",
            "event": "epoch_completed",
            "epoch": 11,
            "epochs": 12,
            "train_loss": 0.3,
            "validation_loss": 0.4,
        }
    )
    # 진행 event는 stderr로 옵니다. stdout은 마지막 결과 JSON 문서 하나입니다.
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stderr=line + "\n"))

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 12})

    assert response.status_code == 201, response.text
    started = manager.get(response.json()["started"]["job_id"])
    deadline = time.monotonic() + 10
    while manager._active_job_id is not None and time.monotonic() < deadline:
        time.sleep(0.02)

    assert [entry["epoch"] for entry in started.progress["epochs"]] == list(range(1, 12))
    assert started.progress["completed_epochs"] == 11


def test_epochs_past_the_checkpoint_are_dropped_before_training_even_starts(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """checkpoint는 `checkpoint_every`마다 저장되지만 완료는 epoch마다 알려 옵니다.

    그래서 앞선 기록에는 checkpoint보다 뒤의 epoch가 남아 있을 수 있습니다. 그대로
    두면 다시 도는 epoch를 이미 끝난 것으로 세어, 진행률이 실제보다 앞섭니다.
    train이 model을 만드는 동안에도 화면에 보이므로 미리 잘라 냅니다.
    """

    from src.pipelines.web.jobs import runner

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    # 10 epoch마다 저장했으므로 15까지 돌았어도 남아 있는 checkpoint는 10입니다.
    record.settings = {**record.settings, "checkpoint_every": 10}
    record.progress = {
        "available": True,
        "epochs": [{"epoch": number, "validation_loss": 0.6} for number in range(1, 16)],
    }
    # 첫 epoch event가 오기 전에도 계획만 알려 준 상태에서 이미 맞아야 합니다.
    line = json.dumps(
        {"schema": "train.progress/1", "event": "run_started", "run_id": "r", "epochs": 50}
    )
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stderr=line + "\n"))

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 50})

    assert response.status_code == 201, response.text
    started = manager.get(response.json()["started"]["job_id"])
    deadline = time.monotonic() + 10
    while manager._active_job_id is not None and time.monotonic() < deadline:
        time.sleep(0.02)

    assert [entry["epoch"] for entry in started.progress["epochs"]] == list(range(1, 11))
    assert started.progress["completed_epochs"] == 10


def test_a_queued_resume_still_finds_its_curve_after_the_run_id_is_reused(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """앞선 실행을 이름이 아니라 job id로 찾습니다.

    이름으로 찾으면 같은 이름을 가진 다른 기록의 곡선이 붙을 수 있고, 대기열에
    줄을 선 항목은 나중에 시작할 때 그 출처를 잃습니다.
    """

    from src.pipelines.web.jobs import runner
    from src.pipelines.web.jobs.model import JobRecord

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.progress = {"available": True, "epochs": [{"epoch": 1, "validation_loss": 0.6}]}
    # 같은 이름을 가진 남의 기록이 먼저 들어 있어도 헷갈리면 안 됩니다.
    twin = JobRecord(job_id="twin", config_id="twin", run_id=record.run_id)
    twin.progress = {"available": True, "epochs": [{"epoch": 9, "validation_loss": 9.9}]}
    manager._records = {"twin": twin, **manager._records}
    line = json.dumps(
        {"schema": "train.progress/1", "event": "epoch_started", "epoch": 2, "epochs": 12}
    )
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stderr=line + "\n"))

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 12})

    assert response.status_code == 201, response.text
    started = manager.get(response.json()["started"]["job_id"])
    deadline = time.monotonic() + 10
    while manager._active_job_id is not None and time.monotonic() < deadline:
        time.sleep(0.02)

    assert [entry["epoch"] for entry in started.progress["epochs"]] == [1]


def test_a_run_started_from_scratch_gets_no_borrowed_epochs(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """이어서 하지 않는 실행에 남의 곡선을 붙이면 안 됩니다."""

    from src.pipelines.web.jobs import runner

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.progress = {"available": True, "epochs": [{"epoch": 1, "train_loss": 0.5}]}
    created = client.post(
        "/api/train/configs",
        json={"train": {"run_id": "fresh-run", "epochs": 3}, "inputs": {"data": data_inputs}},
    )
    line = json.dumps(
        {"schema": "train.progress/1", "event": "epoch_completed", "epoch": 1, "epochs": 3}
    )
    # 진행 event는 stderr로 옵니다. stdout은 마지막 결과 JSON 문서 하나입니다.
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stderr=line + "\n"))

    started = manager.start(created.json()["config_id"])
    deadline = time.monotonic() + 10
    while manager._active_job_id is not None and time.monotonic() < deadline:
        time.sleep(0.02)

    assert [entry["epoch"] for entry in started.progress["epochs"]] == [1]


def test_resume_route_refuses_a_job_that_is_not_interrupted(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "succeeded"

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 409
    assert "중단된" in response.text
