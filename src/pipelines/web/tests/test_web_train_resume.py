"""이어서 학습(train.resume_from)을 화면에서 시작하는 부분입니다.

`contracts/proposals/010-web-train-resume-mirror.md`가 요청한 내용입니다. train을
import할 수 없으므로 검증 규칙은 여기에 복제하고, 이어서 할 checkpoint 경로도 train이
정한 규칙을 그대로 옮겨 만듭니다.
"""

from __future__ import annotations

import json
import time

import pytest

from src.common import StorageError
from src.pipelines.web import train_config
from src.pipelines.web.errors import TeamSyncAuthError, WebValidationError
from src.pipelines.web.train_config import (
    build_resume_config,
    field_specs,
    next_resume_run_id,
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


def test_resume_config_shortens_a_long_original_name_without_losing_the_suffix():
    source = _runtime_config("local")
    source["train"]["run_id"] = "a" * 128

    resumed = build_resume_config(source)

    name = resumed["train"]["run_id"]
    assert len(name) == 128
    assert name.endswith(".2")
    assert train_config.RUN_ID_PATTERN.fullmatch(name)


def test_resume_names_read_as_a_lineage():
    """A 다음은 A.2, A.2 다음은 A.3입니다. A.2.2가 아닙니다."""

    assert next_resume_run_id("web-run") == "web-run.2"
    assert next_resume_run_id("web-run.2") == "web-run.3"


def test_resume_names_skip_a_number_that_is_already_used():
    """같은 이름을 다시 쓰면 train이 시작을 거부하고 두 실행이 한 이름으로 섞입니다."""

    assert next_resume_run_id("web-run", ["web-run.2", "web-run.3"]) == "web-run.4"


def test_a_name_that_cannot_take_a_suffix_falls_back_to_a_timestamp():
    """이름을 못 지어서 이어 학습이 막히면 안 됩니다."""

    name = next_resume_run_id("이름 with spaces")

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


def _interrupted_job(
    client, manager, monkeypatch, fake_process_factory, data_inputs, checkpoint=True
):
    """중단된 학습 하나를 만듭니다.

    기본으로 이어갈 checkpoint를 남깁니다. 그것이 보통의 모습이고, 이어하기는 어느
    상태에서든 checkpoint가 있어야 하기 때문입니다. 없는 경우를 보려면 `checkpoint=False`.
    """

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
    if checkpoint:
        config = read_runtime_config(record.config_id)
        path = run_id_working_path(config["train"]) / "last_checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"resumable")
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
    assert body["run_id"] == "web-run.2"
    assert body["resumed_from_job_id"] == record.job_id


def test_resume_availability_reports_a_real_checkpoint(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """화면이 단추를 세울지 서버에 물어봅니다.

    완료한 epoch 수로 셈하면 "저장됐을 가능성"만 알 뿐입니다. 이어온 실행처럼 앞선
    실행의 epoch이 섞여 있으면 셈은 맞는데 이 실행의 checkpoint는 없습니다. 두 쪽이
    같은 답을 보려면 실제 저장소를 본 쪽이 알려 줘야 합니다.
    """

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )

    response = client.get(f"/api/train/jobs/{record.job_id}/resume")

    assert response.status_code == 200, response.text
    assert response.json() == {"available": True, "reason": None}


def test_resume_availability_says_why_it_cannot_resume(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs, checkpoint=False
    )
    record.status = "cancelled"

    body = client.get(f"/api/train/jobs/{record.job_id}/resume").json()

    assert body["available"] is False
    assert "checkpoint" in body["reason"]


def test_resume_availability_lets_you_try_when_it_cannot_check(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """확인하지 못한 것과 이어갈 수 없는 것은 다릅니다.

    저장소를 못 읽었다고 단추를 없애면, 눌러서 알아낼 수 있는 것까지 막고 사람은
    새로고침 말고 할 것이 없습니다. 모를 때는 시도할 수 있게 두고 POST가 답합니다.
    """

    from src.pipelines.web.api import routes_train

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "cancelled"

    def explode(_config, _artifacts=None):
        raise StorageError("S3에 닿지 못했습니다.")

    monkeypatch.setattr(routes_train, "resume_checkpoint_exists", explode)

    body = client.get(f"/api/train/jobs/{record.job_id}/resume").json()

    assert body["available"] is True
    assert "확인하지 못했습니다" in body["reason"]


def test_resume_availability_refuses_a_run_that_has_not_finished(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """아직 끝나지 않은 학습은 저장소를 보지도 않고 답합니다."""

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "running"

    body = client.get(f"/api/train/jobs/{record.job_id}/resume").json()

    assert body["available"] is False


# --- 끝까지 간 학습 이어가기 ------------------------------------------------


def _finished_job(client, manager, monkeypatch, fake_process_factory, data_inputs):
    """끝까지 돈 학습 하나입니다. 게시된 checkpoint까지 만들어 둡니다.

    끝난 실행에는 작업 폴더가 없습니다. train이 그 자리를 공개 폴더로 옮기고
    ``last_checkpoint_uri``로 알려 주므로, 이어갈 파일도 거기 있습니다.
    """

    from src.pipelines.web.paths import resolve_within_repo

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "succeeded"
    record.summary = {"completed_epochs": 30, "stopped_early": False}
    published = "artifacts/experiments/completed/web-run/last_checkpoint.pt"
    record.artifacts = {"run_id": "web-run", "last_checkpoint_uri": published}
    path = resolve_within_repo(published, label="checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"resumable")
    return record


def test_a_finished_run_can_be_continued(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """best_epoch이 마지막 epoch이면 더 배울 것이 남아 있다는 뜻입니다."""

    record = _finished_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )

    assert client.get(f"/api/train/jobs/{record.job_id}/resume").json() == {
        "available": True,
        "reason": None,
    }

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 35})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["run_id"] == "web-run.2"
    assert body["resume_from"] == record.artifacts["last_checkpoint_uri"]
    assert read_runtime_config(body["config_id"])["train"]["epochs"] == 35


def test_a_second_resume_skips_the_name_still_waiting_in_the_queue(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """줄만 선 항목은 아직 job 기록이 아닙니다.

    대기열 항목은 실제로 시작할 때에야 `JobRecord`가 되므로, 아는 이름을 job 기록에서만
    세면 두 번째 이어 학습도 A.2를 받습니다. 둘 다 밤새 기다렸다가 뒤엣것이 이름
    충돌로 죽습니다.
    """

    record = _finished_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    # 다른 학습이 돌고 있어 줄만 서는 상태입니다. 대기열을 꺼내 가는 곳이 여기
    # 하나뿐이라, 멈춰 세우면 실제로 줄이 남아 있는 그 순간을 그대로 봅니다.
    monkeypatch.setattr(manager, "_start_next", lambda: None)

    first = client.post(f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 35})
    second = client.post(f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 35})

    assert first.json()["run_id"] == "web-run.2"
    assert second.json()["run_id"] == "web-run.3"
    assert [entry["run_id"] for entry in manager.queue_entries()] == [
        "web-run.2",
        "web-run.3",
    ]


def test_two_resumes_that_start_together_still_get_different_names(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """이름을 고르는 것과 그 이름이 대기열에 보이는 것 사이에 틈이 있으면 안 됩니다.

    FastAPI는 `def` route를 threadpool에서 돌리므로 두 요청이 실제로 동시에 들어옵니다.
    갈라져 있으면 둘 다 "A.2는 아직 없다"를 보고 같은 이름을 만들고, 뒤엣것은 밤새
    기다렸다 이름 충돌로 죽습니다.

    **틈을 일부러 벌려 놓고 봅니다.** 그냥 두 thread를 띄우면 대개 앞뒤로 지나가서,
    문이 없어도 초록으로 통과합니다.
    """

    import threading

    from src.pipelines.web.api import routes_train

    record = _finished_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    monkeypatch.setattr(manager, "_start_next", lambda: None)

    inside = threading.Event()
    release = threading.Event()
    real_write = routes_train.write_runtime_config

    def blocking_write(config):
        # 첫 요청이 이름을 고른 **직후** 멈춰 세웁니다. 여기가 위험한 창입니다.
        if config["train"]["run_id"] == "web-run.2":
            inside.set()
            release.wait(timeout=5)
        return real_write(config)

    monkeypatch.setattr(routes_train, "write_runtime_config", blocking_write)

    names: list[str] = []

    def resume() -> None:
        body = routes_train.resume_job(
            record.job_id, routes_train.ResumeRequest(epochs=35), None
        )
        names.append(body["run_id"])

    first = threading.Thread(target=resume, daemon=True)
    first.start()
    assert inside.wait(timeout=5), "첫 요청이 이름을 고르는 지점에 닿지 못했습니다"

    second = threading.Thread(target=resume, daemon=True)
    second.start()
    # 문이 없으면 이 사이에 두 번째가 같은 이름을 고르고 지나갑니다.
    time.sleep(0.3)
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert sorted(names) == ["web-run.2", "web-run.3"]


def test_continuing_a_finished_run_needs_a_bigger_plan(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """`epochs`는 남은 수가 아니라 전체 목표입니다.

    그대로 이어가면 "이미 지난 epoch보다 크지 않다"며 train이 거절하는데, 그 답은
    job과 config를 만들고 대기열을 다시 돌린 뒤에야 옵니다.
    """

    record = _finished_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )

    empty = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})
    same = client.post(f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 30})

    assert empty.status_code == 409 and "30" in empty.text
    assert same.status_code == 409 and "30" in same.text


def test_a_run_that_stopped_early_cannot_be_continued(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """조기 종료로 끝난 실행의 checkpoint에는 다 쓴 patience가 함께 들어 있습니다.

    **단추를 감추는 것만으로는 모자랍니다.** 시작하는 쪽이 같은 답을 하지 않으면,
    직접 부른 요청 하나가 반드시 실패할 학습을 대기열에 밀어 넣습니다.
    """

    record = _finished_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.summary["stopped_early"] = True

    body = client.get(f"/api/train/jobs/{record.job_id}/resume").json()
    started = client.post(
        f"/api/train/jobs/{record.job_id}/resume", json={"epochs": 35}
    )

    assert body["available"] is False
    assert "patience" in body["reason"]
    assert started.status_code == 409
    assert "patience" in started.text
    assert manager.queue_entries() == []


def test_resume_route_refuses_an_interrupted_job_without_a_checkpoint(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """중단된 학습도 checkpoint가 없으면 이어갈 수 없습니다.

    확인을 건너뛰면 새 설정과 job을 만들고 대기열까지 다시 돌린 뒤에야 train이
    checkpoint를 읽다가 죽습니다. 미리 말할 수 있는 실패입니다.
    """

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs, checkpoint=False
    )

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 409
    assert "checkpoint" in response.text


def test_resume_route_queues_a_failed_run_when_its_checkpoint_exists(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "failed"

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 201, response.text
    assert response.json()["resumed_from_job_id"] == record.job_id


def test_resume_route_refuses_a_failed_run_without_a_checkpoint(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs, checkpoint=False
    )
    record.status = "failed"

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 409
    assert "checkpoint" in response.text


def test_resume_route_queues_a_cancelled_run_when_its_checkpoint_exists(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """중지 단추를 누른 학습도 이어갈 수 있어야 합니다.

    실패와 다를 것이 없습니다. epoch마다 저장한 checkpoint가 그대로 남아 있는데도
    이어갈 방법이 없어, 밤새 돌린 학습을 처음부터 다시 돌려야 했습니다.
    """

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "cancelled"

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 201, response.text
    assert response.json()["resumed_from_job_id"] == record.job_id


def test_resume_route_refuses_a_cancelled_run_without_a_checkpoint(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """checkpoint 주기를 채우기 전에 중지하면 이어갈 것이 없습니다."""

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs, checkpoint=False
    )
    record.status = "cancelled"

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


def test_epochs_past_the_checkpoint_are_dropped_when_training_says_where_it_resumed(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    """checkpoint는 `checkpoint_every`마다 저장되지만 완료는 epoch마다 알려 옵니다.

    그래서 앞선 기록에는 checkpoint보다 뒤의 epoch가 남아 있을 수 있습니다. 그대로
    두면 다시 도는 epoch를 이미 끝난 것으로 세어, 진행률이 실제보다 앞섭니다.
    어느 checkpoint가 남아 있는지는 train만 알므로, 말해 줄 때까지 기다립니다.
    """

    from src.pipelines.web.jobs import runner

    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.progress = {
        "available": True,
        "epochs": [{"epoch": number, "validation_loss": 0.6} for number in range(1, 16)],
    }
    # 15까지 돌았지만 남아 있는 checkpoint는 10이라 train은 11부터 다시 시작합니다.
    lines = "".join(
        json.dumps(event) + "\n"
        for event in (
            {"schema": "train.progress/1", "event": "run_started", "run_id": "r", "epochs": 50},
            {"schema": "train.progress/1", "event": "epoch_started", "epoch": 11, "epochs": 50},
        )
    )
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stderr=lines))

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


def test_resume_route_refuses_a_job_that_has_not_finished(
    client, manager, monkeypatch, fake_process_factory, data_inputs
):
    record = _interrupted_job(
        client, manager, monkeypatch, fake_process_factory, data_inputs
    )
    record.status = "running"

    response = client.post(f"/api/train/jobs/{record.job_id}/resume", json={})

    assert response.status_code == 409
    assert "끝난 학습만" in response.text
