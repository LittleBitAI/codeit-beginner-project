"""팀 동기화의 인증, outbox, masking 회귀 test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.pipelines.web.errors import TeamSyncAuthError, TeamSyncError
from src.pipelines.web.jobs.model import JobRecord
from src.pipelines.web.team_sync import TeamSync, TeamSyncConfig


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, query, variables, *, access_token=None, iam=False):
        self.calls.append(
            {
                "query": query,
                "variables": variables,
                "access_token": access_token,
                "iam": iam,
            }
        )
        cloud_id = variables.get("input", {}).get("cloudRunId")
        if "CreateRun" in query:
            return {"createRun": {"cloudRunId": cloud_id, "status": "starting", "revision": 0}}
        if "PublishRunUpdate" in query:
            return {"publishRunUpdate": {"cloudRunId": cloud_id}}
        return {"publishLogBatch": {"cloudRunId": cloud_id}}


def enabled_config() -> TeamSyncConfig:
    return TeamSyncConfig(
        enabled=True,
        team_id="pill-team",
        endpoint="https://example.appsync-api.ap-northeast-2.amazonaws.com/graphql",
        region="ap-northeast-2",
        user_pool_id="ap-northeast-2_example",
        user_pool_client_id="client",
        cognito_domain="example.auth.ap-northeast-2.amazoncognito.com",
    )


def test_enabled_config_requires_every_non_secret_value():
    with pytest.raises(Exception, match="PILL_TEAM_ID"):
        TeamSyncConfig.from_environment({"PILL_TEAM_SYNC_ENABLED": "true"})


def test_create_run_requires_login_token(isolated_repo):
    sync = TeamSync(enabled_config(), transport=FakeTransport())
    with pytest.raises(TeamSyncAuthError, match="로그인"):
        sync.create_run(
            access_token=None,
            local_job_id="a" * 32,
            run_id="run-a",
            settings={},
            data_inputs={},
        )


def test_actor_name_lets_a_headless_run_register_without_login(isolated_repo):
    # Colab에는 browser login이 없습니다. 이름을 미리 정해 두면 IAM으로 기록합니다.
    transport = FakeTransport()
    sync = TeamSync(
        TeamSyncConfig(**{**enabled_config().__dict__, "actor_name": "지현 (Colab)"}),
        transport=transport,
    )

    cloud_run_id = sync.create_run(
        access_token=None,
        local_job_id="a" * 32,
        run_id="run-a",
        settings={},
        data_inputs={},
    )

    assert cloud_run_id
    call = transport.calls[-1]
    assert call["iam"] is True
    assert call["access_token"] is None
    assert call["variables"]["input"]["actorName"] == "지현 (Colab)"


def test_login_path_does_not_send_a_self_declared_name(isolated_repo):
    transport = FakeTransport()
    sync = TeamSync(
        TeamSyncConfig(**{**enabled_config().__dict__, "actor_name": "지현 (Colab)"}),
        transport=transport,
    )

    sync.create_run(
        access_token="user-token",
        local_job_id="a" * 32,
        run_id="run-a",
        settings={},
        data_inputs={},
    )

    call = transport.calls[-1]
    assert call["access_token"] == "user-token"
    assert "actorName" not in call["variables"]["input"]


def test_outbox_batches_logs_and_masks_sensitive_values(isolated_repo, monkeypatch):
    transport = FakeTransport()
    sync = TeamSync(enabled_config(), transport=transport)
    monkeypatch.setattr(sync, "_ensure_worker", lambda: None)
    record = JobRecord(
        job_id="a" * 32,
        config_id="b" * 32,
        run_id="run-a",
        cloud_run_id="c" * 32,
    )
    sync.enqueue_log(
        record,
        {
            "seq": 1,
            "stream": "stderr",
            "level": "info",
            "text": "Authorization: Bearer secret-token",
            "ts": "2026-08-05T00:00:00.000Z",
        },
    )
    sync.enqueue_log(
        record,
        {
            "seq": 2,
            "stream": "stderr",
            "level": "info",
            "text": r"C:\Users\person\private\file.txt",
            "ts": "2026-08-05T00:00:01.000Z",
        },
    )

    assert sync.publish_pending() == 1
    variables = transport.calls[-1]["variables"]
    lines = json.loads(variables["input"]["lines"])
    assert [line["seq"] for line in lines] == [1, 2]
    assert "secret-token" not in json.dumps(lines)
    assert "C:\\Users\\person" not in json.dumps(lines)
    assert variables["input"]["expiresAt"] > 0


def test_update_payload_carries_metrics_without_local_paths(isolated_repo, monkeypatch):
    transport = FakeTransport()
    sync = TeamSync(enabled_config(), transport=transport)
    monkeypatch.setattr(sync, "_ensure_worker", lambda: None)
    record = JobRecord(
        job_id="a" * 32,
        config_id="b" * 32,
        run_id="run-a",
        status="succeeded",
        cloud_run_id="c" * 32,
    )
    # 평가 기록 전체에는 로컬 경로와 storage 설정이 들어 있습니다. 팀에게는 지표만
    # 보내야 합니다.
    record.evaluation = {
        "status": "succeeded",
        "finished_at": "2026-08-05T00:02:00.000Z",
        "message": "평가를 마쳤습니다.",
        "exit_code": 0,
        "artifacts": {"metrics_uri": r"C:\Users\person\artifacts\metrics.json"},
        "summary": {
            "metrics": {
                "mAP": 0.7348,
                "mAP50": None,
                "mAP75": 0.9726,
                "precision50": None,
                "recall50": None,
            }
        },
        "settings": {"checkpoint_uri": r"C:\Users\person\best.pt"},
        "storage": {"backend": "s3", "s3": {"secret_access_key": "very-secret"}},
    }
    record.registration = {"status": "succeeded"}

    sync.enqueue_update(record)

    assert sync.publish_pending() == 1
    payload = transport.calls[-1]["variables"]["input"]
    evaluation = json.loads(payload["evaluation"])
    assert evaluation["status"] == "succeeded"
    assert evaluation["registration_status"] == "succeeded"
    assert evaluation["metrics"]["mAP"] == 0.7348
    assert evaluation["metrics"]["mAP50"] is None
    encoded = json.dumps(evaluation)
    assert "C:\\Users\\person" not in encoded
    assert "very-secret" not in encoded


def test_update_payload_omits_evaluation_before_it_exists(isolated_repo, monkeypatch):
    transport = FakeTransport()
    sync = TeamSync(enabled_config(), transport=transport)
    monkeypatch.setattr(sync, "_ensure_worker", lambda: None)
    record = JobRecord(
        job_id="a" * 32,
        config_id="b" * 32,
        run_id="run-a",
        cloud_run_id="c" * 32,
    )

    sync.enqueue_update(record)

    assert sync.publish_pending() == 1
    # 빈 값을 실어 보내면 나중에 도착한 heartbeat가 평가 지표를 덮어씁니다.
    assert "evaluation" not in transport.calls[-1]["variables"]["input"]


def test_public_config_never_contains_credentials():
    public = enabled_config().public_dict()
    assert set(public) == {
        "enabled",
        "team_id",
        "appsync_url",
        "region",
        "user_pool_id",
        "user_pool_client_id",
        "cognito_domain",
    }
    assert not any("secret" in key or "credential" in key for key in public)


def test_team_config_endpoint_is_disabled_by_default(client):
    response = client.get("/api/team/config")
    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "team_id": None,
        "appsync_url": None,
        "region": "ap-northeast-2",
        "user_pool_id": None,
        "user_pool_client_id": None,
        "cognito_domain": None,
    }


def test_cloud_record_failure_prevents_local_process_start(
    client, valid_payload, monkeypatch
):
    from src.pipelines.web import team_sync
    from src.pipelines.web.jobs import runner

    class FailingSync:
        def create_run(self, **_kwargs):
            raise TeamSyncError("원격 시작 기록 실패")

        def enqueue_update(self, _record):
            raise AssertionError("시작 기록 실패 뒤 update하면 안 됩니다.")

    spawn = lambda *_args, **_kwargs: pytest.fail("로컬 process가 시작되면 안 됩니다.")
    monkeypatch.setattr(team_sync, "get_team_sync", lambda: FailingSync())
    monkeypatch.setattr(runner, "spawn", spawn)
    created = client.post("/api/train/configs", json=valid_payload).json()

    response = client.post(
        "/api/train/jobs",
        json={"config_id": created["config_id"]},
        headers={"Authorization": "Bearer user-token"},
    )

    assert response.status_code == 503
    assert "원격 시작 기록 실패" in response.json()["message"]
