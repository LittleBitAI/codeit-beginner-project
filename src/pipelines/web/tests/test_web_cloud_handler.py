"""AWS Lambda resolver의 팀 권한과 상태 전이, 그리고 배포되는 schema의 drift test."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from boto3.dynamodb.types import TypeSerializer

from src.pipelines.web.cloud import handler as cloud


CLOUD_DIR = Path(__file__).resolve().parents[1] / "cloud"


class FakeTable:
    def __init__(self) -> None:
        self.items = {}

    def put_item(self, *, Item, ConditionExpression=None):
        # 실제 DynamoDB resource와 같은 type 제약을 적용합니다. AppSync의 AWSJSON은
        # Lambda에 dict/list로 들어오므로 nested float도 여기서 잡아야 합니다.
        TypeSerializer().serialize(Item)
        key = (Item["PK"], Item["SK"])
        if ConditionExpression and key in self.items:
            raise AssertionError("이 test에서는 중복을 만들지 않습니다.")
        self.items[key] = copy.deepcopy(Item)
        return {}

    def get_item(self, *, Key, ConsistentRead=False):
        item = self.items.get((Key["PK"], Key["SK"]))
        return {"Item": copy.deepcopy(item)} if item else {}


def event(field, arguments, *, groups=None):
    return {
        "info": {"fieldName": field},
        "arguments": arguments,
        "identity": {
            "sub": "member-a",
            "claims": {
                "sub": "member-a",
                "cognito:username": "a@example.com",
                "cognito:groups": groups or [],
            },
        },
    }


@pytest.fixture
def table(monkeypatch):
    value = FakeTable()
    monkeypatch.setattr(cloud, "TABLE", value)
    monkeypatch.setenv("TEAM_ID", "pill-team")
    monkeypatch.setenv("TEAM_GROUP", "train-team")
    return value


def iam_event(field, arguments):
    """AppSync가 SigV4 요청에 붙여 주는 identity입니다. claims가 없습니다."""

    return {
        "info": {"fieldName": field},
        "arguments": arguments,
        "identity": {
            "accountId": "070351677013",
            "userArn": "arn:aws:iam::070351677013:user/teamlead",
            "username": "teamlead",
            "sourceIp": ["203.0.113.10"],
        },
    }


def create_input():
    return {
        "cloudRunId": "c" * 32,
        "localJobId": "a" * 32,
        "runId": "run-a",
        "settings": "{}",
        "dataInputs": "{}",
    }


def test_create_run_requires_fixed_team_membership(table):
    with pytest.raises(PermissionError):
        cloud.handler(
            event("createRun", {"teamId": "pill-team", "input": create_input()}), None
        )


def test_iam_caller_creates_a_run_without_browser_login(table):
    # Colab처럼 browser login을 못 하는 곳에서도 팀 화면에 기록이 남아야 합니다.
    payload = create_input()
    payload["actorName"] = "지현 (Colab)"

    created = cloud.handler(iam_event("createRun", {"teamId": "pill-team", "input": payload}), None)

    assert created["actorName"] == "지현 (Colab)"
    assert created["actorSub"] == "arn:aws:iam::070351677013:user/teamlead"
    # Cognito가 확인해 준 이름이 아니라 스스로 적은 이름임을 화면이 알아야 합니다.
    assert created["actorSource"] == "iam"


def test_iam_caller_must_say_who_it_is(table):
    with pytest.raises(ValueError, match="actorName"):
        cloud.handler(
            iam_event("createRun", {"teamId": "pill-team", "input": create_input()}), None
        )


def test_login_wins_over_a_self_declared_name(table):
    payload = create_input()
    payload["actorName"] = "다른 사람"

    created = cloud.handler(
        event("createRun", {"teamId": "pill-team", "input": payload}, groups=["train-team"]),
        None,
    )

    # 로그인 경로에서는 claims가 진실입니다. input의 이름은 무시합니다.
    assert created["actorName"] == "a@example.com"
    assert created["actorSource"] == "cognito"


def test_terminal_update_wins_even_after_revision_gap(table):
    created = cloud.handler(
        event(
            "createRun",
            {"teamId": "pill-team", "input": create_input()},
            groups=["train-team"],
        ),
        None,
    )
    assert created["actorName"] == "a@example.com"

    update = {
        "eventId": "event-1",
        "cloudRunId": "c" * 32,
        "revision": 0,
        "status": "interrupted",
        "startedAt": created["startedAt"],
        "finishedAt": "2026-08-05T00:01:00.000Z",
        "message": "PC 연결 중단",
        "progress": "{}",
        "summary": "{}",
        "artifacts": "{}",
        "heartbeatAt": "2026-08-05T00:01:00.000Z",
    }
    result = cloud.handler(
        event("publishRunUpdate", {"teamId": "pill-team", "input": update}), None
    )
    assert result["status"] == "interrupted"
    assert result["revision"] == 1


def test_appsync_parsed_awsjson_is_stored_as_json_text(table):
    create = create_input()
    create["settings"] = {"learning_rate": 0.0001, "weight_decay": 0.01}
    create["dataInputs"] = {"sizes": [320.0, 320.0]}
    created = cloud.handler(
        event(
            "createRun",
            {"teamId": "pill-team", "input": create},
            groups=["train-team"],
        ),
        None,
    )

    update = {
        "eventId": "event-1",
        "cloudRunId": "c" * 32,
        "revision": 1,
        "status": "running",
        "startedAt": created["createdAt"],
        "finishedAt": None,
        "message": None,
        "progress": {"train_loss": 1.25},
        "summary": {"best_validation_loss": 0.75},
        "artifacts": {"scores": [0.9]},
        "heartbeatAt": "2026-08-05T00:01:00.000Z",
    }
    updated = cloud.handler(
        event("publishRunUpdate", {"teamId": "pill-team", "input": update}), None
    )
    batch = cloud.handler(
        event(
            "publishLogBatch",
            {
                "teamId": "pill-team",
                "input": {
                    "eventId": "log-1",
                    "cloudRunId": "c" * 32,
                    "startSeq": 1,
                    "endSeq": 1,
                    "lines": [{"seq": 1, "loss": 0.5}],
                },
            },
        ),
        None,
    )

    assert json.loads(created["settings"]) == create["settings"]
    assert json.loads(created["dataInputs"]) == create["dataInputs"]
    assert json.loads(updated["progress"]) == update["progress"]
    assert json.loads(updated["summary"]) == update["summary"]
    assert json.loads(updated["artifacts"]) == update["artifacts"]
    assert json.loads(batch["lines"]) == [{"seq": 1, "loss": 0.5}]


def test_evaluation_is_kept_when_a_later_update_omits_it(table):
    created = cloud.handler(
        event(
            "createRun",
            {"teamId": "pill-team", "input": create_input()},
            groups=["train-team"],
        ),
        None,
    )
    assert created["evaluation"] == "{}"

    def update(revision, **extra):
        payload = {
            "eventId": f"event-{revision}",
            "cloudRunId": "c" * 32,
            "revision": revision,
            "status": "succeeded",
            "startedAt": created["createdAt"],
            "finishedAt": "2026-08-05T00:01:00.000Z",
            "message": None,
            "progress": "{}",
            "summary": "{}",
            "artifacts": "{}",
            "heartbeatAt": "2026-08-05T00:01:00.000Z",
            **extra,
        }
        return cloud.handler(
            event("publishRunUpdate", {"teamId": "pill-team", "input": payload}), None
        )

    # 이 배포 전에 만들어진 기록에는 evaluation 속성이 아예 없습니다.
    del table.items[(cloud._pk("pill-team"), cloud._run_sk("c" * 32))]["evaluation"]
    legacy = update(1)
    assert legacy["status"] == "succeeded"

    stored = update(2, evaluation={"status": "succeeded", "metrics": {"mAP": 0.73}})
    assert json.loads(stored["evaluation"])["metrics"]["mAP"] == 0.73

    # 30초마다 오는 heartbeat는 evaluation을 싣지 않습니다. 지우면 안 됩니다.
    heartbeat = update(3)
    assert json.loads(heartbeat["evaluation"])["metrics"]["mAP"] == 0.73


def test_rejects_a_different_team_before_reading_table(table):
    with pytest.raises(PermissionError, match="다른 teamId"):
        cloud.handler(
            event(
                "createRun",
                {"teamId": "another-team", "input": create_input()},
                groups=["train-team"],
            ),
            None,
        )


def _comparable(schema: str) -> str:
    """주석을 걷어내고 공백을 하나로 줄여 형식 차이를 무시합니다."""

    without_comments = [line.split("#", 1)[0] for line in schema.splitlines()]
    return " ".join(" ".join(without_comments).split())


def _inline_definition() -> str:
    """template.yaml 안에 인라인으로 박혀 있는 AppSync schema만 떼어 옵니다."""

    lines = (CLOUD_DIR / "template.yaml").read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == "Definition: |")
    body_indent = len(lines[start]) - len(lines[start].lstrip()) + 2
    body = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) < body_indent:
            break
        body.append(line)
    return "\n".join(body)


def test_deployed_schema_matches_the_reference_file():
    # AppSync에 실제로 올라가는 것은 template.yaml의 인라인 schema이고
    # schema.graphql은 읽기 좋으라고 둔 사본입니다. 사본만 고치고 배포하면
    # 아무것도 바뀌지 않은 채 성공합니다.
    reference = (CLOUD_DIR / "schema.graphql").read_text(encoding="utf-8")
    assert _comparable(_inline_definition()) == _comparable(reference)


def test_subscription_connection_also_requires_group(table):
    with pytest.raises(PermissionError):
        cloud.handler(event("onTeamRunChanged", {"teamId": "pill-team"}), None)
    assert (
        cloud.handler(
            event(
                "onTeamRunChanged",
                {"teamId": "pill-team"},
                groups=["train-team"],
            ),
            None,
        )
        is None
    )
