"""AWS Lambda resolver의 팀 권한과 상태 전이 test."""

from __future__ import annotations

import copy

import pytest

from src.pipelines.web.cloud import handler as cloud


class FakeTable:
    def __init__(self) -> None:
        self.items = {}

    def put_item(self, *, Item, ConditionExpression=None):
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
