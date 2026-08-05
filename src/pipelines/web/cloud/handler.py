"""AppSync Lambda resolver. Cognito 팀 경계와 DynamoDB 상태를 한곳에서 지킵니다."""

from __future__ import annotations

import os
import time
import json
from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError


TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
RUN_PREFIX = "RUN#"
LOG_PREFIX = "LOG#"
TABLE: Any = None


def _table() -> Any:
    global TABLE
    if TABLE is None:
        TABLE = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
    return TABLE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _groups(identity: dict[str, Any]) -> set[str]:
    value = (identity.get("claims") or {}).get("cognito:groups", [])
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value.strip("[]").split(",")
        value = decoded if isinstance(decoded, list) else [decoded]
    return {str(item) for item in value}


def _require_member(event: dict[str, Any]) -> dict[str, Any]:
    identity = event.get("identity") or {}
    if os.environ.get("TEAM_GROUP", "train-team") not in _groups(identity):
        raise PermissionError("이 팀의 학습 기록에 접근할 권한이 없습니다.")
    return identity


def _pk(team_id: str) -> str:
    return f"TEAM#{team_id}"


def _run_sk(cloud_run_id: str) -> str:
    return f"{RUN_PREFIX}{cloud_run_id}"


def _public(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in {"PK", "SK", "expiresAt", "lastEventId"}}


def _create(event: dict[str, Any], team_id: str, data: dict[str, Any]) -> dict[str, Any]:
    identity = _require_member(event)
    claims = identity.get("claims") or {}
    timestamp = _now()
    item = {
        "PK": _pk(team_id),
        "SK": _run_sk(data["cloudRunId"]),
        "teamId": team_id,
        "cloudRunId": data["cloudRunId"],
        "localJobId": data["localJobId"],
        "runId": data["runId"],
        "actorSub": identity.get("sub") or claims.get("sub") or "unknown",
        "actorName": claims.get("cognito:username") or claims.get("email") or "팀원",
        "status": "starting",
        "settings": data["settings"],
        "dataInputs": data["dataInputs"],
        "progress": "{}",
        "summary": "{}",
        "artifacts": "{}",
        "message": None,
        "createdAt": timestamp,
        "startedAt": None,
        "finishedAt": None,
        "heartbeatAt": timestamp,
        "revision": 0,
    }
    try:
        _table().put_item(Item=item, ConditionExpression="attribute_not_exists(PK)")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
        raise ValueError("이미 존재하는 cloud run ID입니다.") from error
    return _public(item)


def _update(team_id: str, data: dict[str, Any]) -> dict[str, Any]:
    key = {"PK": _pk(team_id), "SK": _run_sk(data["cloudRunId"])}
    response = _table().get_item(Key=key, ConsistentRead=True)
    current = response.get("Item")
    if not current:
        raise ValueError("업데이트할 팀 학습 기록이 없습니다.")
    if current.get("lastEventId") == data["eventId"]:
        return _public(current)
    revision = int(data["revision"])
    if revision <= int(current.get("revision", 0)) and data["status"] not in TERMINAL:
        return _public(current)
    updated = dict(current)
    updated.update(
        {
            "status": data["status"],
            "startedAt": data.get("startedAt"),
            "finishedAt": data.get("finishedAt"),
            "message": data.get("message"),
            "progress": data["progress"],
            "summary": data["summary"],
            "artifacts": data["artifacts"],
            "heartbeatAt": data["heartbeatAt"],
            "revision": max(revision, int(current.get("revision", 0)) + 1),
            "lastEventId": data["eventId"],
        }
    )
    _table().put_item(Item=updated)
    return _public(updated)


def _publish_logs(team_id: str, data: dict[str, Any]) -> dict[str, Any]:
    timestamp = _now()
    item = {
        "PK": _pk(team_id),
        "SK": f"{LOG_PREFIX}{data['cloudRunId']}#{int(data['startSeq']):020d}",
        "teamId": team_id,
        "cloudRunId": data["cloudRunId"],
        "startSeq": int(data["startSeq"]),
        "endSeq": int(data["endSeq"]),
        "lines": data["lines"],
        "createdAt": timestamp,
        "expiresAt": int(time.time()) + int(os.environ.get("LOG_RETENTION_DAYS", "30")) * 86400,
        "lastEventId": data["eventId"],
    }
    try:
        _table().put_item(Item=item, ConditionExpression="attribute_not_exists(PK)")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
        existing = _table().get_item(Key={"PK": item["PK"], "SK": item["SK"]}).get("Item")
        if existing:
            item = existing
    return _public(item)


def _team_runs(team_id: str, limit: int) -> list[dict[str, Any]]:
    response = _table().query(
        KeyConditionExpression=Key("PK").eq(_pk(team_id)) & Key("SK").begins_with(RUN_PREFIX)
    )
    items = sorted(response.get("Items", []), key=lambda item: item["createdAt"], reverse=True)
    return [_public(item) for item in items[: max(1, min(limit, 100))]]


def _run(team_id: str, cloud_run_id: str) -> dict[str, Any] | None:
    item = _table().get_item(
        Key={"PK": _pk(team_id), "SK": _run_sk(cloud_run_id)}, ConsistentRead=True
    ).get("Item")
    return _public(item) if item else None


def _logs(team_id: str, cloud_run_id: str, after: int, limit: int) -> list[dict[str, Any]]:
    response = _table().query(
        KeyConditionExpression=Key("PK").eq(_pk(team_id))
        & Key("SK").begins_with(f"{LOG_PREFIX}{cloud_run_id}#")
    )
    items = [item for item in response.get("Items", []) if int(item["endSeq"]) > after]
    items.sort(key=lambda item: int(item["startSeq"]))
    return [_public(item) for item in items[: max(1, min(limit, 500))]]


def handler(event: dict[str, Any], _context: Any) -> Any:
    field = event["info"]["fieldName"]
    args = event.get("arguments") or {}
    team_id = str(args.get("teamId") or "")
    if not team_id:
        raise ValueError("teamId가 필요합니다.")
    configured_team = os.environ.get("TEAM_ID")
    if configured_team and team_id != configured_team:
        raise PermissionError("설정된 팀과 다른 teamId에는 접근할 수 없습니다.")
    if field == "createRun":
        return _create(event, team_id, args["input"])
    if field == "publishRunUpdate":
        return _update(team_id, args["input"])
    if field == "publishLogBatch":
        return _publish_logs(team_id, args["input"])
    _require_member(event)
    if field == "teamRuns":
        return _team_runs(team_id, int(args.get("limit") or 50))
    if field == "run":
        return _run(team_id, args["cloudRunId"])
    if field == "runLogs":
        return _logs(
            team_id,
            args["cloudRunId"],
            int(args.get("afterSeq") or 0),
            int(args.get("limit") or 100),
        )
    if field in {"onTeamRunChanged", "onRunLogBatch"}:
        return None
    raise ValueError(f"지원하지 않는 field입니다: {field}")
