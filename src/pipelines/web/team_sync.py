"""로컬 학습 상태를 AWS AppSync로 안전하게 전달합니다.

동기화는 명시적으로 켰을 때만 동작합니다. 전송할 event는 먼저 저장소 안의 outbox에
기록하므로 browser나 network가 끊겨도 다음 실행에서 이어 보낼 수 있습니다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from .errors import TeamSyncAuthError, TeamSyncError
from .masking import redact
from .paths import web_state_dir


__all__ = ["TeamSync", "TeamSyncConfig", "get_team_sync", "reset_team_sync"]


# AppSync subscription은 스스로 기록을 다시 읽지 않습니다. mutation이 고른 field를
# 그대로 구독자에게 흘려보내고, 고르지 않은 field는 null로 채웁니다. 그래서 여기서
# 좁게 고르면 팀 활동 화면이 settings·summary·evaluation·lines를 null로 받아
# 모델명과 optimizer가 `-`가 되고 실시간 로그가 멈춥니다. 화면이 구독하는 field
# 목록(frontend/src/team/cloud.ts의 RUN_FIELDS·LOG_FIELDS)과 같게 유지하세요.
RUN_FIELDS = """
  teamId cloudRunId localJobId runId actorSub actorName actorSource status settings dataInputs
  progress summary artifacts evaluation message createdAt startedAt finishedAt heartbeatAt
  revision
"""

LOG_FIELDS = "teamId cloudRunId startSeq endSeq lines createdAt"

CREATE_RUN = f"""
mutation CreateRun($teamId: ID!, $input: CreateRunInput!) {{
  createRun(teamId: $teamId, input: $input) {{ {RUN_FIELDS} }}
}}
"""

PUBLISH_UPDATE = f"""
mutation PublishRunUpdate($teamId: ID!, $input: RunUpdateInput!) {{
  publishRunUpdate(teamId: $teamId, input: $input) {{ {RUN_FIELDS} }}
}}
"""

PUBLISH_LOGS = f"""
mutation PublishLogBatch($teamId: ID!, $input: LogBatchInput!) {{
  publishLogBatch(teamId: $teamId, input: $input) {{ {LOG_FIELDS} }}
}}
"""

MAX_BATCH_LINES = 100
MAX_BATCH_BYTES = 64 * 1024

# Evaluate가 metrics.json에 쓰는 이름을 그대로 씁니다. 이름을 한 번 더 번역하면
# 어느 쪽이 진짜인지 알기 어려워집니다. "mAP"가 곧 mAP@[0.75:0.95]입니다.
SHARED_METRIC_KEYS = ("mAP", "mAP50", "mAP75", "precision50", "recall50")


def _team_evaluation(public: dict[str, Any]) -> dict[str, Any]:
    """평가 기록에서 팀 화면이 쓸 지표만 추립니다.

    기록 전체에는 작성자 PC의 경로와 storage 설정이 들어 있어 그대로 보내면
    안 됩니다. 평가를 아직 돌리지 않았으면 빈 dict입니다.
    """

    evaluation = public.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation:
        return {}
    summary = evaluation.get("summary")
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    values = metrics if isinstance(metrics, dict) else {}
    registration = public.get("registration")
    return {
        "status": evaluation.get("status"),
        "finished_at": evaluation.get("finished_at"),
        "message": evaluation.get("message"),
        # 계산되지 않은 지표를 None으로 채워 보내면 화면은 "평가를 마쳤다"고 읽고
        # `-`만 다섯 칸 그립니다. 실제로 나온 값만 싣습니다.
        "metrics": {
            key: values[key]
            for key in SHARED_METRIC_KEYS
            if values.get(key) is not None
        },
        "registration_status": (
            registration.get("status") if isinstance(registration, dict) else None
        ),
    }


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise TeamSyncError(f"팀 동기화 환경 변수 {name}이 필요합니다.")
    return value


@dataclass(frozen=True)
class TeamSyncConfig:
    """팀 동기화의 공개·서버 설정입니다. credential은 포함하지 않습니다."""

    enabled: bool
    team_id: str | None = None
    endpoint: str | None = None
    region: str = "ap-northeast-2"
    user_pool_id: str | None = None
    user_pool_client_id: str | None = None
    cognito_domain: str | None = None
    # 설정하면 browser login 없이도 학습을 팀에 기록합니다. Colab처럼 로그인
    # 화면을 띄울 수 없는 곳을 위한 값이고, 이 이름은 Cognito가 확인해 주지
    # 않습니다. 그래서 선택 값으로 두고, 켠 사람만 그 경로를 씁니다.
    actor_name: str | None = None
    # 로그인은 되지만 **밤새 무인으로** 대기열을 돌리는 컴퓨터를 위한 이름입니다.
    # `actor_name`과 일부러 나눠 두었습니다. `actor_name`은 화면에 "여기는 로그인할
    # 수 없는 환경"이라고 알려 로그인 관문을 걷어 내는데, 로그인이 되는 PC에서 그것을
    # 켜면 사람이 로그인을 잊은 채로 쓰다가 남의 이름으로 기록하게 됩니다. 이 값은
    # 화면에 그런 신호를 보내지 않고, 만료된 token을 IAM으로 대신할 때만 쓰입니다.
    unattended_actor_name: str | None = None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "TeamSyncConfig":
        source = os.environ if environment is None else environment
        enabled = source.get("PILL_TEAM_SYNC_ENABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return cls(enabled=False)
        return cls(
            enabled=True,
            team_id=_required(source, "PILL_TEAM_ID"),
            endpoint=_required(source, "PILL_TEAM_APPSYNC_URL"),
            region=source.get("AWS_REGION", "ap-northeast-2").strip() or "ap-northeast-2",
            user_pool_id=_required(source, "PILL_TEAM_COGNITO_USER_POOL_ID"),
            user_pool_client_id=_required(source, "PILL_TEAM_COGNITO_CLIENT_ID"),
            cognito_domain=_required(source, "PILL_TEAM_COGNITO_DOMAIN"),
            actor_name=(source.get("PILL_TEAM_ACTOR", "").strip() or None),
            unattended_actor_name=(
                source.get("PILL_TEAM_UNATTENDED_ACTOR", "").strip() or None
            ),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "team_id": self.team_id,
            "appsync_url": self.endpoint,
            "region": self.region,
            "user_pool_id": self.user_pool_id,
            "user_pool_client_id": self.user_pool_client_id,
            "cognito_domain": self.cognito_domain,
            # 화면이 로그인 관문을 열지 말지 정할 때 씁니다. 비밀이 아니고 팀
            # 목록에도 그대로 보이는 이름입니다.
            "actor": self.actor_name,
            # 무인 대기열용 이름은 **여기 따로** 실어 보냅니다. `actor`에 합치면
            # 로그인이 되는 PC에서도 관문이 사라집니다. 화면은 이 값을 보고 관문을
            # 여닫지 않으며, 설정이 켜져 있는지 확인하는 용도로만 씁니다.
            "unattended_actor": self.unattended_actor_name,
        }


class GraphQLTransport:
    """Cognito token 또는 AWS SigV4로 AppSync GraphQL을 호출합니다."""

    def __init__(self, config: TeamSyncConfig) -> None:
        self.config = config

    def execute(
        self,
        query: str,
        variables: dict[str, Any],
        *,
        access_token: str | None = None,
        iam: bool = False,
    ) -> dict[str, Any]:
        if not self.config.endpoint:
            raise TeamSyncError("AppSync endpoint가 설정되지 않았습니다.")
        body = json.dumps(
            {"query": query, "variables": variables},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = access_token
        elif iam:
            credentials = boto3.Session(region_name=self.config.region).get_credentials()
            if credentials is None:
                raise TeamSyncAuthError(
                    "AWS credential을 찾지 못했습니다. AWS SSO login 또는 profile을 확인하세요."
                )
            request = AWSRequest(
                method="POST", url=self.config.endpoint, data=body, headers=headers
            )
            SigV4Auth(credentials.get_frozen_credentials(), "appsync", self.config.region).add_auth(
                request
            )
            headers = dict(request.headers.items())
        else:
            raise TeamSyncAuthError("팀 학습을 시작하려면 먼저 로그인해야 합니다.")

        try:
            response = httpx.post(self.config.endpoint, content=body, headers=headers, timeout=15.0)
            # 만료된 Cognito token에 AppSync는 401 UnauthorizedException을 돌려줍니다.
            # 그것까지 "연결하지 못했습니다"로 옮기면, 다시 로그인하면 되는 상황에서
            # network와 서버를 뒤지게 됩니다. 대기열은 이 오류를 만난 자리에서 멈추므로
            # 무엇을 해야 풀리는지가 메시지에 담겨야 합니다.
            if response.status_code in {401, 403}:
                raise TeamSyncAuthError(
                    "팀 기록 로그인이 만료되었거나 거부되었습니다. 다시 로그인한 뒤"
                    " 대기열을 다시 돌려 주세요."
                )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise TeamSyncError("AWS 팀 동기화 API에 연결하지 못했습니다.") from error
        if payload.get("errors"):
            message = str(payload["errors"][0].get("message", "원격 요청이 거부되었습니다."))
            raise TeamSyncError(f"AWS 팀 동기화 요청이 실패했습니다: {message}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise TeamSyncError("AWS 팀 동기화 응답 형식이 올바르지 않습니다.")
        return data


class TeamSync:
    """영속 outbox와 background publisher를 소유합니다."""

    def __init__(
        self,
        config: TeamSyncConfig | None = None,
        *,
        transport: GraphQLTransport | None = None,
    ) -> None:
        self.config = config or TeamSyncConfig.from_environment()
        self.transport = transport or GraphQLTransport(self.config)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _directory(self) -> Path:
        return web_state_dir() / "team-sync"

    def _outbox_path(self) -> Path:
        return self._directory() / "outbox.jsonl"

    def _cursor_path(self) -> Path:
        return self._directory() / "cursor.txt"

    def create_run(
        self,
        *,
        access_token: str | None,
        local_job_id: str,
        run_id: str,
        settings: dict[str, Any],
        data_inputs: dict[str, Any],
    ) -> str | None:
        if not self.enabled:
            return None
        actor_name = self.config.actor_name
        if not access_token and not actor_name:
            raise TeamSyncAuthError(
                "팀 학습을 시작하려면 먼저 로그인해야 합니다. "
                "Colab처럼 로그인할 수 없는 곳이면 PILL_TEAM_ACTOR에 이름을 넣으세요."
            )
        cloud_run_id = uuid4().hex
        payload: dict[str, Any] = {
            "cloudRunId": cloud_run_id,
            "localJobId": local_job_id,
            "runId": run_id,
            "settings": json.dumps(redact(settings), ensure_ascii=False, allow_nan=False),
            "dataInputs": json.dumps(redact(data_inputs), ensure_ascii=False, allow_nan=False),
        }
        # 로그인이 있으면 claims가 이깁니다. 이름을 같이 보내지 않아야 화면에서
        # 남의 이름으로 기록하는 길이 아예 생기지 않습니다.
        if not access_token:
            payload["actorName"] = actor_name
        try:
            data = self._create_run_once(payload, access_token=access_token)
        except TeamSyncAuthError:
            # Cognito access token은 기본 한 시간만 삽니다. 그런데 대기열은 앞 학습이
            # 끝난 **뒤에야** 다음 항목을 시작하므로, 학습이 한 시간을 넘기면 넣어 둘 때
            # 받아 둔 token은 반드시 죽어 있습니다. 여기서 그대로 포기하면 밤새 돌리려고
            # 걸어 둔 목록이 첫 인계에서 멈추고, 아침에 사람이 눌러 줄 때까지 GPU가 놉니다.
            #
            # 진행 상황과 로그는 이미 IAM으로 올라가고 있어(``_publish``) 이 컴퓨터의 AWS
            # credential은 만료가 없습니다. 시작 기록만 브라우저 token을 고집할 이유가
            # 없으므로 SigV4로 한 번 더 시도합니다. 다만 누구의 학습인지 지어내지 않도록,
            # 이름을 미리 지정해 둔 경우에만 그렇게 합니다.
            #
            # 여기서 쓰는 이름은 `PILL_TEAM_ACTOR`가 아니라 `PILL_TEAM_UNATTENDED_ACTOR`
            # 입니다. 앞의 것은 화면의 로그인 관문까지 걷어 내는 Colab용이라, 로그인이
            # 되는 PC에서 이 기능을 켜자고 그것을 쓰면 로그인을 잊은 채로 남의 이름으로
            # 기록하게 됩니다. 두 상황은 필요한 것이 다르므로 값도 나눠 둡니다.
            unattended_name = self.config.unattended_actor_name
            if not access_token or not unattended_name:
                raise
            payload["actorName"] = unattended_name
            data = self._create_run_once(payload, access_token=None)
        created = data.get("createRun")
        if not isinstance(created, dict) or created.get("cloudRunId") != cloud_run_id:
            raise TeamSyncError("AWS가 만든 학습 ID를 확인하지 못했습니다.")
        return cloud_run_id

    def _create_run_once(
        self, payload: dict[str, Any], *, access_token: str | None
    ) -> dict[str, Any]:
        """``createRun`` 한 번. token이 있으면 그것으로, 없으면 IAM으로 보냅니다."""

        return self.transport.execute(
            CREATE_RUN,
            {"teamId": self.config.team_id, "input": dict(payload)},
            access_token=access_token,
            iam=not access_token,
        )

    def enqueue_update(self, record: Any) -> None:
        if not self.enabled or not getattr(record, "cloud_run_id", None):
            return
        record.sync_revision += 1
        public = redact(record.to_dict())
        event_id = f"{record.job_id}:update:{record.sync_revision}"
        payload = {
            "eventId": event_id,
            "cloudRunId": record.cloud_run_id,
            "revision": record.sync_revision,
            "status": record.status,
            "startedAt": record.started_at,
            "finishedAt": record.finished_at,
            "message": record.message,
            "progress": json.dumps(public["progress"], ensure_ascii=False, allow_nan=False),
            "summary": json.dumps(public["summary"], ensure_ascii=False, allow_nan=False),
            "artifacts": json.dumps(public["artifacts"], ensure_ascii=False, allow_nan=False),
            "heartbeatAt": public.get("updated_at") or _utc_now(),
        }
        evaluation = _team_evaluation(public)
        # 평가 전 update에 빈 값을 실으면 나중에 도착한 heartbeat가 지표를 지웁니다.
        if evaluation:
            payload["evaluation"] = json.dumps(
                evaluation, ensure_ascii=False, allow_nan=False
            )
        self._append({"kind": "update", "event_id": event_id, "payload": payload})

    def enqueue_log(self, record: Any, entry: dict[str, Any]) -> None:
        if not self.enabled or not getattr(record, "cloud_run_id", None):
            return
        safe = redact(entry)
        self._append(
            {
                "kind": "log",
                "event_id": f"{record.job_id}:log:{safe['seq']}",
                "cloud_run_id": record.cloud_run_id,
                "line": safe,
            }
        )

    def _append(self, event: dict[str, Any]) -> None:
        encoded = (
            json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self._lock:
            path = self._outbox_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
            self._ensure_worker()
            self._wake.set()

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop, name="team-sync-publisher", daemon=True
        )
        self._worker.start()

    def _read_cursor(self) -> int:
        try:
            value = int(self._cursor_path().read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return 0
        return max(0, value)

    def _write_cursor(self, value: int) -> None:
        path = self._cursor_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(str(value), encoding="ascii")
        os.replace(temporary, path)

    def _next_batch(self) -> tuple[dict[str, Any], int] | None:
        path = self._outbox_path()
        if not path.is_file():
            return None
        offset = self._read_cursor()
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.readline()
            if not raw:
                return None
            try:
                first = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {"kind": "skip"}, stream.tell()
            end = stream.tell()
            if first.get("kind") != "log":
                return first, end

            lines = [first["line"]]
            encoded_size = len(raw)
            cloud_run_id = first.get("cloud_run_id")
            while len(lines) < MAX_BATCH_LINES and encoded_size < MAX_BATCH_BYTES:
                checkpoint = stream.tell()
                candidate_raw = stream.readline()
                if not candidate_raw:
                    break
                try:
                    candidate = json.loads(candidate_raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    break
                if candidate.get("kind") != "log" or candidate.get("cloud_run_id") != cloud_run_id:
                    stream.seek(checkpoint)
                    break
                if encoded_size + len(candidate_raw) > MAX_BATCH_BYTES:
                    stream.seek(checkpoint)
                    break
                lines.append(candidate["line"])
                encoded_size += len(candidate_raw)
                end = stream.tell()
            return {
                "kind": "log_batch",
                "event_id": f"{first['event_id']}:{lines[-1]['seq']}",
                "payload": {
                    "eventId": first["event_id"] + f":{lines[-1]['seq']}",
                    "cloudRunId": cloud_run_id,
                    "startSeq": lines[0]["seq"],
                    "endSeq": lines[-1]["seq"],
                    "lines": json.dumps(lines, ensure_ascii=False, allow_nan=False),
                    "expiresAt": int(time.time()) + 30 * 24 * 60 * 60,
                },
            }, end

    def publish_pending(self, *, limit: int = 100) -> int:
        if not self.enabled:
            return 0
        sent = 0
        with self._lock:
            while sent < limit:
                item = self._next_batch()
                if item is None:
                    break
                event, end = item
                kind = event.get("kind")
                if kind == "update":
                    self.transport.execute(
                        PUBLISH_UPDATE,
                        {"teamId": self.config.team_id, "input": event["payload"]},
                        iam=True,
                    )
                elif kind == "log_batch":
                    self.transport.execute(
                        PUBLISH_LOGS,
                        {"teamId": self.config.team_id, "input": event["payload"]},
                        iam=True,
                    )
                self._write_cursor(end)
                sent += 1
        return sent

    def _worker_loop(self) -> None:
        delay = 1.0
        while True:
            try:
                sent = self.publish_pending()
                delay = 1.0
                if sent == 0:
                    self._wake.wait(timeout=30.0)
                    self._wake.clear()
            except TeamSyncError:
                self._wake.wait(timeout=delay)
                self._wake.clear()
                delay = min(delay * 2, 30.0)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


_TEAM_SYNC: TeamSync | None = None


def get_team_sync() -> TeamSync:
    global _TEAM_SYNC
    if _TEAM_SYNC is None:
        _TEAM_SYNC = TeamSync()
    return _TEAM_SYNC


def reset_team_sync() -> None:
    """Test와 설정 재로딩을 위해 singleton 참조만 비웁니다."""

    global _TEAM_SYNC
    _TEAM_SYNC = None
