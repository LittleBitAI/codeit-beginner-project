"""학습 job 한 건의 상태 표현."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


__all__ = [
    "ACTIVE_STATUSES",
    "JobRecord",
    "STATUS_LABELS",
    "TERMINAL_STATUSES",
    "utc_now_text",
]


# queued -> running -> succeeded | failed | cancelled
# 서버가 중간에 죽으면 남아 있던 queued/running 기록은 interrupted가 됩니다.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_INTERRUPTED = "interrupted"

ACTIVE_STATUSES = frozenset({STATUS_QUEUED, STATUS_RUNNING})
TERMINAL_STATUSES = frozenset(
    {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED}
)

STATUS_LABELS = {
    STATUS_QUEUED: "대기",
    STATUS_RUNNING: "실행 중",
    STATUS_SUCCEEDED: "성공",
    STATUS_FAILED: "실패",
    STATUS_CANCELLED: "취소됨",
    STATUS_INTERRUPTED: "중단됨",
}


def utc_now_text() -> str:
    """기록에 남길 UTC 시각 문자열을 만듭니다."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class JobRecord:
    """디스크에 저장되고 화면에 그대로 나가는 job 기록."""

    job_id: str
    config_id: str
    run_id: str
    status: str = STATUS_QUEUED
    created_at: str = field(default_factory=utc_now_text)
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    message: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    data_inputs: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    log_lines: int = 0
    orphan_note: str | None = None
    cloud_run_id: str | None = None
    sync_revision: int = 0

    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def elapsed_seconds(self, *, now: str | None = None) -> float | None:
        """시작 시각 기준 경과 시간. 아직 시작 전이면 ``None``입니다."""

        if not self.started_at:
            return None
        end = self.finished_at or now or utc_now_text()
        try:
            start_time = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            return None
        return round((end_time - start_time).total_seconds(), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "config_id": self.config_id,
            "run_id": self.run_id,
            "status": self.status,
            "status_label": STATUS_LABELS.get(self.status, self.status),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds(),
            "exit_code": self.exit_code,
            "message": self.message,
            "artifacts": dict(self.artifacts),
            "summary": dict(self.summary),
            "settings": dict(self.settings),
            "data_inputs": dict(self.data_inputs),
            "progress": dict(self.progress),
            "log_lines": self.log_lines,
            "orphan_note": self.orphan_note,
            "cloud_run_id": self.cloud_run_id,
            "sync_revision": self.sync_revision,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobRecord":
        return cls(
            job_id=str(payload["job_id"]),
            config_id=str(payload.get("config_id", "")),
            run_id=str(payload.get("run_id", "")),
            status=str(payload.get("status", STATUS_QUEUED)),
            created_at=str(payload.get("created_at") or utc_now_text()),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            exit_code=payload.get("exit_code"),
            message=payload.get("message"),
            artifacts=dict(payload.get("artifacts") or {}),
            summary=dict(payload.get("summary") or {}),
            settings=dict(payload.get("settings") or {}),
            data_inputs=dict(payload.get("data_inputs") or {}),
            progress=dict(payload.get("progress") or {}),
            log_lines=int(payload.get("log_lines") or 0),
            orphan_note=payload.get("orphan_note"),
            cloud_run_id=payload.get("cloud_run_id"),
            sync_revision=int(payload.get("sync_revision") or 0),
        )
