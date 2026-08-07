"""서버 진입점이 시작에 실패했을 때 남기는 흔적에 대한 test."""

from __future__ import annotations

import json

import pytest

from src.pipelines.web import server
from src.pipelines.web.jobs import store
from src.pipelines.web.jobs.model import JobRecord


def test_a_server_that_cannot_bind_leaves_running_records_alone(
    manager, isolated_repo, monkeypatch
):
    """포트를 못 잡고 죽을 서버가 남의 학습 상태를 망치면 안 됩니다.

    Colab 안내 문서의 GUI 셀을 다시 실행하면 두 번째 서버가 뜹니다. 그 서버는
    포트 8000이 이미 잡혀 있어 곧 죽지만, 예전에는 죽기 전에 이미 디스크 기록을
    interrupted로 덮고 팀에도 그렇게 알렸습니다. 멀쩡히 돌던 학습이 중단된 것처럼
    보였습니다.
    """

    record = JobRecord(
        job_id="a" * 32,
        config_id="b" * 32,
        run_id="colab-run",
        status="running",
        started_at="2026-08-07T00:00:00.000Z",
        process_id=4242,
    )
    store.save_record(record)

    import uvicorn

    def refuse(*_args, **_kwargs):
        raise OSError("[Errno 98] Address already in use")

    monkeypatch.setattr(uvicorn, "run", refuse)

    with pytest.raises(OSError):
        server.main(["--no-frontend"])

    path = store.job_directory(record.job_id) / "record.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert saved["message"] is None
