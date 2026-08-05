"""여러 학습 기록이 저장 후 비교 API까지 도달하는 Web E2E 시나리오.

실제 GPU 학습 대신 subprocess stdout만 대역으로 바꿉니다. 설정 검증, config 저장,
job thread, 결과 parsing, record 영구 저장, 서버 재로딩과 비교 adapter는 실제 코드를
그대로 통과합니다.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from src.pipelines.web.jobs import manager as manager_module
from src.pipelines.web.jobs import runner
from src.pipelines.web.jobs.manager import JobManager


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "test"
    / "fixtures"
    / "multiExperiment.json"
)


def load_scenario() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def completed_stdout(experiment: dict[str, Any]) -> str:
    run_id = experiment["run_id"]
    return json.dumps(
        {
            "status": "ok",
            "artifacts": {
                "train": {
                    "run_id": run_id,
                    "best_checkpoint_uri": f"artifacts/e2e/results/{run_id}/best.pt",
                    "last_checkpoint_uri": f"artifacts/e2e/results/{run_id}/last.pt",
                    "training_history_uri": f"artifacts/e2e/results/{run_id}/history.json",
                }
            },
            "summary": {"train": experiment["summary"]},
            "message": "E2E fixture 학습 완료",
        },
        ensure_ascii=False,
    )


def wait_until_idle(manager: JobManager, job_id: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if not record.is_active() and manager.active_job() is None:
            return
        time.sleep(0.01)
    raise AssertionError(f"E2E job이 {timeout}초 안에 끝나지 않았습니다: {job_id}")


def test_two_experiments_survive_reload_and_reach_comparison_api(
    client, manager, monkeypatch, fake_process_factory
):
    scenario = load_scenario()
    processes = deque(
        fake_process_factory(stdout=completed_stdout(experiment))
        for experiment in scenario["experiments"]
    )
    monkeypatch.setattr(runner, "spawn", lambda *args, **kwargs: processes.popleft())

    started_ids: list[str] = []
    for experiment in scenario["experiments"]:
        config_response = client.post(
            "/api/train/configs",
            json={
                "train": {"run_id": experiment["run_id"], **experiment["train"]},
                "inputs": {"data": scenario["dataset"]},
            },
        )
        assert config_response.status_code == 201, config_response.text

        start_response = client.post(
            "/api/train/jobs", json={"config_id": config_response.json()["config_id"]}
        )
        assert start_response.status_code == 201, start_response.text
        job_id = start_response.json()["job_id"]
        started_ids.append(job_id)
        wait_until_idle(manager, job_id)

    # 메모리 상태를 버리고 새 서버가 디스크 record를 다시 읽는 상황을 재현합니다.
    reloaded = JobManager()
    monkeypatch.setattr(manager_module, "_MANAGER", reloaded)

    response = client.get("/api/train/experiments")

    assert response.status_code == 200
    experiments = response.json()["experiments"]
    assert {item["experiment_id"] for item in experiments} == set(started_ids)
    assert {item["run_id"] for item in experiments} == {
        item["run_id"] for item in scenario["experiments"]
    }
    assert {item["status"] for item in experiments} == {"succeeded"}
    identities = [item["dataset"]["identity"] for item in experiments]
    if any(identity is None for identity in identities):
        dataset_relationship = "unknown"
    elif len(set(identities)) == 1:
        dataset_relationship = "same"
    else:
        dataset_relationship = "different"
    assert dataset_relationship == scenario["expectation"]["dataset_relationship"]
    assert all(item["dataset"]["artifacts_complete"] for item in experiments)

    by_run = {item["run_id"]: item for item in experiments}
    for expected in scenario["experiments"]:
        actual = by_run[expected["run_id"]]
        assert actual["optimizer"]["learning_rate"] == expected["train"]["learning_rate"]
        assert actual["training"]["batch_size"] == expected["train"]["batch_size"]
        assert actual["metrics"]["best_validation_loss"] == expected["summary"][
            "best_validation_loss"
        ]
        assert actual["optimizer"]["name"] == "SGD"
        assert actual["optimizer"]["source"] == "legacy_fallback"

    best = min(experiments, key=lambda item: item["metrics"]["best_validation_loss"])
    assert best["run_id"] == scenario["expectation"]["best_run_id"]
    assert "train_manifest_uri" not in response.text
    assert "artifacts/e2e/pills" not in response.text
