"""Web pipeline test에서 쓰는 fixture.

실제 저장소의 ``artifacts/``를 건드리지 않도록 저장소 root를 ``tmp_path``로 바꿉니다.
``paths.config_dir()``와 ``paths.jobs_dir()``는 호출 시점에 module 전역을 읽으므로
이 방법으로 모든 쓰기 위치가 함께 옮겨집니다.

경로만으로는 부족해서 주변 환경 변수도 함께 걷어 냅니다. 이 pipeline은 storage
backend와 팀 기록 여부를 ``PILL_`` 환경 변수로 고르기 때문입니다.
"""

from __future__ import annotations

import io
import os
import threading
import time

import pytest

from src.pipelines.web import paths
from src.pipelines.web.jobs import manager as manager_module
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


# 이 prefix로 시작하는 값은 모두 이 project가 동작을 고르려고 두는 knob입니다.
PROJECT_ENVIRONMENT_PREFIX = "PILL_"


@pytest.fixture(autouse=True)
def neutral_environment(monkeypatch):
    """주변 ``PILL_`` 환경 변수를 test 동안 걷어 냅니다.

    개발자 기계에 실제 값이 있으면 ``PILL_STORAGE_S3_BUCKET`` 때문에 registry backend가
    s3가 되어 test가 팀의 진짜 실험 목록을 읽고, ``PILL_TEAM_SYNC_ENABLED`` 때문에
    학습 시작이 access token을 요구합니다. CI에는 그 값이 없어 통과하고 개발자 기계에서만
    깨지므로 원인을 찾기 어렵습니다.

    이름을 나열하지 않고 prefix로 지우는 이유는, 새 knob이 생겼을 때 이 목록을 고치는
    것을 잊어도 새지 않게 하려는 것입니다. 값이 필요한 test는 본문에서
    ``monkeypatch.setenv``로 직접 넣습니다. 이 fixture보다 나중에 실행되므로 그대로
    적용됩니다.

    AWS credential 변수는 건드리지 않습니다. boto3의 기본 chain이 다루는 값이고 web이
    동작을 고르는 데 쓰지 않습니다.
    """

    for name in [
        name for name in os.environ if name.startswith(PROJECT_ENVIRONMENT_PREFIX)
    ]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    """저장소 root를 임시 directory로 바꿉니다."""

    monkeypatch.setattr(paths, "REPOSITORY_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def manager(isolated_repo, monkeypatch):
    """Test마다 완전히 새 JobManager를 씁니다."""

    fresh = manager_module.JobManager()
    monkeypatch.setattr(manager_module, "_MANAGER", fresh)
    return fresh


@pytest.fixture
def preparation_runner(isolated_repo, monkeypatch):
    """Test마다 새 준비 runner를 쓰고, 남은 thread가 다음 test로 넘어가지 않게 합니다.

    준비는 background thread에서 돌기 때문에, 정리하지 않으면 다음 test가 바꾼
    저장소 root를 그 thread가 보게 됩니다.
    """

    from src.pipelines.web import data_jobs

    fresh = data_jobs.PreparationRunner()
    monkeypatch.setattr(data_jobs, "_RUNNER", fresh)
    yield fresh

    deadline = time.monotonic() + 10
    while fresh.status().get("status") == "running" and time.monotonic() < deadline:
        time.sleep(0.02)


@pytest.fixture
def evaluation_runner(isolated_repo, monkeypatch):
    """평가도 background thread에서 돌므로 test마다 새로 두고 정리합니다."""

    from src.pipelines.web import evaluation

    fresh = evaluation.EvaluationRunner()
    monkeypatch.setattr(evaluation, "_RUNNER", fresh)
    yield fresh

    deadline = time.monotonic() + 10
    while fresh.status().get("status") == "running" and time.monotonic() < deadline:
        time.sleep(0.02)


@pytest.fixture
def client(manager, preparation_runner, evaluation_runner):
    """Frontend 없이 API만 올린 TestClient."""

    from fastapi.testclient import TestClient

    from src.pipelines.web.api.app import create_app

    with TestClient(create_app(serve_frontend=False)) as test_client:
        yield test_client


@pytest.fixture
def data_inputs():
    """유효한 data artifact 입력 4개."""

    return {key: f"artifacts/data/{key}.json" for key in DATA_ARTIFACT_KEYS}


@pytest.fixture
def valid_payload(data_inputs):
    return {"train": {"epochs": 2, "run_id": "test-run"}, "inputs": {"data": data_inputs}}


class FakeProcess:
    """``subprocess.Popen``을 대신하는 최소 구현.

    실제 학습을 돌리지 않고 job 수명주기만 확인합니다.
    """

    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        block_until_signalled: bool = False,
    ) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.pid = 4242
        self._exit_code = exit_code
        self._block = block_until_signalled
        self._released = threading.Event()
        self.terminate_calls = 0
        self.kill_calls = 0

    def wait(self, timeout: float | None = None) -> int:
        if self._block and not self._released.wait(timeout if timeout else 10):
            import subprocess

            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 10)
        return self._exit_code

    def release(self, exit_code: int | None = None) -> None:
        if exit_code is not None:
            self._exit_code = exit_code
        self._released.set()

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.release()

    def kill(self) -> None:
        self.kill_calls += 1
        self.release()


@pytest.fixture
def fake_process_factory():
    return FakeProcess
