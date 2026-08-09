"""런타임이 통째로 사라져도 그 학습을 이어서 할 수 있는지 확인합니다.

Colab은 세션이 끊기면 디스크가 함께 사라집니다. 화면의 "이어서 학습" 버튼은
``artifacts/web``의 job 기록과 설정 파일을 읽으므로, 그 두 개가 살아남지 않으면
새 런타임에서는 누를 대상 자체가 없습니다.
"""

from __future__ import annotations

import json

import pytest

from src.common import StorageError
from src.pipelines.web import paths, state_sync
from src.pipelines.web.errors import WebStateError
from src.pipelines.web.jobs import manager as manager_module
from src.pipelines.web.jobs import store
from src.pipelines.web.jobs.model import JobRecord
from src.pipelines.web.train_config import read_runtime_config, write_runtime_config


WORKSPACE = "hyunwoo-colab"
JOB_ID = "a" * 32


class FakeBucket:
    """S3 대신 쓰는 dict 하나짜리 bucket입니다.

    실제 ``S3Storage``처럼 ``list``는 전체 URI를 돌려주고 쓰기는 상대 key를 받습니다.
    """

    def __init__(self) -> None:
        self.objects: dict[str, object] = {}

    def write_json(self, destination, value, *, overwrite=False):
        key = str(destination)
        if not overwrite and key in self.objects:
            raise StorageError(f"이미 있습니다: {key}")
        self.objects[key] = value
        return f"s3://bucket/{key}"

    def read_json(self, source):
        key = str(source).removeprefix("s3://bucket/")
        if key not in self.objects:
            raise StorageError(f"없습니다: {key}")
        return self.objects[key]

    def list(self, prefix=""):
        return sorted(
            f"s3://bucket/{key}" for key in self.objects if key.startswith(str(prefix))
        )


@pytest.fixture
def bucket(monkeypatch):
    """workspace 이름을 정하고, mirror가 쓸 bucket을 가짜로 바꿉니다."""

    fake = FakeBucket()
    monkeypatch.setenv(state_sync.WORKSPACE_VARIABLE, WORKSPACE)
    monkeypatch.setattr(state_sync, "_storage", lambda: fake)
    return fake


def _job_key(job_id: str = JOB_ID) -> str:
    return f"{state_sync.STATE_PREFIX}/{WORKSPACE}/jobs/{job_id}.json"


def _running_record(config_id: str, *, status: str = "running") -> JobRecord:
    return JobRecord(
        job_id=JOB_ID,
        config_id=config_id,
        run_id="colab-20260810T010203Z-9f21",
        status=status,
        process_id=4242,
    )


def test_a_new_runtime_finds_the_interrupted_job_and_can_read_its_config(
    bucket, isolated_repo, monkeypatch, tmp_path
):
    config = {
        "storage": {"backend": "s3", "s3": {"prefix": ""}},
        "train": {"run_id": "colab-20260810T010203Z-9f21", "epochs": 30},
    }
    config_id = write_runtime_config(config)
    store.save_record(_running_record(config_id))

    # 런타임이 사라집니다. 새 VM에는 artifacts/web이 통째로 없습니다.
    monkeypatch.setattr(paths, "REPOSITORY_ROOT", tmp_path / "new-runtime")

    restarted = manager_module.JobManager()
    restarted.load()

    [restored] = restarted.list_jobs()
    assert restored.status == "interrupted"
    assert restored.run_id == "colab-20260810T010203Z-9f21"
    # 같은 기계에서 서버만 죽은 것이 아닙니다. 없는 process를 찾아보라고 하면
    # 팀원은 시간만 씁니다.
    assert "런타임이 사라졌습니다" in (restored.message or "")
    # 이어서 학습은 이 설정을 그대로 읽어 새 이름으로 다시 시작합니다.
    assert read_runtime_config(restored.config_id) == config


def test_a_restored_record_forgets_the_dead_runtime_process_id(
    bucket, isolated_repo, monkeypatch, tmp_path
):
    store.save_record(_running_record(write_runtime_config({"train": {}})))
    monkeypatch.setattr(paths, "REPOSITORY_ROOT", tmp_path / "new-runtime")

    state_sync.restore()

    payload = json.loads(
        (paths.jobs_dir() / JOB_ID / "record.json").read_text(encoding="utf-8")
    )
    # 새 VM에서 그 번호는 아무 상관 없는 process의 것입니다. 그대로 두면 화면이
    # 죽은 학습을 두고 "아직 돌고 있습니다"라고 안내합니다.
    assert payload["process_id"] is None


def test_the_mirror_never_replaces_a_record_this_runtime_already_has(
    bucket, isolated_repo
):
    config_id = write_runtime_config({"train": {}})
    store.save_record(_running_record(config_id, status="succeeded"))
    bucket.objects[_job_key()] = {
        "job_id": JOB_ID,
        "config_id": config_id,
        "run_id": "colab-20260810T010203Z-9f21",
        "status": "queued",
    }

    state_sync.restore()

    # 돌고 있는 server가 진실입니다. 오래된 사본이 끝난 학습을 대기로 되돌리면
    # 그 자리에서 같은 학습이 한 번 더 시작됩니다.
    assert store.load_record(JOB_ID).status == "succeeded"


def test_nothing_leaves_this_machine_without_a_workspace_name(
    isolated_repo, monkeypatch
):
    fake = FakeBucket()
    monkeypatch.setattr(state_sync, "_storage", lambda: fake)

    store.save_record(_running_record(write_runtime_config({"train": {}})))

    # 기본값은 꺼짐입니다. 이름을 정하지 않은 사람의 기록이 팀 bucket에 올라가면
    # 안 됩니다.
    assert fake.objects == {}


@pytest.mark.parametrize("name", ["../other-team", "team/colab", "a" * 65])
def test_a_workspace_name_that_could_reach_another_prefix_is_refused(monkeypatch, name):
    monkeypatch.setenv(state_sync.WORKSPACE_VARIABLE, name)

    # 이 이름은 그대로 S3 key가 됩니다. 남의 칸을 가리키면 남의 기록을 덮어씁니다.
    with pytest.raises(WebStateError, match=state_sync.WORKSPACE_VARIABLE):
        state_sync.workspace()
