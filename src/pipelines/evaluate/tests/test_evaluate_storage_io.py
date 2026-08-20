"""Local 경로 처리와 저장소 root 경계 test입니다."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipelines.evaluate.errors import ArtifactWriteError, InputArtifactError
from src.pipelines.evaluate.storage_io import ArtifactStore, is_remote_uri, join_uri


def test_local_path_resolves_relative_to_repository_root(repository_root: Path):
    store = ArtifactStore()

    assert store.local_path("data/val.jsonl") == (repository_root / "data/val.jsonl").resolve()


def test_normalize_uri_returns_repository_relative_path(repository_root: Path):
    store = ArtifactStore()

    assert store.normalize_uri(repository_root / "artifacts/evaluate/m.json") == (
        "artifacts/evaluate/m.json"
    )


def test_normalize_uri_keeps_s3_uri(repository_root: Path):
    assert ArtifactStore().normalize_uri("s3://bucket/key.json") == "s3://bucket/key.json"
    assert is_remote_uri("S3://bucket/key.json")
    assert join_uri("artifacts/evaluate/run/", "metrics.json") == (
        "artifacts/evaluate/run/metrics.json"
    )


@pytest.mark.parametrize("uri", ["../outside.json", "data/../../outside.json"])
def test_reading_outside_the_repository_is_rejected(repository_root: Path, uri: str):
    (repository_root.parent / "outside.json").write_text("{}", encoding="utf-8", newline="\n")
    store = ArtifactStore()

    with pytest.raises(InputArtifactError, match="저장소 root 밖의 local 경로"):
        store.read_text(uri)


def test_writing_outside_the_repository_is_rejected(repository_root: Path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "metrics.json"
    store = ArtifactStore()

    with pytest.raises(InputArtifactError, match="저장소 root 밖의 local 경로"):
        store.write_json(str(outside), {"leaked": True})

    assert not outside.exists()


def test_write_json_refuses_to_overwrite_by_default(repository_root: Path):
    store = ArtifactStore()
    store.write_json("artifacts/evaluate/m.json", {"first": True})

    with pytest.raises(ArtifactWriteError, match="이미 있습니다"):
        store.write_json("artifacts/evaluate/m.json", {"second": True})

    assert store.read_json("artifacts/evaluate/m.json") == {"first": True}


def test_write_json_writes_utf8_without_bom_and_lf(repository_root: Path):
    store = ArtifactStore()
    store.write_json("artifacts/evaluate/m.json", {"name": "타이레놀"})

    raw = (repository_root / "artifacts/evaluate/m.json").read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert "타이레놀" in raw.decode("utf-8")


def test_write_text_writes_utf8_without_bom_and_refuses_overwrite(repository_root: Path):
    store = ArtifactStore()
    store.write_text("submissions/run/submission.csv", "이름,value\n약,1\n")

    raw = (repository_root / "submissions/run/submission.csv").read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert "약,1\n" in raw.decode("utf-8")
    with pytest.raises(ArtifactWriteError, match="이미 있습니다"):
        store.write_text("submissions/run/submission.csv", "replacement\n")


def test_remove_local_ignores_paths_outside_the_repository(repository_root: Path):
    """정리 과정에서 저장소 밖 file을 지우지 않는지 확인합니다."""
    outside = repository_root.parent / "keep-me.json"
    outside.write_text("{}", encoding="utf-8", newline="\n")

    ArtifactStore().remove_local("../keep-me.json")

    assert outside.exists()


def test_a_local_run_can_name_an_s3_artifact_it_never_opens():
    """열지 않고 이름만 견주는 자리는 local backend로 돌아도 답이 나와야 합니다.

    합칠 예측은 어느 checkpoint의 증거인지 적어 두고, 융합은 그 이름으로 같은
    checkpoint를 두 번 세지 않게 막습니다. 그 checkpoint를 여는 일은 없습니다.
    여기서 멈추면 자격 증명이 없는 사람은 이미 만들어 둔 예측조차 합칠 수 없습니다.
    """
    store = ArtifactStore({"storage": {"backend": "local", "local": {"root": "artifacts"}}})

    identity = store.artifact_identity("s3://team/experiments/a/best_checkpoint.pt")

    # 표기가 달라도 같은 자리면 같은 값이라는 약속은 그대로여야 합니다.
    assert identity == store.artifact_identity("S3://team/experiments/a/best_checkpoint.pt")
    assert identity != store.artifact_identity("s3://team/experiments/b/best_checkpoint.pt")
    assert identity != store.artifact_identity("s3://other/experiments/a/best_checkpoint.pt")


def test_a_broken_s3_uri_is_still_reported():
    store = ArtifactStore({"storage": {"backend": "local", "local": {"root": "artifacts"}}})

    with pytest.raises(InputArtifactError, match="저장 위치를 확인하지 못했습니다"):
        store.artifact_identity("s3://team/key.json?version=2")
