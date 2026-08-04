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


def test_remove_local_ignores_paths_outside_the_repository(repository_root: Path):
    """정리 과정에서 저장소 밖 file을 지우지 않는지 확인합니다."""
    outside = repository_root.parent / "keep-me.json"
    outside.write_text("{}", encoding="utf-8", newline="\n")

    ArtifactStore().remove_local("../keep-me.json")

    assert outside.exists()
