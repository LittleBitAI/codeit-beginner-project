"""고를 수 있는 전처리 폴더 목록입니다.

경로를 손으로 붙여넣는 것 말고는 dataset을 고를 방법이 없었습니다.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from src.pipelines.web import datasets


ROOT = datasets.PROCESSED_ROOT
FILES = ("train_manifest.json", "validation_manifest.json", "class_map.json", "dataset_summary.json")


def write_dataset(root, name: str, *names: str) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    for file_name in names:
        (directory / file_name).write_text(json.dumps({}), encoding="utf-8")


# --- local backend --------------------------------------------------------


def test_a_complete_dataset_is_offered_and_a_partial_one_is_marked(isolated_repo, monkeypatch):
    monkeypatch.delenv("PILL_STORAGE_S3_BUCKET", raising=False)
    root = isolated_repo / ROOT
    write_dataset(root, "v5-seed42-8020-group", *FILES, "test_manifest.json")
    write_dataset(root, "v9-broken", "train_manifest.json")

    listing = datasets.list_processed_datasets()

    assert listing["backend"] == "local"
    found = {item["name"]: item for item in listing["datasets"]}
    assert set(found) == {"v5-seed42-8020-group", "v9-broken"}
    assert found["v5-seed42-8020-group"]["complete"] is True
    assert found["v5-seed42-8020-group"]["has_test_manifest"] is True
    assert found["v9-broken"]["complete"] is False
    # 무엇이 없어서 못 고르는지 화면이 말해 줄 수 있어야 합니다.
    assert "class_map_uri" in found["v9-broken"]["missing"]


def test_a_dataset_with_an_eda_report_says_so(isolated_repo, monkeypatch):
    """이미 분석한 판인지 알면 EDA를 다시 돌릴지 판단할 수 있습니다."""

    monkeypatch.delenv("PILL_STORAGE_S3_BUCKET", raising=False)
    root = isolated_repo / ROOT
    write_dataset(root, "v5", *FILES)
    write_dataset(root, "v5/eda", "report.json")

    found = {item["name"]: item for item in datasets.list_processed_datasets()["datasets"]}

    assert found["v5"]["has_eda_report"] is True


def test_an_empty_or_missing_root_is_not_an_error(isolated_repo, monkeypatch):
    """전처리를 한 번도 안 한 사람에게 오류를 보여 주면 안 됩니다."""

    monkeypatch.delenv("PILL_STORAGE_S3_BUCKET", raising=False)

    listing = datasets.list_processed_datasets()

    assert listing["datasets"] == []
    assert listing["problems"] == []


# --- s3 backend -----------------------------------------------------------


def test_s3_datasets_are_grouped_from_one_listing(isolated_repo, monkeypatch):
    """폴더마다 JSON을 열면 판이 늘수록 화면이 느려집니다. 목록 한 번만 씁니다."""

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "team-bucket")
    prefix = f"s3://team-bucket/{ROOT}"
    storage = Mock()
    storage.list = Mock(
        return_value=[f"{prefix}v5-group/{name}" for name in (*FILES, "test_manifest.json")]
        + [f"{prefix}v5-group/eda/report.json", f"{prefix}v3-group/train_manifest.json"]
    )
    monkeypatch.setattr(datasets, "create_storage", lambda config: storage, raising=False)
    monkeypatch.setattr("src.common.create_storage", lambda config: storage)

    listing = datasets.list_processed_datasets()

    assert listing["backend"] == "s3"
    assert listing["root"] == prefix
    assert storage.list.call_count == 1
    storage.read_json.assert_not_called()
    found = {item["name"]: item for item in listing["datasets"]}
    assert found["v5-group"]["complete"] is True
    assert found["v5-group"]["has_eda_report"] is True
    assert found["v3-group"]["complete"] is False


def test_an_s3_failure_is_reported_without_leaking_the_reason(isolated_repo, monkeypatch):
    """오류 원문에는 bucket 경로가 섞일 수 있습니다."""

    from src.common import StorageError

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "team-bucket")

    def explode(config):
        raise StorageError("s3://team-bucket/secret/path 접근 거부")

    monkeypatch.setattr("src.common.create_storage", explode)

    listing = datasets.list_processed_datasets()

    assert listing["datasets"] == []
    assert listing["problems"]
    assert "secret" not in listing["problems"][0]


# --- route ----------------------------------------------------------------


def test_the_route_serves_the_list(client, isolated_repo, monkeypatch):
    monkeypatch.delenv("PILL_STORAGE_S3_BUCKET", raising=False)
    write_dataset(isolated_repo / ROOT, "v5-seed42-8020-group-angle", *FILES)

    body = client.get("/api/data/datasets").json()

    assert [item["name"] for item in body["datasets"]] == ["v5-seed42-8020-group-angle"]
    assert body["datasets"][0]["directory"].endswith("v5-seed42-8020-group-angle/")


@pytest.mark.parametrize("key", ("name", "directory", "complete", "missing", "has_eda_report"))
def test_every_entry_carries_what_the_screen_draws(client, isolated_repo, monkeypatch, key):
    monkeypatch.delenv("PILL_STORAGE_S3_BUCKET", raising=False)
    write_dataset(isolated_repo / ROOT, "v5", *FILES)

    entry = client.get("/api/data/datasets").json()["datasets"][0]

    assert key in entry
