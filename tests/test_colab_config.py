"""Colab 학습 config 생성기 test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_colab_config as maker  # noqa: E402


DATASET = "v1-seed42-8020"
BUCKET = "pill-detection-team-a83f"


class FakeStorage:
    """S3 대신 쓰는 가짜 storage입니다. 있는 것만 있다고 답합니다."""

    def __init__(self, present: set[str] | None = None, entries: list[str] | None = None):
        self.present = present if present is not None else set()
        self.entries = entries or []

    def exists(self, location: str) -> bool:
        return location in self.present

    def list(self, prefix: str = "") -> list[str]:
        return [item for item in self.entries if item.startswith(prefix)]


def dataset_uris(dataset: str = DATASET) -> set[str]:
    base = f"s3://{BUCKET}/{maker.PROCESSED_PREFIX}/{dataset}"
    return {f"{base}/{name}.json" for name in maker.REQUIRED_ARTIFACTS}


def build(tmp_path: Path, storage, **overrides) -> dict:
    options = {
        "dataset": DATASET,
        "epochs": 30,
        "batch_size": 2,
        "architecture": "retinanet_resnet50_fpn_v2",
        "optimizer": "AdamW",
        "output": tmp_path / "colab.json",
        "bucket": BUCKET,
    }
    options.update(overrides)
    return maker.build_config(storage=storage, **options)


def test_config_points_every_input_at_the_shared_bucket(tmp_path):
    config = build(tmp_path, FakeStorage(present=dataset_uris()))

    assert config["storage"] == {"backend": "s3", "s3": {"prefix": ""}}
    for name in maker.REQUIRED_ARTIFACTS:
        key = maker.ARTIFACT_KEYS[name]
        assert config["inputs"]["data"][key].startswith(f"s3://{BUCKET}/")
        assert DATASET in config["inputs"]["data"][key]


def test_run_id_says_where_it_ran_and_never_repeats(tmp_path):
    storage = FakeStorage(present=dataset_uris())
    first = build(tmp_path, storage)["train"]["run_id"]
    second = build(tmp_path, storage)["train"]["run_id"]

    # 같은 run_id면 train이 기존 결과를 지키려고 실행을 거부합니다.
    assert first != second
    assert first.startswith("colab-")
    assert maker.RUN_ID_PATTERN.match(first)


def test_only_chosen_values_are_written(tmp_path):
    config = build(tmp_path, FakeStorage(present=dataset_uris()))

    # train이 가진 기본값을 여기서 베끼면 두 곳이 어긋납니다. 고른 값과 Colab
    # 실행에 필요한 값만 적고 나머지는 train의 기본값에 맡깁니다.
    assert set(config["train"]) == {
        "run_id",
        "architecture",
        "optimizer",
        "epochs",
        "batch_size",
        "device",
        "output_prefix",
    }
    assert config["train"]["device"] == "cuda"


def test_optional_values_appear_only_when_given(tmp_path):
    storage = FakeStorage(present=dataset_uris())
    without = build(tmp_path, storage)
    with_rate = build(tmp_path, storage, learning_rate=0.0005)

    assert "learning_rate" not in without["train"]
    assert with_rate["train"]["learning_rate"] == 0.0005


def test_missing_dataset_names_what_is_missing(tmp_path):
    present = dataset_uris()
    present.discard(next(iter(sorted(present))))

    with pytest.raises(maker.ColabConfigError) as problem:
        build(tmp_path, FakeStorage(present=present))

    assert DATASET in str(problem.value)
    assert ".json" in str(problem.value)


def test_missing_bucket_is_reported_before_anything_else(tmp_path):
    with pytest.raises(maker.ColabConfigError, match="PILL_STORAGE_S3_BUCKET"):
        maker.resolve_bucket({})


def test_incomplete_datasets_are_separated_from_usable_ones():
    base = f"s3://{BUCKET}/{maker.PROCESSED_PREFIX}"
    complete = dataset_uris("v1-seed42-8020")
    storage = FakeStorage(
        present=complete,
        entries=sorted(complete) + [f"{base}/v1/nested/train_manifest.json"],
    )

    report = maker.inspect_datasets(storage, BUCKET)

    # 'v1'은 dataset이 아니라 상위 prefix라 필수 artifact가 하나도 없습니다.
    assert dict(report)["v1-seed42-8020"] == []
    assert sorted(dict(report)["v1"]) == sorted(maker.REQUIRED_ARTIFACTS)


def test_written_file_is_utf8_without_bom(tmp_path):
    destination = tmp_path / "nested" / "colab.json"
    config = build(tmp_path, FakeStorage(present=dataset_uris()), output=destination)
    maker.write_config(destination, config)

    raw = destination.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert json.loads(raw.decode("utf-8"))["train"]["epochs"] == 30
