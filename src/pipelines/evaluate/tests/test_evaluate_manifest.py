"""Manifest와 class map schema 검증 test입니다."""

from __future__ import annotations

import json

import pytest

from src.pipelines.evaluate.errors import InputArtifactError
from src.pipelines.evaluate.manifest import parse_class_map, parse_manifest


VALID_RECORD = {
    "image_id": "img-1",
    "image_uri": "data/val/img-1.jpg",
    "width": 100,
    "height": 100,
    "annotations": [{"category_id": 1, "bbox": [10, 10, 20, 20]}],
}


def _jsonl(*records: dict) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def test_parse_manifest_accepts_valid_records():
    second = dict(VALID_RECORD, image_id="img-2", annotations=[])
    records = parse_manifest(_jsonl(VALID_RECORD, second), source="manifest.jsonl")

    assert [record["image_key"] for record in records] == ["img-1", "img-2"]
    assert records[0]["annotations"][0]["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert records[1]["annotations"] == []


def test_parse_manifest_ignores_blank_lines():
    text = _jsonl(VALID_RECORD) + "\n   \n"

    assert len(parse_manifest(text, source="manifest.jsonl")) == 1


def test_parse_manifest_rejects_empty_manifest():
    with pytest.raises(InputArtifactError, match="record가 없습니다"):
        parse_manifest("\n\n", source="manifest.jsonl")


def test_parse_manifest_rejects_broken_json():
    with pytest.raises(InputArtifactError, match="유효한 JSON이 아닙니다"):
        parse_manifest('{"image_id": "img-1"', source="manifest.jsonl")


def test_parse_manifest_rejects_missing_field():
    broken = {key: value for key, value in VALID_RECORD.items() if key != "width"}

    with pytest.raises(InputArtifactError, match="필수 field가 없습니다"):
        parse_manifest(_jsonl(broken), source="manifest.jsonl")


def test_parse_manifest_rejects_duplicate_image_id():
    with pytest.raises(InputArtifactError, match="중복되었습니다"):
        parse_manifest(_jsonl(VALID_RECORD, VALID_RECORD), source="manifest.jsonl")


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ([10, 10, 20], "bbox는"),
        ([10, 10, 0, 20], "0보다 커야"),
        ([10, 10, "20", 20], "유한한 숫자"),
        ([90, 90, 20, 20], "이미지 범위"),
    ],
)
def test_parse_manifest_rejects_invalid_bbox(bbox, message):
    record = dict(VALID_RECORD, annotations=[{"category_id": 1, "bbox": bbox}])

    with pytest.raises(InputArtifactError, match=message):
        parse_manifest(_jsonl(record), source="manifest.jsonl")


def test_parse_class_map_supports_both_formats():
    listed = parse_class_map(
        {"classes": [{"id": 1, "name": "tylenol"}]}, source="class_map.json"
    )
    flat = parse_class_map({"1": "tylenol"}, source="class_map.json")

    assert listed == flat == {1: "tylenol"}


def test_parse_class_map_rejects_invalid_document():
    with pytest.raises(InputArtifactError, match="object여야 합니다"):
        parse_class_map(["tylenol"], source="class_map.json")
