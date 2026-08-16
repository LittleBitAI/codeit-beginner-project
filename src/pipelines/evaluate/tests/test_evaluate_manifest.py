"""Manifest와 class map schema 검증 test입니다."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.pipelines.evaluate.errors import InputArtifactError
from src.pipelines.evaluate.manifest import (
    parse_class_map,
    parse_manifest,
    parse_test_manifest,
    sample_records,
)


VALID_RECORD = {
    "image_id": "img-1",
    "image_uri": "data/val/img-1.jpg",
    "width": 100,
    "height": 100,
    "annotations": [{"category_id": 1, "bbox": [10, 10, 20, 20]}],
}

VALID_COCO = {
    "images": [
        {"id": 1, "file_name": "images/img-1.jpg", "width": 100, "height": 80}
    ],
    "annotations": [
        {"id": 10, "image_id": 1, "category_id": 7, "bbox": [10, 5, 20, 30]}
    ],
    "categories": [{"id": 7, "name": "pill"}],
}

VALID_TEST_COCO = {
    "images": [
        {"id": 12, "file_name": "images/0012.jpg", "width": 100, "height": 80}
    ],
    "annotations": [],
    "categories": [{"id": 3, "name": "pill-a"}, {"id": 7, "name": "pill-b"}],
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


def test_single_jsonl_record_keeps_coco_named_extra_fields():
    record = dict(VALID_RECORD, images=["legacy metadata"], categories=["legacy metadata"])

    records = parse_manifest(_jsonl(record), source="manifest.jsonl")

    assert records[0]["image_id"] == "img-1"


def test_parse_manifest_rejects_empty_manifest():
    with pytest.raises(InputArtifactError, match="record가 없습니다"):
        parse_manifest("\n\n", source="manifest.jsonl")


def test_parse_manifest_rejects_broken_json():
    with pytest.raises(InputArtifactError, match="유효한 JSON이 아닙니다"):
        parse_manifest('{"image_id": "img-1"', source="manifest.jsonl")


def test_parse_manifest_rejects_missing_field():
    broken = {key: value for key, value in VALID_RECORD.items() if key != "width"}

    with pytest.raises(InputArtifactError, match=r"#line\[1\].*필수 field가 없습니다"):
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


def test_parse_manifest_accepts_coco_object_and_resolves_relative_image_uri():
    records = parse_manifest(
        json.dumps(VALID_COCO), source="data/validation/instances.json"
    )

    assert records == [
        {
            "image_id": 1,
            "image_key": "1",
            "image_uri": "data/validation/images/img-1.jpg",
            "width": 100,
            "height": 80,
            "annotations": [{"category_id": 7, "bbox": [10.0, 5.0, 20.0, 30.0]}],
        }
    ]


def test_parse_manifest_resolves_coco_image_relative_to_s3_manifest():
    records = parse_manifest(
        json.dumps(VALID_COCO), source="s3://pill-data/validation/instances.json"
    )

    assert records[0]["image_uri"] == "s3://pill-data/validation/images/img-1.jpg"


def test_parse_test_manifest_accepts_unlabelled_coco_and_preserves_category_ids():
    records, category_ids = parse_test_manifest(
        json.dumps(VALID_TEST_COCO), source="data/test/instances.json"
    )

    assert records[0]["image_id"] == 12
    assert records[0]["annotations"] == []
    assert category_ids == frozenset({3, 7})


def test_parse_test_manifest_rejects_labels_and_non_numeric_filename():
    labelled = deepcopy(VALID_TEST_COCO)
    labelled["annotations"] = [
        {"id": 1, "image_id": 12, "category_id": 3, "bbox": [1, 1, 2, 2]}
    ]
    named = deepcopy(VALID_TEST_COCO)
    named["images"][0]["file_name"] = "images/pill.jpg"

    with pytest.raises(InputArtifactError, match=r"annotations=\[\]"):
        parse_test_manifest(json.dumps(labelled), source="test.json")
    with pytest.raises(InputArtifactError, match="숫자 file_name"):
        parse_test_manifest(json.dumps(named), source="test.json")


def test_parse_test_manifest_rejects_empty_categories():
    document = deepcopy(VALID_TEST_COCO)
    document["categories"] = []

    with pytest.raises(InputArtifactError, match="categories가 비어"):
        parse_test_manifest(json.dumps(document), source="test.json")


def test_parse_test_manifest_rejects_filename_stem_that_does_not_match_image_id():
    document = deepcopy(VALID_TEST_COCO)
    document["images"][0]["file_name"] = "images/0013.jpg"

    with pytest.raises(InputArtifactError, match="file_name 숫자 stem과 image id가 다릅니다"):
        parse_test_manifest(json.dumps(document), source="test.json")


def test_parse_test_manifest_rejects_duplicate_filename():
    document = deepcopy(VALID_TEST_COCO)
    document["images"].append(
        {"id": 13, "file_name": "images/0012.jpg", "width": 100, "height": 80}
    )

    with pytest.raises(InputArtifactError, match="file_name이 중복"):
        parse_test_manifest(json.dumps(document), source="test.json")


def test_parse_test_manifest_rejects_duplicate_numeric_stem():
    document = deepcopy(VALID_TEST_COCO)
    document["images"].append(
        {"id": 13, "file_name": "other/0012.png", "width": 100, "height": 80}
    )

    with pytest.raises(InputArtifactError, match="숫자 stem이 중복"):
        parse_test_manifest(json.dumps(document), source="test.json")


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("annotations", "image_id", 999, "존재하지 않는 image_id"),
        ("annotations", "category_id", 999, "존재하지 않는 category_id"),
        ("annotations", "bbox", [90, 70, 20, 20], "이미지 범위"),
        ("images", "width", 0, "0보다 큰 정수"),
        ("images", "height", 1.5, "0보다 큰 정수"),
    ],
)
def test_parse_manifest_rejects_invalid_coco_references_and_sizes(
    section, field, value, message
):
    document = deepcopy(VALID_COCO)
    document[section][0][field] = value

    with pytest.raises(InputArtifactError, match=message):
        parse_manifest(json.dumps(document), source="instances.json")


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("images", "image id가 중복"),
        ("categories", "category id가 중복"),
        ("annotations", "annotation id가 중복"),
    ],
)
def test_parse_manifest_rejects_duplicate_coco_ids(section, message):
    document = deepcopy(VALID_COCO)
    document[section].append(dict(document[section][0]))

    with pytest.raises(InputArtifactError, match=message):
        parse_manifest(json.dumps(document), source="instances.json")


@pytest.mark.parametrize("section", ["images", "categories", "annotations"])
@pytest.mark.parametrize("invalid_id", [True, -1, 1.5, "1"])
def test_parse_manifest_rejects_invalid_coco_id_types(section, invalid_id):
    document = deepcopy(VALID_COCO)
    document[section][0]["id"] = invalid_id

    with pytest.raises(InputArtifactError, match="id는 0 이상의 정수"):
        parse_manifest(json.dumps(document), source="instances.json")


@pytest.mark.parametrize("iscrowd", [True, -1, 2, 0.0, "0"])
def test_parse_manifest_rejects_invalid_coco_iscrowd(iscrowd):
    document = deepcopy(VALID_COCO)
    document["annotations"][0]["iscrowd"] = iscrowd

    with pytest.raises(InputArtifactError, match="iscrowd는 0 또는 1"):
        parse_manifest(json.dumps(document), source="instances.json")


@pytest.mark.parametrize("field", ["images", "annotations", "categories"])
def test_parse_manifest_rejects_missing_coco_top_level_field(field):
    document = deepcopy(VALID_COCO)
    document.pop(field)

    with pytest.raises(InputArtifactError, match="COCO manifest 필수 field"):
        parse_manifest(json.dumps(document), source="instances.json")


def test_parse_class_map_supports_both_formats():
    listed = parse_class_map(
        {"classes": [{"id": 1, "name": "tylenol"}]}, source="class_map.json"
    )
    flat = parse_class_map({"1": "tylenol"}, source="class_map.json")

    assert listed == flat == {1: "tylenol"}


def test_parse_class_map_rejects_invalid_document():
    with pytest.raises(InputArtifactError, match="object여야 합니다"):
        parse_class_map(["tylenol"], source="class_map.json")


@pytest.mark.parametrize("category_id", [True, False, 1.5, 1.0, "1.5", "", "abc", None, [1]])
def test_parse_class_map_rejects_non_integer_ids(category_id):
    """int()로는 통과하지만 뜻이 달라지는 값(true, 1.5 등)을 막습니다."""
    with pytest.raises(InputArtifactError, match="category id는 정수여야 합니다"):
        parse_class_map({"classes": [{"id": category_id, "name": "tylenol"}]}, source="c.json")


def test_parse_class_map_accepts_integer_text_ids():
    assert parse_class_map({" 12 ": "tylenol", "-3": "aspirin"}, source="c.json") == {
        12: "tylenol",
        -3: "aspirin",
    }


def test_parse_class_map_rejects_duplicate_ids():
    document = {"classes": [{"id": 1, "name": "tylenol"}, {"id": "1", "name": "aspirin"}]}

    with pytest.raises(InputArtifactError, match="중복되었습니다"):
        parse_class_map(document, source="class_map.json")


# --- 표본 ------------------------------------------------------------------


def _records(counts: dict[int, int]) -> list[dict]:
    """class마다 그 수만큼 image를 만듭니다. 한 image에는 class 하나입니다."""

    records = []
    for category_id, count in counts.items():
        for number in range(count):
            records.append(
                {
                    "image_id": f"c{category_id}-{number}",
                    "image_uri": "data/val/x.jpg",
                    "width": 100,
                    "height": 100,
                    "annotations": [{"category_id": category_id, "bbox": [0, 0, 5, 5]}],
                }
            )
    return records


def test_sample_gives_a_rare_class_a_place_before_a_common_one_gets_a_second():
    """무작위로 줄이면 드문 class가 통째로 빠집니다.

    빠진 class의 AP는 측정값이 아니라 없는 값이 되어, 그 표본으로는 후보끼리
    견줄 수 없게 됩니다.
    """

    records = _records({1: 100, 2: 1, 3: 1})

    sample = sample_records(records, 6, seed=42)

    assert len(sample) == 6
    classes = [record["annotations"][0]["category_id"] for record in sample]
    assert set(classes) == {1, 2, 3}


def test_the_same_seed_looks_at_the_same_images():
    """후보들이 서로 다른 시험지를 보면 점수를 나란히 놓을 수 없습니다."""

    records = _records({1: 20, 2: 20})

    first = sample_records(records, 8, seed=42)
    second = sample_records(records, 8, seed=42)
    other = sample_records(records, 8, seed=7)

    assert [record["image_id"] for record in first] == [
        record["image_id"] for record in second
    ]
    assert [record["image_id"] for record in first] != [
        record["image_id"] for record in other
    ]


def test_a_sample_bigger_than_the_manifest_keeps_every_image():
    records = _records({1: 3})

    assert len(sample_records(records, 10, seed=42)) == 3
