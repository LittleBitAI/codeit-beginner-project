"""Data pipeline 준비 경로의 테스트.

실제 AWS에는 접속하지 않습니다. 저장소 관례대로 `unittest.mock`으로 만든
in-memory S3 storage 대역을 써서 준비 경로의 흐름과 산출물을 검증합니다.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
from PIL import Image

from src.common import LocalStorage, StorageError, validate_pipeline_result
from src.common.storage import S3Storage
from src.pipelines.data import preparation, run
from src.pipelines.data.errors import DataError
from src.pipelines.data.preparation import REPOSITORY_ROOT
from src.pipelines.data.test_manifest import build_test_manifest


BUCKET = "test-bucket"
RAW_PREFIX = "datasets/pill_detection/raw/v1/"
CATEGORY_NAMES = {1: "pill_a", 2: "pill_b", 3: "pill_c"}
# registry가 local artifact URI를 거부하는 두 형태입니다.
# (src/pipelines/registry/record.py의 resolve_local_uri)
WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


# --- 대역 storage와 원본 fixture ------------------------------------------


def make_fake_s3_storage(objects: dict[str, Any] | None = None) -> tuple[S3Storage, dict]:
    """실제 network 호출 없이 동작하는 S3Storage 대역을 만듭니다."""

    storage = S3Storage(BUCKET, client=Mock())
    stored: dict[str, Any] = copy.deepcopy(objects or {})

    def uri_of(location: Any) -> str:
        text = str(location)
        return text if text.startswith("s3://") else f"s3://{BUCKET}/{text}"

    def write_json(destination, value, *, overwrite=False):
        uri = uri_of(destination)
        if uri in stored and not overwrite:
            raise StorageError(f"S3 object가 이미 있습니다: {uri}")
        stored[uri] = copy.deepcopy(value)
        return uri

    def read_json(source):
        uri = uri_of(source)
        if uri not in stored:
            raise StorageError(f"S3 object가 없습니다: {uri}")
        return copy.deepcopy(stored[uri])

    def download_file(source, destination, *, overwrite=False):
        uri = uri_of(source)
        destination_path = Path(destination)
        if uri not in stored:
            raise StorageError(f"S3 object가 없습니다: {uri}")
        if destination_path.exists() and not overwrite:
            raise StorageError(f"download 대상이 이미 있습니다: {destination_path.name}")
        value = stored[uri]
        if not isinstance(value, bytes):
            raise StorageError(f"image object가 bytes가 아닙니다: {uri}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(value)
        return destination_path

    storage.write_json = Mock(side_effect=write_json)
    storage.read_json = Mock(side_effect=read_json)
    storage.download_file = Mock(side_effect=download_file)
    storage.exists = Mock(side_effect=lambda location: uri_of(location) in stored)
    storage.list = Mock(
        side_effect=lambda prefix="": sorted(
            uri for uri in stored if uri.startswith(uri_of(prefix))
        )
    )
    return storage, stored


def annotation_document(image_index: int, category_ids: list[int]) -> dict[str, Any]:
    """이미지 한 장에 대한 원본 COCO 문서를 만듭니다."""

    return {
        "images": [
            {
                "id": image_index,
                "file_name": f"img_{image_index:03d}.jpg",
                "width": 100,
                "height": 100,
            }
        ],
        "annotations": [
            {
                "id": order,
                "image_id": image_index,
                "category_id": category_id,
                "bbox": [10, 10 + order, 20, 20],
                "iscrowd": 0,
            }
            for order, category_id in enumerate(category_ids, start=1)
        ],
        "categories": [
            {"id": category_id, "name": CATEGORY_NAMES[category_id]}
            for category_id in sorted(set(category_ids))
        ],
    }


def image_bytes(width: int, height: int, *, image_format: str = "PNG") -> bytes:
    """Fake storage에 넣을 실제 image bytes를 만듭니다."""

    output = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(output, format=image_format)
    return output.getvalue()


def raw_objects(
    categories_by_image: dict[int, list[int]] | None = None,
    *,
    image_count: int = 40,
    test_image_count: int = 5,
) -> dict[str, Any]:
    """train_images, train_annotations, test_images를 갖춘 원본을 만듭니다."""

    if categories_by_image is None:
        categories_by_image = {}
        for index in range(1, image_count + 1):
            primary = (index - 1) % 3 + 1
            categories = [primary]
            if index % 4 == 0:
                categories.append(index % 3 + 1)
            categories_by_image[index] = categories

    objects: dict[str, Any] = {}
    for index, category_ids in categories_by_image.items():
        image_uri = f"s3://{BUCKET}/{RAW_PREFIX}train_images/img_{index:03d}.jpg"
        annotation_uri = (
            f"s3://{BUCKET}/{RAW_PREFIX}train_annotations/img_{index:03d}.json"
        )
        objects[image_uri] = {"placeholder": "image bytes"}
        objects[annotation_uri] = annotation_document(index, category_ids)
    for index in range(1, test_image_count + 1):
        objects[f"s3://{BUCKET}/{RAW_PREFIX}test_images/{index:03d}.png"] = image_bytes(
            30 + index, 40 + index
        )
    return objects


def prepare_config(split_ratio: Any = "8:2", **extra: Any) -> dict[str, Any]:
    data_section: dict[str, Any] = {"prepare": True}
    if split_ratio is not None:
        data_section["split_ratio"] = split_ratio
    data_section.update(extra)
    return {
        "execution": {"mode": "local"},
        "storage": {"backend": "s3", "s3": {"bucket": BUCKET}},
        "data": data_section,
    }


def run_with_fake_storage(storage: S3Storage, config: dict[str, Any]) -> dict[str, Any]:
    with patch.object(preparation, "create_storage", return_value=storage):
        return run(config)


def prepare(config: dict[str, Any], objects: dict[str, Any] | None = None):
    storage, stored = make_fake_s3_storage(objects if objects is not None else raw_objects())
    return run_with_fake_storage(storage, config), stored


def artifact_document(stored: dict[str, Any], result: dict[str, Any], key: str) -> Any:
    return stored[result["artifacts"][key]]


SPLIT_MANIFEST_FILE_NAMES = {
    "train_manifest_uri": "train_manifest.json",
    "validation_manifest_uri": "validation_manifest.json",
}


# --- 분할 비율 옵션 ---------------------------------------------------------


@pytest.mark.parametrize(
    ("split_ratio", "validation_images", "train_images"),
    [("8:2", 8, 32), ("9:1", 4, 36)],
)
def test_split_ratio_option_controls_validation_size(
    split_ratio, validation_images, train_images
):
    result, stored = prepare(prepare_config(split_ratio))

    assert result["status"] == "ok", result["message"]
    assert validate_pipeline_result(result, pipeline_name="data") is result
    assert result["summary"]["split_ratio"] == split_ratio
    assert result["summary"]["validation_ratio"] == preparation.SPLIT_RATIO_OPTIONS[split_ratio]
    assert result["summary"]["validation_images"] == validation_images
    assert result["summary"]["train_images"] == train_images

    validation = artifact_document(stored, result, "validation_manifest_uri")
    train = artifact_document(stored, result, "train_manifest_uri")
    assert len(validation["images"]) == validation_images
    assert len(train["images"]) == train_images
    assert not {image["id"] for image in train["images"]} & {
        image["id"] for image in validation["images"]
    }

    summary_document = artifact_document(stored, result, "dataset_summary_uri")
    assert summary_document["split"]["split_ratio"] == split_ratio
    assert summary_document["split"]["validation_ratio"] == (
        preparation.SPLIT_RATIO_OPTIONS[split_ratio]
    )
    assert summary_document["split"]["seed"] == preparation.DEFAULT_SEED


@pytest.mark.parametrize(
    "split_ratio", ["7:3", "0.2", "20%", "8:2:1", 0.2, 0.1, 1, None, True, "", "  "]
)
def test_other_split_ratios_are_rejected(split_ratio):
    result, stored = prepare(prepare_config(split_ratio))

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert '"8:2"' in result["message"] and '"9:1"' in result["message"]
    assert result["summary"]["allowed_split_ratios"] == ["8:2", "9:1"]
    assert validate_pipeline_result(result, pipeline_name="data") is result
    assert not [uri for uri in stored if "/processed/" in uri]


def test_split_ratio_tolerates_surrounding_whitespace_only():
    result, _ = prepare(prepare_config(" 9:1 "))

    assert result["status"] == "ok", result["message"]
    assert result["summary"]["split_ratio"] == "9:1"


def test_split_ratio_is_required_even_when_preparation_is_requested():
    result, _ = prepare({"data": {"prepare": True}})

    assert result["status"] == "error"
    assert "split_ratio" in result["message"]


def test_two_options_are_stored_in_different_locations():
    objects = raw_objects()
    storage, stored = make_fake_s3_storage(objects)

    first = run_with_fake_storage(storage, prepare_config("8:2"))
    second = run_with_fake_storage(storage, prepare_config("9:1"))

    assert first["status"] == "ok" and second["status"] == "ok"
    assert set(first["artifacts"]) == set(second["artifacts"])
    for key, uri in first["artifacts"].items():
        assert uri != second["artifacts"][key]
    assert "8020" in first["summary"]["processed_prefix"]
    assert "9010" in second["summary"]["processed_prefix"]
    # 원본은 그대로 두고, 두 옵션의 산출물 5개씩만 새로 생겼습니다.
    processed = sorted(uri for uri in stored if "/processed/" in uri)
    assert len(processed) == 10
    assert set(first["artifacts"].values()) | set(second["artifacts"].values()) == set(
        processed
    )


# --- 재현성 ----------------------------------------------------------------


def test_same_input_seed_and_ratio_produce_the_same_split():
    objects = raw_objects()

    first, first_stored = prepare(prepare_config("8:2"), objects)
    second, second_stored = prepare(prepare_config("8:2"), objects)

    assert first["artifacts"] == second["artifacts"]
    for key in (
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
        "test_manifest_uri",
    ):
        assert artifact_document(first_stored, first, key) == artifact_document(
            second_stored, second, key
        )
    first_summary = artifact_document(first_stored, first, "dataset_summary_uri")
    second_summary = artifact_document(second_stored, second, "dataset_summary_uri")
    # generated_at만 실행 시각이고 나머지 내용은 모두 같습니다.
    first_summary.pop("generated_at")
    second_summary.pop("generated_at")
    assert first_summary == second_summary


def test_different_seed_changes_the_split_and_the_location():
    objects = raw_objects()

    default_seed, default_stored = prepare(prepare_config("8:2"), objects)
    other_seed, other_stored = prepare(prepare_config("8:2", seed=7), objects)

    assert default_seed["artifacts"] != other_seed["artifacts"]
    assert "seed7" in other_seed["summary"]["processed_prefix"]
    default_validation = artifact_document(
        default_stored, default_seed, "validation_manifest_uri"
    )
    other_validation = artifact_document(
        other_stored, other_seed, "validation_manifest_uri"
    )
    assert len(default_validation["images"]) == len(other_validation["images"])


# --- 유출 방지와 category 보장 ---------------------------------------------


def test_test_images_never_enter_any_split():
    result, stored = prepare(prepare_config("8:2"))

    assert result["summary"]["test_images_used"] == 0
    for key in ("train_manifest_uri", "validation_manifest_uri"):
        manifest = artifact_document(stored, result, key)
        for image in manifest["images"]:
            assert "test_images/" not in image["file_name"]
            assert "/train_images/" in image["file_name"]
    summary_document = artifact_document(stored, result, "dataset_summary_uri")
    assert summary_document["raw"]["test_images_used"] == 0


def test_only_test_image_bytes_are_read_and_test_annotations_are_never_read():
    objects = raw_objects()
    objects[f"s3://{BUCKET}/{RAW_PREFIX}test_annotations/secret.json"] = {
        "must_not_be_read": True
    }
    storage, _ = make_fake_s3_storage(objects)

    result = run_with_fake_storage(storage, prepare_config("8:2"))

    assert result["status"] == "ok"
    storage.list.assert_called_once_with(RAW_PREFIX)
    read_locations = [str(call.args[0]) for call in storage.read_json.call_args_list]
    assert read_locations
    assert all("test_images/" not in location for location in read_locations)
    assert all("/train_annotations/" in location for location in read_locations)
    downloaded_locations = [
        str(call.args[0]) for call in storage.download_file.call_args_list
    ]
    assert len(downloaded_locations) == 5
    assert all("/test_images/" in location for location in downloaded_locations)
    assert all("test_annotations/" not in location for location in read_locations)


@pytest.mark.parametrize("split_ratio", ["8:2", "9:1"])
def test_every_category_appears_in_both_splits(split_ratio):
    result, stored = prepare(prepare_config(split_ratio))

    train = artifact_document(stored, result, "train_manifest_uri")
    validation = artifact_document(stored, result, "validation_manifest_uri")
    expected = set(CATEGORY_NAMES)
    assert {category["id"] for category in train["categories"]} == expected
    assert {category["id"] for category in validation["categories"]} == expected
    assert {annotation["category_id"] for annotation in train["annotations"]} == expected
    assert {
        annotation["category_id"] for annotation in validation["annotations"]
    } == expected

    summary_document = artifact_document(stored, result, "dataset_summary_uri")
    for category in summary_document["categories"]:
        assert category["train_image_count"] >= 1
        assert category["validation_image_count"] >= 1


def test_category_present_in_only_one_image_is_rejected():
    categories_by_image = {index: [(index - 1) % 2 + 1] for index in range(1, 11)}
    categories_by_image[5] = [1, 3]

    result, stored = prepare(prepare_config("8:2"), raw_objects(categories_by_image))

    assert result["status"] == "error"
    assert "이미지가 2장 이상 필요합니다" in result["message"]
    assert "3" in result["message"]
    assert not [uri for uri in stored if "/processed/" in uri]


# --- 산출물 형식 ------------------------------------------------------------


def test_build_test_manifest_converts_two_images_and_unsorted_class_map():
    image_ten = f"s3://{BUCKET}/{RAW_PREFIX}test_images/10.png"
    image_two = f"s3://{BUCKET}/{RAW_PREFIX}test_images/2.png"
    storage, _ = make_fake_s3_storage(
        {
            image_ten: image_bytes(31, 41),
            image_two: image_bytes(12, 22),
        }
    )

    manifest = build_test_manifest(
        storage,
        [image_ten, image_two],
        {"10": "pill_ten", 2: "pill_two"},
        publish_file_name=str,
    )

    assert manifest == {
        "info": {
            "description": "Pill detection test COCO manifest",
            "split": "test",
        },
        "images": [
            {
                "id": 2,
                "file_name": image_two,
                "width": 12,
                "height": 22,
            },
            {
                "id": 10,
                "file_name": image_ten,
                "width": 31,
                "height": 41,
            },
        ],
        "annotations": [],
        "categories": [
            {"id": 2, "name": "pill_two", "supercategory": "pill"},
            {"id": 10, "name": "pill_ten", "supercategory": "pill"},
        ],
    }


def test_prepare_generates_test_manifest_from_decoded_images():
    result, stored = prepare(prepare_config("8:2"))

    assert result["status"] == "ok", result["message"]
    assert set(result["artifacts"]) == {
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
        "dataset_summary_uri",
        "test_manifest_uri",
    }
    manifest = artifact_document(stored, result, "test_manifest_uri")
    class_map = artifact_document(stored, result, "class_map_uri")
    assert manifest["annotations"] == []
    assert manifest["images"] == [
        {
            "id": index,
            "file_name": f"s3://{BUCKET}/{RAW_PREFIX}test_images/{index:03d}.png",
            "width": 30 + index,
            "height": 40 + index,
        }
        for index in range(1, 6)
    ]
    assert manifest["categories"] == [
        {"id": int(category_id), "name": name, "supercategory": "pill"}
        for category_id, name in sorted(class_map.items(), key=lambda item: int(item[0]))
    ]


def test_empty_test_image_listing_is_rejected_before_publishing():
    result, stored = prepare(
        prepare_config("8:2"), raw_objects(test_image_count=0)
    )

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "test_images" in result["message"]
    assert not [uri for uri in stored if "/processed/" in uri]


def test_nonnumeric_test_image_stem_is_rejected_before_publishing():
    objects = raw_objects(test_image_count=1)
    source = f"s3://{BUCKET}/{RAW_PREFIX}test_images/001.png"
    objects[f"s3://{BUCKET}/{RAW_PREFIX}test_images/image-1.png"] = objects.pop(source)

    result, stored = prepare(prepare_config("8:2"), objects)

    assert result["status"] == "error"
    assert "stem 전체가 숫자" in result["message"]
    assert not [uri for uri in stored if "/processed/" in uri]


def test_duplicate_numeric_test_ids_are_rejected_before_publishing():
    objects = raw_objects(test_image_count=1)
    objects[f"s3://{BUCKET}/{RAW_PREFIX}test_images/1.jpg"] = image_bytes(9, 9)

    result, stored = prepare(prepare_config("8:2"), objects)

    assert result["status"] == "error"
    assert "숫자 id가 중복" in result["message"]
    assert "001.png" in result["message"] and "1.jpg" in result["message"]
    assert not [uri for uri in stored if "/processed/" in uri]


def test_duplicate_test_filenames_are_rejected_before_publishing():
    objects = raw_objects(test_image_count=1)
    objects[f"s3://{BUCKET}/{RAW_PREFIX}test_images/copy/001.png"] = image_bytes(9, 9)

    result, stored = prepare(prepare_config("8:2"), objects)

    assert result["status"] == "error"
    assert "file 이름이 중복" in result["message"]
    assert not [uri for uri in stored if "/processed/" in uri]


def test_unreadable_test_image_is_rejected_before_publishing():
    objects = raw_objects(test_image_count=1)
    objects[f"s3://{BUCKET}/{RAW_PREFIX}test_images/001.png"] = b"not an image"

    result, stored = prepare(prepare_config("8:2"), objects)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "읽을 수 없습니다" in result["message"]
    assert "001.png" in result["message"]
    assert not [uri for uri in stored if "/processed/" in uri]


@pytest.mark.parametrize(
    "class_map",
    [None, {}, {"not-an-id": "pill_a"}, {1: "   "}],
)
def test_missing_or_invalid_class_map_raises_typed_data_error(class_map):
    storage, _ = make_fake_s3_storage()

    with pytest.raises(DataError, match="category"):
        build_test_manifest(
            storage,
            [],
            class_map,
            publish_file_name=str,
        )


def test_duplicate_numeric_class_map_ids_raise_typed_data_error():
    storage, _ = make_fake_s3_storage()

    with pytest.raises(DataError, match="숫자 id가 중복"):
        build_test_manifest(
            storage,
            [],
            {"1": "pill_a", 1: "pill_b"},
            publish_file_name=str,
        )


def test_duplicate_class_map_names_raise_typed_data_error():
    storage, _ = make_fake_s3_storage()

    with pytest.raises(DataError, match="name이 중복"):
        build_test_manifest(
            storage,
            [],
            {1: "pill", 2: "pill"},
            publish_file_name=str,
        )


def test_same_basename_in_train_and_test_keeps_the_split_paths_separate():
    objects = raw_objects(test_image_count=1)
    train_source = f"s3://{BUCKET}/{RAW_PREFIX}train_images/img_001.jpg"
    train_location = f"s3://{BUCKET}/{RAW_PREFIX}train_images/001.png"
    objects[train_location] = objects.pop(train_source)
    annotation_location = (
        f"s3://{BUCKET}/{RAW_PREFIX}train_annotations/img_001.json"
    )
    objects[annotation_location]["images"][0]["file_name"] = "001.png"

    result, stored = prepare(prepare_config("8:2"), objects)

    assert result["status"] == "ok", result["message"]
    test_manifest = artifact_document(stored, result, "test_manifest_uri")
    assert test_manifest["images"][0]["file_name"] == (
        f"s3://{BUCKET}/{RAW_PREFIX}test_images/001.png"
    )
    train_images = [
        image
        for key in ("train_manifest_uri", "validation_manifest_uri")
        for image in artifact_document(stored, result, key)["images"]
    ]
    assert next(image for image in train_images if image["id"] == 1)["file_name"] == (
        train_location
    )


def test_manifest_matches_the_format_the_train_pipeline_reads():
    result, stored = prepare(prepare_config("8:2"))

    class_map = artifact_document(stored, result, "class_map_uri")
    for key in ("train_manifest_uri", "validation_manifest_uri"):
        manifest = artifact_document(stored, result, key)
        assert set(manifest) >= {"images", "annotations", "categories"}
        for field in ("images", "annotations", "categories"):
            assert isinstance(manifest[field], list) and manifest[field]

        category_ids = set()
        for category in manifest["categories"]:
            assert isinstance(category["id"], int) and category["id"] >= 0
            assert isinstance(category["name"], str) and category["name"].strip()
            category_ids.add(category["id"])
        assert len(category_ids) == len(manifest["categories"])
        assert {str(category_id) for category_id in category_ids} == set(class_map)

        image_ids = set()
        for image in manifest["images"]:
            assert isinstance(image["id"], int) and image["id"] >= 0
            assert image["id"] not in image_ids
            image_ids.add(image["id"])
            # 상대경로로 두면 manifest 위치(processed/) 기준으로 풀려 원본
            # 이미지(raw/)를 찾지 못합니다.
            assert image["file_name"].startswith("s3://")
            assert isinstance(image["width"], int) and image["width"] > 0
            assert isinstance(image["height"], int) and image["height"] > 0

        annotation_ids = set()
        for annotation in manifest["annotations"]:
            assert isinstance(annotation["id"], int) and annotation["id"] >= 0
            assert annotation["id"] not in annotation_ids
            annotation_ids.add(annotation["id"])
            assert annotation["image_id"] in image_ids
            assert annotation["category_id"] in category_ids
            bbox = annotation["bbox"]
            assert isinstance(bbox, list) and len(bbox) == 4
            x, y, width, height = bbox
            assert x >= 0 and y >= 0 and width > 0 and height > 0
            assert annotation["iscrowd"] in {0, 1}


def test_class_map_uses_category_id_to_name_form():
    result, stored = prepare(prepare_config("8:2"))

    class_map = artifact_document(stored, result, "class_map_uri")

    assert class_map == {"1": "pill_a", "2": "pill_b", "3": "pill_c"}
    assert all(isinstance(value, str) and value.strip() for value in class_map.values())
    assert len(set(class_map.values())) == len(class_map)


def test_artifacts_are_json_serialisable_and_contract_shaped():
    result, stored = prepare(prepare_config("8:2"))

    assert list(result) == ["status", "artifacts", "summary", "message"]
    assert set(result["artifacts"]) == set(preparation.ARTIFACT_FILE_NAMES)
    for key, uri in result["artifacts"].items():
        assert isinstance(uri, str) and uri.strip()
        assert uri.endswith(preparation.ARTIFACT_FILE_NAMES[key])
        assert json.loads(json.dumps(stored[uri], ensure_ascii=False)) == stored[uri]
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result


def test_dataset_summary_records_source_ratio_and_seed():
    result, stored = prepare(prepare_config("9:1", seed=11))

    summary_document = artifact_document(stored, result, "dataset_summary_uri")

    assert summary_document["source_prefix"] == RAW_PREFIX
    assert summary_document["processed_prefix"] == result["summary"]["processed_prefix"]
    # checksums는 원본에 따라 값이 달라지므로 아래 전용 test에서 확인합니다.
    assert {
        key: value
        for key, value in summary_document["split"].items()
        if key != "checksums"
    } == {
        "method": "deterministic_multilabel_distribution_preserving",
        "split_ratio": "9:1",
        "validation_ratio": 0.1,
        "seed": 11,
    }
    assert summary_document["raw"]["listed_train_images"] == 40
    assert summary_document["raw"]["listed_test_images"] == 5
    assert summary_document["raw"]["annotation_documents"] == 40
    assert set(summary_document["artifacts"]) == {
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
        "test_manifest_uri",
    }


# --- 덮어쓰기 방지 ----------------------------------------------------------


def test_legacy_four_artifacts_backfill_only_the_missing_test_manifest():
    """기존 학습 증거는 그대로 두고 누락된 test manifest 하나만 보충합니다."""

    objects = raw_objects()
    storage, stored = make_fake_s3_storage(objects)
    initial = run_with_fake_storage(storage, prepare_config("8:2"))
    test_manifest_uri = initial["artifacts"]["test_manifest_uri"]
    del stored[test_manifest_uri]
    before = copy.deepcopy(stored)
    storage.read_json.reset_mock()
    storage.write_json.reset_mock()
    storage.download_file.reset_mock()

    result = run_with_fake_storage(storage, prepare_config("8:2"))

    assert result["status"] == "ok", result["message"]
    assert result["summary"]["mode"] == "backfill_test_manifest"
    assert result["summary"]["test_manifest_images"] == 5
    assert set(result["artifacts"]) == set(preparation.ARTIFACT_FILE_NAMES)
    assert result["artifacts"] == initial["artifacts"]
    assert {uri: value for uri, value in stored.items() if uri != test_manifest_uri} == before
    assert stored[test_manifest_uri]["annotations"] == []
    class_map_location = (
        f"datasets/pill_detection/processed/v1-seed42-8020/"
        f"{preparation.ARTIFACT_FILE_NAMES['class_map_uri']}"
    )
    storage.read_json.assert_has_calls([call(class_map_location)])
    # train annotation과 기존 manifest/summary는 백필에서 읽지도 수정하지도 않습니다.
    assert storage.read_json.call_count == 1
    storage.write_json.assert_called_once()
    assert storage.write_json.call_args.kwargs["overwrite"] is False


def test_incomplete_legacy_artifacts_are_not_mistaken_for_a_safe_backfill():
    objects = raw_objects()
    storage, stored = make_fake_s3_storage(objects)
    initial = run_with_fake_storage(storage, prepare_config("8:2"))
    del stored[initial["artifacts"]["test_manifest_uri"]]
    del stored[initial["artifacts"]["validation_manifest_uri"]]
    before = copy.deepcopy(stored)

    result = run_with_fake_storage(storage, prepare_config("8:2"))

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "이미 있습니다" in result["message"]
    assert stored == before


def test_existing_artifacts_are_not_overwritten_silently():
    objects = raw_objects()
    storage, stored = make_fake_s3_storage(objects)
    first = run_with_fake_storage(storage, prepare_config("8:2"))
    before = copy.deepcopy(stored)

    second = run_with_fake_storage(storage, prepare_config("8:2"))

    assert first["status"] == "ok"
    assert second["status"] == "error"
    assert "이미 있습니다" in second["message"]
    assert "overwrite" in second["message"]
    assert stored == before


def test_overwrite_option_replaces_existing_artifacts():
    objects = raw_objects()
    storage, stored = make_fake_s3_storage(objects)
    first = run_with_fake_storage(storage, prepare_config("8:2"))

    second = run_with_fake_storage(storage, prepare_config("8:2", overwrite=True))

    assert second["status"] == "ok"
    assert second["artifacts"] == first["artifacts"]
    assert second["summary"]["overwrite"] is True
    assert len([uri for uri in stored if "/processed/" in uri]) == 5


# --- 잘못된 config와 storage 실패 -------------------------------------------


@pytest.mark.parametrize(
    ("data_section", "expected"),
    [
        ({"prepare": "yes", "split_ratio": "8:2"}, "prepare"),
        ({"prepare": True, "split_ratio": "8:2", "seed": -1}, "seed"),
        ({"prepare": True, "split_ratio": "8:2", "seed": 1.5}, "seed"),
        ({"prepare": True, "split_ratio": "8:2", "overwrite": "yes"}, "overwrite"),
        ({"prepare": True, "split_ratio": "8:2", "raw_prefix": ""}, "raw_prefix"),
        (
            {"prepare": True, "split_ratio": "8:2", "raw_prefix": "somewhere/else/"},
            "raw_prefix",
        ),
        (
            {
                "prepare": True,
                "split_ratio": "8:2",
                "raw_prefix": "datasets/pill_detection/raw/v1/test_images/",
            },
            "raw_prefix",
        ),
        (
            {"prepare": True, "split_ratio": "8:2", "processed_root": "../escape/"},
            "processed_root",
        ),
    ],
)
def test_invalid_preparation_config_returns_error_result(data_section, expected):
    result, stored = prepare({"data": data_section})

    assert result["status"] == "error"
    assert expected in result["message"]
    assert validate_pipeline_result(result, pipeline_name="data") is result
    assert not [uri for uri in stored if "/processed/" in uri]


def test_missing_raw_objects_return_a_readable_error():
    result, _ = prepare(prepare_config("8:2"), {})

    assert result["status"] == "error"
    assert "train_images" in result["message"]


def test_storage_failure_is_reported_without_raising():
    storage, _ = make_fake_s3_storage(raw_objects())
    storage.list = Mock(side_effect=StorageError("s3://secret-bucket/실패"))

    result = run_with_fake_storage(storage, prepare_config("8:2"))

    assert result["status"] == "error"
    assert "StorageError" in result["message"]
    assert "secret-bucket" not in result["message"]
    assert validate_pipeline_result(result, pipeline_name="data") is result


def test_broken_annotation_document_returns_error_with_file_name_only():
    objects = raw_objects()
    objects[f"s3://{BUCKET}/{RAW_PREFIX}train_annotations/img_003.json"] = {
        "images": [],
        "annotations": [],
    }

    result, _ = prepare(prepare_config("8:2"), objects)

    assert result["status"] == "error"
    assert "img_003.json" in result["message"]
    assert f"s3://{BUCKET}" not in result["message"]


def test_image_with_out_of_bounds_bbox_is_excluded_and_reported():
    objects = raw_objects()
    broken_uri = f"s3://{BUCKET}/{RAW_PREFIX}train_annotations/img_005.json"
    document = objects[broken_uri]
    document["annotations"][0]["bbox"] = [90, 90, 50, 50]

    result, stored = prepare(prepare_config("8:2"), objects)

    assert result["status"] == "ok"
    assert result["summary"]["excluded_images"] == 1
    summary_document = artifact_document(stored, result, "dataset_summary_uri")
    excluded = summary_document["excluded_images"]
    assert [entry["file_name"] for entry in excluded] == ["img_005.jpg"]
    assert excluded[0]["reasons"][0]["reason"] == "bbox outside image bounds"
    for key in ("train_manifest_uri", "validation_manifest_uri"):
        manifest = artifact_document(stored, result, key)
        assert all("img_005.jpg" not in image["file_name"] for image in manifest["images"])


# --- 기존 동작 유지 ---------------------------------------------------------


def test_dummy_mode_wins_over_preparation():
    config = prepare_config("8:2")
    config["execution"] = {"mode": "dummy"}

    result, stored = prepare(config)

    assert result == {
        "status": "ok",
        "artifacts": {},
        "summary": {"pipeline": "data", "mode": "dummy"},
        "message": "data pipeline dummy 실행 완료",
    }
    assert not [uri for uri in stored if "/processed/" in uri]


def test_s3_backend_keeps_s3_uris_unchanged():
    result, stored = prepare(prepare_config("8:2"))

    assert all(
        uri.startswith(f"s3://{BUCKET}/") for uri in result["artifacts"].values()
    )
    manifest = artifact_document(stored, result, "train_manifest_uri")
    assert all(image["file_name"].startswith("s3://") for image in manifest["images"])


# --- local backend URI 계약 --------------------------------------------------
#
# LocalStorage는 storage root 기준으로 resolve한 절대 경로를 돌려줍니다. 절대
# 경로와 Windows drive 경로는 소비자(registry)가 계약 위반으로 거부하므로,
# 내보내는 URI는 저장소 root 기준 상대 POSIX 경로여야 합니다.


@pytest.fixture
def local_storage_root():
    """저장소 안에 임시 local storage root를 만듭니다.

    저장소 밖(system 임시 directory)에 두면 저장소 기준 상대 URI를 만들 수 없어
    이 test가 검증하려는 상황 자체를 만들 수 없습니다.
    """

    parent = REPOSITORY_ROOT / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="data-preparation-test-", dir=parent))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def build_local_raw(root: Path, image_count: int = 20) -> None:
    """Local storage root 안에 원본 train 이미지와 annotation을 만듭니다."""

    images = root / RAW_PREFIX / "train_images"
    annotations = root / RAW_PREFIX / "train_annotations"
    test_images = root / RAW_PREFIX / "test_images"
    for directory in (images, annotations, test_images):
        directory.mkdir(parents=True, exist_ok=True)
    (test_images / "1.png").write_bytes(image_bytes(17, 19))
    for index in range(1, image_count + 1):
        (images / f"img_{index:03d}.jpg").write_bytes(b"fake image bytes")
        document = annotation_document(index, [(index - 1) % 2 + 1])
        (annotations / f"img_{index:03d}.json").write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8", newline="\n"
        )


def local_config(root: Path, split_ratio: str = "8:2") -> dict[str, Any]:
    return {
        "execution": {"mode": "local"},
        "storage": {"backend": "local", "local": {"root": str(root)}},
        "data": {"prepare": True, "split_ratio": split_ratio},
    }


@pytest.fixture
def clean_storage_environment(monkeypatch):
    """환경 변수가 config보다 우선하므로 test에서는 비웁니다."""

    for name in (
        "PILL_STORAGE_BACKEND",
        "PILL_STORAGE_LOCAL_ROOT",
        "PILL_STORAGE_S3_BUCKET",
        "PILL_STORAGE_S3_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)


def test_local_backend_publishes_repository_relative_artifact_uris(
    local_storage_root, clean_storage_environment
):
    build_local_raw(local_storage_root)

    result = run(local_config(local_storage_root))

    assert result["status"] == "ok", result["message"]
    assert validate_pipeline_result(result, pipeline_name="data") is result
    assert set(result["artifacts"]) == set(preparation.ARTIFACT_FILE_NAMES)
    for key, uri in result["artifacts"].items():
        # registry의 resolve_local_uri가 거부하는 형태가 아니어야 합니다.
        assert not Path(uri).is_absolute(), f"{key}가 절대 경로입니다: {uri}"
        assert not WINDOWS_DRIVE_PATTERN.match(uri), f"{key}에 drive 문자가 있습니다"
        assert "\\" not in uri
        assert not uri.startswith("s3://")
        resolved = (REPOSITORY_ROOT / uri).resolve()
        assert resolved.is_file()
        assert resolved.relative_to(REPOSITORY_ROOT)


def test_local_backend_manifest_images_resolve_from_the_manifest_directory(
    local_storage_root, clean_storage_environment
):
    build_local_raw(local_storage_root)

    result = run(local_config(local_storage_root))

    assert result["status"] == "ok", result["message"]
    for key in ("train_manifest_uri", "validation_manifest_uri"):
        manifest_path = (REPOSITORY_ROOT / result["artifacts"][key]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["images"]
        for image in manifest["images"]:
            file_name = image["file_name"]
            # 개인 컴퓨터 경로가 artifact 안에 박히면 다른 컴퓨터에서 못 씁니다.
            assert not Path(file_name).is_absolute()
            assert not WINDOWS_DRIVE_PATTERN.match(file_name)
            assert "\\" not in file_name
            # train의 _resolve_image와 같은 방식으로 풉니다.
            resolved = (manifest_path.parent / file_name).resolve()
            assert resolved.is_file(), f"이미지를 찾지 못했습니다: {file_name}"
            assert resolved.relative_to(REPOSITORY_ROOT)
            assert "/train_images/" in resolved.as_posix()
            assert "test_images" not in resolved.as_posix()

    test_manifest_path = (REPOSITORY_ROOT / result["artifacts"]["test_manifest_uri"]).resolve()
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    assert test_manifest["annotations"] == []
    for image in test_manifest["images"]:
        resolved = (test_manifest_path.parent / image["file_name"]).resolve()
        assert resolved.is_file()
        assert "/test_images/" in resolved.as_posix()


def test_local_test_manifest_json_uses_utf8_without_bom_and_lf(
    local_storage_root, clean_storage_environment
):
    build_local_raw(local_storage_root)

    result = run(local_config(local_storage_root))

    assert result["status"] == "ok", result["message"]
    manifest_path = (REPOSITORY_ROOT / result["artifacts"]["test_manifest_uri"]).resolve()
    payload = manifest_path.read_bytes()
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in payload
    assert payload.endswith(b"\n")
    assert json.loads(payload.decode("utf-8"))["annotations"] == []


def test_local_backend_image_paths_are_relative_to_the_manifest_not_the_root(
    local_storage_root, clean_storage_environment
):
    """`processed/`에 있는 manifest에서 `raw/`로 거슬러 올라가야 합니다."""

    build_local_raw(local_storage_root)

    result = run(local_config(local_storage_root))

    manifest = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["train_manifest_uri"]).read_text(
            encoding="utf-8"
        )
    )
    for image in manifest["images"]:
        assert image["file_name"].startswith("../")
        assert image["file_name"].endswith(".jpg")
        assert "/raw/v1/train_images/" in image["file_name"]


def test_local_storage_root_outside_the_repository_is_rejected(
    clean_storage_environment,
):
    outside_root = Path(tempfile.mkdtemp(prefix="data-preparation-outside-"))
    try:
        build_local_raw(outside_root)

        result = run(local_config(outside_root))

        assert result["status"] == "error"
        assert "저장소 안" in result["message"]
        assert str(outside_root) not in result["message"]
        assert validate_pipeline_result(result, pipeline_name="data") is result
        assert not list((outside_root / "datasets/pill_detection/processed").glob("**/*"))
    finally:
        shutil.rmtree(outside_root, ignore_errors=True)


# --- split manifest checksum -------------------------------------------------
#
# 같은 seed와 비율이어도 원본이 바뀌면 split은 달라집니다. 요약에 남긴 sha256이
# 그 변화를 드러내는 유일한 기록이므로, 기록한 값은 실제 저장된 file의 byte와
# 같아야 합니다. 같은 입력이 같은 결과를 내는지는
# test_same_input_seed_and_ratio_produce_the_same_split가 이미 지킵니다.


def test_split_checksums_match_the_written_manifest_files(
    local_storage_root, clean_storage_environment
):
    """기록한 hash는 실제 file을 `sha256sum`한 값과 같아야 합니다."""

    build_local_raw(local_storage_root)

    result = run(local_config(local_storage_root))

    assert result["status"] == "ok", result["message"]
    summary_path = (REPOSITORY_ROOT / result["artifacts"]["dataset_summary_uri"]).resolve()
    checksums = json.loads(summary_path.read_text(encoding="utf-8"))["split"]["checksums"]
    for key, file_name in SPLIT_MANIFEST_FILE_NAMES.items():
        written = (REPOSITORY_ROOT / result["artifacts"][key]).resolve().read_bytes()
        assert checksums[file_name]["sha256"] == hashlib.sha256(written).hexdigest()
        assert checksums[file_name]["bytes"] == len(written)


def test_pass_through_still_works_when_preparation_is_not_requested():
    artifacts = {
        "train_manifest_uri": "artifacts/data/train.json",
        "validation_manifest_uri": "artifacts/data/validation.json",
        "class_map_uri": "artifacts/data/class_map.json",
        "dataset_summary_uri": "artifacts/data/summary.json",
    }
    config = {
        "execution": {"mode": "local"},
        "data": {"prepare": False, "split_ratio": "8:2"},
        "inputs": {"data": dict(artifacts)},
    }

    result, stored = prepare(config)

    assert result["status"] == "ok"
    assert result["artifacts"] == artifacts
    assert result["summary"] == {"pipeline": "data", "mode": "integration"}
    assert not [uri for uri in stored if "/processed/" in uri]
