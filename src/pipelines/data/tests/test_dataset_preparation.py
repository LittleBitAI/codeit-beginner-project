"""Data pipeline 준비 경로의 테스트.

실제 AWS에는 접속하지 않습니다. 저장소 관례대로 `unittest.mock`으로 만든
in-memory S3 storage 대역을 써서 준비 경로의 흐름과 산출물을 검증합니다.
"""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.common import StorageError, validate_pipeline_result
from src.common.storage import S3Storage
from src.pipelines.data import preparation, run


BUCKET = "test-bucket"
RAW_PREFIX = "datasets/pill_detection/raw/v1/"
CATEGORY_NAMES = {1: "pill_a", 2: "pill_b", 3: "pill_c"}


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

    storage.write_json = Mock(side_effect=write_json)
    storage.read_json = Mock(side_effect=read_json)
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
        objects[f"s3://{BUCKET}/{RAW_PREFIX}test_images/test_{index:03d}.jpg"] = {
            "placeholder": "image bytes"
        }
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
    # 원본은 그대로 두고, 두 옵션의 산출물 4개씩만 새로 생겼습니다.
    processed = sorted(uri for uri in stored if "/processed/" in uri)
    assert len(processed) == 8
    assert set(first["artifacts"].values()) | set(second["artifacts"].values()) == set(
        processed
    )


# --- 재현성 ----------------------------------------------------------------


def test_same_input_seed_and_ratio_produce_the_same_split():
    objects = raw_objects()

    first, first_stored = prepare(prepare_config("8:2"), objects)
    second, second_stored = prepare(prepare_config("8:2"), objects)

    assert first["artifacts"] == second["artifacts"]
    for key in ("train_manifest_uri", "validation_manifest_uri", "class_map_uri"):
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


def test_test_images_are_never_read_from_storage():
    storage, _ = make_fake_s3_storage(raw_objects())

    result = run_with_fake_storage(storage, prepare_config("8:2"))

    assert result["status"] == "ok"
    read_locations = [str(call.args[0]) for call in storage.read_json.call_args_list]
    assert read_locations
    assert all("test_images/" not in location for location in read_locations)
    assert all("/train_annotations/" in location for location in read_locations)


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
    assert summary_document["split"] == {
        "method": "deterministic_multilabel_distribution_preserving",
        "split_ratio": "9:1",
        "validation_ratio": 0.1,
        "seed": 11,
    }
    assert summary_document["raw"]["listed_train_images"] == 40
    assert summary_document["raw"]["annotation_documents"] == 40
    assert set(summary_document["artifacts"]) == {
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
    }


# --- 덮어쓰기 방지 ----------------------------------------------------------


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
    assert len([uri for uri in stored if "/processed/" in uri]) == 4


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
