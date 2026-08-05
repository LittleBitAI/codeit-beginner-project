"""Test image 목록에서 annotation 없는 COCO manifest를 만듭니다."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.common import Storage

from .coco import location_name
from .errors import DatasetPreparationError


_NUMERIC_STEM = re.compile(r"[0-9]+")


def _categories(class_map: Mapping[str | int, Any] | None) -> list[dict[str, Any]]:
    """Class map의 category id/name을 검증하고 COCO 형식으로 정렬합니다."""

    if not isinstance(class_map, Mapping) or not class_map:
        raise DatasetPreparationError(
            "test manifest에는 category class_map object가 하나 이상 필요합니다."
        )
    categories_by_id: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    for raw_category_id, name in class_map.items():
        if isinstance(raw_category_id, bool):
            raise DatasetPreparationError(
                "test manifest category id는 0 이상의 정수 또는 숫자 문자열이어야 "
                "합니다."
            )
        if isinstance(raw_category_id, int):
            category_id = raw_category_id
        elif isinstance(raw_category_id, str) and _NUMERIC_STEM.fullmatch(
            raw_category_id
        ):
            category_id = int(raw_category_id)
        else:
            raise DatasetPreparationError(
                "test manifest category id는 0 이상의 정수 또는 숫자 문자열이어야 "
                "합니다."
            )
        if category_id < 0 or not isinstance(name, str) or not name.strip():
            raise DatasetPreparationError(
                "test manifest category에는 0 이상의 id와 비어 있지 않은 name이 "
                "필요합니다."
            )
        if category_id in categories_by_id:
            raise DatasetPreparationError(
                f"test manifest category의 숫자 id가 중복됩니다: {category_id}"
            )
        if name in names:
            raise DatasetPreparationError(
                f"test manifest category name이 중복됩니다: {name}"
            )
        categories_by_id[category_id] = {
            "id": category_id,
            "name": name,
            "supercategory": "pill",
        }
        names.add(name)
    return [categories_by_id[category_id] for category_id in sorted(categories_by_id)]


def _image_candidates(
    image_locations: Sequence[str],
) -> list[tuple[int, str, str]]:
    """Test image 이름과 id를 검증한 뒤 id 순서로 돌려줍니다."""

    if not image_locations:
        raise DatasetPreparationError("test_images에 image가 하나 이상 필요합니다.")

    names: set[str] = set()
    ids: dict[int, str] = {}
    candidates: list[tuple[int, str, str]] = []
    for location in image_locations:
        text = str(location)
        name = location_name(text)
        stem = Path(name).stem
        if _NUMERIC_STEM.fullmatch(stem) is None:
            raise DatasetPreparationError(
                f"test image file 이름의 stem 전체가 숫자여야 합니다: {name}"
            )
        image_id = int(stem)
        if name in names:
            raise DatasetPreparationError(f"test image file 이름이 중복됩니다: {name}")
        previous_name = ids.get(image_id)
        if previous_name is not None:
            raise DatasetPreparationError(
                "test image의 숫자 id가 중복됩니다: "
                f"{previous_name}, {name} (id {image_id})"
            )
        names.add(name)
        ids[image_id] = name
        candidates.append((image_id, name, text))
    return sorted(candidates, key=lambda item: item[0])


def _decoded_size(storage: Storage, location: str, destination: Path, name: str) -> tuple[int, int]:
    """Image를 끝까지 decode하고 실제 width와 height를 돌려줍니다."""

    storage.download_file(location, destination)
    try:
        with Image.open(destination) as image:
            image.load()
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise DatasetPreparationError(f"test image를 읽을 수 없습니다: {name}") from error
    if width <= 0 or height <= 0:
        raise DatasetPreparationError(f"test image 크기가 올바르지 않습니다: {name}")
    return width, height


def build_test_manifest(
    storage: Storage,
    image_locations: Sequence[str],
    class_map: Mapping[str | int, Any] | None,
    *,
    publish_file_name: Callable[[str], str],
) -> dict[str, Any]:
    """Test image를 검증하고 downstream에서 읽을 수 있는 COCO 문서를 만듭니다."""

    manifest_categories = _categories(class_map)
    candidates = _image_candidates(image_locations)
    images: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pill-test-manifest-") as temporary_directory:
        directory = Path(temporary_directory)
        for order, (image_id, name, location) in enumerate(candidates):
            destination = directory / f"{order}{Path(name).suffix.lower()}"
            width, height = _decoded_size(storage, location, destination, name)
            images.append(
                {
                    "id": image_id,
                    "file_name": publish_file_name(location),
                    "width": width,
                    "height": height,
                }
            )
    return {
        "info": {
            "description": "Pill detection test COCO manifest",
            "split": "test",
        },
        "images": images,
        "annotations": [],
        "categories": manifest_categories,
    }
