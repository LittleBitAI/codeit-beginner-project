"""원본의 이미지별 COCO 문서를 하나의 COCO dataset으로 합칩니다.

원본 annotation은 이미지 한 장마다 별도 JSON 문서로 저장되어 있습니다. 이
module은 그 문서들을 읽어 하나의 `images`/`annotations`/`categories`로 합치고,
bounding box가 이미지 밖으로 나가는 등 학습에 쓸 수 없는 이미지를 제외 목록으로
분리합니다. Storage 접근은 하지 않고 이미 읽어 온 문서만 다룹니다.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .errors import DatasetPreparationError


__all__ = ["ConsolidatedDataset", "consolidate", "location_name"]


def _is_index(value: Any) -> bool:
    """COCO id로 쓸 수 있는 0 이상의 정수인지 확인합니다."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_size(value: Any) -> bool:
    """이미지 width/height로 쓸 수 있는 양의 정수인지 확인합니다."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_number(value: Any) -> bool:
    """bbox 값으로 쓸 수 있는 유한한 숫자인지 확인합니다."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def location_name(location: str) -> str:
    """Storage URI 또는 경로에서 file 이름만 뽑습니다.

    오류 message와 로그에는 이 값만 사용해 개인 컴퓨터 절대경로가 새어 나가지
    않게 합니다.
    """

    return PurePosixPath(str(location).replace("\\", "/")).name


@dataclass(frozen=True)
class ConsolidatedDataset:
    """합쳐진 COCO dataset과 제외 내역입니다."""

    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)
    excluded_images: list[dict[str, Any]] = field(default_factory=list)
    unreferenced_image_count: int = 0


def _location_by_name(image_locations: Sequence[str]) -> dict[str, str]:
    """이미지 file 이름으로 storage 위치를 찾을 수 있는 색인을 만듭니다."""

    candidates: defaultdict[str, list[str]] = defaultdict(list)
    for location in image_locations:
        candidates[location_name(location)].append(str(location))
    duplicates = sorted(name for name, found in candidates.items() if len(found) != 1)
    if duplicates:
        raise DatasetPreparationError(
            "원본 train image의 file 이름은 서로 달라야 합니다. 중복된 이름 "
            f"{len(duplicates)}개(예: {', '.join(duplicates[:3])})"
        )
    return {name: found[0] for name, found in candidates.items()}


def _fields(document: Any, source_name: str) -> tuple[list, list, list]:
    """COCO 문서에서 필수 list field 세 개를 꺼냅니다."""

    if not isinstance(document, Mapping):
        raise DatasetPreparationError(
            f"annotation 문서의 최상위 값은 object여야 합니다: {source_name}"
        )
    missing = [
        name
        for name in ("images", "annotations", "categories")
        if not isinstance(document.get(name), list)
    ]
    if missing:
        raise DatasetPreparationError(
            f"annotation 문서에 list field가 없거나 형식이 다릅니다: {source_name} "
            f"({', '.join(missing)})"
        )
    return document["images"], document["annotations"], document["categories"]


def _collect_images(
    entries: list[Any],
    *,
    source_name: str,
    location_by_name: Mapping[str, str],
    images_by_name: dict[str, dict[str, Any]],
    name_by_id: dict[int, str],
) -> dict[int, str]:
    """문서 안의 이미지 정보를 전체 dataset 기준으로 합칩니다."""

    document_images: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise DatasetPreparationError(
                f"image 항목은 object여야 합니다: {source_name}"
            )
        image_id = entry.get("id")
        file_name = entry.get("file_name")
        width = entry.get("width")
        height = entry.get("height")
        if not _is_index(image_id):
            raise DatasetPreparationError(
                f"image id는 0 이상의 정수여야 합니다: {source_name}"
            )
        if not isinstance(file_name, str) or not file_name.strip():
            raise DatasetPreparationError(
                f"image file_name은 비어 있지 않은 문자열이어야 합니다: {source_name}"
            )
        if not _is_size(width) or not _is_size(height):
            raise DatasetPreparationError(
                f"image width와 height는 양의 정수여야 합니다: {source_name}"
            )

        name = location_name(file_name)
        location = location_by_name.get(name)
        if location is None:
            raise DatasetPreparationError(
                f"annotation이 가리키는 이미지가 train_images에 없습니다: {name}"
            )

        canonical = {
            "id": image_id,
            "file_name": location,
            "width": width,
            "height": height,
        }
        existing = images_by_name.get(name)
        if existing is not None and existing != canonical:
            raise DatasetPreparationError(
                f"같은 이미지에 서로 다른 metadata가 있습니다: {name}"
            )
        previous_name = name_by_id.get(image_id)
        if previous_name is not None and previous_name != name:
            raise DatasetPreparationError(
                f"하나의 image id가 서로 다른 이미지를 가리킵니다: {image_id}"
            )
        images_by_name[name] = canonical
        name_by_id[image_id] = name
        document_images[image_id] = name
    return document_images


def _collect_categories(
    entries: list[Any],
    *,
    source_name: str,
    categories_by_id: dict[int, dict[str, Any]],
) -> dict[int, str]:
    """문서 안의 category 정의를 전체 dataset 기준으로 합칩니다."""

    document_categories: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise DatasetPreparationError(
                f"category 항목은 object여야 합니다: {source_name}"
            )
        category_id = entry.get("id")
        name = entry.get("name")
        if not _is_index(category_id) or not isinstance(name, str) or not name.strip():
            raise DatasetPreparationError(
                f"category id는 0 이상의 정수, name은 비어 있지 않은 문자열이어야 "
                f"합니다: {source_name}"
            )
        canonical = {"id": category_id, "name": name.strip()}
        existing = categories_by_id.get(category_id)
        if existing is not None and existing != canonical:
            raise DatasetPreparationError(
                f"같은 category id에 서로 다른 name이 있습니다: {category_id}"
            )
        categories_by_id[category_id] = canonical
        document_categories[category_id] = canonical["name"]
    return document_categories


def _collect_annotations(
    entries: list[Any],
    *,
    source_name: str,
    document_images: Mapping[int, str],
    document_categories: Mapping[int, str],
    images_by_name: Mapping[str, dict[str, Any]],
    raw_annotations: list[dict[str, Any]],
    invalid_images: dict[str, list[dict[str, Any]]],
) -> None:
    """문서 안의 annotation을 검증하고 전체 목록에 모읍니다."""

    for entry in entries:
        if not isinstance(entry, Mapping):
            raise DatasetPreparationError(
                f"annotation 항목은 object여야 합니다: {source_name}"
            )
        image_id = entry.get("image_id")
        category_id = entry.get("category_id")
        bbox = entry.get("bbox")
        iscrowd = entry.get("iscrowd", 0)
        if image_id not in document_images:
            raise DatasetPreparationError(
                f"annotation이 같은 문서에 없는 image id를 가리킵니다: {source_name}"
            )
        if category_id not in document_categories:
            raise DatasetPreparationError(
                f"annotation이 같은 문서에 없는 category id를 가리킵니다: {source_name}"
            )
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(_is_number(value) for value in bbox)
        ):
            raise DatasetPreparationError(
                f"bbox는 유한한 숫자 4개의 list여야 합니다: {source_name}"
            )
        if not isinstance(iscrowd, int) or isinstance(iscrowd, bool) or iscrowd not in {0, 1}:
            raise DatasetPreparationError(
                f"iscrowd는 0 또는 1이어야 합니다: {source_name}"
            )

        x, y, width, height = (float(value) for value in bbox)
        image_name = document_images[image_id]
        image = images_by_name[image_name]
        reason: str | None = None
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            reason = "negative position or non-positive size"
        elif x + width > image["width"] or y + height > image["height"]:
            reason = "bbox outside image bounds"
        if reason is not None:
            invalid_images[image_name].append(
                {
                    "source_annotation_file": source_name,
                    "original_bbox": [float(value) for value in bbox],
                    "reason": reason,
                }
            )
        raw_annotations.append(
            {
                "source_annotation_id": entry.get("id"),
                "image_id": image_id,
                "image_name": image_name,
                "category_id": category_id,
                "bbox": [x, y, width, height],
                "iscrowd": iscrowd,
            }
        )


def consolidate(
    documents: Sequence[tuple[str, Any]],
    image_locations: Sequence[str],
) -> ConsolidatedDataset:
    """이미지별 COCO 문서를 하나의 COCO dataset으로 합칩니다.

    bbox가 이미지 밖으로 나가거나 크기가 0 이하인 이미지, 그리고 남은 annotation이
    하나도 없는 이미지는 학습에 쓸 수 없으므로 제외하고 그 이유를 함께 돌려줍니다.
    """

    location_by_name = _location_by_name(image_locations)
    images_by_name: dict[str, dict[str, Any]] = {}
    name_by_id: dict[int, str] = {}
    categories_by_id: dict[int, dict[str, Any]] = {}
    raw_annotations: list[dict[str, Any]] = []
    invalid_images: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for source, document in documents:
        source_name = location_name(source)
        image_entries, annotation_entries, category_entries = _fields(document, source_name)
        document_images = _collect_images(
            image_entries,
            source_name=source_name,
            location_by_name=location_by_name,
            images_by_name=images_by_name,
            name_by_id=name_by_id,
        )
        document_categories = _collect_categories(
            category_entries,
            source_name=source_name,
            categories_by_id=categories_by_id,
        )
        _collect_annotations(
            annotation_entries,
            source_name=source_name,
            document_images=document_images,
            document_categories=document_categories,
            images_by_name=images_by_name,
            raw_annotations=raw_annotations,
            invalid_images=invalid_images,
        )

    excluded_names = set(invalid_images)
    retained_counts = Counter(
        annotation["image_name"]
        for annotation in raw_annotations
        if annotation["image_name"] not in excluded_names
    )
    for name in sorted(images_by_name):
        if name not in excluded_names and retained_counts[name] == 0:
            invalid_images[name].append(
                {
                    "source_annotation_file": None,
                    "original_bbox": None,
                    "reason": "no usable annotation",
                }
            )
    excluded_names = set(invalid_images)

    included_images = [
        image for name, image in sorted(images_by_name.items()) if name not in excluded_names
    ]
    if len({image["id"] for image in included_images}) != len(included_images):
        raise DatasetPreparationError("포함된 이미지의 image id가 서로 겹칩니다.")

    retained = [
        annotation
        for annotation in raw_annotations
        if annotation["image_name"] not in excluded_names
    ]
    retained.sort(
        key=lambda item: (
            item["image_id"],
            item["category_id"],
            str(item["source_annotation_id"]),
        )
    )
    annotations = [
        {
            "id": index,
            "image_id": annotation["image_id"],
            "category_id": annotation["category_id"],
            "bbox": annotation["bbox"],
            "iscrowd": annotation["iscrowd"],
        }
        for index, annotation in enumerate(retained, start=1)
    ]
    used_category_ids = {annotation["category_id"] for annotation in annotations}
    categories = [categories_by_id[category_id] for category_id in sorted(used_category_ids)]
    excluded = [
        {
            "file_name": name,
            "image_id": images_by_name[name]["id"],
            "reasons": reasons,
        }
        for name, reasons in sorted(invalid_images.items())
    ]
    if not included_images or not annotations or not categories:
        raise DatasetPreparationError(
            "학습에 쓸 수 있는 이미지, annotation, category가 모두 필요합니다. "
            f"(이미지 {len(included_images)}개, annotation {len(annotations)}개, "
            f"category {len(categories)}개)"
        )
    return ConsolidatedDataset(
        images=included_images,
        annotations=annotations,
        categories=categories,
        excluded_images=excluded,
        unreferenced_image_count=len(image_locations) - len(images_by_name),
    )
