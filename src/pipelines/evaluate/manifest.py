"""Validation manifest와 class map을 읽고 schema를 검증합니다.

Manifest는 JSON Lines(JSONL) 형식이며 한 줄이 이미지 한 장을 뜻합니다.

    {"image_id": "img-1", "image_uri": "datasets/.../a.jpg", "width": 640,
     "height": 480,
     "annotations": [{"category_id": 1, "bbox": [10, 10, 40, 40]}]}

`bbox`는 COCO와 같은 `[x, y, width, height]` (pixel 단위)입니다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import InputArtifactError
from .storage_io import ArtifactStore


REQUIRED_RECORD_FIELDS = ("image_id", "image_uri", "width", "height")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def normalize_image_key(value: Any) -> str:
    """image_id를 비교 가능한 문자열 key로 바꿉니다."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise InputArtifactError(f"image_id는 문자열 또는 정수여야 합니다: {value!r}")
    key = str(value).strip()
    if not key:
        raise InputArtifactError("image_id는 비어 있을 수 없습니다.")
    return key


def _parse_annotation(value: Any, *, label: str, width: int, height: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputArtifactError(f"{label}: annotation 항목은 object여야 합니다.")

    category_id = value.get("category_id")
    if not isinstance(category_id, int) or isinstance(category_id, bool):
        raise InputArtifactError(f"{label}: category_id는 정수여야 합니다: {category_id!r}")

    bbox = value.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise InputArtifactError(f"{label}: bbox는 [x, y, width, height] 형식이어야 합니다.")
    if not all(_is_number(item) for item in bbox):
        raise InputArtifactError(f"{label}: bbox 값은 모두 유한한 숫자여야 합니다.")

    x, y, box_width, box_height = (float(item) for item in bbox)
    if box_width <= 0 or box_height <= 0:
        raise InputArtifactError(f"{label}: bbox의 width와 height는 0보다 커야 합니다.")
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        raise InputArtifactError(f"{label}: bbox가 이미지 범위를 벗어났습니다.")

    return {"category_id": category_id, "bbox": [x, y, box_width, box_height]}


def parse_manifest(text: str, *, source: str) -> list[dict[str, Any]]:
    """JSONL manifest 문자열을 검증하고 정규화된 record 목록을 반환합니다."""
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        label = f"{source}#line[{line_number}]"
        try:
            document = json.loads(line)
        except json.JSONDecodeError as error:
            raise InputArtifactError(f"{label}: 유효한 JSON이 아닙니다.") from error
        if not isinstance(document, Mapping):
            raise InputArtifactError(f"{label}: manifest record는 object여야 합니다.")

        missing = [name for name in REQUIRED_RECORD_FIELDS if name not in document]
        if missing:
            raise InputArtifactError(f"{label}: 필수 field가 없습니다: {', '.join(missing)}")

        image_key = normalize_image_key(document["image_id"])
        if image_key in seen_keys:
            raise InputArtifactError(f"{label}: image_id가 중복되었습니다: {image_key}")
        seen_keys.add(image_key)

        image_uri = document["image_uri"]
        if not isinstance(image_uri, str) or not image_uri.strip():
            raise InputArtifactError(f"{label}: image_uri는 비어 있지 않은 문자열이어야 합니다.")

        width, height = document["width"], document["height"]
        if not _positive_int(width) or not _positive_int(height):
            raise InputArtifactError(f"{label}: width와 height는 0보다 큰 정수여야 합니다.")

        raw_annotations = document.get("annotations", [])
        if not isinstance(raw_annotations, list):
            raise InputArtifactError(f"{label}: annotations는 list여야 합니다.")
        annotations = [
            _parse_annotation(item, label=f"{label}#annotation[{index}]", width=width, height=height)
            for index, item in enumerate(raw_annotations)
        ]

        records.append(
            {
                "image_id": document["image_id"],
                "image_key": image_key,
                "image_uri": image_uri.strip(),
                "width": width,
                "height": height,
                "annotations": annotations,
            }
        )

    if not records:
        raise InputArtifactError(f"manifest에 record가 없습니다: {source}")
    return records


def load_manifest(store: ArtifactStore, uri: str) -> list[dict[str, Any]]:
    """Manifest artifact를 읽고 검증합니다."""
    return parse_manifest(store.read_text(uri), source=uri)


def parse_class_map(document: Any, *, source: str) -> dict[int, str]:
    """class map 문서를 {category_id: name} 형태로 정규화합니다.

    `{"classes": [{"id": 1, "name": "..."}]}`와 `{"1": "..."}` 두 형식을 지원합니다.
    """
    entries: list[tuple[Any, Any]] = []
    if isinstance(document, Mapping) and isinstance(document.get("classes"), list):
        for index, item in enumerate(document["classes"]):
            if not isinstance(item, Mapping) or "id" not in item or "name" not in item:
                raise InputArtifactError(
                    f"{source}#classes[{index}]: id와 name을 가진 object여야 합니다."
                )
            entries.append((item["id"], item["name"]))
    elif isinstance(document, Mapping):
        entries.extend(document.items())
    else:
        raise InputArtifactError(f"{source}: class map의 최상위 값은 object여야 합니다.")

    class_map: dict[int, str] = {}
    for raw_id, raw_name in entries:
        try:
            category_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise InputArtifactError(f"{source}: category id는 정수여야 합니다: {raw_id!r}") from error
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise InputArtifactError(f"{source}: class 이름은 비어 있지 않은 문자열이어야 합니다.")
        class_map[category_id] = raw_name.strip()

    if not class_map:
        raise InputArtifactError(f"{source}: class map이 비어 있습니다.")
    return class_map


def load_class_map(store: ArtifactStore, uri: str) -> dict[int, str]:
    """class map artifact를 읽고 검증합니다."""
    return parse_class_map(store.read_json(uri), source=uri)


__all__ = [
    "load_class_map",
    "load_manifest",
    "normalize_image_key",
    "parse_class_map",
    "parse_manifest",
]
