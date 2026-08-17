"""Validation manifest와 class map을 읽고 schema를 검증합니다.

Manifest는 기존 JSON Lines(JSONL) 형식과 COCO 단일 JSON object를 지원합니다.
JSONL은 한 줄이 이미지 한 장을 뜻합니다.

    {"image_id": "img-1", "image_uri": "datasets/.../a.jpg", "width": 640,
     "height": 480,
     "annotations": [{"category_id": 1, "bbox": [10, 10, 40, 40]}]}

`bbox`는 COCO와 같은 `[x, y, width, height]` (pixel 단위)입니다.
"""

from __future__ import annotations

import json
import math
import posixpath
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .errors import InputArtifactError
from .storage_io import ArtifactStore, is_remote_uri


REQUIRED_RECORD_FIELDS = ("image_id", "image_uri", "width", "height")
REQUIRED_COCO_FIELDS = ("images", "annotations", "categories")
_INTEGER_TEXT = re.compile(r"[+-]?\d+")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _coco_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def normalize_image_key(value: Any) -> str:
    """image_id를 비교 가능한 문자열 key로 바꿉니다."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise InputArtifactError(f"image_id는 문자열 또는 정수여야 합니다: {value!r}")
    key = str(value).strip()
    if not key:
        raise InputArtifactError("image_id는 비어 있을 수 없습니다.")
    return key


def _strict_category_id(value: Any, *, source: str) -> int:
    """정수 또는 정수 문자열만 category id로 받아들입니다.

    `True`나 `1.5`처럼 int()로는 통과하지만 뜻이 달라지는 값은 거부합니다.
    """
    if isinstance(value, bool):
        raise InputArtifactError(f"{source}: category id는 정수여야 합니다: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _INTEGER_TEXT.fullmatch(value.strip()):
        return int(value.strip())
    raise InputArtifactError(f"{source}: category id는 정수여야 합니다: {value!r}")


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


def _parse_jsonl_manifest(text: str, *, source: str) -> list[dict[str, Any]]:
    """기존 JSONL manifest를 검증하고 정규화합니다."""
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


def _resolve_coco_image_uri(
    file_name: str,
    *,
    source: str,
    store: ArtifactStore | None,
) -> str:
    """COCO file_name을 manifest가 있는 directory 기준 URI로 해석합니다."""
    if is_remote_uri(file_name):
        return file_name

    normalized_name = file_name.replace("\\", "/")
    if is_remote_uri(source):
        parsed = urlsplit(source)
        if not parsed.netloc:
            raise InputArtifactError(f"{source}: 유효한 S3 manifest URI가 아닙니다.")
        parent = posixpath.dirname(parsed.path.lstrip("/"))
        key = posixpath.normpath(
            posixpath.join(parent, normalized_name.lstrip("/"))
        )
        if key in {"", ".", ".."} or key.startswith("../"):
            raise InputArtifactError(
                f"{source}: COCO file_name이 S3 bucket 범위를 벗어났습니다: {file_name}"
            )
        return f"s3://{parsed.netloc}/{key}"

    if store is not None:
        manifest_path = store.local_path(source)
        image_path = store.local_path(str(manifest_path.parent / Path(normalized_name)))
        return store.normalize_uri(image_path)

    return (Path(source).parent / Path(normalized_name)).as_posix()


def _parse_coco_manifest(
    document: Mapping[str, Any],
    *,
    source: str,
    store: ArtifactStore | None,
) -> list[dict[str, Any]]:
    missing = [field for field in REQUIRED_COCO_FIELDS if field not in document]
    if missing:
        raise InputArtifactError(
            f"{source}: COCO manifest 필수 field가 없습니다: {', '.join(missing)}"
        )
    for field in REQUIRED_COCO_FIELDS:
        if not isinstance(document[field], list):
            raise InputArtifactError(f"{source}: COCO {field}는 list여야 합니다.")

    category_ids: set[int] = set()
    category_names: set[str] = set()
    for index, entry in enumerate(document["categories"]):
        label = f"{source}#categories[{index}]"
        if not isinstance(entry, Mapping):
            raise InputArtifactError(f"{label}: category 항목은 object여야 합니다.")
        category_id = entry.get("id")
        if not _coco_id(category_id):
            raise InputArtifactError(f"{label}: id는 0 이상의 정수여야 합니다.")
        if category_id in category_ids:
            raise InputArtifactError(f"{label}: category id가 중복되었습니다: {category_id}")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InputArtifactError(f"{label}: name은 비어 있지 않은 문자열이어야 합니다.")
        normalized_name = name.strip()
        if normalized_name in category_names:
            raise InputArtifactError(f"{label}: category name이 중복되었습니다: {normalized_name}")
        category_ids.add(category_id)
        category_names.add(normalized_name)

    records: list[dict[str, Any]] = []
    records_by_id: dict[int, dict[str, Any]] = {}
    for index, entry in enumerate(document["images"]):
        label = f"{source}#images[{index}]"
        if not isinstance(entry, Mapping):
            raise InputArtifactError(f"{label}: image 항목은 object여야 합니다.")
        image_id = entry.get("id")
        if not _coco_id(image_id):
            raise InputArtifactError(f"{label}: id는 0 이상의 정수여야 합니다.")
        if image_id in records_by_id:
            raise InputArtifactError(f"{label}: image id가 중복되었습니다: {image_id}")
        file_name = entry.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise InputArtifactError(
                f"{label}: file_name은 비어 있지 않은 문자열이어야 합니다."
            )
        width, height = entry.get("width"), entry.get("height")
        if not _positive_int(width) or not _positive_int(height):
            raise InputArtifactError(f"{label}: width와 height는 0보다 큰 정수여야 합니다.")

        record = {
            "image_id": image_id,
            "image_key": normalize_image_key(image_id),
            "image_uri": _resolve_coco_image_uri(
                file_name.strip(), source=source, store=store
            ),
            "width": width,
            "height": height,
            "annotations": [],
        }
        records.append(record)
        records_by_id[image_id] = record

    if not records:
        raise InputArtifactError(f"COCO manifest에 image가 없습니다: {source}")

    annotation_ids: set[int] = set()
    for index, entry in enumerate(document["annotations"]):
        label = f"{source}#annotations[{index}]"
        if not isinstance(entry, Mapping):
            raise InputArtifactError(f"{label}: annotation 항목은 object여야 합니다.")
        annotation_id = entry.get("id")
        if not _coco_id(annotation_id):
            raise InputArtifactError(f"{label}: id는 0 이상의 정수여야 합니다.")
        if annotation_id in annotation_ids:
            raise InputArtifactError(
                f"{label}: annotation id가 중복되었습니다: {annotation_id}"
            )
        image_id = entry.get("image_id")
        if not _coco_id(image_id) or image_id not in records_by_id:
            raise InputArtifactError(
                f"{label}: 존재하지 않는 image_id를 참조합니다: {image_id!r}"
            )
        category_id = entry.get("category_id")
        if not _coco_id(category_id) or category_id not in category_ids:
            raise InputArtifactError(
                f"{label}: 존재하지 않는 category_id를 참조합니다: {category_id!r}"
            )
        iscrowd = entry.get("iscrowd", 0)
        if (
            not isinstance(iscrowd, int)
            or isinstance(iscrowd, bool)
            or iscrowd not in {0, 1}
        ):
            raise InputArtifactError(f"{label}: iscrowd는 0 또는 1이어야 합니다.")

        image_record = records_by_id[image_id]
        annotation = _parse_annotation(
            entry,
            label=label,
            width=image_record["width"],
            height=image_record["height"],
        )
        image_record["annotations"].append(annotation)
        annotation_ids.add(annotation_id)

    return records


def _looks_like_coco(document: Any) -> bool:
    if not isinstance(document, Mapping):
        return False
    # 기존 JSONL의 단일 record도 전체 text로는 유효한 JSON object입니다.
    # JSONL field가 하나라도 있으면 먼저 기존 parser로 보내 누락 field의
    # line 단위 오류 형식을 유지합니다.
    if any(field in document for field in REQUIRED_RECORD_FIELDS):
        return False
    return any(field in document for field in REQUIRED_COCO_FIELDS)


def parse_manifest(
    text: str,
    *,
    source: str,
    store: ArtifactStore | None = None,
) -> list[dict[str, Any]]:
    """JSONL 또는 COCO manifest를 검증해 같은 record 목록으로 정규화합니다."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None

    if _looks_like_coco(document):
        return _parse_coco_manifest(document, source=source, store=store)
    return _parse_jsonl_manifest(text, source=source)


def load_manifest(store: ArtifactStore, uri: str) -> list[dict[str, Any]]:
    """Manifest artifact를 읽고 검증합니다."""
    return parse_manifest(store.read_text(uri), source=uri, store=store)


def parse_test_manifest(
    text: str,
    *,
    source: str,
    store: ArtifactStore | None = None,
) -> tuple[list[dict[str, Any]], frozenset[int]]:
    """라벨 없는 competition COCO test manifest를 검증합니다."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise InputArtifactError(f"{source}: 유효한 COCO JSON이 아닙니다.") from error
    if not isinstance(document, Mapping):
        raise InputArtifactError(f"{source}: test manifest는 COCO object여야 합니다.")
    if document.get("annotations") != []:
        raise InputArtifactError(
            f"{source}: test manifest는 label 없이 annotations=[]여야 합니다."
        )

    records = _parse_coco_manifest(document, source=source, store=store)
    if not document["categories"]:
        raise InputArtifactError(f"{source}: test manifest categories가 비어 있습니다.")
    seen_file_names: set[str] = set()
    seen_numeric_stems: set[int] = set()
    for index, image in enumerate(document["images"]):
        file_name = str(image["file_name"]).replace("\\", "/")
        stem = posixpath.splitext(posixpath.basename(file_name))[0]
        if not stem.isdecimal():
            raise InputArtifactError(
                f"{source}#images[{index}]: 숫자 file_name이 필요합니다: {file_name!r}"
            )
        numeric_stem = int(stem)
        if file_name in seen_file_names:
            raise InputArtifactError(
                f"{source}#images[{index}]: file_name이 중복되었습니다: {file_name}"
            )
        if numeric_stem in seen_numeric_stems:
            raise InputArtifactError(
                f"{source}#images[{index}]: file_name 숫자 stem이 중복되었습니다: {stem}"
            )
        if numeric_stem != image["id"]:
            raise InputArtifactError(
                f"{source}#images[{index}]: file_name 숫자 stem과 image id가 다릅니다: "
                f"{stem} != {image['id']}"
            )
        seen_file_names.add(file_name)
        seen_numeric_stems.add(numeric_stem)
    return records, frozenset(int(category["id"]) for category in document["categories"])


def load_test_manifest(
    store: ArtifactStore,
    uri: str,
) -> tuple[list[dict[str, Any]], frozenset[int]]:
    """Competition test manifest artifact를 읽고 검증합니다."""
    return parse_test_manifest(store.read_text(uri), source=uri, store=store)


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
        category_id = _strict_category_id(raw_id, source=source)
        if category_id in class_map:
            raise InputArtifactError(f"{source}: category id가 중복되었습니다: {category_id}")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise InputArtifactError(f"{source}: class 이름은 비어 있지 않은 문자열이어야 합니다.")
        class_map[category_id] = raw_name.strip()

    if not class_map:
        raise InputArtifactError(f"{source}: class map이 비어 있습니다.")
    return class_map


def load_class_map(store: ArtifactStore, uri: str) -> dict[int, str]:
    """class map artifact를 읽고 검증합니다."""
    return parse_class_map(store.read_json(uri), source=uri)


def sample_records(
    records: Sequence[Mapping[str, Any]], size: int, *, seed: int
) -> list[dict[str, Any]]:
    """검증 image를 ``size``장만 고릅니다. class가 고르게 들어가도록 집습니다.

    후보 checkpoint 20개를 전수로 재면 GPU로도 한 시간이 넘습니다. 그런데 그냥
    무작위로 줄이면 **드문 class가 통째로 빠집니다.** 알약 118종에 300장이면 한
    class가 한 장도 못 들어가는 일이 흔하고, 그 class의 AP는 측정값이 아니라 없는
    값이 되어 후보끼리 견줄 수 없게 됩니다.

    그래서 **먼저 모든 class를 덮고, 남는 자리를 채웁니다.** 사진 한 장에 알약이 넷까지
    들어가므로 한 장이 여러 class를 한꺼번에 덮습니다. 이미 덮인 class 차례에 또 한 장을
    집으면 그 자리만큼 아직 못 덮은 class가 밀려나므로, 덮인 class는 건너뜁니다. 모든
    class가 한 번씩 나온 뒤에 남는 자리는 무작위로 채워 실제 분포를 따르게 둡니다.

    같은 ``seed``면 같은 표본이므로 후보들은 모두 같은 시험지를 봅니다.
    """

    if size >= len(records):
        return [dict(record) for record in records]
    shuffled = [dict(record) for record in records]
    # 전역 ``random``을 쓰면 이 표본이 호출 순서에 따라 달라집니다.
    random.Random(seed).shuffle(shuffled)

    queues: dict[int, list[dict[str, Any]]] = {}
    for record in shuffled:
        for category_id in sorted(
            {annotation["category_id"] for annotation in record["annotations"]}
        ):
            queues.setdefault(category_id, []).append(record)

    chosen: dict[Any, dict[str, Any]] = {}
    covered: set[int] = set()
    # **한 바퀴면 끝납니다.** 이 바퀴에서 각 class는 자기 사진을 받아 덮이거나, 다른
    # 사진이 이미 덮었거나, 쓸 사진이 남아 있지 않습니다. 그래서 두 바퀴째에 할 일이
    # 없습니다.
    for category_id in sorted(queues):
        if len(chosen) == size:
            break
        if category_id in covered:
            # 다른 사진이 이미 덮은 class입니다. 여기서 또 집으면 아직 한 장도 없는
            # class가 그만큼 밀려납니다.
            continue
        queue = queues[category_id]
        while queue and queue[-1]["image_id"] in chosen:
            queue.pop()
        if not queue:
            continue
        record = queue.pop()
        chosen[record["image_id"]] = record
        covered.update(
            annotation["category_id"] for annotation in record["annotations"]
        )
    # 라벨이 하나도 없는 image와, class를 다 덮고 남은 자리입니다. 무작위로 채워
    # 실제 분포를 따르게 둡니다 — 배경만 있는 사진에서 나온 오탐도 점수에 들어갑니다.
    for record in shuffled:
        if len(chosen) == size:
            break
        chosen.setdefault(record["image_id"], record)

    # manifest 순서를 지킵니다. 뒤섞인 순서로 추론하면 진행 로그와 예측 파일의
    # 순서가 실행마다 달라져 두 결과를 나란히 놓고 볼 수 없습니다.
    return [
        chosen[record["image_id"]]
        for record in records
        if record["image_id"] in chosen
    ]


__all__ = [
    "load_class_map",
    "load_manifest",
    "load_test_manifest",
    "normalize_image_key",
    "parse_class_map",
    "parse_manifest",
    "parse_test_manifest",
    "sample_records",
]
