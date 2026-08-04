"""COCO manifest loading for the train pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.common import LocalStorage, Storage


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_INTEGER_TEXT = re.compile(r"[+-]?\d+")


def _is_s3(location: str) -> bool:
    return location.lower().startswith("s3://")


def _repo_path(location: str | Path) -> Path:
    candidate = Path(location)
    if candidate.is_absolute():
        raise ValueError(f"local artifact URI must be repository-relative: {location}")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"local artifact URI leaves the repository: {location}") from error
    return resolved


def _local_artifact_path(location: str | Path, storage: Storage) -> Path:
    """Resolve local input while containing absolute paths in LocalStorage.root."""
    candidate = Path(location)
    if not candidate.is_absolute():
        return _repo_path(candidate)
    if not isinstance(storage, LocalStorage):
        raise ValueError(f"absolute local artifact URI requires LocalStorage: {location}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(storage.root)
    except ValueError as error:
        raise ValueError(f"absolute local artifact URI leaves the storage root: {location}") from error
    return resolved


def read_json_artifact(location: str, storage: Storage) -> Any:
    """Read a local repository-relative or S3 JSON artifact."""
    if _is_s3(location):
        return storage.read_json(location)

    path = _local_artifact_path(location, storage)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"JSON artifact does not exist: {location}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON artifact is not valid UTF-8 JSON: {location}") from error
    except OSError as error:
        raise ValueError(f"JSON artifact could not be read: {location}") from error


def load_class_map(location: str, storage: Storage) -> dict[str, int]:
    document = read_json_artifact(location, storage)
    if not isinstance(document, Mapping) or not document:
        raise ValueError("class map must be a non-empty JSON object")

    if all(isinstance(value, str) for value in document.values()):
        categories: list[tuple[int, str]] = []
        category_ids: set[int] = set()
        names: set[str] = set()
        for raw_id, raw_name in document.items():
            if not isinstance(raw_id, str) or not _INTEGER_TEXT.fullmatch(raw_id.strip()):
                raise ValueError("class map category ids must be non-negative integers")
            category_id = int(raw_id.strip())
            if category_id < 0:
                raise ValueError("class map category ids must be non-negative integers")
            if category_id in category_ids:
                raise ValueError("class map category ids must be unique")
            name = raw_name.strip()
            if not name:
                raise ValueError("class map names must be non-empty strings")
            if name in names:
                raise ValueError("class map names must be unique")
            categories.append((category_id, name))
            category_ids.add(category_id)
            names.add(name)
        categories.sort(key=lambda category: category[0])
        return {
            name: label
            for label, (_, name) in enumerate(categories, start=1)
        }

    if any(isinstance(value, str) for value in document.values()):
        raise ValueError("class map cannot mix model labels and class names")

    class_map: dict[str, int] = {}
    for name, label in document.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("class map names must be non-empty strings")
        if not isinstance(label, int) or isinstance(label, bool) or label <= 0:
            raise ValueError("class map labels must be positive integers; 0 is background")
        class_map[name] = label

    labels = sorted(class_map.values())
    if labels != list(range(1, len(labels) + 1)):
        raise ValueError("class map labels must be unique and contiguous from 1")
    return class_map


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _coco_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _s3_relative(base_uri: str, relative: str) -> str:
    parsed = urlsplit(base_uri)
    parent = parsed.path.rsplit("/", 1)[0].strip("/")
    key = "/".join(part for part in (parent, relative.replace("\\", "/").lstrip("/")) if part)
    return f"s3://{parsed.netloc}/{key}"


class CocoDetectionDataset(Dataset):
    """Strict COCO detection dataset backed by local files or S3 objects."""

    def __init__(
        self,
        manifest_uri: str,
        class_map: Mapping[str, int],
        storage: Storage,
        cache_directory: Path,
    ) -> None:
        self.manifest_uri = manifest_uri
        self.class_map = dict(class_map)
        self.storage = storage
        self.cache_directory = cache_directory
        document = read_json_artifact(manifest_uri, storage)
        if not isinstance(document, Mapping):
            raise ValueError(f"COCO manifest root must be an object: {manifest_uri}")
        for field in ("images", "annotations", "categories"):
            if not isinstance(document.get(field), list):
                raise ValueError(f"COCO manifest field must be a list: {field}")

        self._category_labels = self._validate_categories(document["categories"])
        self._images = self._validate_images(document["images"])
        self._annotations = self._validate_annotations(document["annotations"])
        if not self._images:
            raise ValueError(f"COCO manifest contains no images: {manifest_uri}")

    def _validate_categories(self, entries: list[Any]) -> dict[Any, int]:
        category_labels: dict[Any, int] = {}
        names: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"categories[{index}] must be an object")
            category_id = entry.get("id")
            name = entry.get("name")
            if not _coco_id(category_id) or category_id in category_labels:
                raise ValueError("COCO category ids must be non-negative integers and unique")
            if not isinstance(name, str) or name not in self.class_map or name in names:
                raise ValueError(f"COCO category name is missing, duplicated, or absent from class map: {name!r}")
            category_labels[category_id] = self.class_map[name]
            names.add(name)
        if names != set(self.class_map):
            raise ValueError("COCO categories and class map names must match exactly")
        self._category_ids = {
            label: category_id for category_id, label in category_labels.items()
        }
        return category_labels

    def _resolve_image(self, file_name: str) -> str:
        if _is_s3(file_name):
            return file_name
        if self.manifest_uri.lower().startswith("s3://"):
            return _s3_relative(self.manifest_uri, file_name)

        manifest_path = _local_artifact_path(self.manifest_uri, self.storage)
        candidate = (manifest_path.parent / file_name).resolve()
        if Path(self.manifest_uri).is_absolute() or Path(file_name).is_absolute():
            return str(_local_artifact_path(candidate, self.storage))
        try:
            return candidate.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"COCO image path leaves the repository: {file_name}") from error

    def _validate_images(self, entries: list[Any]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        ids: set[Any] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"images[{index}] must be an object")
            image_id = entry.get("id")
            file_name = entry.get("file_name")
            if not _coco_id(image_id) or image_id in ids:
                raise ValueError("COCO image ids must be non-negative integers and unique")
            if not isinstance(file_name, str) or not file_name.strip():
                raise ValueError(f"images[{index}].file_name must be a non-empty string")
            for dimension in ("width", "height"):
                value = entry.get(dimension)
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                    raise ValueError(f"images[{index}].{dimension} must be a positive integer")
            ids.add(image_id)
            images.append(
                {
                    "id": image_id,
                    "location": self._resolve_image(file_name),
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                }
            )
        self._image_ids = ids
        return images

    def _validate_annotations(self, entries: list[Any]) -> dict[Any, list[dict[str, Any]]]:
        annotations: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        annotation_ids: set[Any] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"annotations[{index}] must be an object")
            annotation_id = entry.get("id")
            image_id = entry.get("image_id")
            category_id = entry.get("category_id")
            bbox = entry.get("bbox")
            if not _coco_id(annotation_id) or annotation_id in annotation_ids:
                raise ValueError("COCO annotation ids must be non-negative integers and unique")
            if image_id not in self._image_ids:
                raise ValueError(f"annotation references an unknown image_id: {image_id!r}")
            if category_id not in self._category_labels:
                raise ValueError(f"annotation references an unknown category_id: {category_id!r}")
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(_number(value) for value in bbox):
                raise ValueError(f"annotations[{index}].bbox must be four finite numbers")
            x, y, width, height = bbox
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"annotations[{index}].bbox must have non-negative x/y and positive size")
            iscrowd = entry.get("iscrowd", 0)
            if not isinstance(iscrowd, int) or isinstance(iscrowd, bool) or iscrowd not in {0, 1}:
                raise ValueError(f"annotations[{index}].iscrowd must be 0 or 1")
            annotation_ids.add(annotation_id)
            annotations[image_id].append(
                {
                    "box": [float(x), float(y), float(x + width), float(y + height)],
                    "label": self._category_labels[category_id],
                    "area": float(width * height),
                    "iscrowd": iscrowd,
                }
            )
        return dict(annotations)

    @property
    def image_locations(self) -> set[str]:
        return {str(image["location"]) for image in self._images}

    @property
    def category_ids(self) -> dict[int, int]:
        """Map contiguous model labels back to original COCO category ids."""
        return dict(self._category_ids)

    def __len__(self) -> int:
        return len(self._images)

    def _local_image_path(self, location: str) -> Path:
        if not _is_s3(location):
            return _local_artifact_path(location, self.storage)
        suffix = Path(urlsplit(location).path).suffix or ".image"
        digest = hashlib.sha256(location.encode("utf-8")).hexdigest()
        destination = self.cache_directory / f"{digest}{suffix}"
        if not destination.is_file():
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            self.storage.download_file(location, destination)
        return destination

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_record = self._images[index]
        location = str(image_record["location"])
        try:
            with Image.open(self._local_image_path(location)) as source:
                image = source.convert("RGB")
                width, height = image.size
                pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                tensor = pixels.reshape(height, width, 3).permute(2, 0, 1).float().div(255)
        except (OSError, ValueError) as error:
            raise ValueError(f"image artifact is missing or corrupt: {location}") from error

        expected_width = image_record["width"]
        expected_height = image_record["height"]
        if expected_width is not None and expected_width != width:
            raise ValueError(f"COCO width does not match image artifact: {location}")
        if expected_height is not None and expected_height != height:
            raise ValueError(f"COCO height does not match image artifact: {location}")

        entries = self._annotations.get(image_record["id"], [])
        boxes = torch.tensor([entry["box"] for entry in entries], dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() and ((boxes[:, 2] > width).any() or (boxes[:, 3] > height).any()):
            raise ValueError(f"COCO bbox lies outside image bounds: {location}")
        target = {
            "boxes": boxes,
            "labels": torch.tensor([entry["label"] for entry in entries], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([entry["area"] for entry in entries], dtype=torch.float32),
            "iscrowd": torch.tensor([entry["iscrowd"] for entry in entries], dtype=torch.int64),
        }
        return tensor, target
