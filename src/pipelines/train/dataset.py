"""COCO manifest loading for the train pipeline."""

from __future__ import annotations

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
from torchvision.transforms import functional as transform_functional

from src.common import LocalStorage, Storage

from .image_cache import ImageCacheSession


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_INTEGER_TEXT = re.compile(r"[+-]?\d+")
# 알약마다 하나씩 있는 값입니다. 알약을 빼면 이 값도 같이 빼야 알약과 이름이
# 어긋나지 않습니다. `image_id`는 이미지마다 하나라 여기 없습니다.
PER_OBJECT_TARGET_KEYS = ("boxes", "labels", "area", "iscrowd")


def _rotate_boxes(
    boxes: torch.Tensor, turns: int, *, height: int, width: int
) -> torch.Tensor:
    """``torch.rot90``이 이미지를 돌린 것과 같은 방향으로 bbox를 돌립니다.

    좌표는 픽셀 번호가 아니라 경계값이라 ``width - x``로 뒤집습니다. 뒤집기가 쓰는
    규칙과 같습니다.
    """

    left, top, right, bottom = boxes.unbind(dim=-1)
    if turns == 1:
        return torch.stack((top, width - right, bottom, width - left), dim=-1)
    if turns == 2:
        return torch.stack(
            (width - right, height - bottom, width - left, height - top), dim=-1
        )
    return torch.stack((height - bottom, left, height - top, right), dim=-1)


class DetectionAugmentation:
    """Tensor image와 detection bbox에 같은 무작위 변환을 적용합니다."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)

    def __call__(
        self,
        image: torch.Tensor,
        target: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.settings.get("preset") == "none":
            return image, {name: value.clone() for name, value in target.items()}

        augmented = image
        copied = {name: value.clone() for name, value in target.items()}
        augmented, copied = self._quarter_turn(augmented, copied)
        augmented, copied = self._flip(augmented, copied)
        augmented, copied = self._crop(augmented, copied)
        if self._happens("color_probability"):
            augmented = self._color(augmented)
        if self._happens("noise_probability"):
            augmented = self._noise(augmented)
        return augmented, copied

    def _amount(self, name: str) -> float:
        """preset에 없는 설정은 0으로 읽습니다.

        version 1 preset에는 version 2가 더한 key가 없습니다. 0으로 읽어 두면 옛
        preset을 고치지 않아도 되고, 이미 남긴 checkpoint의 기록도 그대로입니다.
        """

        return float(self.settings.get(name, 0.0))

    def _happens(self, name: str) -> bool:
        """그 변환을 할지 정합니다. 확률이 0이면 무작위 수를 뽑지도 않습니다.

        뽑지 않아야 `pill_basic`이 이 preset이 생기기 전과 똑같은 무작위 수열로
        돕니다. 그러지 않으면 같은 seed로 다시 돌린 옛 실험이 재현되지 않습니다.
        """

        probability = self._amount(name)
        return probability > 0.0 and torch.rand(()).item() < probability

    def _quarter_turn(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """이미지를 90°의 배수로 돌립니다.

        90°의 배수는 픽셀을 옮겨 놓기만 해서 화질도 bbox도 그대로입니다. 임의 각도
        회전은 쓰지 않습니다. 우리 annotation은 축에 나란한 bbox뿐이라, 비스듬히
        돌린 알약에 다시 bbox를 씌우면 안이 배경으로 가득 차고, 그렇게 헐거워진
        bbox는 IoU 0.75~0.95로 매기는 이 대회 점수를 그대로 깎습니다.
        """

        if not self._happens("quarter_turn_probability"):
            return image, target
        # 1, 2, 3 중 하나입니다. gate 확률 0.75와 합치면 네 방향이 모두 25%입니다.
        turns = min(3, 1 + int(torch.rand(()).item() * 3))
        height, width = image.shape[-2:]
        boxes = target["boxes"]
        if boxes.numel():
            target["boxes"] = _rotate_boxes(boxes, turns, height=height, width=width)
        return torch.rot90(image, turns, dims=(-2, -1)), target

    def _flip(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boxes = target["boxes"]
        height, width = image.shape[-2:]
        if self._happens("horizontal_flip_probability"):
            image = torch.flip(image, dims=(-1,))
            if boxes.numel():
                left = width - boxes[:, 2].clone()
                right = width - boxes[:, 0].clone()
                boxes[:, 0], boxes[:, 2] = left, right
        if self._happens("vertical_flip_probability"):
            image = torch.flip(image, dims=(-2,))
            if boxes.numel():
                top = height - boxes[:, 3].clone()
                bottom = height - boxes[:, 1].clone()
                boxes[:, 1], boxes[:, 3] = top, bottom
        return image, target

    def _crop(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """이미지 가장자리를 잘라 냅니다.

        EDA가 잰 대회 test와 학습 데이터의 알약 크기 비율은 0.99라 확대가 필요해서가
        아니라, 늘 같은 자리에 같은 크기로 놓인 사진을 통째로 외우지 못하게 하려고
        씁니다. 가로세로 비율은 그대로 둡니다. 알약의 생김새 자체가 class 단서라
        찌그러뜨리면 안 됩니다.

        잘린 알약은 남기지 않고 통째로 버립니다. 알약을 가리는 것은 표면의 각인인데,
        가장자리가 잘리면 각인도 함께 잘려 label과 맞지 않는 그림이 됩니다. 남는
        알약이 하나도 없으면 아예 자르지 않습니다.
        """

        if not self._happens("crop_probability"):
            return image, target
        minimum = self._amount("crop_minimum_scale")
        if not 0.0 < minimum < 1.0:
            return image, target
        height, width = image.shape[-2:]
        scale = minimum + (1.0 - minimum) * torch.rand(()).item()
        crop_height = max(1, min(height, round(height * scale)))
        crop_width = max(1, min(width, round(width * scale)))
        # 뽑은 값이 1.0이면 시작점이 한 칸 넘어가 crop이 그만큼 짧아집니다. 실제
        # `torch.rand`는 1.0을 주지 않지만 test는 줍니다.
        top = min(height - crop_height, int(torch.rand(()).item() * (height - crop_height + 1)))
        left = min(width - crop_width, int(torch.rand(()).item() * (width - crop_width + 1)))
        boxes = target["boxes"]
        kept = (
            (boxes[:, 0] >= left)
            & (boxes[:, 1] >= top)
            & (boxes[:, 2] <= left + crop_width)
            & (boxes[:, 3] <= top + crop_height)
        )
        if not bool(kept.any()):
            return image, target
        cropped = dict(target)
        for name in PER_OBJECT_TARGET_KEYS:
            value = target.get(name)
            if isinstance(value, torch.Tensor) and value.shape[:1] == boxes.shape[:1]:
                cropped[name] = value[kept]
        cropped["boxes"] = boxes[kept] - boxes.new_tensor([left, top, left, top])
        return image[..., top : top + crop_height, left : left + crop_width], cropped

    def _noise(self, image: torch.Tensor) -> torch.Tensor:
        """센서 잡음 정도의 약한 잡음을 더합니다.

        같은 조합을 세 각도로만 찍은 데이터라 한 장면을 그대로 외우기 쉽습니다.
        더한 뒤에는 값을 [0, 1]로 되돌립니다. 뒤따르는 정규화가 그 범위를 전제합니다.
        """

        sigma = self._amount("noise_sigma")
        if sigma <= 0.0:
            return image
        return (image + torch.randn_like(image) * sigma).clamp(0.0, 1.0)

    def _factor(self, amount: float) -> float:
        return 1.0 + (torch.rand(()).item() * 2.0 - 1.0) * amount

    def _color(self, image: torch.Tensor) -> torch.Tensor:
        image = transform_functional.adjust_brightness(
            image, self._factor(self.settings["brightness"])
        )
        image = transform_functional.adjust_contrast(
            image, self._factor(self.settings["contrast"])
        )
        image = transform_functional.adjust_saturation(
            image, self._factor(self.settings["saturation"])
        )
        hue = (torch.rand(()).item() * 2.0 - 1.0) * self.settings["hue"]
        return transform_functional.adjust_hue(image, hue)


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
        image_cache: ImageCacheSession,
        augmentation: Mapping[str, Any] | None = None,
    ) -> None:
        self.manifest_uri = manifest_uri
        self.class_map = dict(class_map)
        self.storage = storage
        self.image_cache = image_cache
        self.augmentation = DetectionAugmentation(augmentation) if augmentation else None
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
        return self.image_cache.fetch(location, self.storage)

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
        if self.augmentation is not None:
            return self.augmentation(tensor, target)
        return tensor, target
