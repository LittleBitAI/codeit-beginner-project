"""Validation이 train과 얼마나 비슷한지 재어 summary에 남길 값을 만듭니다.

조합 단위로 잘 나눠도, 같은 알약의 겉모습이 양쪽에 나타나면 model은 validation에서
새로운 것을 보지 않습니다. 그런 검증 세트의 점수는 높게 나오지만 대회 점수와
무관합니다. 그 사실을 dataset 자신이 들고 다니게 하려고 잽니다.

이미지를 전부 열어야 하므로 기본값은 꺼짐입니다. 켠 실행만 이 비용을 냅니다.
"""

from __future__ import annotations

import posixpath
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.common import StorageError

from .errors import DatasetPreparationError


#: crop을 이 크기로 맞춰 비교합니다. 알약이 놓인 위치와 크기가 달라도 같은 그림인지
#: 보려는 것이라 원본 해상도로 비교하면 안 됩니다.
THUMBNAIL_SIZE = 32

#: 0~255 척도의 평균 픽셀 차이입니다. 이보다 가까우면 눈으로 구분되지 않습니다.
NEAR_DUPLICATE_DISTANCE = 3.0


def _thumbnail(image: Image.Image, bbox: Sequence[float]) -> np.ndarray | None:
    """annotation 한 개의 자리를 잘라 비교용 벡터로 만듭니다."""

    x, y, width, height = (int(round(float(value))) for value in bbox)
    if width <= 0 or height <= 0:
        return None
    patch = image.crop((x, y, x + width, y + height)).convert("RGB")
    resized = patch.resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32).reshape(-1)


def _collect(
    storage: Any,
    images: Sequence[Mapping[str, Any]],
    annotations_by_image: Mapping[Any, list[Mapping[str, Any]]],
    keep: set[Any],
    on_progress: Callable[[int, int], None] | None,
) -> list[tuple[int, np.ndarray]]:
    """split 한쪽의 crop을 모읍니다. 이미지는 받는 즉시 지웁니다.

    manifest의 `file_name`이 아니라 **원본 storage 위치**를 씁니다. manifest 값은
    manifest directory 기준 상대 경로(`../../raw/...`)라 LocalStorage에 그대로
    넘기면 storage root 밖이라며 거부됩니다.
    """

    chosen = [image for image in images if image["id"] in keep]
    rows: list[tuple[int, np.ndarray]] = []
    for done, image in enumerate(chosen, start=1):
        location = str(image["file_name"])
        annotations = annotations_by_image.get(image["id"])
        if annotations:
            with tempfile.TemporaryDirectory(prefix="similarity-") as scratch:
                local = Path(scratch) / posixpath.basename(location.replace("\\", "/"))
                try:
                    storage.download_file(location, local)
                    with Image.open(local) as opened:
                        opened.load()
                        for annotation in annotations:
                            vector = _thumbnail(opened, annotation["bbox"])
                            if vector is not None:
                                rows.append((int(annotation["category_id"]), vector))
                except (StorageError, OSError, UnidentifiedImageError) as error:
                    raise DatasetPreparationError(
                        f"유사도를 재려고 이미지를 여는 데 실패했습니다: {location}"
                    ) from error
        if on_progress is not None:
            on_progress(done, len(chosen))
    return rows


def measure_validation_similarity(
    storage: Any,
    images: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    train_image_ids: set[Any],
    validation_image_ids: set[Any],
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any] | None:
    """validation crop마다 같은 class의 가장 비슷한 train crop까지의 거리를 잽니다.

    다른 class와 비슷한 것은 그냥 닮은 알약이라 문제가 아닙니다. 같은 class 안에서만
    봅니다. train에 그 class가 없으면 잴 수 없으므로 세지 않습니다.

    잴 수 있는 crop이 하나도 없으면 ``None``을 돌려줍니다. 0으로 적으면 "완전히
    똑같다"로 읽혀 사실과 정반대가 됩니다.
    """

    annotations_by_image: defaultdict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    def report(stage: str) -> Callable[[int, int], None] | None:
        if on_progress is None:
            return None
        return lambda done, total: on_progress(stage, done, total)

    train = _collect(
        storage, images, annotations_by_image, train_image_ids, report("similarity_train")
    )
    validation = _collect(
        storage,
        images,
        annotations_by_image,
        validation_image_ids,
        report("similarity_validation"),
    )
    if not train or not validation:
        return None

    banks: dict[int, np.ndarray] = {}
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for category_id, vector in train:
        grouped[category_id].append(vector)
    for category_id, vectors in grouped.items():
        banks[category_id] = np.stack(vectors)

    distances: list[float] = []
    for category_id, vector in validation:
        bank = banks.get(category_id)
        if bank is None:
            continue
        distances.append(float(np.abs(bank - vector).mean(axis=1).min()))
    if not distances:
        return None

    values = np.asarray(distances, dtype=np.float64)
    return {
        "measured_crops": int(values.size),
        "median_distance": round(float(np.median(values)), 4),
        "near_duplicate_distance": NEAR_DUPLICATE_DISTANCE,
        "near_duplicate_ratio_3": round(
            float((values < NEAR_DUPLICATE_DISTANCE).mean()), 4
        ),
    }
