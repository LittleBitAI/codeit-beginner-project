"""Competition detection 결과를 고정된 submission CSV 형식으로 만듭니다."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence, Set
from typing import Any

from .errors import PredictionError


SUBMISSION_HEADER = (
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
)


def _sort_key(
    indexed_prediction: tuple[int, Mapping[str, Any]],
    *,
    category_ids: Set[int],
) -> tuple[Any, ...]:
    """Competition row를 재현 가능한 순서로 정렬할 key를 만듭니다."""
    index, prediction = indexed_prediction
    image_id = prediction.get("image_id")
    if not isinstance(image_id, int) or isinstance(image_id, bool):
        raise PredictionError(f"submission image_id는 정수여야 합니다: {image_id!r}")
    category_id = prediction.get("category_id")
    if category_id not in category_ids:
        raise PredictionError(
            "test manifest categories에 없는 category_id를 model이 반환했습니다: "
            f"{category_id!r}"
        )
    bbox = prediction.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise PredictionError("submission bbox는 [x, y, width, height] 형식이어야 합니다.")
    try:
        score = float(prediction["score"])
        box_key = tuple(float(value) for value in bbox)
    except (KeyError, TypeError, ValueError) as error:
        raise PredictionError("submission bbox와 score는 숫자여야 합니다.") from error
    return (image_id, -score, int(category_id), *box_key, index)


def render_submission_csv(
    predictions: Sequence[Mapping[str, Any]],
    *,
    category_ids: Set[int],
) -> str:
    """예측을 header와 결정적 순서를 가진 UTF-8 CSV text로 바꿉니다."""
    indexed = list(enumerate(predictions))
    indexed.sort(key=lambda item: _sort_key(item, category_ids=category_ids))

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(SUBMISSION_HEADER)
    for annotation_id, (_, prediction) in enumerate(indexed, start=1):
        bbox = prediction["bbox"]
        writer.writerow(
            (
                annotation_id,
                prediction["image_id"],
                prediction["category_id"],
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                prediction["score"],
            )
        )
    return output.getvalue()


__all__ = ["SUBMISSION_HEADER", "render_submission_csv"]
