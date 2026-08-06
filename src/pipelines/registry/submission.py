"""제출 CSV가 competition 계약을 지키는지 검사하는 registry 내부 module입니다.

규격은 `contracts/proposals/001-competition-artifact-contract.md`가 정한 것을
그대로 따르며, 여기서 새 규칙을 만들지 않습니다. CSV를 만드는 쪽은 evaluate
pipeline이고 registry는 읽기만 합니다. 파일을 고치거나 지우지 않습니다.

이 module은 registry pipeline 안에서만 사용합니다.
"""

from __future__ import annotations

import codecs
import csv
import io
import math
from pathlib import Path
from typing import Any

from .record import CorruptedArtifactError, InvalidSubmissionError


# 계약이 정한 header입니다. 순서까지 정확히 같아야 합니다.
SUBMISSION_HEADER: tuple[str, ...] = (
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
)

# 과제가 image 한 장당 알약 4개까지 예측하도록 정해 두었습니다.
MAX_DETECTIONS_PER_IMAGE = 4

_INTEGER_COLUMNS = ("annotation_id", "image_id", "category_id")
_FLOAT_COLUMNS = ("bbox_x", "bbox_y", "bbox_w", "bbox_h", "score")


def _parse_integer(value: str, *, column: str, line: int) -> int:
    """정수 column 하나를 읽습니다. `1.0` 같은 실수 표기도 거부합니다."""

    try:
        return int(value.strip())
    except ValueError as error:
        raise InvalidSubmissionError(
            f"submission CSV {line}번째 줄의 {column}은(는) 정수여야 하는데 "
            f"{value!r}을(를) 받았습니다."
        ) from error


def _parse_number(value: str, *, column: str, line: int) -> float:
    """실수 column 하나를 읽습니다. NaN과 Infinity는 표준 JSON이 아니라 거부합니다."""

    try:
        number = float(value.strip())
    except ValueError as error:
        raise InvalidSubmissionError(
            f"submission CSV {line}번째 줄의 {column}은(는) 숫자여야 하는데 "
            f"{value!r}을(를) 받았습니다."
        ) from error
    if not math.isfinite(number):
        raise InvalidSubmissionError(
            f"submission CSV {line}번째 줄의 {column}에 NaN 또는 Infinity가 있습니다."
        )
    return number


def _read_rows(path: Path) -> list[list[str]]:
    """CSV를 BOM 없는 UTF-8로 읽어 행 목록으로 돌려줍니다."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CorruptedArtifactError(
            f"submission CSV를 읽지 못했습니다: {path.name}"
        ) from error

    if raw.startswith(codecs.BOM_UTF8):
        raise InvalidSubmissionError(
            "submission CSV는 BOM 없는 UTF-8이어야 합니다."
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidSubmissionError(
            "submission CSV를 UTF-8로 읽지 못했습니다."
        ) from error

    return list(csv.reader(io.StringIO(text, newline="")))


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """계약이 정한 행 정렬 순서를 tuple 하나로 만듭니다.

    image_id 오름차순, 같은 image 안에서는 score 내림차순, 이어서 category_id와
    bbox 좌표 오름차순입니다.
    """

    return (
        row["image_id"],
        -row["score"],
        row["category_id"],
        row["bbox_x"],
        row["bbox_y"],
        row["bbox_w"],
        row["bbox_h"],
    )


def check_submission_csv(path: Path) -> dict[str, Any]:
    """제출 CSV가 계약을 지키는지 검사하고 요약 수치를 돌려줍니다.

    계약에 적힌 항목만 확인합니다. category_id가 test manifest에 있는지와 score
    threshold 적용 여부는 evaluate의 책임이라 여기서 다시 판정하지 않습니다.
    """

    rows = _read_rows(path)
    if not rows:
        raise InvalidSubmissionError("submission CSV에 header가 없습니다.")

    header = tuple(field.strip() for field in rows[0])
    if header != SUBMISSION_HEADER:
        raise InvalidSubmissionError(
            "submission CSV의 header가 계약과 다릅니다. "
            f"기대: {','.join(SUBMISSION_HEADER)} / 실제: {','.join(header)}"
        )

    detections_per_image: dict[int, int] = {}
    previous_key: tuple[Any, ...] | None = None

    for index, raw_row in enumerate(rows[1:], start=1):
        line = index + 1  # header가 1번째 줄입니다.
        if len(raw_row) != len(SUBMISSION_HEADER):
            raise InvalidSubmissionError(
                f"submission CSV {line}번째 줄의 field 개수가 "
                f"{len(SUBMISSION_HEADER)}개가 아니라 {len(raw_row)}개입니다."
            )

        row = dict(zip(SUBMISSION_HEADER, raw_row))
        parsed: dict[str, Any] = {
            column: _parse_integer(row[column], column=column, line=line)
            for column in _INTEGER_COLUMNS
        }
        parsed.update(
            {
                column: _parse_number(row[column], column=column, line=line)
                for column in _FLOAT_COLUMNS
            }
        )

        if parsed["annotation_id"] != index:
            raise InvalidSubmissionError(
                f"submission CSV의 annotation_id는 최종 정렬 순서대로 1..N이어야 "
                f"합니다. {line}번째 줄에서 {index}을(를) 기대했지만 "
                f"{parsed['annotation_id']}을(를) 받았습니다."
            )

        image_id = parsed["image_id"]
        detections_per_image[image_id] = detections_per_image.get(image_id, 0) + 1
        if detections_per_image[image_id] > MAX_DETECTIONS_PER_IMAGE:
            raise InvalidSubmissionError(
                f"image_id={image_id}의 검출이 image당 상한 "
                f"{MAX_DETECTIONS_PER_IMAGE}개를 넘었습니다 ({line}번째 줄)."
            )

        current_key = _sort_key(parsed)
        if previous_key is not None and current_key < previous_key:
            raise InvalidSubmissionError(
                f"submission CSV {line}번째 줄의 정렬이 계약과 다릅니다. "
                "image_id 오름차순, score 내림차순, category_id와 bbox 좌표 "
                "오름차순이어야 합니다."
            )
        previous_key = current_key

    return {
        "row_count": len(rows) - 1,
        "image_count": len(detections_per_image),
        "max_detections_per_image": max(detections_per_image.values(), default=0),
    }
