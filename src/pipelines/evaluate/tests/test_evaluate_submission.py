"""Competition submission CSV 생성 규칙 test입니다."""

from __future__ import annotations

import csv
import io

import pytest

from src.pipelines.evaluate.errors import PredictionError
from src.pipelines.evaluate.submission import render_submission_csv


def test_submission_rows_are_deterministic_and_keep_original_values():
    predictions = [
        {
            "image_id": 20,
            "image_key": "20",
            "category_id": 7,
            "bbox": [0.123456789012345, 2.0, 3.0, 4.0],
            "score": 0.9,
        },
        {
            "image_id": 10,
            "image_key": "10",
            "category_id": 7,
            "bbox": [2.0, 2.0, 3.0, 4.0],
            "score": 0.8,
        },
        {
            "image_id": 10,
            "image_key": "10",
            "category_id": 3,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "score": 0.8,
        },
        {
            "image_id": 10,
            "image_key": "10",
            "category_id": 7,
            "bbox": [3.0, 2.0, 3.0, 4.0],
            "score": 0.95,
        },
    ]

    text = render_submission_csv(predictions, category_ids=frozenset({3, 7}))
    rows = list(csv.reader(io.StringIO(text)))

    assert rows[0] == [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    assert [(row[1], row[2], row[7]) for row in rows[1:]] == [
        ("10", "7", "0.95"),
        ("10", "3", "0.8"),
        ("10", "7", "0.8"),
        ("20", "7", "0.9"),
    ]
    assert [row[0] for row in rows[1:]] == ["1", "2", "3", "4"]
    assert rows[-1][3] == "0.123456789012345"
    assert "\r\n" not in text


def test_zero_detections_produce_header_only_csv():
    text = render_submission_csv([], category_ids=frozenset({3, 7}))

    assert text == (
        "annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score\n"
    )


def test_submission_rejects_category_not_declared_by_test_manifest():
    prediction = {
        "image_id": 10,
        "image_key": "10",
        "category_id": 99,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "score": 0.8,
    }

    with pytest.raises(PredictionError, match="test manifest categories에 없는 category_id"):
        render_submission_csv([prediction], category_ids=frozenset({3, 7}))
