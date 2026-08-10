"""검증 세트 유사도 감사 도구 test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validation_similarity as audit  # noqa: E402


def _crop(value: int, size: int = 40) -> Image.Image:
    return Image.new("RGB", (size, size), color=(value, value, value))


def test_thumbnail_ignores_position_and_size_but_not_content():
    """같은 그림이 크기만 달라도 같게, 내용이 다르면 다르게 나와야 합니다."""

    small = audit.thumbnail(_crop(120, size=30))
    large = audit.thumbnail(_crop(120, size=90))
    different = audit.thumbnail(_crop(200, size=30))

    assert float(np.abs(small - large).mean()) < 1.0
    assert float(np.abs(small - different).mean()) > 50.0


def test_nearest_distance_only_compares_within_the_same_class():
    """다른 class와 가까운 것은 누수가 아닙니다. 같은 class 안에서만 봅니다."""

    train = [
        {"category_id": 1, "thumb": np.zeros(4, dtype=np.float32)},
        {"category_id": 2, "thumb": np.full(4, 10.0, dtype=np.float32)},
    ]
    validation = [{"category_id": 2, "thumb": np.full(4, 12.0, dtype=np.float32)}]

    distances = audit.nearest_same_class(validation, train)

    assert distances.tolist() == pytest.approx([2.0])


def test_a_class_missing_from_train_is_reported_not_silently_scored():
    """train에 없는 class는 거리를 잴 수 없습니다. 0으로 세면 누수처럼 보입니다."""

    train = [{"category_id": 1, "thumb": np.zeros(4, dtype=np.float32)}]
    validation = [{"category_id": 99, "thumb": np.zeros(4, dtype=np.float32)}]

    distances = audit.nearest_same_class(validation, train)

    assert np.isinf(distances).all()
    report = audit.build_report(distances, thresholds=(3.0,))
    assert report["comparable_crops"] == 0
    assert report["near_duplicate_ratio"]["3.0"] is None


def test_report_counts_only_comparable_crops():
    """비교할 수 없는 crop을 분모에 넣으면 비율이 낮아 보입니다."""

    distances = np.array([0.5, 2.0, 9.0, np.inf], dtype=np.float32)

    report = audit.build_report(distances, thresholds=(1.0, 3.0))

    assert report["validation_crops"] == 4
    assert report["comparable_crops"] == 3
    assert report["near_duplicate_ratio"]["1.0"] == pytest.approx(1 / 3)
    assert report["near_duplicate_ratio"]["3.0"] == pytest.approx(2 / 3)
    assert report["median_distance"] == pytest.approx(2.0)


def test_verdict_names_what_the_number_means():
    """숫자만 주면 아무도 읽지 않습니다. 믿어도 되는지까지 말해야 합니다."""

    leaky = audit.build_report(np.zeros(10, dtype=np.float32), thresholds=(3.0,))
    honest = audit.build_report(np.full(10, 40.0, dtype=np.float32), thresholds=(3.0,))

    assert "믿" in leaky["verdict"] or "구분" in leaky["verdict"]
    assert leaky["verdict"] != honest["verdict"]


def test_a_sampled_run_never_clears_the_validation_set():
    """표본은 train 비교 대상을 줄여 유사도를 실제보다 낮게 만듭니다.

    v4를 250장으로 재면 37%, 전수로 재면 79%가 나왔습니다. 표본 결과로 "써도 된다"고
    말하면 그 판정 자체가 틀립니다. 하한선이라고만 말해야 합니다.
    """

    distances = np.full(10, 40.0, dtype=np.float32)

    sampled = audit.build_report(distances, thresholds=(3.0,), sampled=True)
    full = audit.build_report(distances, thresholds=(3.0,), sampled=False)

    assert "하한" in sampled["verdict"]
    assert "쓸 수 있습니다" not in sampled["verdict"]
    assert "쓸 수 있습니다" in full["verdict"]
