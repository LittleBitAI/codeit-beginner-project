"""여러 실행의 예측을 하나로 합치는 규칙 test입니다."""

from __future__ import annotations

import pytest

from src.pipelines.evaluate.fusion import fuse_predictions


def prediction(
    image_id: int,
    category_id: int,
    bbox: list[float],
    score: float,
) -> dict:
    return {
        "image_id": image_id,
        "image_key": str(image_id),
        "category_id": category_id,
        "bbox": list(bbox),
        "score": score,
    }


def test_boxes_that_agree_become_one_with_the_average_score():
    """두 실행이 같은 자리를 가리키면 하나로 합칩니다."""
    fused = fuse_predictions(
        [
            [prediction(1, 7, [0, 0, 10, 10], 0.8)],
            [prediction(1, 7, [0, 0, 10, 10], 0.6)],
        ]
    )

    assert len(fused) == 1
    assert fused[0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
    # 둘 다 찾았으므로 평균 그대로입니다.
    assert fused[0]["score"] == pytest.approx(0.7)


def test_a_box_only_one_run_found_is_less_certain():
    """혼자 찾은 상자는 확신도를 낮춥니다.

    이것이 융합의 핵심입니다. 한 실행이 외운 것을 그대로 믿으면 합치는 뜻이 없습니다.
    """
    fused = fuse_predictions(
        [
            [prediction(1, 7, [0, 0, 10, 10], 0.8)],
            [],
        ]
    )

    assert len(fused) == 1
    # 둘 중 하나만 찾았으므로 절반입니다.
    assert fused[0]["score"] == pytest.approx(0.4)


def test_the_fused_box_leans_towards_the_more_certain_run():
    """상자 좌표는 확신도로 가중 평균합니다."""
    fused = fuse_predictions(
        [
            [prediction(1, 7, [0, 0, 10, 10], 0.9)],
            [prediction(1, 7, [2, 0, 10, 10], 0.1)],
        ]
    )

    assert len(fused) == 1
    # (0.9 * 0 + 0.1 * 2) / 1.0
    assert fused[0]["bbox"][0] == pytest.approx(0.2)


def test_boxes_that_do_not_overlap_enough_stay_apart():
    fused = fuse_predictions(
        [
            [prediction(1, 7, [0, 0, 10, 10], 0.8)],
            [prediction(1, 7, [40, 40, 10, 10], 0.8)],
        ]
    )

    assert len(fused) == 2


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(prediction(1, 7, [0, 0, 10, 10], 0.8),
                     prediction(1, 3, [0, 0, 10, 10], 0.8), id="다른-class"),
        pytest.param(prediction(1, 7, [0, 0, 10, 10], 0.8),
                     prediction(2, 7, [0, 0, 10, 10], 0.8), id="다른-이미지"),
    ],
)
def test_only_the_same_class_in_the_same_image_is_fused(first: dict, second: dict):
    """겹쳐 보여도 class나 이미지가 다르면 다른 것입니다."""
    fused = fuse_predictions([[first], [second]])

    assert len(fused) == 2


def test_one_run_finding_it_twice_does_not_count_as_agreement():
    """한 실행이 두 번 찾은 것을 둘이 동의한 것으로 세지 않습니다.

    그렇게 세면 자기 자신과 동의해 확신도가 올라갑니다.
    """
    fused = fuse_predictions(
        [
            [
                prediction(1, 7, [0, 0, 10, 10], 0.8),
                prediction(1, 7, [1, 0, 10, 10], 0.8),
            ],
            [],
        ]
    )

    assert len(fused) == 1
    # 실행 하나가 찾았으므로 둘 중 하나입니다. 두 번 찾았다고 올라가지 않습니다.
    assert fused[0]["score"] == pytest.approx(0.4)


def test_fused_predictions_keep_the_fields_a_submission_needs():
    """제출 CSV가 쓰는 field를 그대로 갖고 나옵니다.

    `image_key`는 정규화한 값이라 CSV에 쓸 수 없습니다. manifest의 `image_id`가
    없으면 합친 결과로 제출을 만들 수 없습니다.
    """
    fused = fuse_predictions([[prediction(1, 7, [0, 0, 10, 10], 0.8)]])

    assert set(fused[0]) >= {"image_id", "category_id", "bbox", "score"}
    assert fused[0]["image_id"] == 1
