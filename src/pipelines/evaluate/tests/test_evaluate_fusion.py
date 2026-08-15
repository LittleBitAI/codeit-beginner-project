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

    그렇게 세면 자기 자신과 동의해 확신도가 올라갑니다. 확신도를 서로 다르게 두어
    평균과 좌표까지 그 실행이 끌어당기지 않는지 함께 봅니다.
    """
    fused = fuse_predictions(
        [
            [
                prediction(1, 7, [0, 0, 10, 10], 0.8),
                prediction(1, 7, [1, 0, 10, 10], 0.2),
            ],
            [],
        ]
    )

    assert len(fused) == 1
    # 실행 하나가 찾았으므로 둘 중 하나입니다. 두 번 찾았다고 올라가지 않습니다.
    # 낮은 쪽이 평균을 끌어내리지도 않습니다 — 그 실행에서 가장 확신한 것만 셉니다.
    assert fused[0]["score"] == pytest.approx(0.4)
    # 좌표도 마찬가지로 그 실행의 최선만 반영합니다.
    assert fused[0]["bbox"][0] == pytest.approx(0.0)


def test_a_box_joins_the_cluster_it_overlaps_most():
    """임계치를 넘는 첫 cluster가 아니라 가장 많이 겹치는 cluster에 붙습니다.

    먼저 만들어졌다는 이유로 덜 가까운 자리에 묶이면 좌표와 확신도가 달라지고,
    상위 4개가 뒤바뀝니다.
    """
    # 확신도 순서상 [0,...]이 먼저 자리를 잡고 [6,...]이 두 번째 cluster가 됩니다.
    # 마지막 상자는 둘 다와 임계치를 넘지만 뒤쪽에 더 가깝습니다.
    fused = fuse_predictions(
        [
            [prediction(1, 7, [0, 0, 10, 10], 0.9)],
            [prediction(1, 7, [6, 0, 10, 10], 0.8)],
            [prediction(1, 7, [4, 0, 10, 10], 0.7)],
        ],
        iou_threshold=0.4,
    )

    assert len(fused) == 2
    # 마지막 상자가 뒤쪽 cluster에 붙었으므로 그쪽이 둘의 평균이 됩니다.
    joined = [item for item in fused if item["bbox"][0] > 3.0][0]
    assert joined["score"] == pytest.approx((0.8 + 0.7) / 2 * 2 / 3)


def test_fused_predictions_keep_the_fields_a_submission_needs():
    """제출 CSV가 쓰는 field를 그대로 갖고 나옵니다.

    `image_key`는 정규화한 값이라 CSV에 쓸 수 없습니다. manifest의 `image_id`가
    없으면 합친 결과로 제출을 만들 수 없습니다.
    """
    fused = fuse_predictions([[prediction(1, 7, [0, 0, 10, 10], 0.8)]])

    assert set(fused[0]) >= {"image_id", "category_id", "bbox", "score"}
    assert fused[0]["image_id"] == 1
