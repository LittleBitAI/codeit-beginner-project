"""이미지 단위로 train/validation을 결정적으로 나눕니다.

한 이미지에 여러 category가 함께 있을 수 있으므로(다중 라벨) 단순 무작위 분할은
희귀 category를 한쪽 split에서 통째로 빠뜨릴 수 있습니다. 그래서 희귀 category부터
validation에 배치해 **모든 category가 train과 validation 양쪽에 반드시 나타나게**
한 뒤, 남은 자리를 목표 비율에 가장 가깝게 채웁니다.

같은 이미지 목록, 같은 seed, 같은 비율이면 언제나 같은 결과가 나옵니다. 무작위는
seed를 고정한 `random.Random`의 순서 섞기 한 번에만 쓰이고, 나머지 선택은 모두
결정적인 점수 비교입니다.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import DatasetPreparationError


__all__ = ["SplitResult", "split_images"]


@dataclass(frozen=True)
class SplitResult:
    """분할 결과와 category별 이미지 수입니다."""

    train_image_ids: set[int]
    validation_image_ids: set[int]
    train_category_counts: dict[int, int]
    validation_category_counts: dict[int, int]


def split_images(
    images: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> SplitResult:
    """이미지를 train과 validation으로 나눕니다."""

    categories_by_image: defaultdict[int, set[int]] = defaultdict(set)
    for annotation in annotations:
        categories_by_image[annotation["image_id"]].add(annotation["category_id"])

    image_ids = [image["id"] for image in images]
    if len(image_ids) < 2:
        raise DatasetPreparationError(
            "train과 validation을 나누려면 이미지가 2장 이상 필요합니다."
        )
    if any(not categories_by_image[image_id] for image_id in image_ids):
        raise DatasetPreparationError(
            "분할 대상 이미지에는 annotation이 하나 이상 있어야 합니다."
        )

    category_totals = Counter(
        category_id
        for image_id in image_ids
        for category_id in categories_by_image[image_id]
    )
    scarce = sorted(
        category_id for category_id, total in category_totals.items() if total < 2
    )
    if scarce:
        raise DatasetPreparationError(
            "모든 category가 train과 validation 양쪽에 나타나려면 category마다 "
            "이미지가 2장 이상 필요합니다. 이미지가 1장뿐인 category id: "
            + ", ".join(str(category_id) for category_id in scarce)
        )

    target_size = min(max(1, round(len(image_ids) * validation_ratio)), len(image_ids) - 1)
    # 반올림한 category별 목표치의 합은 validation 이미지 수보다 커질 수 있으므로,
    # 정확한 목표가 아니라 채우기 단계의 분포 목표로만 사용합니다.
    target_by_category = {
        category_id: min(max(1, round(total * validation_ratio)), total - 1)
        for category_id, total in category_totals.items()
    }

    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    tie_rank = {image_id: rank for rank, image_id in enumerate(shuffled)}
    validation: set[int] = set()
    validation_counts: Counter[int] = Counter()

    def keeps_train_coverage(image_id: int) -> bool:
        """이 이미지를 validation에 넣어도 train에 category가 남는지 봅니다."""

        return all(
            validation_counts[category_id] + 1 <= category_totals[category_id] - 1
            for category_id in categories_by_image[image_id]
        )

    def uncovered() -> set[int]:
        return {
            category_id
            for category_id in category_totals
            if validation_counts[category_id] < 1
        }

    while uncovered():
        if len(validation) >= target_size:
            raise DatasetPreparationError(
                "선택한 validation 비율로는 모든 category를 validation에 넣을 수 "
                f"없습니다. validation 이미지 {target_size}장, category "
                f"{len(category_totals)}개"
            )
        remaining = uncovered()
        rarest = min(remaining, key=lambda value: (category_totals[value], value))
        candidates = [
            image_id
            for image_id in image_ids
            if image_id not in validation
            and rarest in categories_by_image[image_id]
            and keeps_train_coverage(image_id)
        ]
        if not candidates:
            raise DatasetPreparationError(
                f"category {rarest}를 train과 validation 양쪽에 넣을 수 없습니다."
            )

        def coverage_score(image_id: int) -> tuple[float, float, float, int]:
            categories = categories_by_image[image_id]
            gain = sum(1 for value in categories if value in remaining)
            scarcity = sum(1.0 / category_totals[value] for value in categories)
            overshoot = sum(
                max(0, validation_counts[value] + 1 - target_by_category[value])
                for value in categories
            )
            return (float(gain), scarcity, -float(overshoot), -tie_rank[image_id])

        chosen = max(candidates, key=coverage_score)
        validation.add(chosen)
        validation_counts.update(categories_by_image[chosen])

    while len(validation) < target_size:
        candidates = [
            image_id
            for image_id in image_ids
            if image_id not in validation and keeps_train_coverage(image_id)
        ]
        if not candidates:
            raise DatasetPreparationError(
                "train에서 category를 비우지 않고는 validation을 목표 크기까지 "
                "채울 수 없습니다."
            )

        def distribution_error(image_id: int) -> tuple[float, int]:
            simulated = {
                category_id: validation_counts[category_id]
                + (1 if category_id in categories_by_image[image_id] else 0)
                for category_id in category_totals
            }
            error = sum(
                (
                    (simulated[category_id] - target_by_category[category_id])
                    / target_by_category[category_id]
                )
                ** 2
                for category_id in category_totals
            )
            return (error, tie_rank[image_id])

        chosen = min(candidates, key=distribution_error)
        validation.add(chosen)
        validation_counts.update(categories_by_image[chosen])

    train = set(image_ids) - validation
    train_counts = Counter(
        category_id
        for image_id in train
        for category_id in categories_by_image[image_id]
    )
    if not train or not validation:
        raise DatasetPreparationError("train과 validation split은 모두 비어 있으면 안 됩니다.")
    for category_id, total in category_totals.items():
        if train_counts[category_id] < 1 or validation_counts[category_id] < 1:
            raise DatasetPreparationError(
                f"category {category_id}가 한쪽 split에만 있습니다."
            )
        if train_counts[category_id] + validation_counts[category_id] != total:
            raise DatasetPreparationError(
                f"category {category_id}의 이미지 수 계산이 맞지 않습니다."
            )
    return SplitResult(
        train_image_ids=train,
        validation_image_ids=validation,
        train_category_counts=dict(train_counts),
        validation_category_counts=dict(validation_counts),
    )
