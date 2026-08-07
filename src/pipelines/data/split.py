"""그룹 단위로 train/validation을 결정적으로 나눕니다.

같은 알약 조합을 각도와 조명만 바꿔 여러 장 찍은 원본에서는, 이미지 한 장씩
나누면 거의 같은 사진이 train과 validation 양쪽에 들어가 validation 점수가 실제
성능보다 좋게 나옵니다. 그래서 **file 이름 접두사가 같은 이미지를 한 그룹으로
묶어 통째로 한쪽 split에만** 넣습니다. 그룹 규칙(`GroupRule`)을 주지 않으면
이미지 한 장이 그대로 한 그룹이 되어 예전과 같은 결과가 나옵니다.

한 이미지에 여러 category가 함께 있을 수 있으므로(다중 라벨) 단순 무작위 분할은
희귀 category를 한쪽 split에서 통째로 빠뜨릴 수 있습니다. 그래서 희귀 category부터
validation에 배치해 **모든 category가 train과 validation 양쪽에 반드시 나타나게**
한 뒤, 남은 자리를 목표 비율에 가장 가깝게 채웁니다. 그룹은 통째로 움직이므로
validation 이미지 수가 목표에 정확히 맞지 않을 수 있고, 그때는 목표에 가장 가까운
지점에서 멈춥니다.

같은 이미지 목록, 같은 seed, 같은 비율, 같은 그룹 규칙이면 언제나 같은 결과가
나옵니다. 무작위는 seed를 고정한 `random.Random`의 순서 섞기 한 번에만 쓰이고,
나머지 선택은 모두 결정적인 점수 비교입니다.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .errors import DatasetPreparationError


__all__ = ["GroupRule", "SplitResult", "split_images"]


@dataclass(frozen=True)
class GroupRule:
    """file 이름 접두사로 그룹을 정하는 규칙입니다.

    `delimiter`로 file 이름(확장자 제외)을 자른 뒤 앞에서 `tokens`개를 다시 이어
    붙인 값이 그룹 이름입니다. 예를 들어 `delimiter="_"`, `tokens=1`이면
    `K-001900-010224_0_2_70.png`의 그룹은 `K-001900-010224`입니다. 구분자가 없는
    이름은 잘리지 않으므로 그 이미지 혼자 한 그룹이 됩니다.
    """

    delimiter: str
    tokens: int

    def key(self, file_name: str) -> str:
        """이미지 위치에서 그룹 이름을 뽑습니다.

        구분자가 없거나 앞부분이 비어 있는 이름은 규칙에 맞지 않는 이름입니다.
        그런 file은 확장자를 포함한 위치 전체를 별도 namespace에 넣어 그 file
        하나만의 그룹이 되게 합니다. 정상 접두사와 fallback 이름이 우연히 같거나
        stem이 같고 확장자만 달라도 서로 합쳐지지 않습니다.
        """

        path = PurePosixPath(str(file_name).replace("\\", "/"))
        stem = path.stem
        prefix = self.delimiter.join(stem.split(self.delimiter)[: self.tokens])
        if self.delimiter not in stem or not prefix:
            return f"file:{path.as_posix()}"
        return f"group:{prefix}"


@dataclass(frozen=True)
class SplitResult:
    """분할 결과와 category별 이미지 수입니다."""

    train_image_ids: set[int]
    validation_image_ids: set[int]
    train_category_counts: dict[int, int]
    validation_category_counts: dict[int, int]
    group_count: int
    train_group_count: int
    validation_group_count: int


def _grouped_image_ids(
    images: Sequence[Mapping[str, Any]], group_rule: GroupRule | None
) -> dict[str, list[int]]:
    """이미지를 그룹 이름별로 묶습니다. 규칙이 없으면 한 장이 한 그룹입니다.

    돌려주는 dict의 **순서가 곧 분할의 기준 순서**입니다. 이름으로 정렬하면
    `image:1`, `image:10`, `image:2`처럼 사전식이 되어, 이미지 단위 분할이 같은
    seed로도 예전과 다른 결과를 내놓습니다. 그래서 이미지 단위에서는 이름이
    아니라 image id 숫자 순서로 넣습니다.
    """

    if group_rule is None:
        return {
            f"image:{image_id}": [image_id]
            for image_id in sorted(image["id"] for image in images)
        }

    groups: defaultdict[str, list[int]] = defaultdict(list)
    for image in images:
        groups[group_rule.key(str(image["file_name"]))].append(image["id"])
    return {name: sorted(members) for name, members in sorted(groups.items())}


def split_images(
    images: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
    group_rule: GroupRule | None = None,
) -> SplitResult:
    """이미지를 그룹 단위로 train과 validation으로 나눕니다."""

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

    groups = _grouped_image_ids(images, group_rule)
    group_names = list(groups)
    if len(group_names) < 2:
        raise DatasetPreparationError(
            "train과 validation을 나누려면 그룹이 2개 이상 필요합니다. 지금은 "
            f"이미지 {len(image_ids)}장이 모두 그룹 '{group_names[0]}' 하나입니다."
        )
    # 그룹이 통째로 움직이므로 이후 계산의 단위는 이미지가 아니라 그룹입니다.
    categories_by_group = {
        name: set().union(*(categories_by_image[image_id] for image_id in members))
        for name, members in groups.items()
    }
    group_sizes = {name: len(members) for name, members in groups.items()}

    category_totals = Counter(
        category_id
        for name in group_names
        for category_id in categories_by_group[name]
    )
    scarce = sorted(
        category_id for category_id, total in category_totals.items() if total < 2
    )
    if scarce:
        # 이미지 단위 분할에서는 이미지 한 장이 곧 그룹 하나입니다.
        raise DatasetPreparationError(
            "모든 category가 train과 validation 양쪽에 나타나려면 category마다 "
            "서로 다른 그룹이 2개 이상 필요합니다. 그룹이 1개뿐인 category id: "
            + ", ".join(str(category_id) for category_id in scarce)
        )

    target_size = min(max(1, round(len(image_ids) * validation_ratio)), len(image_ids) - 1)
    # 트레이드오프: 그룹을 통째로 옮기므로 이미지 단위 분할처럼 class 분포와
    # 목표 비율을 동시에 정확히 맞출 수는 없습니다. 그래서 우선순위를 이렇게
    # 정했습니다.
    #
    # 1. 누수 방지가 먼저입니다. 같은 그룹은 절대 나누지 않습니다.
    # 2. 그다음이 category 보장입니다. 한 category라도 한쪽 split에서 빠지면
    #    그 category의 지표를 아예 잴 수 없으므로, 채우기보다 앞에 둡니다.
    # 3. 그다음이 class 분포입니다. 채우기 단계에서 category별 목표 대비 상대
    #    오차의 제곱합이 가장 작은 그룹을 고릅니다.
    # 4. 마지막이 비율입니다. 목표 장수를 정확히 맞추는 대신 목표에 가장 가까운
    #    지점에서 멈춥니다. 그룹 크기가 고르지 않으면 몇 장 어긋날 수 있고,
    #    category를 모두 담느라 목표를 넘길 수도 있습니다.
    #
    # 즉 validation 장수와 분포의 정확도를 조금 포기하고, 형제 사진이 양쪽에
    # 나뉘어 검증 점수가 부풀려지는 문제를 없앴습니다.
    #
    # 반올림한 category별 목표치의 합은 validation 그룹 수보다 커질 수 있으므로,
    # 정확한 목표가 아니라 채우기 단계의 분포 목표로만 사용합니다.
    target_by_category = {
        category_id: min(max(1, round(total * validation_ratio)), total - 1)
        for category_id, total in category_totals.items()
    }

    # `_grouped_image_ids`가 이미 기준 순서대로 넣어 두었습니다. 여기서 다시
    # 정렬하면 그 순서가 깨지므로 그대로 씁니다.
    shuffled = list(group_names)
    random.Random(seed).shuffle(shuffled)
    tie_rank = {name: rank for rank, name in enumerate(shuffled)}
    validation: set[str] = set()
    validation_images = 0
    validation_counts: Counter[int] = Counter()

    def keeps_train_coverage(name: str) -> bool:
        """이 그룹을 validation에 넣어도 train에 category가 남는지 봅니다."""

        return all(
            validation_counts[category_id] + 1 <= category_totals[category_id] - 1
            for category_id in categories_by_group[name]
        )

    def uncovered() -> set[int]:
        return {
            category_id
            for category_id in category_totals
            if validation_counts[category_id] < 1
        }

    failed_coverage_states: set[frozenset[str]] = set()

    def cover_categories() -> bool:
        """탐욕 점수 순서로 시도하되 막히면 이전 선택으로 돌아갑니다."""

        nonlocal validation_images
        remaining = uncovered()
        if not remaining:
            return True

        state = frozenset(validation)
        if state in failed_coverage_states:
            return False

        rarest = min(remaining, key=lambda value: (category_totals[value], value))
        candidates = [
            name
            for name in group_names
            if name not in validation
            and rarest in categories_by_group[name]
            and keeps_train_coverage(name)
        ]
        if not candidates:
            failed_coverage_states.add(state)
            return False

        def coverage_score(name: str) -> tuple[float, float, float, int]:
            categories = categories_by_group[name]
            gain = sum(1 for value in categories if value in remaining)
            scarcity = sum(1.0 / category_totals[value] for value in categories)
            overshoot = sum(
                max(0, validation_counts[value] + 1 - target_by_category[value])
                for value in categories
            )
            return (float(gain), scarcity, -float(overshoot), -tie_rank[name])

        for chosen in sorted(candidates, key=coverage_score, reverse=True):
            validation.add(chosen)
            validation_images += group_sizes[chosen]
            validation_counts.update(categories_by_group[chosen])
            if cover_categories():
                return True
            validation_counts.subtract(categories_by_group[chosen])
            validation_images -= group_sizes[chosen]
            validation.remove(chosen)

        failed_coverage_states.add(state)
        return False

    # 이 단계에는 목표 장수 제한을 두지 않습니다. 그룹이 크면 목표를 넘길 수
    # 있지만, 한 category가 validation에서 통째로 빠지는 것보다 비율이 조금
    # 어긋나는 편이 낫습니다. 실제 비율은 요약에 남습니다. 가장 좋은 탐욕 선택이
    # 뒤 category의 유일한 후보를 막으면 다음 후보로 돌아가 가능한 분할을 찾습니다.
    if not cover_categories():
        raise DatasetPreparationError(
            "모든 category를 train과 validation 양쪽에 넣는 그룹 분할을 찾을 수 "
            "없습니다."
        )

    while validation_images < target_size:
        # 그룹을 통째로 넣어야 하므로 목표를 지나칠 수 있습니다. 목표에 더
        # 가까워지는 그룹만 후보로 두면, 가장 가까운 지점에서 저절로 멈춥니다.
        distance = abs(validation_images - target_size)
        candidates = [
            name
            for name in group_names
            if name not in validation
            and keeps_train_coverage(name)
            and abs(validation_images + group_sizes[name] - target_size) < distance
        ]
        if not candidates:
            break

        def distribution_error(name: str) -> tuple[float, int]:
            simulated = {
                category_id: validation_counts[category_id]
                + (1 if category_id in categories_by_group[name] else 0)
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
            return (error, tie_rank[name])

        chosen = min(candidates, key=distribution_error)
        validation.add(chosen)
        validation_images += group_sizes[chosen]
        validation_counts.update(categories_by_group[chosen])

    train_groups = [name for name in group_names if name not in validation]
    validation_image_ids = {
        image_id for name in validation for image_id in groups[name]
    }
    train_image_ids = set(image_ids) - validation_image_ids
    if not train_image_ids or not validation_image_ids:
        raise DatasetPreparationError("train과 validation split은 모두 비어 있으면 안 됩니다.")

    train_counts = Counter(
        category_id
        for image_id in train_image_ids
        for category_id in categories_by_image[image_id]
    )
    image_validation_counts = Counter(
        category_id
        for image_id in validation_image_ids
        for category_id in categories_by_image[image_id]
    )
    image_totals = Counter(
        category_id
        for image_id in image_ids
        for category_id in categories_by_image[image_id]
    )
    for category_id, total in image_totals.items():
        if train_counts[category_id] < 1 or image_validation_counts[category_id] < 1:
            raise DatasetPreparationError(
                f"category {category_id}가 한쪽 split에만 있습니다."
            )
        if train_counts[category_id] + image_validation_counts[category_id] != total:
            raise DatasetPreparationError(
                f"category {category_id}의 이미지 수 계산이 맞지 않습니다."
            )
    return SplitResult(
        train_image_ids=train_image_ids,
        validation_image_ids=validation_image_ids,
        train_category_counts=dict(train_counts),
        validation_category_counts=dict(image_validation_counts),
        group_count=len(group_names),
        train_group_count=len(train_groups),
        validation_group_count=len(validation),
    )
