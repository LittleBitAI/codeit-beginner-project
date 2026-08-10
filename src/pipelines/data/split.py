"""그룹 단위로 train/validation을 결정적으로 나눕니다.

같은 알약 조합을 각도와 조명만 바꿔 여러 장 찍은 원본에서는, 이미지 한 장씩
나누면 거의 같은 사진이 train과 validation 양쪽에 들어가 validation 점수가 실제
성능보다 좋게 나옵니다. 그래서 **file 이름 접두사가 같은 이미지를 한 그룹으로
묶어 통째로 한쪽 split에만** 넣습니다. 그룹 규칙(`GroupRule`)을 주지 않으면
이미지 한 장이 그대로 한 그룹이 되어 예전과 같은 결과가 나옵니다.

한 이미지에 여러 category가 함께 있을 수 있으므로(다중 라벨) 단순 무작위 분할은
희귀 category를 한쪽 split에서 통째로 빠뜨릴 수 있습니다. 그래서 희귀 category부터
validation에 배치해 **validation에 갈 수 있는 category가 train과 validation 양쪽에
반드시 나타나게** 한 뒤, 남은 자리를 목표 비율에 가장 가깝게 채웁니다. 그룹은 통째로
움직이므로 validation 이미지 수가 목표에 정확히 맞지 않을 수 있고, 그때는 목표에
가장 가까운 지점에서 멈춥니다.

그룹을 쪼개지 않고는 **validation에 갈 수 없는 category**는 그룹을 통째로 옮기는 한
양쪽 split에 나타날 방법이 원리적으로 없습니다. 등장하는 그룹이 하나뿐인 category가
그렇고, 그런 category를 품은 그룹은 validation 후보에서 빠지므로 자기 그룹이 전부
그런 그룹인 category도 마찬가지입니다. 그래서 그런 category는 실행을 실패시키는 대신
**train 쪽에만** 둡니다(`train_only_category_ids`). 학습에는 쓰지만 validation 지표는
잴 수 없는 category라는 뜻이므로, 결과에 목록으로 남겨 요약에서 확인할 수 있게 합니다.

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


__all__ = ["AngleRule", "GroupRule", "SplitResult", "hold_out_angle", "split_images"]


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
class AngleRule:
    """file 이름에서 촬영 각도를 뽑는 규칙입니다.

    원본 이름은 `<조합 코드>_<촬영 조건들>` 형태이고 각도는 **끝에서 세 번째**
    토큰입니다. 예를 들어 `K-000250-000573_0_2_0_2_70_000_200`의 각도는 `70`입니다.
    조합 코드 뒤의 토큰 수는 원본 판마다 다르지만 뒤쪽 세 개는 같으므로, 앞에서
    세지 않고 뒤에서 셉니다.

    규칙에 맞지 않는 이름은 각도를 `None`으로 둡니다. 모르는 것을 validation에
    넣으면 그 각도가 학습에도 있었는지 말할 수 없게 되므로, 부르는 쪽이 train에
    둡니다.
    """

    delimiter: str
    #: 뒤에서부터 센 위치입니다. 1이 마지막 토큰입니다.
    position_from_end: int

    def key(self, file_name: str) -> str | None:
        path = PurePosixPath(str(file_name).replace("\\", "/"))
        parts = path.stem.split(self.delimiter)
        if len(parts) < self.position_from_end + 1:
            return None
        value = parts[-self.position_from_end]
        return value or None


@dataclass(frozen=True)
class SplitResult:
    """분할 결과와 category별 이미지 수입니다."""

    train_image_ids: set[int]
    validation_image_ids: set[int]
    train_category_counts: dict[int, int]
    validation_category_counts: dict[int, int]
    # 그룹을 쪼개지 않고는 validation에 갈 수 없어 train에만 둔 category id입니다.
    # 오름차순입니다.
    train_only_category_ids: tuple[int, ...]
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
    groups_by_category: defaultdict[int, list[str]] = defaultdict(list)
    for name in group_names:
        for category_id in categories_by_group[name]:
            groups_by_category[category_id].append(name)
    # 그룹이 하나뿐인 category를 품은 그룹은 validation에 갈 수 없습니다. 그 그룹을
    # 옮기면 category가 train에서 통째로 사라지기 때문이며, 아래
    # `keeps_train_coverage`가 `1 <= 0`으로 그 그룹을 후보에서 영구히 뺍니다.
    blocked_groups = {
        name
        for name in group_names
        if any(
            category_totals[category_id] < 2
            for category_id in categories_by_group[name]
        )
    }
    # 그룹을 쪼개지 않는 한 양쪽 split에 나타날 방법이 없는 category입니다. 그룹이
    # 하나뿐인 경우가 대표적이지만, 그룹이 2개 이상이어도 그 그룹이 전부 위에서
    # 막힌 그룹이면 마찬가지로 validation에 갈 수 없습니다. 예전에는 그런 category
    # 때문에 준비 전체가 실패했지만, 그러면 몇 종 때문에 나머지 데이터까지 쓸 수
    # 없게 됩니다. 그래서 이제는 그 category만 train 쪽에 두고 나머지는 평소대로
    # 나눕니다. 이미지 단위 분할에서는 이미지 한 장이 곧 그룹 하나입니다.
    train_only_categories = tuple(
        sorted(
            category_id
            for category_id, total in category_totals.items()
            if total < 2
            or all(name in blocked_groups for name in groups_by_category[category_id])
        )
    )
    # 양쪽 split에 반드시 나타나야 하는 category입니다. 이후 보장·분포 계산은
    # 모두 이 집합만 대상으로 합니다.
    coverable = set(category_totals) - set(train_only_categories)

    target_size = min(max(1, round(len(image_ids) * validation_ratio)), len(image_ids) - 1)
    # 트레이드오프: 그룹을 통째로 옮기므로 이미지 단위 분할처럼 class 분포와
    # 목표 비율을 동시에 정확히 맞출 수는 없습니다. 그래서 우선순위를 이렇게
    # 정했습니다.
    #
    # 1. 누수 방지가 먼저입니다. 같은 그룹은 절대 나누지 않습니다.
    # 2. 그다음이 category 보장입니다. 한 category라도 한쪽 split에서 빠지면
    #    그 category의 지표를 아예 잴 수 없으므로, 채우기보다 앞에 둡니다.
    #    단 그룹을 쪼개지 않고는 애초에 양쪽에 둘 수 없는 category는 train 전용으로
    #    빼고 계산합니다(위 `train_only_categories`).
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
        category_id: min(
            max(1, round(category_totals[category_id] * validation_ratio)),
            category_totals[category_id] - 1,
        )
        for category_id in coverable
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
            for category_id in coverable
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
                if value in target_by_category
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
                for category_id in coverable
            }
            error = sum(
                (
                    (simulated[category_id] - target_by_category[category_id])
                    / target_by_category[category_id]
                )
                ** 2
                for category_id in coverable
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
        if validation_image_ids or not train_only_categories:
            raise DatasetPreparationError(
                "train과 validation split은 모두 비어 있으면 안 됩니다."
            )
        # 모든 그룹이 그룹 1개짜리 category를 하나씩 갖고 있으면 validation으로 옮길
        # 수 있는 그룹이 남지 않습니다. 이때는 데이터 자체로 분할이 불가능합니다.
        raise DatasetPreparationError(
            "validation에 넣을 수 있는 그룹이 없습니다. 모든 그룹이 그룹 1개짜리 "
            "category를 갖고 있어 train에 남아야 합니다. train 전용 category id: "
            + ", ".join(str(category_id) for category_id in train_only_categories)
        )

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
        if category_id in train_only_categories:
            # train 전용 category는 train에만 있어야 합니다. validation에
            # 나타났다면 그룹이 쪼개졌다는 뜻이므로 누수입니다.
            if train_counts[category_id] < 1 or image_validation_counts[category_id]:
                raise DatasetPreparationError(
                    f"train 전용 category {category_id}는 train에만 있어야 합니다."
                )
        elif train_counts[category_id] < 1 or image_validation_counts[category_id] < 1:
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
        train_only_category_ids=train_only_categories,
        group_count=len(group_names),
        train_group_count=len(train_groups),
        validation_group_count=len(validation),
    )


def hold_out_angle(
    split_result: SplitResult,
    images: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    group_rule: GroupRule,
    angle_rule: AngleRule,
    validation_angle: str,
) -> tuple[SplitResult, int]:
    """이미 나뉜 그룹 분할에서 촬영 각도 하나를 validation 전용으로 떼어냅니다.

    validation은 그 각도만 남기고, train은 그 각도를 **전부** 버립니다. 그래야
    validation이 "학습에서 못 본 시점"을 재게 됩니다. 한 장이라도 train에 남으면
    그 시점을 이미 배운 model을 재는 것이라 그룹 분할과 다를 바가 없습니다.

    그 대가로 이미지를 잃습니다. train 그룹의 그 각도와 validation 그룹의 나머지
    각도가 모두 버려집니다. 버린 수를 함께 돌려주어 부르는 쪽이 기록하게 합니다.

    각도를 알 수 없는 이름은 train에 남깁니다. 모르는 것을 validation에 넣으면 그
    각도가 학습에도 있었는지 말할 수 없게 됩니다.
    """

    angle_by_image: dict[int, str | None] = {}
    group_by_image: dict[int, str] = {}
    for image in images:
        file_name = str(image["file_name"])
        angle_by_image[image["id"]] = angle_rule.key(file_name)
        group_by_image[image["id"]] = group_rule.key(file_name)

    train_image_ids = {
        image_id
        for image_id in split_result.train_image_ids
        if angle_by_image.get(image_id) != validation_angle
    }
    validation_image_ids = {
        image_id
        for image_id in split_result.validation_image_ids
        if angle_by_image.get(image_id) == validation_angle
    }
    dropped = (
        len(split_result.train_image_ids)
        + len(split_result.validation_image_ids)
        - len(train_image_ids)
        - len(validation_image_ids)
    )
    if not train_image_ids or not validation_image_ids:
        raise DatasetPreparationError(
            f"각도 '{validation_angle}'을 빼면 한쪽 split이 비어 있게 됩니다. "
            "다른 각도를 고르거나 그룹 분할을 쓰세요."
        )

    categories_by_image: defaultdict[Any, set[int]] = defaultdict(set)
    for annotation in annotations:
        categories_by_image[annotation["image_id"]].add(annotation["category_id"])

    train_counts = Counter(
        category_id
        for image_id in train_image_ids
        for category_id in categories_by_image[image_id]
    )
    validation_counts = Counter(
        category_id
        for image_id in validation_image_ids
        for category_id in categories_by_image[image_id]
    )
    # 각도를 빼면서 train에서 사라진 category는 model이 배울 수 없습니다. 조용히
    # 내보내면 그 category의 점수는 언제나 0이 되고 원인은 보이지 않습니다.
    #
    # 비교 대상은 각도를 빼기 **전의** category 집합입니다. validation에 남은 것만
    # 보면, train 그룹에서는 빼는 각도에만 있고 validation 그룹에서는 다른 각도에만
    # 있어 양쪽에서 동시에 사라지는 category를 놓칩니다. 그런 class는 class map에는
    # 남아 학습 예시 없는 class가 됩니다.
    before = set(split_result.train_category_counts) | set(
        split_result.validation_category_counts
    )
    lost_from_train = sorted(before - set(train_counts))
    if lost_from_train:
        raise DatasetPreparationError(
            f"각도 '{validation_angle}'을 빼면 category "
            + ", ".join(str(value) for value in lost_from_train)
            + "가 train에서 사라집니다. 다른 각도를 고르세요."
        )
    # validation에서 사라진 category는 지표를 잴 수 없을 뿐이라, 그룹 분할이 쓰는
    # 것과 같은 자리에 적어 둡니다.
    train_only = tuple(
        sorted(
            set(split_result.train_only_category_ids)
            | (set(train_counts) - set(validation_counts))
        )
    )

    def groups_of(image_ids: set[int]) -> int:
        return len({group_by_image[image_id] for image_id in image_ids})

    return (
        SplitResult(
            train_image_ids=train_image_ids,
            validation_image_ids=validation_image_ids,
            train_category_counts=dict(train_counts),
            validation_category_counts=dict(validation_counts),
            train_only_category_ids=train_only,
            group_count=split_result.group_count,
            train_group_count=groups_of(train_image_ids),
            validation_group_count=groups_of(validation_image_ids),
        ),
        dropped,
    )
