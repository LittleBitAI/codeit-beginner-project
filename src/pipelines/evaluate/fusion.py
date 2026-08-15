"""여러 실행의 예측을 하나로 합칩니다.

로컬 검증은 0.93에서 포화인데 Kaggle은 0.61입니다. 남은 유력한 설명이 모델이 실물
개체를 외웠다는 것이고, 그렇다면 **서로 다르게 외운 모델을 합치면** 외운 부분이
상쇄됩니다. 상자를 평균 내는 것은 덤입니다 — 대회 지표가 IoU 0.75~0.95로 엄격해서
좌표가 조금만 나아져도 점수가 움직입니다.

Weighted Box Fusion입니다. 겹치는 상자를 버리지 않고(NMS와 다릅니다) 확신도로 가중
평균해 하나로 만들고, **몇 개의 실행이 동의했는지**로 확신도를 다시 매깁니다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


#: 이 값을 넘게 겹치면 같은 알약을 가리킨 것으로 봅니다.
DEFAULT_IOU_THRESHOLD = 0.55


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    """xywh 상자 둘의 IoU입니다."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - overlap
    return overlap / union if union > 0 else 0.0


class _Cluster:
    """같은 알약을 가리킨다고 본 상자들입니다."""

    def __init__(self) -> None:
        self.boxes: list[tuple[Sequence[float], float]] = []
        # 어느 실행이 이 자리를 찾았는지입니다. 한 실행이 두 번 찾아도 한 번으로
        # 셉니다 — 그러지 않으면 자기 자신과 동의해 확신도가 올라갑니다.
        self.sources: set[int] = set()

    def add(self, bbox: Sequence[float], score: float, source: int) -> None:
        self.boxes.append((bbox, score))
        self.sources.add(source)

    @property
    def box(self) -> list[float]:
        """확신도로 가중 평균한 상자입니다."""

        total = sum(score for _, score in self.boxes)
        if total <= 0.0:
            count = len(self.boxes)
            return [sum(bbox[i] for bbox, _ in self.boxes) / count for i in range(4)]
        return [
            sum(bbox[i] * score for bbox, score in self.boxes) / total for i in range(4)
        ]

    def score(self, source_count: int) -> float:
        """평균 확신도를 **동의한 실행 수**로 깎습니다.

        혼자 찾은 상자를 그대로 믿으면 합치는 뜻이 없습니다. 한 실행이 외운 것이
        그대로 올라옵니다.
        """

        average = sum(score for _, score in self.boxes) / len(self.boxes)
        return average * min(len(self.sources), source_count) / source_count


def fuse_predictions(
    prediction_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """실행별 예측 목록을 받아 합친 예측 하나를 돌려줍니다.

    같은 이미지의 같은 class끼리만 합칩니다. 겹쳐 보여도 class가 다르면 다른
    알약이고, 이미지가 다르면 아예 다른 사진입니다.
    """

    if not prediction_groups:
        return []
    source_count = len(prediction_groups)

    # (이미지, class)별로 모읍니다. 확신도가 높은 것부터 자리를 잡아야 낮은 것이
    # 거기에 붙습니다. 순서가 뒤집히면 약한 상자가 기준이 됩니다.
    by_target: dict[tuple[str, int], list[tuple[float, int, Sequence[float]]]] = {}
    # 제출 CSV는 정규화한 key가 아니라 manifest의 `image_id`를 그대로 씁니다.
    image_ids: dict[str, Any] = {}
    for source, predictions in enumerate(prediction_groups):
        for prediction in predictions:
            image_key = str(prediction["image_key"])
            image_ids.setdefault(image_key, prediction["image_id"])
            key = (image_key, int(prediction["category_id"]))
            by_target.setdefault(key, []).append(
                (float(prediction["score"]), source, prediction["bbox"])
            )

    fused: list[dict[str, Any]] = []
    for (image_key, category_id), entries in by_target.items():
        # score 내림차순, 같으면 들어온 순서를 지켜 결과가 실행마다 흔들리지 않게
        # 합니다.
        entries.sort(key=lambda entry: -entry[0])
        clusters: list[_Cluster] = []
        for score, source, bbox in entries:
            for cluster in clusters:
                if _iou(cluster.box, bbox) > iou_threshold:
                    cluster.add(bbox, score, source)
                    break
            else:
                cluster = _Cluster()
                cluster.add(bbox, score, source)
                clusters.append(cluster)

        for cluster in clusters:
            fused.append(
                {
                    "image_id": image_ids[image_key],
                    "image_key": image_key,
                    "category_id": category_id,
                    "bbox": cluster.box,
                    "score": cluster.score(source_count),
                }
            )
    return fused


__all__ = ["DEFAULT_IOU_THRESHOLD", "fuse_predictions"]
