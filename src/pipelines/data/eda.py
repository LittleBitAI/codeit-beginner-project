"""전처리 dataset 하나를 model 없이 뜯어보고 EDA 리포트를 만듭니다.

model이 낸 예측으로 model의 실패를 설명하면 순환 논증입니다. 여기서 재는 값은
manifest에 적힌 사실과 원본 이미지의 픽셀뿐이라, 어떤 학습 결과와도 무관하게 같은
답이 나옵니다. 그래서 "학습이 잘못됐나"와 "데이터가 다른가"를 갈라 말할 수 있습니다.

대회 test 이미지도 읽습니다. **annotation은 읽지 않고**, 여기서 잰 값은 학습이나
전처리 결정에 들어가지 않으며 리포트에만 남습니다. 무엇을 읽었는지는 리포트의
`sources`에 그대로 기록합니다.
"""

from __future__ import annotations

import posixpath
import statistics
import tempfile
from math import isfinite
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.common import Storage, StorageError, create_storage

from .errors import EdaError
from .preparation import REPOSITORY_ROOT
from .progress import ProgressEmitter


__all__ = ["check_manifest", "eda_requested", "measure_image", "run_eda"]


#: 리포트를 넣는 폴더 이름입니다. 전처리 결과 옆에 두어야 화면이 찾아갑니다.
REPORT_DIRECTORY = "eda"

#: 리포트 파일 이름입니다.
REPORT_NAME = "report.json"

#: 리포트 schema. 화면이 이 값을 보고 읽을 수 있는지 판단합니다.
SCHEMA_VERSION = "1.0"

#: 픽셀을 잴 때 이미지를 줄이는 배수입니다. 알약 지름이 200px가 넘으므로 8배로
#: 줄여도 모양이 남고, 한 장에 드는 시간이 64분의 1이 됩니다.
DOWNSCALE = 8

#: 전경 안의 구멍(각인, 반사)을 메우는 반지름입니다. 줄인 이미지 기준이라 2px면
#: 원본에서 16px입니다.
CLOSING_RADIUS = 2

#: 픽셀로 잰 전경이 정답 넓이의 이 배수 안에 들어와야 크기 비교를 믿습니다.
#: 벗어나면 리포트가 비교 대신 "재지 못했다"고 적습니다.
CALIBRATION_LIMITS = (0.5, 1.5)

#: 픽셀을 잴 train 이미지 수의 기본값입니다. 분포만 보면 되므로 전수는 필요 없습니다.
DEFAULT_IMAGE_SAMPLE = 200

#: 파일명에서 조합 이름을 끊는 구분자입니다. `split.py`의 group 규칙과 같습니다.
GROUP_KEY_DELIMITER = "_"


def _is_remote(location: str) -> bool:
    return "://" in location


def _artifact_location(uri: str) -> str:
    """저장소가 실제로 열 수 있는 위치로 바꿉니다.

    artifact URI는 저장소 root 기준(`artifacts/datasets/...`)인데 `LocalStorage`는
    자기 root(보통 `artifacts/`) 기준으로 풉니다. 그대로 넘기면 `artifacts/artifacts/…`를
    찾습니다. S3 URI는 손대지 않습니다.
    """

    text = str(uri).strip()
    if _is_remote(text):
        return text
    path = Path(text)
    resolved = path if path.is_absolute() else (REPOSITORY_ROOT / path)
    resolved = resolved.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise EdaError(f"저장소 밖을 가리키는 artifact 경로입니다: {path.name}") from error
    return str(resolved)


def _image_location(manifest_uri: str, file_name: str) -> str:
    """manifest에 적힌 이미지 위치를 실제로 열 수 있는 위치로 바꿉니다.

    `file_name`은 **manifest 폴더 기준**입니다(`../../raw/v1/train_images/a.png`).
    manifest는 `processed/` 아래, 이미지는 `raw/` 아래라 그대로 넘기면 엉뚱한 곳을
    가리킵니다. 이미 완전한 URI면 그대로 씁니다.
    """

    name = str(file_name).strip()
    if _is_remote(name):
        return name
    if _is_remote(manifest_uri):
        scheme, _, rest = manifest_uri.partition("://")
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(rest), name))
        return f"{scheme}://{joined}"
    base = Path(_artifact_location(manifest_uri)).parent
    return _artifact_location(str(base / name))


def eda_requested(config: Any) -> bool:
    """EDA 경로를 실행할지 확인합니다. 명시적으로 켰을 때만 참입니다."""

    value = _data_config(config).get("eda", False)
    if not isinstance(value, bool):
        raise EdaError("config['data']['eda']는 true 또는 false여야 합니다.")
    return value


def _data_config(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    data = config.get("data")
    return data if isinstance(data, Mapping) else {}


# --- 픽셀 측정 -------------------------------------------------------------


def _otsu_threshold(values: np.ndarray) -> float:
    """전경과 배경을 가르는 값을 Otsu 방법으로 고릅니다.

    촬영 부스 배경이 균일해서 histogram이 두 봉우리로 갈립니다. 임계값을 손으로
    정하면 dataset 판마다 다시 맞춰야 하므로 데이터가 정하게 둡니다.
    """

    top = float(values.max())
    if top <= 0:
        return 0.0
    counts, edges = np.histogram(values, bins=256, range=(0.0, top))
    weights = counts.astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return top / 2.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(weights)
    weight_high = total - weight_low
    sum_low = np.cumsum(weights * centers)
    sum_total = sum_low[-1]
    usable = (weight_low > 0) & (weight_high > 0)
    if not usable.any():
        return top / 2.0
    mean_low = np.where(usable, sum_low / np.maximum(weight_low, 1e-9), 0.0)
    mean_high = np.where(usable, (sum_total - sum_low) / np.maximum(weight_high, 1e-9), 0.0)
    variance = weight_low * weight_high * (mean_low - mean_high) ** 2
    variance = np.where(usable, variance, -1.0)
    return float(centers[int(np.argmax(variance))])


def _closed(mask: np.ndarray, *, radius: int = CLOSING_RADIUS) -> np.ndarray:
    """덩어리를 `radius`만큼 불렸다가 도로 줄입니다.

    알약마다 그림자와 반사 테두리가 얇은 틈을 두고 떨어져 나옵니다. 그대로 세면
    알약 하나가 물체 두 개가 됩니다. 불렸다 줄이면 그 틈만 메워지고 크기는 남습니다.
    """

    def sweep(source: np.ndarray, combine) -> np.ndarray:
        result = source.copy()
        result[1:, :] = combine(result[1:, :], source[:-1, :])
        result[:-1, :] = combine(result[:-1, :], source[1:, :])
        result[:, 1:] = combine(result[:, 1:], source[:, :-1])
        result[:, :-1] = combine(result[:, :-1], source[:, 1:])
        return result

    grown = mask
    for _ in range(radius):
        grown = sweep(grown, np.logical_or)
    for _ in range(radius):
        grown = sweep(grown, np.logical_and)
    return grown


def measure_image(image: Image.Image, *, downscale: int = DOWNSCALE) -> dict[str, Any]:
    """사진 한 장에서 물체가 차지하는 넓이와 배경·물체의 색을 잽니다.

    `foreground_fraction`은 물체가 덮은 넓이의 비율(0~1)입니다.

    annotation을 쓰지 않으므로 정답이 없는 test 이미지에도 그대로 씁니다.

    **물체를 하나씩 세지 않습니다.** 세려면 붙어 있는 픽셀을 덩어리로 묶어야 하는데,
    알약마다 딸린 그림자와 반사 테두리가 얇은 틈을 두고 떨어져 나와 하나가 둘로도
    셋으로도 됩니다. 실제로 재어 보니 무엇을 한 덩어리로 볼지에 따라 결과가 절반씩
    오갔습니다. 한 장에 든 알약 수가 같은 dataset이라면 넓이 비율만으로 "물체가
    더 크게 찍혔는가"를 답할 수 있고, 그 값은 나누는 방식에 흔들리지 않습니다.

    밝기가 아니라 **배경과 색이 얼마나 다른지**로 가릅니다. 밝기로 가르면 흰 알약이
    밝은 촬영 부스와 같은 값이 되어 통째로 사라집니다. 배경색은 가장자리 픽셀에서
    얻습니다 — 알약이 액자 밖까지 나가지는 않으므로 거기는 늘 배경입니다.

    색도 함께 돌려줍니다. 촬영 부스나 조명이 판마다 다르면 물체 크기가 같아도 model이
    보는 그림이 달라지는데, 크기만 재고 있으면 그 차이를 통째로 놓칩니다.
    """

    scale = max(1, int(downscale))
    width, height = image.size
    small = image.convert("RGB").resize(
        (max(1, width // scale), max(1, height // scale)), Image.BILINEAR
    )
    pixels = np.asarray(small, dtype=np.float32)
    border = np.concatenate(
        [pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]]
    )
    distance = _background_distance(pixels)
    mask = _closed(distance > _otsu_threshold(distance))
    return {
        "foreground_fraction": float(mask.mean()),
        "background_color": [round(float(value), 2) for value in np.median(border, axis=0)],
        "foreground_color": (
            [round(float(value), 2) for value in pixels[mask].mean(axis=0)]
            if mask.any()
            else None
        ),
    }


def _background_distance(values: np.ndarray) -> np.ndarray:
    """가장자리에서 얻은 배경값으로부터의 거리입니다.

    알약이 액자 밖까지 나가지는 않으므로 가장자리는 늘 배경입니다.
    """

    border = np.concatenate(
        [values[0, :, :], values[-1, :, :], values[:, 0, :], values[:, -1, :]]
    )
    return np.sqrt(((values - np.median(border, axis=0)) ** 2).sum(axis=2))


def _measure_images(
    storage: Storage,
    locations: Sequence[str],
    *,
    stage: str,
    on_progress: Any | None = None,
) -> list[dict[str, Any]]:
    """이미지를 한 장씩 받아 재고 곧바로 지웁니다. 원본을 남기지 않습니다."""

    measured: list[dict[str, Any]] = []
    for done, location in enumerate(locations, start=1):
        with tempfile.TemporaryDirectory(prefix="eda-") as scratch:
            local = Path(scratch) / posixpath.basename(location.replace("\\", "/"))
            try:
                storage.download_file(location, local)
                with Image.open(local) as opened:
                    opened.load()
                    measured.append(measure_image(opened))
            except (StorageError, OSError, UnidentifiedImageError) as error:
                raise EdaError(
                    f"EDA로 이미지를 여는 데 실패했습니다: {posixpath.basename(location)}"
                ) from error
        if on_progress is not None:
            on_progress(stage, done, len(locations))
    return measured


# --- 요약 도우미 -----------------------------------------------------------


def _distribution(values: Sequence[float]) -> dict[str, Any] | None:
    """분포를 화면이 그대로 그릴 수 있는 다섯 숫자로 줄입니다.

    잰 값이 없으면 0이 아니라 ``None``입니다. 0으로 적으면 "넓이가 0"으로 읽혀
    사실과 정반대가 됩니다.
    """

    # NaN과 Infinity는 표준 JSON이 아니라 리포트를 저장할 때 통째로 깨뜨립니다.
    ordered = sorted(
        float(value) for value in values if isinstance(value, (int, float)) and isfinite(value)
    )
    if not ordered:
        return None
    pick = lambda fraction: ordered[int(fraction * (len(ordered) - 1))]  # noqa: E731
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p10": round(pick(0.10), 4),
        "median": round(statistics.median(ordered), 4),
        "p90": round(pick(0.90), 4),
        "max": round(ordered[-1], 4),
    }


def _sample(values: Sequence[Any], limit: int) -> list[Any]:
    """앞에서 자르지 않고 고르게 건너뛰며 고릅니다. 같은 입력은 같은 표본입니다."""

    if limit <= 0 or len(values) <= limit:
        return list(values)
    step = len(values) / limit
    return [values[int(index * step)] for index in range(limit)]


def _group_name(file_name: str) -> str:
    """파일명에서 조합 이름을 꺼냅니다. split이 쓰는 규칙과 같아야 합니다."""

    base = posixpath.basename(str(file_name).replace("\\", "/"))
    return base.split(GROUP_KEY_DELIMITER)[0]


def _capture_condition(file_name: str) -> str:
    """조합 이름을 뺀 나머지가 촬영 조건(각도·조명·배경)입니다."""

    base = posixpath.basename(str(file_name).replace("\\", "/"))
    parts = base.split(GROUP_KEY_DELIMITER)
    return GROUP_KEY_DELIMITER.join(parts[1:]) if len(parts) > 1 else ""


def _annotations_by_image(manifest: Mapping[str, Any]) -> dict[Any, list[Mapping[str, Any]]]:
    """annotation을 이미지별로 묶습니다. 쓸 수 없는 값이면 여기서 거절합니다.

    `image_id`는 묶음의 열쇠라 해시할 수 있어야 합니다. `category_id`는 번호로
    정렬까지 하므로 **정수만** 받습니다. 숫자와 문자열이 섞이면 정렬에서
    `TypeError`가 나는데, 그것을 여기서 통일해 주면 어느 쪽이 옳은 번호인지 우리가
    정하는 셈이 됩니다. 손상된 manifest는 고쳐서 다시 만들어야 합니다.
    """

    grouped: defaultdict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in manifest.get("annotations") or []:
        if not isinstance(annotation, Mapping):
            continue
        image_id = annotation.get("image_id")
        if not isinstance(image_id, (int, str)) or isinstance(image_id, bool):
            raise EdaError("manifest의 annotation image_id가 숫자나 문자열이 아닙니다.")
        category_id = annotation.get("category_id")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise EdaError("manifest의 annotation category_id가 정수가 아닙니다.")
        grouped[image_id].append(annotation)
    return dict(grouped)


def check_manifest(manifest: Mapping[str, Any], label: str) -> None:
    """리포트를 만들기 전에 manifest가 앞뒤가 맞는지 확인합니다.

    annotation이 없는 image를 가리키면 그 annotation은 어느 이미지에도 붙지 않아
    "물체 0개"로 집계됩니다. 실행은 성공으로 끝나고 화면에는 틀린 숫자가 뜹니다.
    믿을 수 있는 숫자를 내는 것이 이 리포트의 존재 이유이므로, 조용히 세는 대신
    거절합니다.
    """

    images = _images(manifest)
    known = {image["id"] for image in images}
    if len(known) != len(images):
        # 같은 id의 image가 둘이면 그 annotation이 두 번 세어집니다. 검증은 통과하고
        # 이미지 수, class 빈도, 물체 크기가 조용히 부풀어 오릅니다.
        raise EdaError(
            f"{label} manifest에 같은 id를 가진 image가 둘 이상 있습니다. "
            "전처리 결과가 온전한지 확인하세요."
        )
    dangling = sorted(
        {str(key) for key in _annotations_by_image(manifest) if key not in known}
    )
    if dangling:
        raise EdaError(
            f"{label} manifest의 annotation이 없는 image를 가리킵니다"
            f"(image_id {', '.join(dangling[:3])}). 전처리 결과가 온전한지 확인하세요."
        )


def _images(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """manifest의 image 목록. 뒤에서 쓰는 key가 없으면 여기서 거절합니다.

    화면의 문서 분류는 `images`와 `annotations`가 배열인지만 봅니다. 그래서 안이
    비어 있는 manifest도 고를 수 있고, 그대로 두면 `image["file_name"]`에서
    `KeyError`가 `run()` 경계 밖으로 나갑니다.
    """

    images = [image for image in (manifest.get("images") or []) if isinstance(image, Mapping)]
    for image in images:
        # 있기만 하면 되는 것이 아니라 쓸 수 있는 값이어야 합니다. `id: []`는
        # 묶음의 열쇠로 쓸 수 없고, `width: "bad"`는 넓이 계산에서 터집니다.
        if not isinstance(image.get("id"), (int, str)) or isinstance(image.get("id"), bool):
            raise EdaError("manifest의 image id가 숫자나 문자열이 아닙니다.")
        if not str(image.get("file_name") or "").strip():
            raise EdaError("manifest의 image 항목에 file_name이 없습니다.")
        for key in ("width", "height"):
            value = image.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise EdaError(f"manifest의 image {key}가 숫자가 아닙니다.")
    return images


# --- 리포트 절 -------------------------------------------------------------


def _shape_section(train: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    """한 이미지에 물체가 몇 개 들어 있는지. 이 dataset의 기본 모양입니다."""

    section: dict[str, Any] = {}
    for name, manifest in (("train", train), ("validation", validation)):
        by_image = _annotations_by_image(manifest)
        counts = [len(by_image.get(image["id"], [])) for image in _images(manifest)]
        # 같은 class가 한 이미지에 두 번 나오는지. 정답에서 0이면, 예측이 그렇게
        # 하는 것은 그 자체로 틀린 것입니다.
        repeated = sum(
            1
            for image in _images(manifest)
            if len({a.get("category_id") for a in by_image.get(image["id"], [])})
            != len(by_image.get(image["id"], []))
        )
        section[name] = {
            "images": len(counts),
            "annotations": sum(counts),
            "objects_per_image": dict(sorted(Counter(counts).items())),
            "images_with_a_repeated_class": repeated,
        }
    return section


def _class_section(
    train: Mapping[str, Any], validation: Mapping[str, Any], class_map: Mapping[str, Any]
) -> dict[str, Any]:
    """class마다 몇 장에 나오는지와 그 치우침입니다.

    "학습이 적어서 못 맞힌다"는 설명은 여기 숫자로만 검증할 수 있습니다.
    """

    def image_counts(manifest: Mapping[str, Any]) -> Counter:
        by_image = _annotations_by_image(manifest)
        counter: Counter = Counter()
        for image in _images(manifest):
            for category in {a.get("category_id") for a in by_image.get(image["id"], [])}:
                counter[category] += 1
        return counter

    train_counts = image_counts(train)
    validation_counts = image_counts(validation)
    names = _class_names(class_map)
    categories = sorted(
        set(names) | set(train_counts) | set(validation_counts),
        key=lambda value: (-train_counts.get(value, 0), value),
    )
    per_class = [
        {
            "category_id": category,
            "name": names.get(category),
            "train_images": train_counts.get(category, 0),
            "validation_images": validation_counts.get(category, 0),
        }
        for category in categories
    ]
    counts = [row["train_images"] for row in per_class]
    smallest = min(counts) if counts else 0
    return {
        "class_count": len(per_class),
        "train_images_per_class": _distribution(counts),
        # 가장 많은 class가 가장 적은 class의 몇 배인지. 균등화를 논하려면 먼저
        # 이 값이 큰지 봐야 합니다.
        "imbalance_ratio": round(max(counts) / smallest, 2) if smallest else None,
        "classes_missing_from_train": [
            row["category_id"] for row in per_class if row["train_images"] == 0
        ],
        "classes_missing_from_validation": [
            row["category_id"] for row in per_class if row["validation_images"] == 0
        ],
        "per_class": per_class,
    }


def _class_names(class_map: Mapping[str, Any]) -> dict[int, str | None]:
    """class map을 `{번호: 이름}`으로 읽습니다.

    저장소에는 두 모양이 다 있습니다. 지금 것은 `{"250": "이름"}`(번호 → 이름)이고,
    예전 것은 `{"pill": 1}`(이름 → 번호)입니다. 한 모양만 가정하고 `int()`를 부르면
    다른 모양에서 `ValueError`가 `run()` 밖으로 나갑니다.
    """

    names: dict[int, str | None] = {}
    for key, value in class_map.items():
        if isinstance(value, int) and not isinstance(value, bool):
            names[value] = str(key)
            continue
        text = str(key).strip()
        if text.lstrip("-").isdigit():
            names[int(text)] = str(value) if value is not None else None
    return names


def _combination_section(
    train: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    """같은 조합을 여러 각도로 찍은 dataset인지, 그리고 split이 그것을 지켰는지."""

    groups: dict[str, set[str]] = {}
    conditions: Counter = Counter()
    per_split: dict[str, Any] = {}
    for name, manifest in (("train", train), ("validation", validation)):
        names = [str(image.get("file_name", "")) for image in _images(manifest)]
        split_groups = Counter(_group_name(value) for value in names)
        for value in names:
            conditions[_capture_condition(value)] += 1
            groups.setdefault(_group_name(value), set()).add(name)
        per_split[name] = {
            "groups": len(split_groups),
            "images_per_group": _distribution(list(split_groups.values())),
        }
    # 조합 하나가 양쪽에 걸치면 검증 점수가 부풀어 오릅니다. 0이어야 합니다.
    leaked = sorted(name for name, splits in groups.items() if len(splits) > 1)
    return {
        **per_split,
        "groups_in_both_splits": len(leaked),
        "leaked_group_sample": leaked[:5],
        "capture_conditions": dict(conditions.most_common()),
    }


def _appearance_section(
    train_measured: Sequence[Mapping[str, Any]],
    test_measured: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """촬영 부스와 조명이 양쪽에서 같은지 봅니다.

    물체 크기가 같아도 배경색이나 조명이 다르면 model이 보는 그림은 다릅니다.
    크기만 재고 있으면 그 차이를 통째로 놓칩니다.
    """

    def average(rows: Sequence[Mapping[str, Any]], key: str) -> list[float] | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            return None
        return [round(float(np.median([value[index] for value in values])), 2) for index in range(3)]

    section = {
        "train_background_color": average(train_measured, "background_color"),
        "test_background_color": average(test_measured, "background_color"),
        "train_foreground_color": average(train_measured, "foreground_color"),
        "test_foreground_color": average(test_measured, "foreground_color"),
    }
    for what in ("background", "foreground"):
        left = section[f"train_{what}_color"]
        right = section[f"test_{what}_color"]
        # 0~441 척도의 RGB 거리입니다. 눈에 띄는 색 차이는 보통 20을 넘습니다.
        section[f"{what}_color_distance"] = (
            round(float(np.linalg.norm(np.array(left) - np.array(right))), 2)
            if left and right
            else None
        )
    return section


def _size_section(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    train_images: Sequence[Mapping[str, Any]],
    train_measured: Sequence[Mapping[str, Any]],
    test_measured: Sequence[Mapping[str, Any]],
    *,
    has_test: bool,
) -> dict[str, Any]:
    """물체가 얼마나 크게 찍혔는지를 정답과 픽셀 양쪽에서 잽니다.

    train과 validation은 정답 bbox가 있으므로 그대로 잽니다. test는 정답이 없으므로
    픽셀에서 잽니다. 둘을 바로 비교하면 재는 방법이 달라 무엇을 본 것인지 알 수
    없으므로, **같은 픽셀 방법을 train에도 돌려** 나란히 둡니다.

    그리고 그 픽셀 방법이 믿을 만한지를 train에서 먼저 확인합니다(``calibration``).
    정답이 있는 곳에서 정답을 못 맞히는 자로 test를 재고 결론을 내면, 그 결론은
    데이터가 아니라 자의 모양입니다.
    """

    def annotation_fractions(manifest: Mapping[str, Any]) -> dict[Any, float]:
        """이미지마다 정답 bbox가 덮는 넓이의 비율입니다."""

        by_image = _annotations_by_image(manifest)
        fractions: dict[Any, float] = {}
        for image in _images(manifest):
            area = float(image.get("width", 0)) * float(image.get("height", 0))
            if area <= 0:
                continue
            covered = sum(
                _box_area(annotation) or 0.0 for annotation in by_image.get(image["id"], [])
            )
            fractions[image["id"]] = covered / area
        return fractions

    train_truth = annotation_fractions(train)
    section: dict[str, Any] = {
        "train_annotation_fraction": _distribution(list(train_truth.values())),
        "validation_annotation_fraction": _distribution(
            list(annotation_fractions(validation).values())
        ),
    }

    fractions = [row["foreground_fraction"] for row in train_measured]
    paired = [
        (measured, train_truth[image["id"]])
        for image, measured in zip(train_images, fractions)
        if image["id"] in train_truth and train_truth[image["id"]] > 0
    ]
    # 1.0이면 픽셀로 잰 전경이 정답 사각형이 덮는 넓이와 같다는 뜻입니다. 알약은
    # 사각형보다 작고 그림자는 사각형 밖으로 나가므로 정확히 1.0이 되지는 않습니다.
    calibration = (
        round(statistics.median(m) / statistics.median(t), 4)
        if (m := [pair[0] for pair in paired]) and (t := [pair[1] for pair in paired])
        else None
    )
    low, high = CALIBRATION_LIMITS
    trustworthy = calibration is not None and low <= calibration <= high
    section["calibration"] = {
        "images": len(paired),
        "measured_over_annotation": calibration,
        "limits": [low, high],
        "trustworthy": trustworthy,
    }
    section["train_foreground_fraction"] = _distribution(fractions)

    if not has_test:
        section["test_foreground_fraction"] = None
        section["test_over_train"] = None
        return section

    test_fractions = [row["foreground_fraction"] for row in test_measured]
    section["test_foreground_fraction"] = _distribution(test_fractions)
    if not (trustworthy and fractions and test_fractions):
        # 자를 못 믿는데 비율만 적어 두면 그 숫자만 인용됩니다.
        section["test_over_train"] = None
        return section

    area_ratio = statistics.median(test_fractions) / statistics.median(fractions)
    section["test_over_train"] = {
        # 같은 픽셀 방법으로 잰 값끼리의 비율이라 재는 방법의 차이가 지워집니다.
        "area_ratio": round(area_ratio, 4),
        "length_ratio": round(area_ratio**0.5, 4),
    }
    return section


def _box_area(annotation: Mapping[str, Any]) -> float | None:
    box = annotation.get("bbox")
    if not isinstance(box, Sequence) or len(box) != 4:
        return None
    try:
        area = float(box[2]) * float(box[3])
    except (TypeError, ValueError):
        return None
    # 손상된 manifest의 무한대가 분포를 통째로 NaN으로 만듭니다.
    return area if isfinite(area) else None


# --- 실행 -----------------------------------------------------------------


def _required_uri(config: Any, key: str) -> str:
    inputs = config.get("inputs") if isinstance(config, Mapping) else None
    data = inputs.get("data") if isinstance(inputs, Mapping) else None
    value = data.get(key) if isinstance(data, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise EdaError(
            f"EDA를 하려면 config['inputs']['data']['{key}']에 전처리 artifact "
            "URI가 있어야 합니다. 화면에서 전처리 dataset을 먼저 고르세요."
        )
    return value.strip()


def _report_uri(train_manifest_uri: str) -> str:
    """전처리 결과 옆의 ``eda/report.json``을 가리킵니다."""

    directory = posixpath.dirname(train_manifest_uri.replace("\\", "/"))
    return posixpath.join(directory, REPORT_DIRECTORY, REPORT_NAME)


def _sample_size(config: Any) -> int:
    value = _data_config(config).get("eda_image_sample", DEFAULT_IMAGE_SAMPLE)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EdaError("config['data']['eda_image_sample']은 1 이상의 정수여야 합니다.")
    return value


def _overwrite(config: Any) -> bool:
    value = _data_config(config).get("overwrite", False)
    if not isinstance(value, bool):
        raise EdaError("config['data']['overwrite']는 true 또는 false여야 합니다.")
    return value


def _error_result(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "artifacts": {},
        "summary": {"pipeline": "data", "mode": "eda"},
        "message": message,
    }


def run_eda(config: dict, *, progress: ProgressEmitter | None = None) -> dict:
    """고른 전처리 dataset을 model 없이 뜯어보고 리포트 하나를 남깁니다."""

    emitter = ProgressEmitter() if progress is None else progress
    emitter.emit("step_started", step="eda")
    try:
        return _run_eda(config, emitter)
    except EdaError as error:
        return _error_result(str(error))
    except StorageError as error:
        # 원문에는 사용자 이름이 든 절대 경로가 들어 있습니다. type만 남깁니다.
        return _error_result(
            f"EDA 중 저장소 접근에 실패했습니다({type(error).__name__}). "
            "고른 dataset의 artifact가 그 자리에 있는지 확인하세요."
        )


def _run_eda(config: dict, progress: ProgressEmitter) -> dict:
    train_uri = _required_uri(config, "train_manifest_uri")
    validation_uri = _required_uri(config, "validation_manifest_uri")
    class_map_uri = _required_uri(config, "class_map_uri")
    sample = _sample_size(config)
    overwrite = _overwrite(config)

    storage = create_storage(config if isinstance(config, Mapping) else {})
    # artifact URI는 저장소 root 기준이고 storage는 자기 root 기준으로 풉니다.
    report_uri = _report_uri(train_uri)
    if not overwrite and storage.exists(_artifact_location(report_uri)):
        raise EdaError(
            "이미 EDA 리포트가 있습니다. 다시 만들려면 data.overwrite를 켜세요: "
            f"{posixpath.basename(report_uri)}"
        )

    train = storage.read_json(_artifact_location(train_uri))
    validation = storage.read_json(_artifact_location(validation_uri))
    class_map = storage.read_json(_artifact_location(class_map_uri))
    for document in (train, validation):
        # 화면은 이 두 배열을 보고 manifest라고 판단하지만, config를 손으로 쓰면
        # 어떤 문서든 올 수 있습니다. 배열이 아니면 순회에서 TypeError가 납니다.
        if (
            not isinstance(document, Mapping)
            or not isinstance(document.get("images"), list)
            or not isinstance(document.get("annotations"), list)
        ):
            raise EdaError(
                "manifest에 images와 annotations 배열이 있어야 합니다. 고른 파일이 "
                "전처리 manifest가 맞는지 확인하세요."
            )
    if not isinstance(class_map, Mapping):
        raise EdaError("class map을 읽었지만 JSON 객체가 아닙니다.")

    inputs = config.get("inputs") or {}
    test_uri = (inputs.get("data") or {}).get("test_manifest_uri")
    test = None
    if isinstance(test_uri, str) and test_uri.strip():
        loaded = storage.read_json(_artifact_location(test_uri.strip()))
        # test는 image 목록만 씁니다. 그 배열이 없으면 없는 것으로 둡니다.
        test = (
            loaded
            if isinstance(loaded, Mapping) and isinstance(loaded.get("images"), list)
            else None
        )

    # 이미지를 열기 전에 확인합니다. 몇 분 걸린 뒤에 거절하면 그 시간이 버려집니다.
    check_manifest(train, "학습")
    check_manifest(validation, "검증")
    if test is not None:
        # test는 **image 목록만** 봅니다. annotation을 확인에 쓰면 그 값이 실행의
        # 성패를 가르게 되고, 그것은 "대회 test annotation을 쓰지 않는다"는 규칙과
        # 리포트가 적는 `test_annotations_used: false`를 함께 어깁니다.
        _images(test)

    # 이미지는 한 번만 열고 크기와 색을 함께 잽니다. 다시 열면 시간이 두 배입니다.
    on_progress = progress.read_progress
    train_images = _sample(_images(train), sample)
    train_measured = _measure_images(
        storage,
        [_image_location(train_uri, image["file_name"]) for image in train_images],
        stage="train_pixels",
        on_progress=on_progress,
    )
    test_measured = (
        _measure_images(
            storage,
            [
                _image_location(str(test_uri), image["file_name"])
                for image in _images(test)
            ],
            stage="test_pixels",
            on_progress=on_progress,
        )
        if test is not None
        else []
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_directory": posixpath.dirname(train_uri.replace("\\", "/")),
        "shape": _shape_section(train, validation),
        "classes": _class_section(train, validation, class_map),
        "combinations": _combination_section(train, validation),
        "object_size": _size_section(
            train,
            validation,
            train_images,
            train_measured,
            test_measured,
            has_test=test is not None,
        ),
        "appearance": _appearance_section(train_measured, test_measured),
        # 무엇을 읽었는지 남깁니다. 대회 test 이미지는 픽셀만 읽고 annotation은
        # 읽지 않으며, 여기서 잰 값은 학습이나 전처리 결정에 들어가지 않습니다.
        "sources": {
            "train_manifest_uri": train_uri,
            "validation_manifest_uri": validation_uri,
            "class_map_uri": class_map_uri,
            "test_manifest_uri": test_uri if test is not None else None,
            # manifest 파일 자체는 통째로 읽습니다. 거짓말하지 않으려고, 거기에
            # annotation이 몇 개 들어 있었는지와 그것을 **쓰지 않았다**는 사실을
            # 따로 적습니다. 이 리포트 어디에도 test annotation에서 나온 값은 없습니다.
            "test_annotations_in_manifest": (
                len(test.get("annotations") or []) if test is not None else None
            ),
            "test_annotations_used": False,
            "train_image_sample": sample,
        },
    }
    storage.write_json(_artifact_location(report_uri), report, overwrite=overwrite)

    size = report["object_size"]
    ratio = (size.get("test_over_train") or {}).get("length_ratio")
    appearance = report["appearance"]
    # 받은 dataset artifact를 그대로 다시 공개합니다. EDA는 아무것도 바꾸지 않지만,
    # main_pipeline은 성공한 data stage가 그 URI들을 냈는지 확인하므로 빼면 멈춥니다.
    artifacts = {
        key: value
        for key, value in (config.get("inputs") or {}).get("data", {}).items()
        if isinstance(value, str) and value.strip()
    }
    return {
        "status": "ok",
        "artifacts": {**artifacts, "eda_report_uri": report_uri},
        "summary": {
            "pipeline": "data",
            "mode": "eda",
            "class_count": report["classes"]["class_count"],
            "imbalance_ratio": report["classes"]["imbalance_ratio"],
            "groups_in_both_splits": report["combinations"]["groups_in_both_splits"],
            "test_over_train_length_ratio": ratio,
            "background_color_distance": appearance["background_color_distance"],
        },
        "message": f"EDA 리포트를 만들었습니다: {report_uri}",
    }
