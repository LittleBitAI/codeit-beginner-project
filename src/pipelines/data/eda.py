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
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.common import Storage, StorageError, create_storage

from .errors import EdaError
from .progress import ProgressEmitter


__all__ = ["eda_requested", "foreground_fraction", "run_eda"]


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


def foreground_fraction(image: Image.Image, *, downscale: int = DOWNSCALE) -> float:
    """이미지에서 물체가 차지하는 넓이의 비율입니다. 0과 1 사이입니다.

    annotation을 쓰지 않으므로 정답이 없는 test 이미지에도 그대로 씁니다.

    **물체를 하나씩 세지 않습니다.** 세려면 붙어 있는 픽셀을 덩어리로 묶어야 하는데,
    알약마다 딸린 그림자와 반사 테두리가 얇은 틈을 두고 떨어져 나와 하나가 둘로도
    셋으로도 됩니다. 실제로 재어 보니 무엇을 한 덩어리로 볼지에 따라 결과가 절반씩
    오갔습니다. 한 장에 든 알약 수가 같은 dataset이라면 넓이 비율만으로 "물체가
    더 크게 찍혔는가"를 답할 수 있고, 그 값은 나누는 방식에 흔들리지 않습니다.

    밝기가 아니라 **배경과 색이 얼마나 다른지**로 가릅니다. 밝기로 가르면 흰 알약이
    밝은 촬영 부스와 같은 값이 되어 통째로 사라집니다. 배경색은 가장자리 픽셀에서
    얻습니다 — 알약이 액자 밖까지 나가지는 않으므로 거기는 늘 배경입니다.
    """

    scale = max(1, int(downscale))
    width, height = image.size
    small = image.convert("RGB").resize(
        (max(1, width // scale), max(1, height // scale)), Image.BILINEAR
    )
    pixels = np.asarray(small, dtype=np.float32)
    distance = _background_distance(pixels)
    mask = _closed(distance > _otsu_threshold(distance))
    return float(mask.mean())


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
) -> list[float]:
    """이미지를 한 장씩 받아 재고 곧바로 지웁니다. 원본을 남기지 않습니다."""

    measured: list[float] = []
    for done, location in enumerate(locations, start=1):
        with tempfile.TemporaryDirectory(prefix="eda-") as scratch:
            local = Path(scratch) / posixpath.basename(location.replace("\\", "/"))
            try:
                storage.download_file(location, local)
                with Image.open(local) as opened:
                    opened.load()
                    measured.append(foreground_fraction(opened))
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

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
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
    grouped: defaultdict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in manifest.get("annotations") or []:
        if isinstance(annotation, Mapping):
            grouped[annotation.get("image_id")].append(annotation)
    return dict(grouped)


def _images(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [image for image in (manifest.get("images") or []) if isinstance(image, Mapping)]


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
    categories = sorted(
        {int(key) for key in class_map} | set(train_counts) | set(validation_counts),
        key=lambda value: (-train_counts.get(value, 0), value),
    )
    per_class = [
        {
            "category_id": category,
            "name": class_map.get(str(category)),
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


def _size_section(
    storage: Storage,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any] | None,
    *,
    sample: int,
    on_progress: Any | None,
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

    train_images = _sample(_images(train), sample)
    train_measured = _measure_images(
        storage,
        [str(image["file_name"]) for image in train_images],
        stage="train_pixels",
        on_progress=on_progress,
    )
    paired = [
        (measured, train_truth[image["id"]])
        for image, measured in zip(train_images, train_measured)
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
    section["train_foreground_fraction"] = _distribution(train_measured)

    if test is None:
        section["test_foreground_fraction"] = None
        section["test_over_train"] = None
        return section

    test_measured = _measure_images(
        storage,
        [str(image["file_name"]) for image in _images(test)],
        stage="test_pixels",
        on_progress=on_progress,
    )
    section["test_foreground_fraction"] = _distribution(test_measured)
    if not (trustworthy and train_measured and test_measured):
        # 자를 못 믿는데 비율만 적어 두면 그 숫자만 인용됩니다.
        section["test_over_train"] = None
        return section

    area_ratio = statistics.median(test_measured) / statistics.median(train_measured)
    section["test_over_train"] = {
        # 같은 픽셀 방법으로 잰 값끼리의 비율이라 재는 방법의 차이가 지워집니다.
        "area_ratio": round(area_ratio, 4),
        "length_ratio": round(area_ratio**0.5, 4),
    }
    return section


def _box_area(annotation: Mapping[str, Any]) -> float | None:
    box = annotation.get("bbox")
    if isinstance(box, Sequence) and len(box) == 4:
        return float(box[2]) * float(box[3])
    return None


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
        return _error_result(f"EDA 중 저장소 접근에 실패했습니다: {error}")


def _run_eda(config: dict, progress: ProgressEmitter) -> dict:
    train_uri = _required_uri(config, "train_manifest_uri")
    validation_uri = _required_uri(config, "validation_manifest_uri")
    class_map_uri = _required_uri(config, "class_map_uri")
    sample = _sample_size(config)
    overwrite = _overwrite(config)

    storage = create_storage(config if isinstance(config, Mapping) else {})
    report_uri = _report_uri(train_uri)
    if not overwrite and storage.exists(report_uri):
        raise EdaError(
            "이미 EDA 리포트가 있습니다. 다시 만들려면 data.overwrite를 켜세요: "
            f"{posixpath.basename(report_uri)}"
        )

    train = storage.read_json(train_uri)
    validation = storage.read_json(validation_uri)
    class_map = storage.read_json(class_map_uri)
    if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
        raise EdaError("manifest를 읽었지만 JSON 객체가 아닙니다.")
    if not isinstance(class_map, Mapping):
        raise EdaError("class map을 읽었지만 JSON 객체가 아닙니다.")

    inputs = config.get("inputs") or {}
    test_uri = (inputs.get("data") or {}).get("test_manifest_uri")
    test = None
    if isinstance(test_uri, str) and test_uri.strip():
        loaded = storage.read_json(test_uri.strip())
        test = loaded if isinstance(loaded, Mapping) else None

    on_progress = progress.read_progress
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_directory": posixpath.dirname(train_uri.replace("\\", "/")),
        "shape": _shape_section(train, validation),
        "classes": _class_section(train, validation, class_map),
        "combinations": _combination_section(train, validation),
        "object_size": _size_section(
            storage, train, validation, test, sample=sample, on_progress=on_progress
        ),
        # 무엇을 읽었는지 남깁니다. 대회 test 이미지는 픽셀만 읽고 annotation은
        # 읽지 않으며, 여기서 잰 값은 학습이나 전처리 결정에 들어가지 않습니다.
        "sources": {
            "train_manifest_uri": train_uri,
            "validation_manifest_uri": validation_uri,
            "class_map_uri": class_map_uri,
            "test_manifest_uri": test_uri if test is not None else None,
            "test_annotations_read": False,
            "train_image_sample": sample,
        },
    }
    storage.write_json(report_uri, report, overwrite=overwrite)

    size = report["object_size"]
    ratio = (size.get("test_over_train") or {}).get("length_ratio")
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
        },
        "message": f"EDA 리포트를 만들었습니다: {report_uri}",
    }
