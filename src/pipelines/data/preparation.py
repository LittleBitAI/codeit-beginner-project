"""S3 원본에서 학습·평가용 artifact 5개를 만들어 storage에 저장합니다.

`config["data"]["prepare"]`가 `true`일 때만 동작하는 준비 경로입니다. 원본
`train_images/`와 `train_annotations/`로 train/validation을 만들되, 같은 알약
조합을 찍은 사진이 양쪽 split에 나뉘지 않도록 file 이름 접두사를 그룹으로 삼아
그룹 단위로 나눕니다(`split.py`). 그리고 `test_images/`의 크기만 decode해 아래
다섯 file을 저장합니다. 이전 버전이 만든 네 artifact만 정확히 남아 있으면 그
파일들은 보존하고 test manifest만 보충합니다.

- `train_manifest.json`, `validation_manifest.json`: COCO 형식 manifest
- `test_manifest.json`: annotation이 없는 COCO 형식 test manifest
- `class_map.json`: `{"<category id>": "<category name>"}`
- `dataset_summary.json`: 원본 위치, 분할 방식과 그룹 수, 비율, seed, 분포와
  split manifest의 sha256을 담은 요약

Storage 접근은 `src/common/storage.py`의 `create_storage(config)`로만 합니다.
개별 pipeline은 `boto3`를 직접 쓰지 않습니다. competition 평가용 `test_images/`는
test manifest에만 기록하고 어떤 학습 split에도 넣지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common import LocalStorage, Storage, StorageError, create_storage

from .coco import ConsolidatedDataset, consolidate
from .crop_bank import CROP_BANK_FILE_NAME, DEFAULT_PER_CLASS, build_crop_bank
from .errors import DatasetPreparationError
from .progress import ProgressEmitter
from .similarity import measure_validation_similarity
from .split import AngleRule, GroupRule, SplitResult, hold_out_angle, split_images
from .test_manifest import build_test_manifest


__all__ = [
    "ARTIFACT_FILE_NAMES",
    "DEFAULT_SEED",
    "LocationPublisher",
    "PreparationSettings",
    "REPOSITORY_ROOT",
    "SPLIT_METHOD_OPTIONS",
    "SPLIT_RATIO_OPTIONS",
    "build_publisher",
    "preparation_error_result",
    "preparation_requested",
    "resolve_settings",
    "run_preparation",
]


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# 원본과 산출물의 기본 위치입니다. config로 바꿀 수 있지만 두 값 모두
# 공용 logical prefix인 `datasets/` 아래여야 합니다.
DEFAULT_RAW_PREFIX = "datasets/pill_detection/raw/v1/"
DEFAULT_PROCESSED_ROOT = "datasets/pill_detection/processed/"
DEFAULT_SEED = 42
DATASET_PREFIX = "datasets/"
# 산출물 directory 이름에 쓸 dataset 버전 segment입니다: `v1`, `v2`, …
_VERSION_SEGMENT = re.compile(r"v\d+")

# 이번 작업의 핵심: validation 비율은 아래 두 값 중 하나만 허용합니다.
SPLIT_RATIO_OPTIONS: dict[str, float] = {"8:2": 0.2, "9:1": 0.1}
# 비율이 다르면 산출물도 다르므로 저장 directory 이름에 비율을 드러냅니다.
_RATIO_DIRECTORY_TOKENS: dict[str, str] = {"8:2": "8020", "9:1": "9010"}

# 같은 알약 조합을 각도와 조명만 바꿔 여러 장 찍은 원본이라, 이미지 한 장씩
# 나누면 거의 같은 사진이 train과 validation 양쪽에 들어갑니다. 그래서 기본은
# 그룹 분할입니다. `"image"`는 예전 이미지 단위 분할이며, 이미 만들어 둔 산출물을
# 다시 만들 때만 씁니다.
SPLIT_METHOD_OPTIONS = ("group", "image", "group-angle")
DEFAULT_SPLIT_METHOD = "group"
# 그룹 이름은 file 이름의 첫 `_` 앞부분, 즉 알약 조합 코드입니다.
# 예: K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
#     -> K-001900-016548-019607-029451
# 원본 이름 규칙이 바뀌면 이 두 값을 바꿉니다. 파일별로 예외를 두지 않습니다.
GROUP_KEY_DELIMITER = "_"
GROUP_KEY_TOKENS = 1
# 촬영 각도는 file 이름의 끝에서 세 번째 토큰입니다.
# 예: K-001900-016548_0_2_0_2_70_000_200 -> 70
# 조합 코드 뒤의 토큰 수는 원본 판마다 다르지만 뒤쪽 세 개는 같습니다.
ANGLE_KEY_DELIMITER = "_"
ANGLE_KEY_POSITION_FROM_END = 3
# `group-angle`에서 validation 전용으로 뗄 각도입니다.
DEFAULT_VALIDATION_ANGLE = "90"
# 분할 방식이 다르면 내용이 완전히 다른 dataset이므로 directory 이름으로
# 구분합니다. 이미지 분할은 이름을 그대로 두어 기존 산출물을 덮지 않습니다.
_METHOD_DIRECTORY_TOKENS: dict[str, str] = {
    "group": "-group",
    "image": "",
    "group-angle": "-group-angle",
}

ARTIFACT_FILE_NAMES: dict[str, str] = {
    "train_manifest_uri": "train_manifest.json",
    "validation_manifest_uri": "validation_manifest.json",
    "class_map_uri": "class_map.json",
    "dataset_summary_uri": "dataset_summary.json",
    "test_manifest_uri": "test_manifest.json",
}
# 켠 실행에서만 나오는 선택 artifact라 위 표에 넣지 않습니다. 저 표는 "언제나
# 나오는 다섯"이라는 뜻이고, test가 그 뜻으로 개수를 셉니다.
CROP_BANK_ARTIFACT_KEY = "crop_bank_uri"
LEGACY_ARTIFACT_KEYS = frozenset(
    {
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
        "dataset_summary_uri",
    }
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TRAIN_IMAGE_MARKER = "train_images/"
TRAIN_ANNOTATION_MARKER = "train_annotations/"
TEST_IMAGE_MARKER = "test_images/"
# competition 평가용 원본입니다. 학습·검증 어디에도 들어가면 안 됩니다.
FORBIDDEN_MARKERS = ("test_images/", "test_annotations/")
MAX_READ_WORKERS = 16
# 1.1에서 `split.checksums`가, 1.2에서 `split.grouping`, `split.strategy`,
# `split.validation_image_ratio`가 추가됐습니다. 1.2에서 `split.method`는 분할
# 방식("group"/"image")을 뜻하고, 1.1까지 그 자리에 있던 알고리즘 이름은
# `split.strategy`로 옮겼습니다. 1.3에서 `split.train_only_categories`가
# 추가됐습니다. 나머지 key는 그대로입니다.
# 1.4에서 split.validation_similarity가 생겼습니다. 재지 않은 실행에는 없습니다.
SUMMARY_SCHEMA_VERSION = "1.4"
CHECKSUM_ALGORITHM = "sha256"
# hash를 남길 split 산출물입니다. 순서가 요약에 그대로 드러납니다.
SPLIT_CHECKSUM_KEYS: tuple[tuple[str, str], ...] = (
    ("train", "train_manifest_uri"),
    ("validation", "validation_manifest_uri"),
)


@dataclass(frozen=True)
class PreparationSettings:
    """준비 실행에 필요한 config 값입니다."""

    split_ratio: str
    validation_ratio: float
    seed: int
    raw_prefix: str
    processed_prefix: str
    overwrite: bool
    split_method: str
    group_rule: GroupRule | None
    # `group-angle`일 때만 값이 있습니다. 그 각도는 train에서 통째로 빠집니다.
    angle_rule: AngleRule | None
    validation_angle: str | None
    # 켜면 이미지를 전부 열어 validation이 train과 얼마나 비슷한지 잽니다. 준비
    # 시간이 크게 늘어나므로 기본값은 꺼짐입니다.
    measure_validation_similarity: bool
    # 켜면 train 정답 상자를 잘라 참조 crop 은행을 함께 만듭니다. 이것도 이미지를
    # 여는 단계라 기본값은 꺼짐입니다.
    crop_bank: bool
    crop_bank_per_class: int


def _data_config(config: Any) -> Mapping[str, Any]:
    """`config["data"]`를 읽습니다. 없으면 빈 설정으로 봅니다."""

    if not isinstance(config, Mapping):
        return {}
    section = config.get("data")
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise DatasetPreparationError("config['data']는 object여야 합니다.")
    return section


def preparation_requested(config: Any) -> bool:
    """준비 경로를 실행할지 확인합니다. 명시적으로 켰을 때만 참입니다."""

    value = _data_config(config).get("prepare", False)
    if not isinstance(value, bool):
        raise DatasetPreparationError(
            "config['data']['prepare']는 true 또는 false여야 합니다."
        )
    return value


def _normalized_prefix(value: Any, key: str, default: str) -> str:
    """Storage prefix를 검증하고 `datasets/...` 형태로 정규화합니다."""

    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise DatasetPreparationError(
            f"config['data']['{key}']는 비어 있지 않은 문자열이어야 합니다."
        )
    prefix = value.strip().replace("\\", "/").lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    if any(part in {"", ".", ".."} for part in prefix.rstrip("/").split("/")):
        raise DatasetPreparationError(
            f"config['data']['{key}']에는 빈 segment나 '.', '..'를 쓸 수 없습니다."
        )
    if not prefix.startswith(DATASET_PREFIX):
        raise DatasetPreparationError(
            f"config['data']['{key}']는 '{DATASET_PREFIX}'로 시작해야 합니다."
        )
    if any(marker in prefix for marker in FORBIDDEN_MARKERS):
        raise DatasetPreparationError(
            f"config['data']['{key}']에는 competition 평가용 test 경로를 쓸 수 없습니다."
        )
    return prefix


def _split_ratio(section: Mapping[str, Any]) -> str:
    """허용된 두 비율 중 하나인지 확인합니다."""

    allowed = ", ".join(f'"{option}"' for option in SPLIT_RATIO_OPTIONS)
    value = section.get("split_ratio")
    if not isinstance(value, str) or value.strip() not in SPLIT_RATIO_OPTIONS:
        raise DatasetPreparationError(
            f"config['data']['split_ratio']는 {allowed} 중 하나여야 합니다. "
            "다른 비율은 지원하지 않습니다."
        )
    return value.strip()


def _seed(section: Mapping[str, Any]) -> int:
    value = section.get("seed", DEFAULT_SEED)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
        raise DatasetPreparationError(
            "config['data']['seed']는 0 이상 2**32 미만의 정수여야 합니다."
        )
    return value


def _overwrite(section: Mapping[str, Any]) -> bool:
    value = section.get("overwrite", False)
    if not isinstance(value, bool):
        raise DatasetPreparationError(
            "config['data']['overwrite']는 true 또는 false여야 합니다."
        )
    return value


def _split_method(section: Mapping[str, Any]) -> str:
    """분할 방식을 읽습니다. 기본은 그룹 분할입니다."""

    value = section.get("split_method", DEFAULT_SPLIT_METHOD)
    if not isinstance(value, str) or value.strip() not in SPLIT_METHOD_OPTIONS:
        allowed = ", ".join(f'"{option}"' for option in SPLIT_METHOD_OPTIONS)
        raise DatasetPreparationError(
            f"config['data']['split_method']는 {allowed} 중 하나여야 합니다."
        )
    return value.strip()


def _angle_rule(split_method: str) -> AngleRule | None:
    """`group-angle`에서만 각도 규칙을 씁니다."""

    if split_method != "group-angle":
        return None
    return AngleRule(
        delimiter=ANGLE_KEY_DELIMITER,
        position_from_end=ANGLE_KEY_POSITION_FROM_END,
    )


def _validation_angle(section: Mapping[str, Any], split_method: str) -> str | None:
    """validation 전용으로 뗄 각도를 읽습니다."""

    if split_method != "group-angle":
        return None
    value = section.get("validation_angle", DEFAULT_VALIDATION_ANGLE)
    if not isinstance(value, str) or not value.strip():
        raise DatasetPreparationError(
            "config['data']['validation_angle']은 비어 있지 않은 문자열이어야 합니다."
        )
    return value.strip()


def _group_rule(split_method: str) -> GroupRule | None:
    """그룹 이름을 뽑는 규칙을 정합니다.

    규칙은 file 이름에서 유도하며 config로 바꾸지 않습니다. 이미지 분할에는
    그룹 개념이 없으므로 `None`이고, 이때는 이미지 한 장이 곧 한 그룹입니다.
    """

    if split_method == "image":
        return None
    return GroupRule(delimiter=GROUP_KEY_DELIMITER, tokens=GROUP_KEY_TOKENS)


def _dataset_version(raw_prefix: str) -> str:
    """원본 prefix에서 dataset 버전 segment(`v1`, `v2`, …)를 읽습니다.

    산출물 directory 이름에 들어가는 값입니다. 여기서 원본을 구분하지 못하면 서로
    다른 원본으로 만든 산출물이 같은 directory를 노리고, `overwrite`를 켠 실행이
    남의 산출물을 덮습니다. `raw/v2/original/`처럼 뒤에 segment가 더 붙어도 같은
    버전으로 봅니다.
    """

    for part in raw_prefix.strip("/").split("/"):
        if _VERSION_SEGMENT.fullmatch(part):
            return part
    raise DatasetPreparationError(
        "config['data']['raw_prefix']에서 dataset 버전을 읽지 못했습니다. "
        f"'v1'처럼 v와 숫자로 된 segment가 있어야 합니다: '{raw_prefix}'"
    )


def resolve_settings(config: Any) -> PreparationSettings:
    """준비 경로 설정을 읽고 검증합니다."""

    section = _data_config(config)
    split_ratio = _split_ratio(section)
    split_method = _split_method(section)
    group_rule = _group_rule(split_method)
    angle_rule = _angle_rule(split_method)
    validation_angle = _validation_angle(section, split_method)
    measure_similarity = section.get("measure_validation_similarity", False)
    if not isinstance(measure_similarity, bool):
        raise DatasetPreparationError(
            "config['data']['measure_validation_similarity']는 true 또는 false여야 합니다."
        )
    crop_bank = section.get("crop_bank", False)
    if not isinstance(crop_bank, bool):
        raise DatasetPreparationError(
            "config['data']['crop_bank']는 true 또는 false여야 합니다."
        )
    per_class = section.get("crop_bank_per_class", DEFAULT_PER_CLASS)
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1:
        raise DatasetPreparationError(
            "config['data']['crop_bank_per_class']는 1 이상의 정수여야 합니다."
        )
    seed = _seed(section)
    raw_prefix = _normalized_prefix(section.get("raw_prefix"), "raw_prefix", DEFAULT_RAW_PREFIX)
    processed_root = _normalized_prefix(
        section.get("processed_root"), "processed_root", DEFAULT_PROCESSED_ROOT
    )
    # 원본 버전과 비율, seed, 분할 방식이 directory 이름에 들어가므로 서로 다른
    # 원본이나 설정으로 만든 산출물은 덮어쓰지 않고 함께 남습니다.
    processed_prefix = (
        f"{processed_root}{_dataset_version(raw_prefix)}-seed{seed}"
        f"-{_RATIO_DIRECTORY_TOKENS[split_ratio]}"
        f"{_METHOD_DIRECTORY_TOKENS[split_method]}/"
    )
    return PreparationSettings(
        split_ratio=split_ratio,
        validation_ratio=SPLIT_RATIO_OPTIONS[split_ratio],
        seed=seed,
        raw_prefix=raw_prefix,
        processed_prefix=processed_prefix,
        overwrite=_overwrite(section),
        split_method=split_method,
        measure_validation_similarity=measure_similarity,
        crop_bank=crop_bank,
        crop_bank_per_class=per_class,
        group_rule=group_rule,
        angle_rule=angle_rule,
        validation_angle=validation_angle,
    )


def _normalized(location: Any) -> str:
    return str(location).replace("\\", "/")


def _select_test_images(entries: Sequence[str]) -> list[str]:
    """나열된 원본 중 test image만 고릅니다. test annotation은 읽지 않습니다."""

    return [
        entry
        for entry in entries
        if TEST_IMAGE_MARKER in _normalized(entry)
        and _normalized(entry).lower().endswith(IMAGE_EXTENSIONS)
    ]


def _raw_objects(
    storage: Storage, raw_prefix: str
) -> tuple[list[str], list[str], list[str]]:
    """원본 prefix에서 train/test 이미지와 train annotation 위치를 고릅니다.

    `test_annotations/`는 어떤 경우에도 읽을 대상에 넣지 않습니다.
    """

    entries = sorted(str(entry) for entry in storage.list(raw_prefix))
    allowed = [
        entry
        for entry in entries
        if not any(marker in _normalized(entry) for marker in FORBIDDEN_MARKERS)
    ]
    image_locations = [
        entry
        for entry in allowed
        if TRAIN_IMAGE_MARKER in _normalized(entry)
        and _normalized(entry).lower().endswith(IMAGE_EXTENSIONS)
    ]
    annotation_locations = [
        entry
        for entry in allowed
        if TRAIN_ANNOTATION_MARKER in _normalized(entry)
        and _normalized(entry).lower().endswith(".json")
    ]
    test_image_locations = _select_test_images(entries)
    if not image_locations or not annotation_locations or not test_image_locations:
        raise DatasetPreparationError(
            "원본 prefix에서 train_images, train_annotations, test_images를 모두 "
            "찾지 못했습니다. "
            f"(train image {len(image_locations)}개, annotation "
            f"{len(annotation_locations)}개, test image {len(test_image_locations)}개)"
        )
    return image_locations, annotation_locations, test_image_locations


def _read_documents(
    storage: Storage,
    annotation_locations: Sequence[str],
    progress: ProgressEmitter,
) -> list[tuple[str, Any]]:
    """이미지별 annotation 문서를 순서를 유지한 채 읽습니다.

    가장 오래 걸리는 구간이라 완료 개수를 세어 진행 로그로 알립니다. 결과는
    제출한 순서 그대로 모으고, 세는 일만 완료 순서로 합니다.
    """

    total = len(annotation_locations)
    workers = max(1, min(MAX_READ_WORKERS, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(storage.read_json, location)
            for location in annotation_locations
        ]
        for done, _ in enumerate(as_completed(futures), start=1):
            progress.read_progress("annotations", done, total)
        documents = [future.result() for future in futures]
    return list(zip(annotation_locations, documents))


def _is_remote(location: Any) -> bool:
    return str(location).lower().startswith("s3://")


@dataclass(frozen=True)
class LocationPublisher:
    """Storage가 돌려준 위치를 소비자가 쓸 수 있는 URI로 바꿉니다.

    S3 backend가 돌려주는 `s3://` URI는 그대로 씁니다. 반면 local backend의
    `list`와 `write_json`은 storage root 기준으로 resolve한 **절대 경로**를
    돌려주는데, 절대 경로와 Windows drive 경로는 다음 소비자가 계약 위반으로
    거부하고 다른 컴퓨터에서도 쓸 수 없습니다. 그래서 local backend에서는

    - artifact URI를 **저장소 root 기준** 상대 POSIX 경로로,
    - manifest 안 이미지 경로를 **manifest 자신의 위치 기준** 상대 POSIX 경로로

    바꿔서 내보냅니다.
    """

    storage_root: Path | None
    manifest_directory: Path | None

    def _absolute(self, location: str) -> Path:
        candidate = Path(location)
        if candidate.is_absolute() or self.storage_root is None:
            return candidate
        return (self.storage_root / candidate).resolve()

    def artifact_uri(self, location: str) -> str:
        """다음 pipeline에 넘길 artifact URI를 만듭니다."""

        text = str(location)
        if _is_remote(text) or self.storage_root is None:
            return text
        path = self._absolute(text)
        try:
            return path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError as error:
            raise DatasetPreparationError(
                "local artifact는 저장소 root 기준 상대 경로여야 하는데 산출물이 "
                "저장소 밖에 있습니다. storage.local.root를 저장소 안 경로로 "
                "설정하세요."
            ) from error

    def image_file_name(self, location: str) -> str:
        """manifest에 적을 이미지 경로를 만듭니다."""

        text = str(location)
        if _is_remote(text) or self.manifest_directory is None:
            return text
        relative = os.path.relpath(self._absolute(text), self.manifest_directory)
        return relative.replace("\\", "/")


def build_publisher(storage: Storage, settings: PreparationSettings) -> LocationPublisher:
    """Backend에 맞는 URI 변환기를 만듭니다. 쓰기 전에 먼저 확인합니다."""

    if not isinstance(storage, LocalStorage):
        return LocationPublisher(storage_root=None, manifest_directory=None)

    root = Path(storage.root).resolve()
    try:
        root.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise DatasetPreparationError(
            "local backend의 storage root는 저장소 안에 있어야 합니다. 소비자는 "
            "저장소 root 기준 상대 경로 URI만 받기 때문입니다. "
            "storage.local.root 또는 PILL_STORAGE_LOCAL_ROOT를 저장소 안 경로로 "
            "설정하세요."
        ) from error
    return LocationPublisher(
        storage_root=root,
        manifest_directory=(root / settings.processed_prefix).resolve(),
    )


def _manifest(
    split: str,
    image_ids: set[int],
    dataset: ConsolidatedDataset,
    settings: PreparationSettings,
    publisher: LocationPublisher,
) -> dict[str, Any]:
    """train이 그대로 읽을 수 있는 COCO manifest를 만듭니다.

    `file_name`은 소비자가 manifest 위치를 기준으로 푸는 값입니다. manifest는
    `processed/` 아래, 이미지는 `raw/` 아래에 있으므로 단순한 파일 이름만 적으면
    엉뚱한 위치를 가리킵니다. S3에서는 `s3://` URI를, local에서는 manifest
    directory 기준 상대 경로(예: `../../raw/v1/train_images/img_001.jpg`)를
    적어서 두 backend 모두에서 이식 가능하게 만듭니다.
    """

    return {
        "info": {
            "description": "Pill detection processed COCO manifest",
            "split": split,
            "seed": settings.seed,
            "split_ratio": settings.split_ratio,
            "validation_ratio": settings.validation_ratio,
        },
        "images": [
            {**image, "file_name": publisher.image_file_name(image["file_name"])}
            for image in dataset.images
            if image["id"] in image_ids
        ],
        "annotations": [
            annotation
            for annotation in dataset.annotations
            if annotation["image_id"] in image_ids
        ],
        "categories": list(dataset.categories),
    }


def _stored_json_bytes(value: Any) -> bytes:
    """Storage가 JSON artifact를 저장할 때 만드는 byte와 같은 값을 만듭니다.

    `LocalStorage`와 `S3Storage`가 쓰는 직렬화(`ensure_ascii=False`, `indent=2`,
    끝 줄바꿈 한 개)를 그대로 따릅니다. 그래야 저장된 file을 `sha256sum`으로
    확인한 값과 요약에 적은 hash가 같습니다.
    """

    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _split_checksums(manifests: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """train/validation manifest의 sha256과 byte 크기를 남깁니다.

    seed와 비율을 고정해도 원본이 바뀌면 split은 달라집니다. 어떤 split으로
    학습했는지 나중에 확인할 수 있는 기록은 이 hash뿐이라, 요약에 같이 남깁니다.
    """

    checksums: dict[str, Any] = {"algorithm": CHECKSUM_ALGORITHM}
    for split, artifact_key in SPLIT_CHECKSUM_KEYS:
        body = _stored_json_bytes(manifests[split])
        checksums[ARTIFACT_FILE_NAMES[artifact_key]] = {
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
        }
    return checksums


def _grouping_summary(
    group_rule: GroupRule | None, split_result: SplitResult
) -> dict[str, Any] | None:
    """어떤 규칙으로 몇 개의 그룹을 나눴는지 남깁니다.

    나중에 "이 모델은 어떤 데이터로 학습했나"를 요약만 보고 답할 수 있어야
    합니다. directory 이름은 방식까지만 알려 주므로, 규칙과 그룹 수는 여기에
    남깁니다. 이미지 분할에는 그룹 개념이 없으므로 `null`입니다.
    """

    if group_rule is None:
        return None
    return {
        "delimiter": group_rule.delimiter,
        "tokens": group_rule.tokens,
        "group_count": split_result.group_count,
        "train_groups": split_result.train_group_count,
        "validation_groups": split_result.validation_group_count,
    }


def _dataset_summary(
    dataset: ConsolidatedDataset,
    split_result: SplitResult,
    settings: PreparationSettings,
    *,
    manifests: Mapping[str, dict[str, Any]],
    artifact_uris: Mapping[str, str],
    raw_image_count: int,
    test_image_count: int,
    annotation_document_count: int,
    generated_at: str,
    dropped_by_angle: int = 0,
    validation_similarity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """어떤 원본을 어떤 비율과 seed로 나눴는지 남기는 요약을 만듭니다."""

    name_by_id = {category["id"]: category["name"] for category in dataset.categories}
    model_label_by_id = {
        category["id"]: label
        for label, category in enumerate(dataset.categories, start=1)
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_prefix": settings.raw_prefix,
        "processed_prefix": settings.processed_prefix,
        "split": {
            "method": settings.split_method,
            "strategy": "deterministic_multilabel_distribution_preserving",
            "split_ratio": settings.split_ratio,
            "validation_ratio": settings.validation_ratio,
            # 그룹을 통째로 옮기므로 목표 비율과 실제 비율이 다를 수 있습니다.
            "validation_image_ratio": round(
                len(split_result.validation_image_ids)
                / (
                    len(split_result.train_image_ids)
                    + len(split_result.validation_image_ids)
                ),
                4,
            ),
            "seed": settings.seed,
            "grouping": _grouping_summary(settings.group_rule, split_result),
            # 그룹을 쪼개지 않고는 validation에 넣을 수 없어 train에만 둔
            # category입니다. 이 category들은 validation 지표를 잴 수 없습니다.
            "train_only_categories": [
                {
                    "id": category_id,
                    "name": name_by_id.get(category_id),
                    "train_image_count": split_result.train_category_counts.get(
                        category_id, 0
                    ),
                }
                for category_id in split_result.train_only_category_ids
            ],
            "checksums": _split_checksums(manifests),
            # 각도를 떼지 않은 실행에는 이 key가 없습니다. 왜 학습 이미지가 줄었는지
            # 나중에 알 수 있어야 합니다.
            **(
                {
                    "angle_holdout": {
                        "validation_angle": settings.validation_angle,
                        "dropped_images": dropped_by_angle,
                    }
                }
                if settings.validation_angle is not None
                else {}
            ),
            # 재지 않은 실행에는 이 key 자체가 없습니다. 0으로 두면 "완전히 똑같다"로
            # 읽혀 사실과 정반대가 됩니다.
            **(
                {"validation_similarity": validation_similarity}
                if validation_similarity is not None
                else {}
            ),
        },
        "raw": {
            "listed_train_images": raw_image_count,
            "listed_test_images": test_image_count,
            "annotation_documents": annotation_document_count,
            "unreferenced_train_images": dataset.unreferenced_image_count,
            "test_images_used": 0,
        },
        "excluded_images": dataset.excluded_images,
        "train_images": len(split_result.train_image_ids),
        "validation_images": len(split_result.validation_image_ids),
        "train_annotations": len(manifests["train"]["annotations"]),
        "validation_annotations": len(manifests["validation"]["annotations"]),
        "category_count": len(dataset.categories),
        "categories": [
            {
                "id": category["id"],
                "name": category["name"],
                "model_label": model_label_by_id[category["id"]],
                # train 전용 category는 validation 수가 0이라 key 자체가 없습니다.
                "train_image_count": split_result.train_category_counts.get(
                    category["id"], 0
                ),
                "validation_image_count": split_result.validation_category_counts.get(
                    category["id"], 0
                ),
            }
            for category in dataset.categories
        ],
        "class_distribution": {
            "train": {
                name_by_id[category_id]: count
                for category_id, count in sorted(split_result.train_category_counts.items())
            },
            "validation": {
                name_by_id[category_id]: count
                for category_id, count in sorted(
                    split_result.validation_category_counts.items()
                )
            },
        },
        "artifacts": dict(artifact_uris),
    }


def _existing_artifact_keys(
    storage: Storage, settings: PreparationSettings
) -> frozenset[str]:
    """처리 경로에 정확히 어떤 Data artifact가 이미 있는지 확인합니다."""

    return frozenset(
        key
        for key, file_name in ARTIFACT_FILE_NAMES.items()
        if storage.exists(f"{settings.processed_prefix}{file_name}")
    )


def _guard_existing(
    settings: PreparationSettings, existing_keys: frozenset[str]
) -> None:
    """이미 있는 산출물을 말없이 덮어쓰지 않도록 먼저 확인합니다."""

    if settings.overwrite:
        return
    if existing_keys:
        existing_files = [ARTIFACT_FILE_NAMES[key] for key in existing_keys]
        raise DatasetPreparationError(
            f"'{settings.processed_prefix}'에 산출물이 이미 있습니다: "
            f"{', '.join(sorted(existing_files))}. 다시 만들려면 "
            "config['data']['overwrite']를 true로 설정하세요."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _existing_artifact_uris(
    storage: Storage,
    settings: PreparationSettings,
    publisher: LocationPublisher,
) -> dict[str, str]:
    """기존 artifact의 실제 URI를 저장소 목록에서 복원합니다."""

    entries = [_normalized(entry) for entry in storage.list(settings.processed_prefix)]
    artifacts: dict[str, str] = {}
    for key in LEGACY_ARTIFACT_KEYS:
        logical_location = f"{settings.processed_prefix}{ARTIFACT_FILE_NAMES[key]}"
        matches = [
            entry
            for entry in entries
            if entry == logical_location or entry.endswith(f"/{logical_location}")
        ]
        if len(matches) != 1:
            raise DatasetPreparationError(
                f"기존 {ARTIFACT_FILE_NAMES[key]} 위치를 하나로 확인하지 못했습니다."
            )
        artifacts[key] = publisher.artifact_uri(matches[0])
    return artifacts


def _backfill_test_manifest(
    storage: Storage,
    settings: PreparationSettings,
    publisher: LocationPublisher,
    progress: ProgressEmitter,
) -> dict[str, Any]:
    """기존 네 artifact를 보존하고 누락된 test manifest만 만듭니다."""

    artifacts = _existing_artifact_uris(storage, settings, publisher)
    class_map_location = (
        f"{settings.processed_prefix}{ARTIFACT_FILE_NAMES['class_map_uri']}"
    )
    class_map = storage.read_json(class_map_location)
    test_image_locations = _select_test_images(
        sorted(str(entry) for entry in storage.list(settings.raw_prefix))
    )
    if not test_image_locations:
        raise DatasetPreparationError("원본 prefix에서 test_images를 찾지 못했습니다.")

    progress.emit("step_started", step="manifests")
    test_manifest = build_test_manifest(
        storage,
        test_image_locations,
        class_map,
        publish_file_name=publisher.image_file_name,
        on_progress=lambda done, total: progress.read_progress(
            "test_images", done, total
        ),
    )
    progress.emit("step_started", step="publish")
    test_location = storage.write_json(
        f"{settings.processed_prefix}{ARTIFACT_FILE_NAMES['test_manifest_uri']}",
        test_manifest,
        overwrite=False,
    )
    artifacts["test_manifest_uri"] = publisher.artifact_uri(test_location)

    summary = {
        "pipeline": "data",
        "mode": "backfill_test_manifest",
        "split_ratio": settings.split_ratio,
        "split_method": settings.split_method,
        "validation_ratio": settings.validation_ratio,
        "seed": settings.seed,
        "source_prefix": settings.raw_prefix,
        "processed_prefix": settings.processed_prefix,
        "overwrite": False,
        "preserved_artifact_keys": sorted(LEGACY_ARTIFACT_KEYS),
        "test_images_used": 0,
        "test_manifest_images": len(test_manifest["images"]),
        "dataset_summary_updated": False,
    }
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": summary,
        "message": (
            f"기존 artifact 4개를 수정하지 않고 test image "
            f"{summary['test_manifest_images']}장의 test_manifest.json만 만들었습니다."
        ),
    }


def prepare_dataset(config: Any, storage: Storage) -> dict[str, Any]:
    """원본을 읽어 artifact 5개를 저장하고 URI와 요약을 돌려줍니다."""

    settings = resolve_settings(config)
    # URI 변환기를 먼저 만들어, 내보낼 수 없는 위치라면 아무것도 쓰기 전에
    # 실패하게 합니다.
    publisher = build_publisher(storage, settings)
    # 준비는 network를 오래 기다리므로, 지금 무엇을 하는지 stderr로 알립니다.
    # 이 stream은 관찰용이며 산출물과 반환값에는 영향을 주지 않습니다.
    progress = ProgressEmitter()
    progress.emit(
        "prepare_started",
        raw_prefix=settings.raw_prefix,
        split_ratio=settings.split_ratio,
        seed=settings.seed,
        split_method=settings.split_method,
    )
    existing_keys = (
        frozenset() if settings.overwrite else _existing_artifact_keys(storage, settings)
    )
    if not settings.overwrite and existing_keys == LEGACY_ARTIFACT_KEYS:
        return _backfill_test_manifest(storage, settings, publisher, progress)
    _guard_existing(settings, existing_keys)

    image_locations, annotation_locations, test_image_locations = _raw_objects(
        storage, settings.raw_prefix
    )
    progress.emit(
        "sources_listed",
        train_images=len(image_locations),
        annotations=len(annotation_locations),
        test_images=len(test_image_locations),
    )
    documents = _read_documents(storage, annotation_locations, progress)
    dataset = consolidate(documents, image_locations)
    progress.emit("step_started", step="split")
    split_result = split_images(
        dataset.images,
        dataset.annotations,
        validation_ratio=settings.validation_ratio,
        seed=settings.seed,
        group_rule=settings.group_rule,
    )
    # 그룹 분할 위에 각도를 하나 더 뗍니다. validation은 그 각도만, train은 그
    # 각도를 뺀 나머지만 남습니다. 이미지를 잃는 대신 학습에서 못 본 시점으로
    # 채점하게 됩니다.
    dropped_by_angle = 0
    if settings.angle_rule is not None and settings.validation_angle is not None:
        split_result, dropped_by_angle = hold_out_angle(
            split_result,
            dataset.images,
            dataset.annotations,
            group_rule=settings.group_rule,
            angle_rule=settings.angle_rule,
            validation_angle=settings.validation_angle,
        )

    progress.emit("step_started", step="manifests")
    manifests = {
        "train": _manifest(
            "train", split_result.train_image_ids, dataset, settings, publisher
        ),
        "validation": _manifest(
            "validation",
            split_result.validation_image_ids,
            dataset,
            settings,
            publisher,
        ),
    }
    # 원본 COCO category id를 그대로 남깁니다. 소비자는 이 id 순서대로 1부터
    # 이어지는 model label을 붙입니다.
    class_map = {
        str(category["id"]): category["name"] for category in dataset.categories
    }
    # 같은 class_map 객체를 그대로 저장하고 test manifest에도 넘겨 두 산출물의
    # category source가 달라질 수 없게 합니다.
    test_manifest = build_test_manifest(
        storage,
        test_image_locations,
        class_map,
        publish_file_name=publisher.image_file_name,
        on_progress=lambda done, total: progress.read_progress(
            "test_images", done, total
        ),
    )

    # 이미지를 전부 여는 단계라 켠 실행만 이 비용을 냅니다. **산출물을 쓰기 전에**
    # 잽니다. 여기서 실패했는데 manifest가 이미 나가 있으면 반쪽짜리 dataset이 남고,
    # 다음 실행은 덮어쓰기 거부에 걸립니다.
    validation_similarity = None
    if settings.measure_validation_similarity:
        progress.emit("step_started", step="similarity")
        validation_similarity = measure_validation_similarity(
            storage,
            dataset.images,
            dataset.annotations,
            train_image_ids=split_result.train_image_ids,
            validation_image_ids=split_result.validation_image_ids,
            on_progress=lambda stage, done, total: progress.read_progress(
                stage, done, total
            ),
        )

    # 이것도 이미지를 전부 여는 단계라 켠 실행만 비용을 냅니다. 유사도와 같은
    # 이유로 **산출물을 하나라도 내보내기 전에** 여기서 끝냅니다. tar는 임시
    # 자리에 두고, 올리는 것은 아래 publish 단계에서 함께 합니다.
    crop_bank_summary: dict[str, Any] | None = None
    crop_bank_archive: Path | None = None
    crop_bank_scratch: tempfile.TemporaryDirectory[str] | None = None
    if settings.crop_bank:
        progress.emit("step_started", step="crop_bank")
        group_rule = settings.group_rule or GroupRule(
            delimiter=GROUP_KEY_DELIMITER, tokens=GROUP_KEY_TOKENS
        )
        group_of = {
            image["id"]: group_rule.key(str(image["file_name"])) for image in dataset.images
        }
        crop_bank_scratch = tempfile.TemporaryDirectory(prefix="crop-bank-archive-")
        crop_bank_archive = Path(crop_bank_scratch.name) / CROP_BANK_FILE_NAME
        crop_bank_summary = build_crop_bank(
            storage,
            dataset.images,
            dataset.annotations,
            train_image_ids=split_result.train_image_ids,
            group_of=group_of,
            per_class=settings.crop_bank_per_class,
            seed=settings.seed,
            archive_path=crop_bank_archive,
            on_progress=lambda done, total: progress.read_progress(
                "crop_bank", done, total
            ),
        )

    progress.emit("step_started", step="publish")
    artifacts: dict[str, str] = {}
    for key, value in (
        ("train_manifest_uri", manifests["train"]),
        ("validation_manifest_uri", manifests["validation"]),
        ("class_map_uri", class_map),
        ("test_manifest_uri", test_manifest),
    ):
        artifacts[key] = publisher.artifact_uri(
            storage.write_json(
                f"{settings.processed_prefix}{ARTIFACT_FILE_NAMES[key]}",
                value,
                overwrite=settings.overwrite,
            )
        )

    if crop_bank_archive is not None and crop_bank_summary is not None:
        artifacts[CROP_BANK_ARTIFACT_KEY] = publisher.artifact_uri(
            storage.upload_file(
                crop_bank_archive,
                f"{settings.processed_prefix}{CROP_BANK_FILE_NAME}",
                overwrite=settings.overwrite,
            )
        )
    if crop_bank_scratch is not None:
        crop_bank_scratch.cleanup()

    summary_document = _dataset_summary(
        dataset,
        split_result,
        settings,
        manifests=manifests,
        artifact_uris=artifacts,
        raw_image_count=len(image_locations),
        test_image_count=len(test_image_locations),
        annotation_document_count=len(annotation_locations),
        generated_at=_utc_now(),
        dropped_by_angle=dropped_by_angle,
        validation_similarity=validation_similarity,
    )
    artifacts["dataset_summary_uri"] = publisher.artifact_uri(
        storage.write_json(
            f"{settings.processed_prefix}{ARTIFACT_FILE_NAMES['dataset_summary_uri']}",
            summary_document,
            overwrite=settings.overwrite,
        )
    )

    summary = {
        "pipeline": "data",
        "mode": "prepare",
        "split_ratio": settings.split_ratio,
        "validation_ratio": settings.validation_ratio,
        "split_method": settings.split_method,
        "seed": settings.seed,
        "source_prefix": settings.raw_prefix,
        "processed_prefix": settings.processed_prefix,
        "overwrite": settings.overwrite,
        "train_images": len(split_result.train_image_ids),
        "validation_images": len(split_result.validation_image_ids),
        "train_groups": split_result.train_group_count,
        "validation_groups": split_result.validation_group_count,
        "excluded_images": len(dataset.excluded_images),
        "category_count": len(dataset.categories),
        # 그룹을 쪼개지 않고는 validation에 갈 수 없어 train에만 둔 category입니다.
        # 자세한 내용은 dataset_summary.json의 `split.train_only_categories`에 있습니다.
        "train_only_categories": list(split_result.train_only_category_ids),
        "test_images_used": 0,
        "test_manifest_images": len(test_manifest["images"]),
    }
    if crop_bank_summary is not None:
        summary["crop_bank"] = crop_bank_summary
    unit = "그룹 단위" if settings.group_rule is not None else "이미지 단위"
    train_only_note = (
        ""
        if not split_result.train_only_category_ids
        else (
            f" 그룹을 쪼개지 않고는 validation에 갈 수 없어 train에만 둔 category "
            f"{len(split_result.train_only_category_ids)}종은 validation 지표를 "
            "잴 수 없습니다."
        )
    )
    message = (
        f"원본 '{settings.raw_prefix}'에서 split_ratio {settings.split_ratio}"
        f"(validation {settings.validation_ratio}), seed {settings.seed}로 "
        f"{unit} 분할해 artifact 5개를 만들어 "
        f"'{settings.processed_prefix}'에 저장했습니다. "
        f"train {summary['train_images']}장({summary['train_groups']}그룹), "
        f"validation {summary['validation_images']}장"
        f"({summary['validation_groups']}그룹)." + train_only_note
    )
    progress.emit(
        "prepare_completed",
        train_images=summary["train_images"],
        validation_images=summary["validation_images"],
        category_count=summary["category_count"],
    )
    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": summary,
        "message": message,
    }


def preparation_error_result(message: str, config: Any) -> dict[str, Any]:
    """실패를 예외 대신 계약에 맞는 결과로 돌려줍니다."""

    section = _data_config(config) if isinstance(config, Mapping) else {}
    requested_ratio = section.get("split_ratio") if isinstance(section, Mapping) else None
    return {
        "status": "error",
        "artifacts": {},
        "summary": {
            "pipeline": "data",
            "mode": "prepare",
            "allowed_split_ratios": list(SPLIT_RATIO_OPTIONS),
            "requested_split_ratio": (
                requested_ratio if isinstance(requested_ratio, str) else None
            ),
        },
        "message": message,
    }


def run_preparation(config: Any) -> dict[str, Any]:
    """준비 경로를 실행하고 계약에 맞는 결과 dict를 반환합니다."""

    try:
        storage = create_storage(config if isinstance(config, Mapping) else {})
        return prepare_dataset(config, storage)
    except DatasetPreparationError as error:
        return preparation_error_result(str(error), config)
    except StorageError as error:
        # Storage 오류 message에는 개인 절대경로가 들어갈 수 있으므로 예외 종류만
        # 알리고 원문은 남기지 않습니다.
        return preparation_error_result(
            "storage 작업에 실패했습니다"
            f"({type(error).__name__}). storage 설정과 접근 권한을 확인하세요.",
            config,
        )
    except Exception as error:  # 계약상 예외를 밖으로 던지지 않습니다.
        return preparation_error_result(
            f"dataset 준비 중 예기치 못한 오류가 발생했습니다({type(error).__name__}).",
            config,
        )
