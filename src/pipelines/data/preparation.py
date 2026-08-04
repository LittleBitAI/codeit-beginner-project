"""S3 원본에서 학습용 artifact 4개를 만들어 storage에 저장합니다.

`config["data"]["prepare"]`가 `true`일 때만 동작하는 준비 경로입니다. 원본
`train_images/`와 `train_annotations/`만 읽고, 합친 COCO dataset을 train과
validation으로 나눈 뒤 아래 네 file을 저장합니다.

- `train_manifest.json`, `validation_manifest.json`: COCO 형식 manifest
- `class_map.json`: `{"<category id>": "<category name>"}`
- `dataset_summary.json`: 원본 위치, 비율, seed, 분포를 담은 요약

Storage 접근은 `src/common/storage.py`의 `create_storage(config)`로만 합니다.
개별 pipeline은 `boto3`를 직접 쓰지 않습니다. competition 평가용 `test_images/`는
어떤 split에도 넣지 않습니다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.common import Storage, StorageError, create_storage

from .coco import ConsolidatedDataset, consolidate
from .errors import DatasetPreparationError
from .split import SplitResult, split_images


__all__ = [
    "ARTIFACT_FILE_NAMES",
    "DEFAULT_SEED",
    "PreparationSettings",
    "SPLIT_RATIO_OPTIONS",
    "preparation_error_result",
    "preparation_requested",
    "resolve_settings",
    "run_preparation",
]


# 원본과 산출물의 기본 위치입니다. config로 바꿀 수 있지만 두 값 모두
# 공용 logical prefix인 `datasets/` 아래여야 합니다.
DEFAULT_RAW_PREFIX = "datasets/pill_detection/raw/v1/"
DEFAULT_PROCESSED_ROOT = "datasets/pill_detection/processed/"
DEFAULT_SEED = 42
DATASET_PREFIX = "datasets/"

# 이번 작업의 핵심: validation 비율은 아래 두 값 중 하나만 허용합니다.
SPLIT_RATIO_OPTIONS: dict[str, float] = {"8:2": 0.2, "9:1": 0.1}
# 비율이 다르면 산출물도 다르므로 저장 directory 이름에 비율을 드러냅니다.
_RATIO_DIRECTORY_TOKENS: dict[str, str] = {"8:2": "8020", "9:1": "9010"}

ARTIFACT_FILE_NAMES: dict[str, str] = {
    "train_manifest_uri": "train_manifest.json",
    "validation_manifest_uri": "validation_manifest.json",
    "class_map_uri": "class_map.json",
    "dataset_summary_uri": "dataset_summary.json",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TRAIN_IMAGE_MARKER = "train_images/"
TRAIN_ANNOTATION_MARKER = "train_annotations/"
# competition 평가용 원본입니다. 학습·검증 어디에도 들어가면 안 됩니다.
FORBIDDEN_MARKERS = ("test_images/", "test_annotations/")
MAX_READ_WORKERS = 16
SUMMARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PreparationSettings:
    """준비 실행에 필요한 config 값입니다."""

    split_ratio: str
    validation_ratio: float
    seed: int
    raw_prefix: str
    processed_prefix: str
    overwrite: bool


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


def resolve_settings(config: Any) -> PreparationSettings:
    """준비 경로 설정을 읽고 검증합니다."""

    section = _data_config(config)
    split_ratio = _split_ratio(section)
    seed = _seed(section)
    raw_prefix = _normalized_prefix(section.get("raw_prefix"), "raw_prefix", DEFAULT_RAW_PREFIX)
    processed_root = _normalized_prefix(
        section.get("processed_root"), "processed_root", DEFAULT_PROCESSED_ROOT
    )
    # 비율과 seed가 directory 이름에 들어가므로 8:2와 9:1 산출물은 서로 덮어쓰지
    # 않습니다.
    processed_prefix = (
        f"{processed_root}v1-seed{seed}-{_RATIO_DIRECTORY_TOKENS[split_ratio]}/"
    )
    return PreparationSettings(
        split_ratio=split_ratio,
        validation_ratio=SPLIT_RATIO_OPTIONS[split_ratio],
        seed=seed,
        raw_prefix=raw_prefix,
        processed_prefix=processed_prefix,
        overwrite=_overwrite(section),
    )


def _normalized(location: Any) -> str:
    return str(location).replace("\\", "/")


def _raw_objects(storage: Storage, raw_prefix: str) -> tuple[list[str], list[str]]:
    """원본 prefix에서 train 이미지와 train annotation 위치를 고릅니다.

    `test_images/` 등 competition 평가용 경로는 여기서 제외되어 이후 단계로
    전달되지 않습니다.
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
    if not image_locations or not annotation_locations:
        raise DatasetPreparationError(
            "원본 prefix에서 train_images와 train_annotations를 모두 찾지 못했습니다. "
            f"(이미지 {len(image_locations)}개, annotation {len(annotation_locations)}개)"
        )
    return image_locations, annotation_locations


def _read_documents(
    storage: Storage, annotation_locations: Sequence[str]
) -> list[tuple[str, Any]]:
    """이미지별 annotation 문서를 순서를 유지한 채 읽습니다."""

    workers = max(1, min(MAX_READ_WORKERS, len(annotation_locations)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        documents = list(executor.map(storage.read_json, annotation_locations))
    return list(zip(annotation_locations, documents))


def _manifest(
    split: str,
    image_ids: set[int],
    dataset: ConsolidatedDataset,
    settings: PreparationSettings,
) -> dict[str, Any]:
    """train이 그대로 읽을 수 있는 COCO manifest를 만듭니다.

    `file_name`에는 원본 이미지의 storage 위치를 그대로 넣습니다. manifest는
    `processed/` 아래, 이미지는 `raw/` 아래에 있으므로 상대경로로 두면 소비자가
    엉뚱한 위치를 가리키게 됩니다.
    """

    return {
        "info": {
            "description": "Pill detection processed COCO manifest",
            "split": split,
            "seed": settings.seed,
            "split_ratio": settings.split_ratio,
            "validation_ratio": settings.validation_ratio,
        },
        "images": [image for image in dataset.images if image["id"] in image_ids],
        "annotations": [
            annotation
            for annotation in dataset.annotations
            if annotation["image_id"] in image_ids
        ],
        "categories": list(dataset.categories),
    }


def _dataset_summary(
    dataset: ConsolidatedDataset,
    split_result: SplitResult,
    settings: PreparationSettings,
    *,
    manifests: Mapping[str, dict[str, Any]],
    artifact_uris: Mapping[str, str],
    raw_image_count: int,
    annotation_document_count: int,
    generated_at: str,
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
            "method": "deterministic_multilabel_distribution_preserving",
            "split_ratio": settings.split_ratio,
            "validation_ratio": settings.validation_ratio,
            "seed": settings.seed,
        },
        "raw": {
            "listed_train_images": raw_image_count,
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
                "train_image_count": split_result.train_category_counts[category["id"]],
                "validation_image_count": split_result.validation_category_counts[
                    category["id"]
                ],
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


def _guard_existing(storage: Storage, settings: PreparationSettings) -> None:
    """이미 있는 산출물을 말없이 덮어쓰지 않도록 먼저 확인합니다."""

    if settings.overwrite:
        return
    existing = [
        file_name
        for file_name in ARTIFACT_FILE_NAMES.values()
        if storage.exists(f"{settings.processed_prefix}{file_name}")
    ]
    if existing:
        raise DatasetPreparationError(
            f"'{settings.processed_prefix}'에 산출물이 이미 있습니다: "
            f"{', '.join(sorted(existing))}. 다시 만들려면 "
            "config['data']['overwrite']를 true로 설정하세요."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_dataset(config: Any, storage: Storage) -> dict[str, Any]:
    """원본을 읽어 artifact 4개를 저장하고 URI와 요약을 돌려줍니다."""

    settings = resolve_settings(config)
    _guard_existing(storage, settings)

    image_locations, annotation_locations = _raw_objects(storage, settings.raw_prefix)
    documents = _read_documents(storage, annotation_locations)
    dataset = consolidate(documents, image_locations)
    split_result = split_images(
        dataset.images,
        dataset.annotations,
        validation_ratio=settings.validation_ratio,
        seed=settings.seed,
    )

    manifests = {
        "train": _manifest("train", split_result.train_image_ids, dataset, settings),
        "validation": _manifest(
            "validation", split_result.validation_image_ids, dataset, settings
        ),
    }
    # 원본 COCO category id를 그대로 남깁니다. 소비자는 이 id 순서대로 1부터
    # 이어지는 model label을 붙입니다.
    class_map = {
        str(category["id"]): category["name"] for category in dataset.categories
    }

    artifacts: dict[str, str] = {}
    for key, value in (
        ("train_manifest_uri", manifests["train"]),
        ("validation_manifest_uri", manifests["validation"]),
        ("class_map_uri", class_map),
    ):
        artifacts[key] = storage.write_json(
            f"{settings.processed_prefix}{ARTIFACT_FILE_NAMES[key]}",
            value,
            overwrite=settings.overwrite,
        )

    summary_document = _dataset_summary(
        dataset,
        split_result,
        settings,
        manifests=manifests,
        artifact_uris=artifacts,
        raw_image_count=len(image_locations),
        annotation_document_count=len(annotation_locations),
        generated_at=_utc_now(),
    )
    artifacts["dataset_summary_uri"] = storage.write_json(
        f"{settings.processed_prefix}{ARTIFACT_FILE_NAMES['dataset_summary_uri']}",
        summary_document,
        overwrite=settings.overwrite,
    )

    summary = {
        "pipeline": "data",
        "mode": "prepare",
        "split_ratio": settings.split_ratio,
        "validation_ratio": settings.validation_ratio,
        "seed": settings.seed,
        "source_prefix": settings.raw_prefix,
        "processed_prefix": settings.processed_prefix,
        "overwrite": settings.overwrite,
        "train_images": len(split_result.train_image_ids),
        "validation_images": len(split_result.validation_image_ids),
        "excluded_images": len(dataset.excluded_images),
        "category_count": len(dataset.categories),
        "test_images_used": 0,
    }
    message = (
        f"원본 '{settings.raw_prefix}'에서 split_ratio {settings.split_ratio}"
        f"(validation {settings.validation_ratio}), seed {settings.seed}로 "
        f"artifact 4개를 만들어 '{settings.processed_prefix}'에 저장했습니다. "
        f"train {summary['train_images']}장, validation {summary['validation_images']}장."
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
