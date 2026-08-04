"""예측(prediction)을 만들거나 기존 예측 artifact를 읽어 정규화합니다.

두 가지 경로를 지원합니다.

1. `evaluate.predictions_input_uri`가 있으면 이미 만들어진 예측 JSON을 읽습니다.
2. 없으면 checkpoint를 torchvision detection model로 불러와 CPU/GPU에서 추론합니다.

예측 JSON은 COCO detection 결과와 같은 flat list이며, 다음 두 형식을 모두
읽을 수 있습니다.

    [{"image_id": "img-1", "category_id": 1, "bbox": [x, y, w, h], "score": 0.9}]
    {"predictions": [ ... 위와 같은 항목 ... ]}
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import InputArtifactError, PredictionError
from .manifest import normalize_image_key
from .storage_io import ArtifactStore


REQUIRED_PREDICTION_FIELDS = ("image_id", "category_id", "bbox", "score")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_predictions(
    document: Any,
    *,
    source: str,
    known_image_keys: set[str],
) -> list[dict[str, Any]]:
    """예측 문서를 검증하고 정규화된 목록을 반환합니다."""
    if isinstance(document, Mapping):
        entries = document.get("predictions")
        if not isinstance(entries, list):
            raise InputArtifactError(f"{source}: predictions key에 list가 필요합니다.")
    elif isinstance(document, list):
        entries = document
    else:
        raise InputArtifactError(f"{source}: 예측 문서는 list 또는 object여야 합니다.")

    predictions: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        label = f"{source}#prediction[{index}]"
        if not isinstance(entry, Mapping):
            raise InputArtifactError(f"{label}: 예측 항목은 object여야 합니다.")

        missing = [name for name in REQUIRED_PREDICTION_FIELDS if name not in entry]
        if missing:
            raise InputArtifactError(f"{label}: 필수 field가 없습니다: {', '.join(missing)}")

        image_key = normalize_image_key(entry["image_id"])
        if image_key not in known_image_keys:
            raise InputArtifactError(
                f"{label}: manifest에 없는 image_id를 참조합니다: {entry['image_id']!r}"
            )

        category_id = entry["category_id"]
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise InputArtifactError(f"{label}: category_id는 정수여야 합니다: {category_id!r}")

        bbox = entry["bbox"]
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise InputArtifactError(f"{label}: bbox는 [x, y, width, height] 형식이어야 합니다.")
        if not all(_is_number(item) for item in bbox):
            raise InputArtifactError(f"{label}: bbox 값은 모두 유한한 숫자여야 합니다.")
        box = [float(item) for item in bbox]
        if box[2] <= 0 or box[3] <= 0:
            raise InputArtifactError(f"{label}: bbox의 width와 height는 0보다 커야 합니다.")

        score = entry["score"]
        if not _is_number(score):
            raise InputArtifactError(f"{label}: score는 유한한 숫자여야 합니다: {score!r}")

        predictions.append(
            {
                "image_id": entry["image_id"],
                "image_key": image_key,
                "category_id": category_id,
                "bbox": box,
                "score": float(score),
            }
        )
    return predictions


def load_predictions(
    store: ArtifactStore,
    uri: str,
    *,
    known_image_keys: set[str],
) -> list[dict[str, Any]]:
    """기존 예측 artifact를 읽습니다."""
    return parse_predictions(store.read_json(uri), source=uri, known_image_keys=known_image_keys)


def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torchvision
    except ImportError as error:  # pragma: no cover - dependency는 requirements에 고정되어 있습니다.
        raise PredictionError(
            "checkpoint 추론에는 requirements.txt의 torch와 torchvision이 필요합니다."
        ) from error
    return torch, torchvision


def _build_model(checkpoint: Mapping[str, Any], *, source: str) -> Any:
    _, torchvision = _import_torch()

    architecture = checkpoint.get("architecture")
    num_classes = checkpoint.get("num_classes")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))

    if not isinstance(architecture, str) or not architecture.strip():
        raise PredictionError(f"{source}: checkpoint에 architecture 문자열이 필요합니다.")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
        raise PredictionError(f"{source}: checkpoint에 0보다 큰 num_classes 정수가 필요합니다.")
    if not isinstance(state_dict, Mapping):
        raise PredictionError(f"{source}: checkpoint에 state_dict가 필요합니다.")

    builder = getattr(torchvision.models.detection, architecture.strip(), None)
    if not callable(builder):
        raise PredictionError(
            f"{source}: torchvision.models.detection이 제공하지 않는 architecture입니다: {architecture}"
        )

    try:
        model = builder(weights=None, weights_backbone=None, num_classes=num_classes)
        model.load_state_dict(dict(state_dict))
    except (RuntimeError, TypeError, ValueError) as error:
        raise PredictionError(f"{source}: checkpoint를 model에 적용하지 못했습니다: {error}") from error

    model.eval()
    return model


def load_checkpoint_document(store: ArtifactStore, uri: str, *, device: str) -> Mapping[str, Any]:
    """Checkpoint를 local로 가져와 읽습니다."""
    torch, _ = _import_torch()

    with tempfile.TemporaryDirectory(prefix="pill-evaluate-checkpoint-") as directory:
        checkpoint_path = store.ensure_local_file(uri, directory)
        try:
            document = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as error:  # noqa: BLE001 - torch 오류를 계약 오류로 변환합니다.
            raise PredictionError(f"checkpoint를 읽지 못했습니다 ({uri}): {error}") from error

    if isinstance(document, Mapping) and isinstance(document.get("model"), Mapping):
        document = document["model"]
    if not isinstance(document, Mapping):
        raise PredictionError(f"checkpoint의 최상위 값은 object여야 합니다: {uri}")
    return document


def predict_with_checkpoint(
    store: ArtifactStore,
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_uri: str,
    device: str = "cpu",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Checkpoint를 불러와 manifest의 모든 이미지에 대해 추론합니다."""
    torch, _ = _import_torch()
    checkpoint = load_checkpoint_document(store, checkpoint_uri, device=device)
    model = _build_model(checkpoint, source=checkpoint_uri)

    torch.manual_seed(seed)
    try:
        resolved_device = torch.device(device)
        model.to(resolved_device)
    except (RuntimeError, ValueError, AssertionError) as error:
        raise PredictionError(f"device를 준비하지 못했습니다 ({device}): {error}") from error

    category_ids = checkpoint.get("category_ids")
    if category_ids is not None and (
        not isinstance(category_ids, Sequence)
        or isinstance(category_ids, (str, bytes))
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in category_ids)
    ):
        raise PredictionError(f"{checkpoint_uri}: category_ids는 정수 list여야 합니다.")

    predictions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pill-evaluate-images-") as directory:
        for record in records:
            image_tensor = _load_image_tensor(store, record, directory)
            with torch.no_grad():
                outputs = model([image_tensor.to(resolved_device)])
            predictions.extend(
                _outputs_to_predictions(outputs[0], record=record, category_ids=category_ids)
            )
    return predictions


def _load_image_tensor(
    store: ArtifactStore,
    record: Mapping[str, Any],
    download_dir: str | Path,
) -> Any:
    _, torchvision = _import_torch()
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency는 requirements에 고정되어 있습니다.
        raise PredictionError("이미지 추론에는 requirements.txt의 Pillow가 필요합니다.") from error

    image_path = store.ensure_local_file(record["image_uri"], download_dir)
    try:
        with Image.open(image_path) as image:
            tensor = torchvision.transforms.functional.pil_to_tensor(image.convert("RGB"))
    except (OSError, ValueError) as error:
        raise PredictionError(
            f"이미지를 열지 못했습니다 ({record['image_uri']}): {error}"
        ) from error
    return tensor.float().div(255.0)


def _outputs_to_predictions(
    output: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    category_ids: Sequence[int] | None,
) -> list[dict[str, Any]]:
    boxes = output["boxes"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()

    predictions: list[dict[str, Any]] = []
    for (left, top, right, bottom), label, score in zip(boxes, labels, scores):
        width, height = float(right) - float(left), float(bottom) - float(top)
        if width <= 0 or height <= 0:
            continue
        label_index = int(label)
        if category_ids is not None:
            if not 0 <= label_index < len(category_ids):
                raise PredictionError(
                    f"model이 category_ids 범위를 벗어난 label을 반환했습니다: {label_index}"
                )
            category_id = int(category_ids[label_index])
        else:
            category_id = label_index
        predictions.append(
            {
                "image_id": record["image_id"],
                "image_key": record["image_key"],
                "category_id": category_id,
                "bbox": [float(left), float(top), width, height],
                "score": float(score),
            }
        )
    return predictions


__all__ = [
    "load_checkpoint_document",
    "load_predictions",
    "parse_predictions",
    "predict_with_checkpoint",
]
