"""crop embedding으로 검출 점수를 다시 매깁니다.

detector 점수는 "상자 안에 알약이 있는가"에는 강하지만 "그 알약이 정말 이
class인가"에는 약합니다. 그래서 검출한 상자를 잘라 crop embedding model에 넣고,
data가 만든 crop 은행에서

* 같은 class의 가장 가까운 참조 crop과의 유사도(`own`)
* 다른 class의 가장 가까운 참조 crop과의 유사도(`other`)

를 재 그 차이 `margin = own - other`를 점수에 곱합니다.

    score' = score x (1 + margin) / 2

margin은 코사인 유사도의 차이라 확신이 서면 1에 가깝고, 다른 class 쪽이 더
가까우면 음수가 됩니다. **상자를 옮기지도, 행을 지우지도 않고 점수만** 바꿉니다.
대회 채점은 class마다 따로 순위를 매기므로 같은 상자라도 점수가 바뀌면 그 class
안의 순위가 바뀌고, 그 순위가 AP를 만듭니다.

checkpoint를 여러 개 주면 margin을 평균 냅니다. backbone이 다른 model은 서로
다른 자리에서 틀리므로, 평균은 한 model만 크게 틀린 자리를 눌러 줍니다.
"""

from __future__ import annotations

import json
import math
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import InputArtifactError, PredictionError
from .predictor import load_checkpoint_document
from .storage_io import ArtifactStore


#: crop 은행 tar 안에서 목록이 놓이는 자리입니다. data가 정한 이름입니다.
CROP_BANK_INDEX = "index.json"

#: 한 번에 model에 넣는 crop 수입니다. 224px crop 32장이면 8GB GPU에서도 돕니다.
BATCH_SIZE = 32

#: 8-way TTA입니다. 알약은 어느 방향으로 놓여도 같은 알약이므로 네 방향 회전과
#: 좌우 뒤집기를 모두 재 평균 냅니다. 학습 때 준 증강과 짝이 맞아야 합니다.
TTA_TURNS = 4
TTA_FLIPS = (False, True)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - requirements에 고정되어 있습니다.
        raise PredictionError("재순위에는 requirements.txt의 torch가 필요합니다.") from error
    return torch


def _import_torchvision() -> Any:
    try:
        import torchvision
    except ImportError as error:  # pragma: no cover - requirements에 고정되어 있습니다.
        raise PredictionError("재순위에는 requirements.txt의 torchvision이 필요합니다.") from error
    return torchvision


def _import_image() -> Any:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - requirements에 고정되어 있습니다.
        raise PredictionError("재순위에는 requirements.txt의 Pillow가 필요합니다.") from error
    return Image


def load_crop_bank(store: ArtifactStore, uri: str, destination: Path) -> dict[str, Any]:
    """crop 은행 tar을 내려받아 풀고 목록 문서를 돌려줍니다.

    `crop_size`와 `crop_margin`을 이 문서에서 읽습니다. 여기 적힌 값과 다르게
    잘라 낸 시험 crop은 참조 crop과 비교할 수 없는데, 그 어긋남은 오류가 아니라
    조금 낮은 점수로만 드러나므로 상수로 베껴 두지 않습니다.
    """

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = store.ensure_local_file(uri, destination)
    try:
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not str(target).startswith(str(destination.resolve())):
                    raise InputArtifactError(
                        f"crop 은행에 폴더 밖을 가리키는 항목이 있습니다: {member.name}"
                    )
            archive.extractall(destination)
        document = json.loads((destination / CROP_BANK_INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError, tarfile.TarError) as error:
        raise InputArtifactError(f"crop 은행을 읽지 못했습니다 ({uri}): {error}") from error

    if not isinstance(document, Mapping):
        raise InputArtifactError(f"crop 은행 목록은 object여야 합니다: {uri}")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise InputArtifactError(f"crop 은행 목록이 비어 있습니다: {uri}")
    for record in records:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("category_id"), int)
            or isinstance(record.get("category_id"), bool)
        ):
            raise InputArtifactError(f"crop 은행 항목의 형식이 올바르지 않습니다: {uri}")
    return dict(document)


def _positive_int(document: Mapping[str, Any], key: str, *, source: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputArtifactError(f"{source}: {key}는 0보다 큰 정수여야 합니다: {value!r}")
    return value


def _ratio(document: Mapping[str, Any], key: str, *, source: str) -> float:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputArtifactError(f"{source}: {key}는 숫자여야 합니다: {value!r}")
    if not math.isfinite(value) or value < 0:
        raise InputArtifactError(f"{source}: {key}는 0 이상의 유한한 값이어야 합니다.")
    return float(value)


def _embedding_model(checkpoint: Mapping[str, Any], *, source: str, device: str) -> Any:
    """embedding checkpoint에서 head를 뗀 특징 추출기를 만듭니다."""

    torch = _import_torch()
    torchvision = _import_torchvision()

    if checkpoint.get("task") != "embedding":
        raise PredictionError(
            f"{source}: 재순위에는 train.task=embedding으로 학습한 checkpoint가 필요합니다."
        )
    backbone = checkpoint.get("backbone")
    if not isinstance(backbone, str) or not backbone.strip():
        raise PredictionError(f"{source}: checkpoint에 backbone 이름이 필요합니다.")
    category_ids = checkpoint.get("category_ids")
    if (
        not isinstance(category_ids, Sequence)
        or isinstance(category_ids, (str, bytes))
        or not category_ids
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in category_ids)
    ):
        raise PredictionError(f"{source}: checkpoint에 category_ids 정수 list가 필요합니다.")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise PredictionError(f"{source}: checkpoint에 state_dict가 필요합니다.")

    factory = getattr(torchvision.models, backbone.strip(), None)
    if not callable(factory):
        raise PredictionError(
            f"{source}: torchvision.models가 제공하지 않는 backbone입니다: {backbone}"
        )
    try:
        model = factory(weights=None)
        # head를 붙인 모양 그대로 실어야 state_dict가 맞습니다. 그 뒤에 뗍니다.
        model.fc = torch.nn.Linear(model.fc.in_features, len(category_ids))
        model.load_state_dict(dict(state_dict))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PredictionError(
            f"{source}: checkpoint를 model에 적용하지 못했습니다: {error}"
        ) from error
    model.fc = torch.nn.Identity()
    model.eval()
    try:
        model.to(torch.device(device))
    except (RuntimeError, ValueError, AssertionError) as error:
        raise PredictionError(f"device를 준비하지 못했습니다 ({device}): {error}") from error
    return model


def _crop(image: Any, bbox: Sequence[float], *, size: int, margin: float) -> Any:
    """상자 하나를 은행과 같은 방식으로 잘라 같은 크기로 맞춥니다."""

    Image = _import_image()
    x, y, width, height = (float(value) for value in bbox)
    pad_x, pad_y = width * margin, height * margin
    box = (
        max(0, int(x - pad_x)),
        max(0, int(y - pad_y)),
        min(image.width, int(x + width + pad_x)),
        min(image.height, int(y + height + pad_y)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        # manifest가 상자를 이미 확인하므로 여기까지 오면 이미지가 예상과 다른
        # 경우입니다. 자를 수 없는 상자는 점수를 그대로 둡니다.
        return None
    return image.crop(box).convert("RGB").resize((size, size), Image.BICUBIC)


def _stack(crops: Sequence[Any], *, mean: Any, std: Any) -> Any:
    """PIL crop 묶음을 정규화된 tensor batch로 만듭니다."""

    import numpy

    torch = _import_torch()
    array = numpy.stack([numpy.asarray(crop, dtype=numpy.uint8) for crop in crops])
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255.0)
    return (tensor - mean) / std


def _embed(model: Any, batch: Any, *, device: str) -> Any:
    """8-way TTA로 특징을 뽑아 하나로 모읍니다. 결과는 L2 정규화된 vector입니다."""

    torch = _import_torch()
    total = None
    for flip in TTA_FLIPS:
        base = torch.flip(batch, dims=[3]) if flip else batch
        for turns in range(TTA_TURNS):
            view = torch.rot90(base, turns, dims=[2, 3]) if turns else base
            with torch.inference_mode():
                features = model(view.to(torch.device(device)))
            features = torch.nn.functional.normalize(features.float().cpu(), dim=1)
            total = features if total is None else total + features
    return torch.nn.functional.normalize(total, dim=1)


def _embed_all(
    model: Any,
    crops: Sequence[Any],
    *,
    device: str,
    mean: Any,
    std: Any,
    on_progress: Callable[[int, int], None] | None,
) -> Any:
    torch = _import_torch()
    pieces = []
    for start in range(0, len(crops), BATCH_SIZE):
        batch = _stack(crops[start : start + BATCH_SIZE], mean=mean, std=std)
        pieces.append(_embed(model, batch, device=device))
        if on_progress is not None:
            on_progress(min(start + BATCH_SIZE, len(crops)), len(crops))
    return torch.cat(pieces)


def _margins(similarity: Any, *, bank_categories: Sequence[int], categories: Sequence[int]) -> Any:
    """행마다 `자기 class 최고 유사도 - 다른 class 최고 유사도`를 냅니다.

    은행에 없는 class는 잴 수 없습니다. 그때 0을 넣으면 "재 봤더니 확신이
    없더라"가 되어 점수가 반으로 깎이므로 `nan`으로 두고, 부르는 쪽이 그 행의
    점수를 그대로 둡니다.
    """

    torch = _import_torch()
    bank = torch.tensor(list(bank_categories))
    result = torch.full((similarity.shape[0],), float("nan"))
    for index, category_id in enumerate(categories):
        own_columns = bank == category_id
        if not bool(own_columns.any()):
            continue
        row = similarity[index]
        other_columns = ~own_columns
        if not bool(other_columns.any()):
            # class가 하나뿐인 은행에서는 비교 대상이 없습니다.
            continue
        result[index] = row[own_columns].max() - row[other_columns].max()
    return result


def rerank_predictions(
    store: ArtifactStore,
    predictions: Sequence[Mapping[str, Any]],
    *,
    records: Sequence[Mapping[str, Any]],
    checkpoint_uris: Sequence[str],
    crop_bank_uri: str,
    device: str = "cpu",
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """예측의 점수만 다시 매기고, 무엇을 했는지 함께 돌려줍니다."""

    torch = _import_torch()
    Image = _import_image()
    rows = [dict(prediction) for prediction in predictions]
    if not rows:
        return rows, {"rows": 0, "reranked_rows": 0, "checkpoints": list(checkpoint_uris)}

    locations = {record["image_key"]: record["image_uri"] for record in records}
    missing = sorted({row["image_key"] for row in rows} - set(locations))
    if missing:
        raise InputArtifactError(
            f"재순위에 필요한 image가 manifest에 없습니다: {', '.join(missing[:3])}"
        )

    with tempfile.TemporaryDirectory(prefix="pill-evaluate-rerank-") as directory:
        root = Path(directory)
        bank_document = load_crop_bank(store, crop_bank_uri, root / "bank")
        crop_size = _positive_int(bank_document, "crop_size", source=crop_bank_uri)
        crop_margin = _ratio(bank_document, "crop_margin", source=crop_bank_uri)
        bank_records = list(bank_document["records"])

        # 시험 crop을 한 번만 잘라 둡니다. checkpoint마다 다시 자르면 같은 이미지를
        # checkpoint 수만큼 다시 열게 됩니다.
        # ponytail: crop을 전부 메모리에 둡니다(224px 기준 한 행에 150KB). 행이
        # 수십만 개가 되면 파일로 내려야 합니다.
        crops: list[Any] = [None] * len(rows)
        by_image: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_image.setdefault(row["image_key"], []).append(index)
        images_directory = root / "images"
        images_directory.mkdir(parents=True, exist_ok=True)
        for done, (image_key, indexes) in enumerate(sorted(by_image.items()), start=1):
            path = store.ensure_local_file(locations[image_key], images_directory)
            try:
                with Image.open(path) as picture:
                    picture.load()
                    for index in indexes:
                        crops[index] = _crop(
                            picture, rows[index]["bbox"], size=crop_size, margin=crop_margin
                        )
            except (OSError, ValueError) as error:
                raise PredictionError(
                    f"이미지를 열지 못했습니다 ({locations[image_key]}): {error}"
                ) from error
            if on_progress is not None:
                on_progress("rerank_crop", done, len(by_image))

        # 자를 수 없었던 행은 재순위에서 빠집니다.
        wanted = [index for index, crop in enumerate(crops) if crop is not None]
        if not wanted:
            raise PredictionError("재순위에 쓸 수 있는 crop이 하나도 없습니다.")

        collected = []
        for uri in checkpoint_uris:
            checkpoint = load_checkpoint_document(store, uri, device=device)
            model = _embedding_model(checkpoint, source=uri, device=device)
            normalisation = checkpoint.get("normalisation") or {}
            mean = torch.tensor(list(normalisation.get("mean", (0.485, 0.456, 0.406))))
            std = torch.tensor(list(normalisation.get("std", (0.229, 0.224, 0.225))))
            mean, std = mean.view(3, 1, 1), std.view(3, 1, 1)
            checkpoint_size = checkpoint.get("crop_size")
            if isinstance(checkpoint_size, int) and checkpoint_size != crop_size:
                raise InputArtifactError(
                    f"{uri}: checkpoint는 {checkpoint_size}px crop으로 학습했는데 은행은 "
                    f"{crop_size}px입니다."
                )

            bank_crops = [
                Image.open(root / "bank" / record["path"]).convert("RGB")
                for record in bank_records
            ]
            reference = _embed_all(
                model,
                bank_crops,
                device=device,
                mean=mean,
                std=std,
                on_progress=(
                    None if on_progress is None
                    else lambda done, total: on_progress("rerank_reference", done, total)
                ),
            )
            for picture in bank_crops:
                picture.close()
            embedded = _embed_all(
                model,
                [crops[index] for index in wanted],
                device=device,
                mean=mean,
                std=std,
                on_progress=(
                    None if on_progress is None
                    else lambda done, total: on_progress("rerank_embed", done, total)
                ),
            )
            collected.append(
                _margins(
                    embedded @ reference.T,
                    bank_categories=[record["category_id"] for record in bank_records],
                    categories=[rows[index]["category_id"] for index in wanted],
                )
            )
            del model, reference, embedded

    margin = torch.stack(collected).mean(dim=0)
    reranked = 0
    values: list[float] = []
    for position, index in enumerate(wanted):
        value = float(margin[position])
        if not math.isfinite(value):
            continue
        # 유사도 차이는 -2..2까지 갈 수 있습니다. 곱하는 값이 음수가 되면 제출할
        # 수 없는 점수가 나오므로 0..1로 자릅니다.
        multiplier = min(1.0, max(0.0, (1.0 + value) / 2.0))
        rows[index]["score"] = float(rows[index]["score"]) * multiplier
        values.append(value)
        reranked += 1

    values.sort()
    summary = {
        "rows": len(rows),
        "reranked_rows": reranked,
        "checkpoints": list(checkpoint_uris),
        "reference_crops": len(bank_records),
        "crop_bank_uri": store.normalize_uri(crop_bank_uri),
        "median_margin": values[len(values) // 2] if values else None,
        "negative_margin_rows": sum(1 for value in values if value < 0),
    }
    return rows, summary


__all__ = ["BATCH_SIZE", "CROP_BANK_INDEX", "load_crop_bank", "rerank_predictions"]
