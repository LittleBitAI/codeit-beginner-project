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
import ntpath
import statistics
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


def _inside(destination: Path, name: str) -> Path:
    """푼 자리 안을 가리키는 경로인지 확인하고 그 경로를 돌려줍니다.

    **문자열 앞자리 비교로는 부족합니다.** 푸는 자리가 `bank`이면 `bank-evil`은
    앞자리가 같아 통과합니다. 경계는 경로 단위로 봐야 하므로 `relative_to`에
    맡깁니다.
    """

    if ntpath.isabs(name) or Path(name).is_absolute():
        raise InputArtifactError(f"crop 은행에 절대 경로가 있습니다: {name}")
    root = destination.resolve()
    target = (destination / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise InputArtifactError(
            f"crop 은행에 폴더 밖을 가리키는 항목이 있습니다: {name}"
        ) from error
    return target


def _safe_member(member: tarfile.TarInfo, destination: Path) -> None:
    """푸는 항목 하나를 검사합니다.

    이름만 봐서는 모자랍니다. symlink와 hardlink는 **가리키는 곳**으로 나갈 수 있고,
    device나 fifo는 애초에 crop 은행에 있을 이유가 없습니다. 은행은 data가 만들지만
    이 함수는 남이 만든 파일도 받습니다.
    """

    if not (member.isfile() or member.isdir()):
        raise InputArtifactError(
            f"crop 은행에 보통 파일이 아닌 항목이 있습니다: {member.name}"
        )
    _inside(destination, member.name)
    if member.linkname:
        _inside(destination, member.linkname)


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
                _safe_member(member, destination)
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
            or not record.get("path")
            or not isinstance(record.get("category_id"), int)
            or isinstance(record.get("category_id"), bool)
        ):
            raise InputArtifactError(f"crop 은행 항목의 형식이 올바르지 않습니다: {uri}")
        # tar을 안전하게 풀었어도 **목록이 다른 곳을 가리킬 수 있습니다.** 그 경로를
        # 여는 것은 재순위 도중이라, 여기서 안 막으면 저장소 밖 파일을 batch마다
        # 읽습니다.
        _inside(destination, str(record["path"]))
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
    # 여기도 `_is_finite_number`를 씁니다. 은행 문서는 JSON이라 자릿수 제한이 없는
    # 정수가 들어올 수 있고, 맨 `math.isfinite`는 그 값에서 스스로 터집니다.
    if not _is_finite_number(value) or value < 0:
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


def _is_finite_number(value: Any) -> bool:
    """유한한 숫자인지 봅니다. **검사가 스스로 터지지 않게** 합니다.

    python의 정수는 크기 제한이 없어서, `10**400` 같은 값에 `math.isfinite`를 부르면
    `OverflowError`가 납니다. 잘못된 checkpoint를 거절하려던 검사가 그 자리에서
    `run()` 밖으로 나가 버립니다.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _normalisation(checkpoint: Mapping[str, Any], *, source: str) -> tuple[Any, Any]:
    """checkpoint가 적어 둔 정규화 값을 읽고 **쓸 수 있는 값인지** 봅니다.

    std에 0이나 유한하지 않은 값이 있으면 특징이 통째로 `nan`이 됩니다. 그러면
    margin도 전부 `nan`이 되고, 못 잰 margin은 건너뛰도록 되어 있으므로 **한 행도
    바꾸지 않은 채 성공으로 끝납니다.** 재순위를 걸었는데 아무 일도 일어나지 않은
    것을 아무도 모릅니다.
    """

    torch = _import_torch()
    raw = checkpoint.get("normalisation")
    if not isinstance(raw, Mapping):
        raise InputArtifactError(f"{source}: checkpoint에 normalisation이 필요합니다.")
    values: list[list[float]] = []
    for key in ("mean", "std"):
        item = raw.get(key)
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 3
            or not all(_is_finite_number(number) for number in item)
        ):
            raise InputArtifactError(
                f"{source}: normalisation.{key}는 유한한 숫자 셋이어야 합니다."
            )
        values.append([float(number) for number in item])
    # 0뿐 아니라 **음수도 막습니다.** 음수 std는 그 channel의 방향을 뒤집어, 오류
    # 없이 특징만 틀어집니다. 표준편차는 정의상 0보다 큽니다.
    if any(number <= 0.0 for number in values[1]):
        raise InputArtifactError(
            f"{source}: normalisation.std는 모두 0보다 커야 합니다."
        )
    mean = torch.tensor(values[0]).view(3, 1, 1)
    std = torch.tensor(values[1]).view(3, 1, 1)
    return mean, std


def _open_bank_crops(root: Path, records: Sequence[Mapping[str, Any]]) -> list[Any]:
    """참조 crop을 엽니다. 없거나 깨진 파일도 계약 오류로 바꿉니다.

    그대로 두면 `FileNotFoundError`나 `UnidentifiedImageError`가 `run()` 밖으로
    나갑니다. 이 pipeline은 그 경계를 넘어 예외를 내보내지 않습니다.
    """

    Image = _import_image()
    opened: list[Any] = []
    try:
        for record in records:
            path = root / str(record["path"])
            with Image.open(path) as picture:
                opened.append(picture.convert("RGB"))
    except (OSError, ValueError) as error:
        for picture in opened:
            picture.close()
        raise InputArtifactError(
            f"참조 crop을 열지 못했습니다 ({error.__class__.__name__}): {error}"
        ) from error
    return opened


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
    """8-way TTA로 특징을 뽑아 하나로 모읍니다. 결과는 L2 정규화된 vector입니다.

    실패를 계약 오류로 바꾸는 일은 부르는 쪽이 합니다 — 여기만 감싸면 batch를
    쌓거나 이어 붙이는 자리에서 난 오류가 그대로 빠져나갑니다.
    """

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
    """행마다 `자기 class 최고 유사도 - 다른 class 최고 유사도`와, **잴 수 있었는지**를 냅니다.

    은행에 없는 class는 잴 수 없습니다. 그때 0을 넣으면 "재 봤더니 확신이
    없더라"가 되어 점수가 반으로 깎이므로 부르는 쪽이 그 행의 점수를 그대로 둡니다.

    "재지 못했다"와 "재려다 값이 망가졌다"를 **따로 냅니다.** 둘을 `nan` 하나로
    합치면, model이 발산해 특징이 전부 `nan`이 된 실행이 "참조가 없는 class뿐이었다"와
    구별되지 않아 한 행도 바꾸지 않은 채 성공으로 끝납니다.
    """

    torch = _import_torch()
    bank = torch.tensor(list(bank_categories))
    result = torch.zeros(similarity.shape[0])
    measurable = torch.zeros(similarity.shape[0], dtype=torch.bool)
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
        measurable[index] = True
    return result, measurable


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
            mean, std = _normalisation(checkpoint, source=uri)
            # **크기는 반드시 대조합니다.** 없거나 정수가 아니라고 건너뛰면, 다른
            # 크기로 학습한 model에 이 은행의 crop을 먹이고도 아무 말이 없습니다.
            # 그 어긋남은 오류가 아니라 조금 낮은 점수로만 드러납니다.
            checkpoint_size = checkpoint.get("crop_size")
            if (
                not isinstance(checkpoint_size, int)
                or isinstance(checkpoint_size, bool)
                or checkpoint_size < 1
            ):
                raise InputArtifactError(
                    f"{uri}: checkpoint가 학습한 crop 크기를 적어 두지 않았습니다."
                )
            if checkpoint_size != crop_size:
                raise InputArtifactError(
                    f"{uri}: checkpoint는 {checkpoint_size}px crop으로 학습했는데 은행은 "
                    f"{crop_size}px입니다."
                )

            bank_crops = _open_bank_crops(root / "bank", bank_records)
            # **특징을 뽑는 구간 전체를 감쌉니다.** 터지는 곳이 model 호출만은
            # 아닙니다. batch를 쌓는 자리, 이어 붙이는 자리, 유사도 행렬을 만드는
            # 자리 모두 메모리를 크게 쓰고, 거기서 난 오류도 `run()` 밖으로 나가면
            # 안 됩니다.
            try:
                reference = _embed_all(
                    model,
                    bank_crops,
                    device=device,
                    mean=mean,
                    std=std,
                    on_progress=(
                        None if on_progress is None
                        else lambda done, total: on_progress(
                            "rerank_reference", done, total
                        )
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
                similarity = embedded @ reference.T
            except (RuntimeError, ValueError, MemoryError) as error:
                raise PredictionError(
                    f"재순위 추론에 실패했습니다 ({error.__class__.__name__}): {error}"
                ) from error
            collected.append(
                _margins(
                    similarity,
                    bank_categories=[record["category_id"] for record in bank_records],
                    categories=[rows[index]["category_id"] for index in wanted],
                )
            )
            del model, reference, embedded, similarity

    margin = torch.stack([item[0] for item in collected]).mean(dim=0)
    # 한 checkpoint라도 재지 못한 행은 재지 못한 것으로 봅니다. 평균에 넣으면 그
    # checkpoint의 0이 다른 checkpoint의 판단을 희석합니다.
    measurable = torch.stack([item[1] for item in collected]).all(dim=0)
    reranked = 0
    values: list[float] = []
    for position, index in enumerate(wanted):
        if not bool(measurable[position]):
            # 참조 crop이 없는 class입니다. 못 잰 것을 0으로 적으면 그 행의 점수가
            # 절반이 되어, "재지 못했다"가 "확신이 없다"로 조용히 바뀝니다.
            continue
        value = float(margin[position])
        if not math.isfinite(value):
            # 잴 수 있는 행인데 값이 망가졌습니다. model이 발산했거나 checkpoint가
            # 깨진 것이고, 조용히 건너뛰면 아무것도 바꾸지 않은 채 성공합니다.
            raise PredictionError(
                "재순위 margin이 숫자가 아닙니다. checkpoint의 가중치나 정규화 값을 "
                "확인하세요."
            )
        # 식은 `(1 + margin) / 2`이고 **자르지 않습니다.** 유사도 차이는 -2..2까지
        # 가므로 곱하는 값은 -0.5..1.5입니다. 위에서 자르면 가장 확신이 센 행이
        # 보통인 행과 같아지고, 아래에서 0으로 막으면 "확실히 다른 class"인 행들이
        # 전부 같은 점수로 뭉쳐 그 사이 순서가 사라집니다. 채점은 순서로 하므로 그
        # 순서가 곧 점수입니다. 제출 계약에도 점수 범위 규칙은 없습니다.
        rows[index]["score"] = float(rows[index]["score"]) * (1.0 + value) / 2.0
        values.append(value)
        reranked += 1

    values.sort()
    summary = {
        "rows": len(rows),
        "reranked_rows": reranked,
        "checkpoints": list(checkpoint_uris),
        "reference_crops": len(bank_records),
        "crop_bank_uri": store.normalize_uri(crop_bank_uri),
        # 짝수 개일 때 가운데 두 값의 평균입니다. `values[len // 2]`는 위쪽 값이라
        # median이 아니고, 보고 숫자가 틀리면 그것을 근거로 다음 판단을 합니다.
        "median_margin": statistics.median(values) if values else None,
        "negative_margin_rows": sum(1 for value in values if value < 0),
    }
    return rows, summary


__all__ = ["BATCH_SIZE", "CROP_BANK_INDEX", "load_crop_bank", "rerank_predictions"]
