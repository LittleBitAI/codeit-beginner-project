"""잘라 낸 알약 crop 하나가 어떤 class인지 재는 자를 학습합니다.

검출기는 976x1280 전체를 한 번에 보지만 이 model은 알약 하나만 224px로 확대해서
봅니다. 그래서 **오차가 검출기와 독립**이고, 검출 결과의 순위를 다시 매기는 데 쓸 수
있습니다. 학습한 것은 class를 맞히는 head가 아니라 그 **직전 특징**이고, 쓰는 쪽은
참조 crop과의 거리를 잽니다 — 그래야 학습에 없던 class도 참조 사진만으로 맞힙니다.

**증강이 이 model의 전부입니다.** 알약은 놓인 방향이 정해져 있지 않고 조명이 바뀌면
색이 흔들립니다. 네 방향 회전과 상하좌우 뒤집기, 밝기·대비 흔들기를 주지 않으면
촬영 각도 세 개를 외우고 끝납니다.
"""

from __future__ import annotations

import json
import ntpath
import os
import random
import tarfile
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.common import StorageError, create_storage
from src.common.train_contract import (
    DEFAULT_EMBEDDING_BACKBONE,
    EMBEDDING_BACKBONES,
    EMBEDDING_DATA_ARTIFACT_KEYS,
    EMBEDDING_SETTING_DEFAULTS,
    EMBEDDING_SETTING_KEYS,
    RUN_ID_PATTERN,
)

from .dataset import REPOSITORY_ROOT


#: crop 은행 안에서 목록이 놓이는 자리입니다. data가 만드는 tar의 규약입니다.
INDEX_MEMBER = "index.json"

#: data가 만드는 은행의 crop 한 변입니다. **여기 값을 checkpoint에 적지 않습니다** —
#: 적으면 다른 크기로 자른 은행으로 학습하고도 224라고 말하게 됩니다. 은행이 적어 둔
#: 값을 읽고, 없으면 거절합니다. 이 상수는 test와 문서가 가리키는 기준값입니다.
DEFAULT_CROP_SIZE = 224

#: ImageNet 정규화 값입니다. backbone이 그 통계로 학습돼 있습니다.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: 밝기와 대비를 이만큼 흔듭니다. 조명이 바뀐 사진을 같은 알약으로 보게 합니다.
BRIGHTNESS_RANGE = (0.7, 1.3)
SHIFT_RANGE = 0.08

#: label smoothing은 특징을 한 점에 몰지 않게 해 참조와의 거리를 재기 좋게 만듭니다.
LABEL_SMOOTHING = 0.1


#: 실행 하나가 자기 이름을 잡았다는 표식과, 끝까지 갔다는 표식입니다. detector가
#: `running/`과 `completed.json`으로 하는 일을 같은 모양으로 둡니다.
COMPLETED_MARKER = "completed.json"
#: 도는 동안의 사본이자 **이름을 잡는 자리**입니다. 첫 조건부 쓰기가 `run_id`를
#: 잡고, 그 순간부터 원격에 자족적인 사본이 하나 있습니다. 표식 파일을 따로 두면
#: 이름만 잡히고 사본은 없는 사이가 벌어집니다.
RUNNING_CHECKPOINT = "running/last_checkpoint.pt"


def _save_checkpoint(payload: dict[str, Any], destination: Path) -> None:
    """임시 file에 쓰고 제자리로 옮깁니다.

    같은 경로에 바로 쓰면 쓰는 도중 죽었을 때 **앞서 멀쩡했던 checkpoint까지**
    반쪽이 됩니다. 30 epoch짜리 학습에서 그 파일이 유일한 사본입니다. detector도
    같은 이유로 임시 file을 거칩니다.
    """

    temporary = destination.with_name(f".{destination.name}.writing")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _guard_storage_root(storage: Any) -> None:
    """저장이 어디서 일어나는지 **쓰기 전에** 봅니다.

    `output_dir`이 저장소 안이어도 `storage.local.root`가 밖이면 저장은 밖에서
    일어납니다. 올린 **뒤에** 자리를 보고 거절하면 이미 저장소 밖에 파일이 남고,
    그 절대 경로가 오류 메시지에 실려 사용자 이름까지 드러납니다.
    """

    root = getattr(storage, "root", None)
    if root is None:
        # S3처럼 local 경로가 없는 backend입니다. 볼 것이 없습니다.
        return
    try:
        Path(root).resolve().relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise EmbeddingTrainingError(
            "storage.local.root가 저장소 밖입니다. 저장소 안으로 두세요."
        ) from error


def _mirror_running(storage: Any, prefix: str, source: Path) -> bool:
    """도는 동안의 마지막 checkpoint를 저장소에도 둡니다. 성공 여부를 돌려줍니다.

    이름만 잡고 사본을 안 올리면, runtime이 끊겼을 때 이름은 막혀 있는데 학습한
    것은 사라집니다. **하나짜리 자족적인 파일**이라 반쯤 갱신된 짝이 생기지
    않습니다.

    올리기가 실패해도 학습은 멈추지 않습니다 — 이 사본은 보험이지 결과가 아니고,
    결과는 마지막에 attempt 자리로 갑니다. 다만 **조용히 넘어가지는 않습니다.**
    한 번도 못 올린 실행은 끊기면 원격에 아무것도 없으므로, 부르는 쪽이 그 사실을
    결과에 적습니다.
    """

    try:
        storage.upload_file(source, f"{prefix}/{RUNNING_CHECKPOINT}", overwrite=True)
    except (StorageError, OSError):
        return False
    return True


def _guard_completed(storage: Any, prefix: str, run_id: str) -> None:
    """이미 끝난 이름인지 봅니다. 읽기만 하므로 아무것도 남기지 않습니다."""

    try:
        finished = storage.exists(f"{prefix}/{COMPLETED_MARKER}")
    except StorageError as error:
        raise EmbeddingTrainingError(
            f"낼 자리를 확인하지 못했습니다: {prefix} ({type(error).__name__})"
        ) from error
    if finished:
        raise EmbeddingTrainingError(
            f"이미 끝난 실행입니다: {run_id}. 다른 run_id를 쓰세요."
        )


def _published(uri: str) -> str:
    """저장 계층이 돌려준 자리를 계약이 정한 표기로 바꿉니다.

    local 저장은 **절대 경로**를 돌려줍니다. 그대로 내보내면 세 가지가 어긋납니다.
    다른 컴퓨터에서 열 수 없고, 저장소 규칙("절대 경로 금지")을 깨고, 화면까지
    흘러가면 OS 사용자 이름이 드러납니다. detector도 같은 규칙으로 상대 경로를
    냅니다(`pipeline._publish_local`).
    """

    if uri.lower().startswith("s3://"):
        return uri
    path = Path(uri)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        # **절대 경로로 돌려주지 않습니다.** `output_dir`은 저장소 안이라도
        # `storage.local.root`가 밖이면 저장은 밖에서 일어납니다. 그때 그 경로를
        # 그대로 내보내면 다른 컴퓨터에서 못 여는 URI와 OS 사용자 이름이 결과에
        # 실려 나갑니다. detector도 이 자리에서 조용히 넘어가지 않습니다.
        raise EmbeddingTrainingError(
            "저장 위치가 저장소 밖입니다. storage.local.root를 저장소 안으로 "
            f"두세요: {path.as_posix()}"
        ) from error


class EmbeddingTrainingError(RuntimeError):
    """임베딩 학습이 시작조차 할 수 없는 경우입니다."""


@dataclass(frozen=True)
class EmbeddingSettings:
    """임베딩 학습에 필요한 config 값입니다."""

    run_id: str
    backbone: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str
    num_workers: int
    pretrained: bool
    checkpoint_every: int
    output_dir: str
    output_prefix: str


def _repository_path(raw: Mapping[str, Any], name: str) -> str:
    """저장소 안을 가리키는 상대 경로만 받습니다.

    detector가 `train.output_dir`에 거는 것과 같은 규칙입니다. 막지 않으면 절대
    경로나 `..`로 저장소 **밖에** checkpoint를 쓰고, 그 자리가 그대로 artifact URI가
    되어 남의 컴퓨터 경로와 사용자 이름까지 결과에 실려 나갑니다.
    """

    value = raw.get(name, EMBEDDING_SETTING_DEFAULTS[name])
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingTrainingError(f"train.{name}은 비어 있지 않은 문자열이어야 합니다.")
    text = value.strip()
    candidate = Path(text)
    if candidate.is_absolute() or ntpath.isabs(text):
        raise EmbeddingTrainingError(f"train.{name}은 저장소 기준 상대 경로여야 합니다.")
    try:
        (REPOSITORY_ROOT / candidate).resolve().relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise EmbeddingTrainingError(
            f"train.{name}이 저장소 밖을 가리킵니다: {text}"
        ) from error
    return text


def _positive_int(raw: Mapping[str, Any], name: str) -> int:
    value = raw.get(name, EMBEDDING_SETTING_DEFAULTS.get(name))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EmbeddingTrainingError(f"train.{name}은 1 이상의 정수여야 합니다.")
    return value


def _positive_number(raw: Mapping[str, Any], name: str) -> float:
    value = raw.get(name, EMBEDDING_SETTING_DEFAULTS.get(name))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise EmbeddingTrainingError(f"train.{name}은 0보다 큰 숫자여야 합니다.")
    return float(value)


def settings(config: Mapping[str, Any]) -> EmbeddingSettings:
    """``config["train"]``을 읽습니다. 쓰지 않는 칸은 조용히 버리지 않고 거부합니다."""

    raw = config.get("train")
    if not isinstance(raw, Mapping):
        raise EmbeddingTrainingError("config['train']은 object여야 합니다.")
    unknown = sorted(set(raw) - set(EMBEDDING_SETTING_KEYS))
    if unknown:
        fields = ", ".join(f"train.{name}" for name in unknown)
        raise EmbeddingTrainingError(
            f"{fields}은(는) task='embedding'이 쓰지 않는 칸입니다."
        )

    run_id = raw.get("run_id")
    # `match`가 아니라 `fullmatch`입니다. python의 `$`는 **끝의 줄바꿈 앞에서도**
    # 맞으므로, `match`로는 `name\n`이 통과해 그 이름이 경로에 실려 갑니다.
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise EmbeddingTrainingError("train.run_id가 이름 규칙에 맞지 않습니다.")

    backbone = raw.get("backbone", DEFAULT_EMBEDDING_BACKBONE)
    if backbone not in EMBEDDING_BACKBONES:
        raise EmbeddingTrainingError(
            f"train.backbone은 {', '.join(EMBEDDING_BACKBONES)} 중 하나여야 합니다."
        )

    device = raw.get("device", EMBEDDING_SETTING_DEFAULTS["device"])
    if device not in ("cpu", "cuda"):
        raise EmbeddingTrainingError("train.device는 cpu 또는 cuda여야 합니다.")
    if device == "cuda" and not torch.cuda.is_available():
        raise EmbeddingTrainingError("train.device=cuda인데 CUDA를 쓸 수 없습니다.")

    pretrained = raw.get("pretrained", EMBEDDING_SETTING_DEFAULTS["pretrained"])
    if not isinstance(pretrained, bool):
        raise EmbeddingTrainingError("train.pretrained는 true 또는 false여야 합니다.")

    workers = raw.get("num_workers", 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise EmbeddingTrainingError("train.num_workers는 0 이상의 정수여야 합니다.")

    seed = raw.get("seed", EMBEDDING_SETTING_DEFAULTS["seed"])
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EmbeddingTrainingError("train.seed는 0 이상의 정수여야 합니다.")

    return EmbeddingSettings(
        run_id=run_id,
        backbone=backbone,
        epochs=_positive_int(raw, "epochs"),
        batch_size=_positive_int(raw, "batch_size"),
        learning_rate=_positive_number(raw, "learning_rate"),
        weight_decay=_positive_number(raw, "weight_decay"),
        seed=seed,
        device=device,
        num_workers=workers,
        pretrained=pretrained,
        checkpoint_every=_positive_int(raw, "checkpoint_every"),
        output_dir=_repository_path(raw, "output_dir"),
        output_prefix=_repository_path(raw, "output_prefix"),
    )


def data_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    """crop 은행과 class map의 자리를 읽습니다."""

    inputs = config.get("inputs")
    data = inputs.get("data") if isinstance(inputs, Mapping) else None
    if not isinstance(data, Mapping):
        raise EmbeddingTrainingError("config['inputs']['data']는 object여야 합니다.")
    missing = [key for key in EMBEDDING_DATA_ARTIFACT_KEYS if not data.get(key)]
    if missing:
        raise EmbeddingTrainingError(
            "임베딩 학습에는 " + ", ".join(missing) + "이(가) 필요합니다."
        )
    return {key: str(data[key]) for key in EMBEDDING_DATA_ARTIFACT_KEYS}


def _inside(destination: Path, name: str) -> Path:
    """푼 자리 안을 가리키는 경로인지 확인하고 그 경로를 돌려줍니다.

    **문자열 앞자리 비교로는 부족합니다.** `dest`와 `dest-evil`은 앞자리가 같아
    통과합니다. 경계는 경로 단위로 봐야 하므로 `relative_to`에 맡깁니다.
    """

    if ntpath.isabs(name) or Path(name).is_absolute():
        raise EmbeddingTrainingError(f"crop 은행에 절대 경로가 있습니다: {name}")
    root = destination.resolve()
    target = (destination / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise EmbeddingTrainingError(
            f"crop 은행에 폴더 밖을 가리키는 경로가 있습니다: {name}"
        ) from error
    return target


def _safe_member(member: tarfile.TarInfo, destination: Path) -> None:
    """푸는 항목 하나를 검사합니다.

    이름만 봐서는 모자랍니다. symlink와 hardlink는 **가리키는 곳**으로 나갈 수 있고,
    device나 fifo는 애초에 crop 은행에 있을 이유가 없습니다. 은행은 우리가 만들지만
    읽는 쪽은 남이 만든 파일도 받습니다.
    """

    if not (member.isfile() or member.isdir()):
        raise EmbeddingTrainingError(
            f"crop 은행에 보통 파일이 아닌 항목이 있습니다: {member.name}"
        )
    _inside(destination, member.name)
    if member.linkname:
        _inside(destination, member.linkname)


def read_crop_bank(storage: Any, uri: str, destination: Path) -> dict[str, Any]:
    """crop 은행 tar를 풀어 **목록 문서를** 돌려줍니다.

    data가 만든 파일 규약을 여기서 다시 적습니다. pipeline끼리는 import하지 않으므로
    manifest field 이름을 옮겨 적는 것과 같은 방식입니다.

    `records`만 돌려주지 않는 것은 은행이 자기 `crop_size`를 적어 두기 때문입니다.
    그 값을 버리면 학습한 크기와 checkpoint에 적는 크기가 갈릴 수 있습니다.
    """

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "crop_bank.tar"
    try:
        storage.download_file(uri, archive)
        with tarfile.open(archive) as opened:
            for member in opened.getmembers():
                _safe_member(member, destination)
            opened.extractall(destination)
        document = json.loads((destination / INDEX_MEMBER).read_text(encoding="utf-8"))
    except (StorageError, OSError, ValueError, tarfile.TarError) as error:
        raise EmbeddingTrainingError(
            f"crop 은행을 읽지 못했습니다: {uri} ({type(error).__name__})"
        ) from error
    finally:
        archive.unlink(missing_ok=True)
    records = document.get("records") if isinstance(document, Mapping) else None
    if not records:
        raise EmbeddingTrainingError(f"crop 은행이 비어 있습니다: {uri}")

    checked: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise EmbeddingTrainingError(f"crop 은행 항목이 object가 아닙니다: {uri}")
        path = record.get("path")
        category_id = record.get("category_id")
        if not isinstance(path, str) or not path:
            raise EmbeddingTrainingError(f"crop 은행 항목에 path가 없습니다: {uri}")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise EmbeddingTrainingError(
                f"crop 은행 항목의 category_id가 정수가 아닙니다: {uri}"
            )
        # tar을 안전하게 풀었어도 **목록이 다른 곳을 가리킬 수 있습니다.** 여는 것은
        # 학습 중이라, 여기서 안 막으면 저장소 밖 파일을 batch마다 읽습니다.
        _inside(destination, path)
        checked.append(dict(record))

    # **없으면 거절합니다.** data가 만드는 은행은 언제나 이 값을 적습니다. 없는데
    # 224로 짐작하면 64px 은행으로 학습하고도 checkpoint는 224라고 말합니다.
    crop_size = document.get("crop_size")
    if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size < 1:
        raise EmbeddingTrainingError(
            f"crop 은행이 crop_size를 적어 두지 않았습니다: {uri}"
        )
    return {"records": checked, "crop_size": crop_size}


def check_class_map(storage: Any, uri: str, categories: list[int]) -> None:
    """은행의 class가 정말 그 dataset의 class인지 봅니다.

    이 입력을 필수로 두고 읽지 않으면, 다른 dataset의 class map을 붙여도 학습이
    그냥 성공합니다. 그러면 checkpoint의 `category_ids`가 어느 dataset의 것인지
    아무도 보증하지 않은 채 재순위까지 흘러갑니다.

    이름은 쓰지 않고 **id 집합만** 봅니다. 학습에는 이름이 필요 없고, 여기서 이름까지
    맞추라고 하면 표기가 조금 다른 판을 이유 없이 막습니다.
    """

    try:
        document = storage.read_json(uri)
    except (StorageError, OSError, ValueError) as error:
        raise EmbeddingTrainingError(
            f"class map을 읽지 못했습니다: {uri} ({type(error).__name__})"
        ) from error
    if not isinstance(document, Mapping) or not document:
        raise EmbeddingTrainingError(f"class map이 비어 있습니다: {uri}")

    # data가 내는 두 형태를 모두 받습니다: {"7": "pill"}과 {"pill": 7}.
    known: set[int] = set()
    for key, value in document.items():
        for candidate in (key, value):
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                known.add(candidate)
            elif isinstance(candidate, str) and candidate.strip().lstrip("-").isdigit():
                known.add(int(candidate))
    missing = sorted(set(categories) - known)
    if missing:
        raise EmbeddingTrainingError(
            "crop 은행의 class가 class map에 없습니다. 다른 dataset의 값을 짝지은 "
            f"것은 아닌지 확인하세요: {', '.join(str(item) for item in missing[:5])}"
        )


class CropDataset(Dataset):
    """crop 하나와 그 class 번호를 돌려줍니다."""

    def __init__(self, root: Path, records: list[dict[str, Any]], labels: dict[int, int]):
        self.root = root
        self.records = records
        self.labels = labels
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(self.root / record["path"]) as picture:
            array = np.asarray(picture.convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
        # 알약은 놓인 방향이 없습니다. 네 방향 회전과 뒤집기는 전부 같은 알약입니다.
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        if random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[1])
        turns = random.randint(0, 3)
        if turns:
            tensor = torch.rot90(tensor, turns, dims=[1, 2])
        tensor = tensor * random.uniform(*BRIGHTNESS_RANGE)
        tensor = tensor + random.uniform(-SHIFT_RANGE, SHIFT_RANGE)
        tensor = tensor.clamp_(0.0, 1.0)
        return (tensor - self.mean) / self.std, self.labels[record["category_id"]]


def build_model(backbone: str, class_count: int, *, pretrained: bool) -> torch.nn.Module:
    """분류 head를 붙인 backbone입니다. 쓰는 쪽은 그 head를 떼고 특징만 씁니다."""

    factory = getattr(torchvision.models, backbone)
    model = factory(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    return model


def _checkpoint(
    model: torch.nn.Module,
    setting: EmbeddingSettings,
    categories: list[int],
    epoch: int,
    accuracy: float,
    crop_size: int,
) -> dict[str, Any]:
    """쓰는 쪽이 이 하나만 읽고 model을 되살릴 수 있어야 합니다."""

    return {
        "task": "embedding",
        "backbone": setting.backbone,
        # 학습한 class 순서입니다. 이것이 없으면 head를 되살릴 수 없고, 특징만 쓰는
        # 쪽도 자기가 몇 종을 본 model인지 알 수 없습니다.
        "category_ids": list(categories),
        # **은행이 적은 값입니다.** 상수를 적으면 다른 크기로 자른 은행으로 학습하고도
        # 224라고 말하게 되고, 쓰는 쪽은 그 값으로 시험 crop을 자릅니다.
        "crop_size": crop_size,
        "normalisation": {"mean": list(MEAN), "std": list(STD)},
        "epoch": epoch,
        "train_accuracy": accuracy,
        "training_config": {
            "backbone": setting.backbone,
            "epochs": setting.epochs,
            "batch_size": setting.batch_size,
            "learning_rate": setting.learning_rate,
            "weight_decay": setting.weight_decay,
            "seed": setting.seed,
            "pretrained": setting.pretrained,
        },
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }


def train_embedding(config: Mapping[str, Any]) -> dict[str, Any]:
    """crop 은행으로 임베딩을 학습하고 checkpoint 둘과 학습 이력을 남깁니다."""

    setting = settings(config)
    inputs = data_inputs(config)
    storage = create_storage(config)

    # detector와 같은 규칙입니다(`trainer.py`). seed만 심고 algorithm mode를 두지
    # 않으면 CUDA가 비결정 kernel을 골라, 같은 seed로 돌린 두 실행의 checkpoint가
    # 달라집니다. 전역 상태이므로 끝나면 되돌립니다.
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    random.seed(setting.seed)
    np.random.seed(setting.seed)
    torch.manual_seed(setting.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        return _train_embedding(config, setting, inputs, storage)
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )


def _train_embedding(
    config: Mapping[str, Any],
    setting: EmbeddingSettings,
    inputs: dict[str, str],
    storage: Any,
) -> dict[str, Any]:
    """`train_embedding`이 결정성 설정을 걸어 둔 채로 부르는 본체입니다."""

    # **아무것도 쓰기 전에** 저장이 어디서 일어나는지 봅니다. 늦게 보면 저장소
    # 밖에 파일을 쓴 뒤에 거절합니다.
    _guard_storage_root(storage)
    prefix = f"{setting.output_prefix}/{setting.run_id}"
    _guard_completed(storage, prefix, setting.run_id)

    with tempfile.TemporaryDirectory(prefix="embedding-") as scratch:
        root = Path(scratch) / "bank"
        bank = read_crop_bank(storage, inputs["crop_bank_uri"], root)
        records, crop_size = bank["records"], bank["crop_size"]
        categories = sorted({int(record["category_id"]) for record in records})
        labels = {category: index for index, category in enumerate(categories)}
        if len(categories) < 2:
            raise EmbeddingTrainingError(
                "임베딩은 class가 둘 이상이어야 학습할 수 있습니다."
            )
        check_class_map(storage, inputs["class_map_uri"], categories)

        device = torch.device(setting.device)
        model = build_model(
            setting.backbone, len(categories), pretrained=setting.pretrained
        ).to(device)
        loader = DataLoader(
            CropDataset(root, records, labels),
            batch_size=setting.batch_size,
            shuffle=True,
            num_workers=setting.num_workers,
            drop_last=len(records) > setting.batch_size,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, setting.epochs * len(loader))
        )
        loss_function = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        history: list[dict[str, Any]] = []
        best_accuracy = -1.0
        best_epoch = 0
        # 원격 사본이 담고 있는 epoch입니다. 이름을 잡는 순간 0으로 시작하고,
        # 그 뒤로는 갱신에 성공한 epoch입니다. **원격 사본 자체가 자기 epoch을
        # 적고 있으므로**, 이 값은 편의일 뿐 근거는 그 파일입니다.
        mirrored_epoch = 0
        started = time.time()
        # **작업 directory는 만들어 잡습니다.** 이미 있으면 중단된 앞선 실행이
        # 거기 있고, 그 안의 checkpoint가 그 학습의 **유일한 사본**입니다. 그대로
        # 이어 쓰면 다시 돌린 사람이 앞선 밤을 지웁니다. `exist_ok=False`라야
        # 만들기와 확인이 한 번에 일어나, 두 실행이 동시에 시작해도 하나만 잡습니다.
        working = REPOSITORY_ROOT / setting.output_dir / f".{setting.run_id}.partial"
        try:
            working.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise EmbeddingTrainingError(
                f"같은 이름의 중단된 학습이 남아 있습니다: {setting.run_id}. "
                "그 안의 checkpoint가 유일한 사본이므로 지우거나 다른 run_id를 "
                "쓰는 것은 사람이 정합니다."
            ) from error
        best_path = working / "best_checkpoint.pt"
        last_path = working / "last_checkpoint.pt"

        # **이름 잡기와 원격 사본을 하나로 둡니다.** 아직 한 batch도 돌지 않았지만
        # 지금 model 그대로가 자족적인 checkpoint이므로, 그것을 조건부로 올려
        # `running/`을 만듭니다. 그 조건부 쓰기 하나가 두 가지를 다 합니다.
        #
        # - 진 쪽은 **첫 batch 전에** 멈춥니다.
        # - 이긴 쪽은 그 순간부터 원격에 사본을 갖습니다. 이름만 남고 학습은
        #   사라지는 창이 없습니다.
        #
        # 표식 파일을 따로 두면 그 사이가 벌어집니다. detector의 "첫 조건부 쓰기가
        # run_id를 claim한다"와 같은 규칙입니다.
        _save_checkpoint(
            _checkpoint(model, setting, categories, 0, 0.0, crop_size), last_path
        )
        try:
            storage.upload_file(
                last_path, f"{prefix}/{RUNNING_CHECKPOINT}", overwrite=False
            )
        except (StorageError, OSError) as error:
            # 원격 이름을 다른 실행이 먼저 잡았습니다. 방금 만든 이 자리는 **이
            # 실행이 만든 것**이라 치웁니다. 남겨 두면 다음 시도가 "중단된 학습이
            # 있다"에 막히는데, 그 안에는 이 실행이 방금 쓴 것밖에 없습니다.
            last_path.unlink(missing_ok=True)
            working.rmdir()
            raise EmbeddingTrainingError(
                f"'{setting.run_id}' 이름을 잡지 못했습니다. 같은 이름의 실행이 돌고 "
                "있거나 끊긴 채 남아 있습니다. 치울지는 사람이 정합니다."
            ) from error

        for epoch in range(1, setting.epochs + 1):
            model.train()
            correct = seen = 0
            total_loss = 0.0
            for batch, target in loader:
                batch, target = batch.to(device), target.to(device)
                output = model(batch)
                loss = loss_function(output, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
                total_loss += float(loss.item()) * len(target)
                correct += int((output.argmax(1) == target).sum().item())
                seen += len(target)
            accuracy = correct / seen if seen else 0.0
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(total_loss / seen, 6) if seen else None,
                    "train_accuracy": round(accuracy, 6),
                }
            )
            if epoch % setting.checkpoint_every == 0 or epoch == setting.epochs:
                _save_checkpoint(
                    _checkpoint(model, setting, categories, epoch, accuracy, crop_size),last_path
                )
                # **저장소에도 그때그때 올려 둡니다.** 이름만 잡아 두고 사본을 안
                # 올리면, Colab runtime이 끊겼을 때 이름은 막혀 있는데 학습한 것은
                # 사라집니다. detector의 `running/`과 같은 자리입니다.
                if _mirror_running(storage, prefix, last_path):
                    mirrored_epoch = epoch
            # **best는 주기와 무관하게 매 epoch 봅니다.** 주기 안에 두면 주기 사이에
            # 나온 가장 좋은 epoch이 best가 되지 못하고, 더 나쁜 model이 조용히
            # best_checkpoint.pt라는 이름으로 나갑니다. 쓰는 쪽은 그 이름을 믿습니다.
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                _save_checkpoint(
                    _checkpoint(model, setting, categories, epoch, accuracy, crop_size),best_path
                )

        if not best_path.exists():
            raise EmbeddingTrainingError("checkpoint를 하나도 남기지 못했습니다.")

        history_document = {
            "run_id": setting.run_id,
            "task": "embedding",
            "backbone": setting.backbone,
            "class_count": len(categories),
            "crop_count": len(records),
            "best_epoch": best_epoch,
            "best_train_accuracy": round(best_accuracy, 6),
            "elapsed_seconds": round(time.time() - started, 1),
            # 원격 사본이 담고 있는 epoch입니다. 이름을 잡을 때 epoch 0으로
            # 시작하므로 **사본은 언제나 있습니다.** 0이면 그 뒤 갱신이 한 번도
            # 성공하지 못했다는 뜻이고, 지금 끊기면 원격에는 학습 전 model만
            # 남습니다.
            "running_mirror_epoch": mirrored_epoch,
            "epochs": history,
        }

        # **한 번 쓰고 마는 attempt 자리에 올립니다.** 고정된 이름에 세 번 올리면
        # 두 번째에서 끊겼을 때 반쪽이 그 이름을 차지해, 그 자리를 다시 쓰려면
        # 사람이 지워야 합니다. attempt 자리는 실행마다 새 이름이라 끊긴 것은
        # 그냥 버려진 폴더로 남습니다. detector의 게시와 같은 모양입니다.
        attempt = f"{prefix}/attempts/{uuid4().hex}"
        artifacts = {
            "run_id": setting.run_id,
            "best_checkpoint_uri": _published(
                storage.upload_file(best_path, f"{attempt}/best_checkpoint.pt")
            ),
            "last_checkpoint_uri": _published(
                storage.upload_file(last_path, f"{attempt}/last_checkpoint.pt")
            ),
            "training_history_uri": _published(
                storage.write_json(f"{attempt}/training_history.json", history_document)
            ),
        }
        # **마지막에 끝났다는 표식을 남깁니다.** 셋이 다 올라간 뒤라야 다음 실행이
        # "이미 끝난 이름"과 "끊긴 이름"을 구별할 수 있습니다.
        storage.write_json(
            f"{prefix}/{COMPLETED_MARKER}",
            {"run_id": setting.run_id, "artifacts": artifacts},
            overwrite=False,
        )
        best_path.unlink(missing_ok=True)
        last_path.unlink(missing_ok=True)
        working.rmdir()

    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": {
            "pipeline": "train",
            "task": "embedding",
            "backbone": setting.backbone,
            "class_count": len(categories),
            "crop_count": len(records),
            "epochs": setting.epochs,
            "best_epoch": best_epoch,
            "best_train_accuracy": round(best_accuracy, 6),
            "seed": setting.seed,
            "device": setting.device,
            # 원격 사본이 담고 있는 epoch입니다. 이름을 잡을 때 0으로 시작하므로
            # **사본이 없는 경우는 없습니다.** 0으로 남았으면 도는 동안의 갱신이
            # 한 번도 성공하지 못했다는 뜻입니다.
            "running_mirror_epoch": mirrored_epoch,
        },
        "message": (
            f"crop {len(records)}개, class {len(categories)}종으로 "
            f"{setting.backbone} 임베딩을 {setting.epochs} epoch 학습했습니다"
            f"(가장 좋은 epoch {best_epoch}, 학습 정확도 {best_accuracy:.3f})."
        ),
    }
