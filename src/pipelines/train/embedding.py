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
import random
import tarfile
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


#: crop 은행 안에서 목록이 놓이는 자리입니다. data가 만드는 tar의 규약입니다.
INDEX_MEMBER = "index.json"

#: 학습에 쓰는 crop 한 변입니다. 은행이 이 크기로 잘라 둡니다.
CROP_SIZE = 224

#: ImageNet 정규화 값입니다. backbone이 그 통계로 학습돼 있습니다.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: 밝기와 대비를 이만큼 흔듭니다. 조명이 바뀐 사진을 같은 알약으로 보게 합니다.
BRIGHTNESS_RANGE = (0.7, 1.3)
SHIFT_RANGE = 0.08

#: label smoothing은 특징을 한 점에 몰지 않게 해 참조와의 거리를 재기 좋게 만듭니다.
LABEL_SMOOTHING = 0.1


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
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
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
        output_dir=str(raw.get("output_dir", EMBEDDING_SETTING_DEFAULTS["output_dir"])),
        output_prefix=str(
            raw.get("output_prefix", EMBEDDING_SETTING_DEFAULTS["output_prefix"])
        ),
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


def read_crop_bank(storage: Any, uri: str, destination: Path) -> list[dict[str, Any]]:
    """crop 은행 tar를 풀어 목록을 돌려줍니다.

    data가 만든 파일 규약을 여기서 다시 적습니다. pipeline끼리는 import하지 않으므로
    manifest field 이름을 옮겨 적는 것과 같은 방식입니다.
    """

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "crop_bank.tar"
    try:
        storage.download_file(uri, archive)
        with tarfile.open(archive) as opened:
            for member in opened.getmembers():
                target = (destination / member.name).resolve()
                if not str(target).startswith(str(destination.resolve())):
                    raise EmbeddingTrainingError(
                        f"crop 은행에 위험한 경로가 있습니다: {member.name}"
                    )
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
    return [dict(record) for record in records]


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
) -> dict[str, Any]:
    """쓰는 쪽이 이 하나만 읽고 model을 되살릴 수 있어야 합니다."""

    return {
        "task": "embedding",
        "backbone": setting.backbone,
        # 학습한 class 순서입니다. 이것이 없으면 head를 되살릴 수 없고, 특징만 쓰는
        # 쪽도 자기가 몇 종을 본 model인지 알 수 없습니다.
        "category_ids": list(categories),
        "crop_size": CROP_SIZE,
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

    random.seed(setting.seed)
    np.random.seed(setting.seed)
    torch.manual_seed(setting.seed)

    with tempfile.TemporaryDirectory(prefix="embedding-") as scratch:
        root = Path(scratch) / "bank"
        records = read_crop_bank(storage, inputs["crop_bank_uri"], root)
        categories = sorted({int(record["category_id"]) for record in records})
        labels = {category: index for index, category in enumerate(categories)}
        if len(categories) < 2:
            raise EmbeddingTrainingError(
                "임베딩은 class가 둘 이상이어야 학습할 수 있습니다."
            )

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
        started = time.time()
        working = Path(setting.output_dir) / f".{setting.run_id}.partial"
        working.mkdir(parents=True, exist_ok=True)
        best_path = working / "best_checkpoint.pt"
        last_path = working / "last_checkpoint.pt"

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
                torch.save(_checkpoint(model, setting, categories, epoch, accuracy), last_path)
                if accuracy > best_accuracy:
                    best_accuracy, best_epoch = accuracy, epoch
                    torch.save(
                        _checkpoint(model, setting, categories, epoch, accuracy), best_path
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
            "epochs": history,
        }

        prefix = f"{setting.output_prefix}/{setting.run_id}"
        artifacts = {
            "run_id": setting.run_id,
            "best_checkpoint_uri": storage.upload_file(
                best_path, f"{prefix}/best_checkpoint.pt", overwrite=False
            ),
            "last_checkpoint_uri": storage.upload_file(
                last_path, f"{prefix}/last_checkpoint.pt", overwrite=False
            ),
            "training_history_uri": storage.write_json(
                f"{prefix}/training_history.json", history_document, overwrite=False
            ),
        }
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
        },
        "message": (
            f"crop {len(records)}개, class {len(categories)}종으로 "
            f"{setting.backbone} 임베딩을 {setting.epochs} epoch 학습했습니다"
            f"(가장 좋은 epoch {best_epoch}, 학습 정확도 {best_accuracy:.3f})."
        ),
    }
