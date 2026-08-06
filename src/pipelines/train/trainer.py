"""재현 가능한 torchvision detection 학습 loop입니다."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .progress import ProgressEmitter


SUPPORTED_OPTIMIZERS = ("AdamW", "SGD", "Adam")


def build_optimizer(
    parameters: list[nn.Parameter], settings: Mapping[str, Any]
) -> torch.optim.Optimizer:
    """정규화된 설정으로 optimizer를 만듭니다."""
    common = {
        "lr": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
    }
    name = settings.get("optimizer", "SGD")
    if name == "SGD":
        return torch.optim.SGD(parameters, momentum=settings["momentum"], **common)
    adam = {
        "betas": (settings.get("beta1", 0.9), settings.get("beta2", 0.999)),
        "eps": settings.get("epsilon", 1e-8),
    }
    if name == "AdamW":
        return torch.optim.AdamW(parameters, **common, **adam)
    if name == "Adam":
        return torch.optim.Adam(parameters, **common, **adam)
    raise ValueError(f"unsupported train optimizer: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(batch: list[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return tuple(zip(*batch))


def _move_target(target: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in target.items()}


def _loss(
    model: nn.Module,
    images: tuple[torch.Tensor, ...],
    targets: tuple[Mapping[str, torch.Tensor], ...],
    device: torch.device,
) -> torch.Tensor:
    moved_images = [image.to(device) for image in images]
    moved_targets = [_move_target(target, device) for target in targets]
    losses = model(moved_images, moved_targets)
    if not isinstance(losses, Mapping) or not losses:
        raise RuntimeError("Faster R-CNN did not return a non-empty loss mapping")
    total = sum(loss for loss in losses.values())
    if not torch.isfinite(total):
        raise RuntimeError("training produced a non-finite loss")
    return total


def _state_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _freeze_batch_norm(model: nn.Module) -> None:
    """Prevent validation images from changing BatchNorm running statistics."""
    batch_norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, batch_norm_types):
            module.eval()


def _train_model(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    settings: Mapping[str, Any],
    progress: ProgressEmitter | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, float | int]]]:
    seed = settings["seed"]
    device = torch.device(settings["device"])
    model.to(device)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        collate_fn=_collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        collate_fn=_collate,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    optimizer = build_optimizer(parameters, settings)

    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_epoch = 0
    best_checkpoint: dict[str, Any] | None = None
    for epoch in range(1, settings["epochs"] + 1):
        epoch_started_at = time.perf_counter()
        if progress is not None:
            progress.emit("epoch_started", epoch=epoch, epochs=settings["epochs"])
        model.train()
        train_total = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, images, targets, device)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu())

        model.train()
        _freeze_batch_norm(model)
        validation_total = 0.0
        with torch.no_grad():
            for images, targets in validation_loader:
                validation_total += float(_loss(model, images, targets, device).detach().cpu())

        train_loss = train_total / len(train_loader)
        validation_loss = validation_total / len(validation_loader)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(epoch_record)
        is_best = validation_loss < best_loss
        if is_best:
            best_loss = validation_loss
            best_epoch = epoch
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": _state_on_cpu(model),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "validation_loss": validation_loss,
            }
        if progress is not None:
            progress.emit(
                "epoch_completed",
                epoch=epoch,
                epochs=settings["epochs"],
                train_loss=train_loss,
                validation_loss=validation_loss,
                best_validation_loss=best_loss,
                best_epoch=best_epoch,
                is_best=is_best,
                epoch_seconds=round(time.perf_counter() - epoch_started_at, 3),
            )

    if best_checkpoint is None:
        raise RuntimeError("training completed without a best checkpoint")
    last_checkpoint = {
        "epoch": settings["epochs"],
        "model_state_dict": _state_on_cpu(model),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "validation_loss": history[-1]["validation_loss"],
    }
    return best_checkpoint, last_checkpoint, history


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    settings: Mapping[str, Any],
    progress: ProgressEmitter | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, float | int]]]:
    """Train deterministically without leaving the global algorithm mode changed."""
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        set_seed(settings["seed"])
        torch.use_deterministic_algorithms(True, warn_only=True)
        return _train_model(model, train_dataset, validation_dataset, settings, progress)
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
