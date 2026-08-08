"""재현 가능한 torchvision detection 학습 loop입니다."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .errors import TrainError
from .progress import ProgressEmitter


SUPPORTED_OPTIMIZERS = ("AdamW", "SGD", "Adam")


class _PrecisionRuntime:
    """정규화된 정밀도 설정을 autocast와 optimizer step에 적용합니다."""

    def __init__(self, precision: Mapping[str, Any], device: torch.device) -> None:
        self._device = device
        dtype_name = precision.get("dtype", "fp32")
        self._dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(dtype_name)
        self._scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if precision.get("grad_scaler") is True
            else None
        )

    def autocast(self) -> Any:
        """AMP가 꺼져 있으면 기존 fp32 실행 context를 그대로 돌려줍니다."""

        if self._dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self._device.type, dtype=self._dtype)

    def backward_and_step(
        self, loss: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        """fp16은 scale하고, 나머지 dtype은 기존 backward/step을 사용합니다."""

        if self._scaler is None:
            loss.backward()
            optimizer.step()
            return
        self._scaler.scale(loss).backward()
        self._scaler.step(optimizer)
        self._scaler.update()


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
) -> tuple[torch.Tensor, dict[str, float]]:
    moved_images = [image.to(device) for image in images]
    moved_targets = [_move_target(target, device) for target in targets]
    losses = model(moved_images, moved_targets)
    if not isinstance(losses, Mapping) or not losses:
        raise TrainError("detector did not return a non-empty loss mapping")
    components: dict[str, float] = {}
    tensors: list[torch.Tensor] = []
    for name, loss in losses.items():
        if not isinstance(name, str) or not name:
            raise TrainError("detector loss names must be non-empty strings")
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TrainError(f"detector loss '{name}' must be a scalar tensor")
        if not bool(torch.isfinite(loss).item()):
            raise TrainError(f"training produced a non-finite loss: {name}")
        tensors.append(loss)
        components[name] = float(loss.detach().cpu())
    total = torch.stack(tensors).sum()
    if not torch.isfinite(total):
        raise TrainError("training produced a non-finite total loss")
    return total, components


def _add_components(
    totals: dict[str, float],
    components: Mapping[str, float],
    *,
    phase: str,
) -> None:
    """Batch마다 loss key가 바뀌어 잘못된 평균이 생기는 것을 막습니다."""
    if totals and set(totals) != set(components):
        raise TrainError(f"detector {phase} loss names changed between batches")
    if not totals:
        totals.update({name: 0.0 for name in components})
    for name, value in components.items():
        totals[name] += value


def _average_components(totals: Mapping[str, float], batches: int) -> dict[str, float]:
    return {name: value / batches for name, value in totals.items()}


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
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
    precision = _PrecisionRuntime(
        settings.get(
            "precision",
            {"mode": "fp32", "dtype": "fp32", "grad_scaler": False},
        ),
        device,
    )

    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_checkpoint: dict[str, Any] | None = None
    early_stopping = settings.get("early_stopping")
    early_reference_loss = math.inf
    epochs_without_improvement = 0
    for epoch in range(1, settings["epochs"] + 1):
        epoch_started_at = time.perf_counter()
        if progress is not None:
            progress.emit("epoch_started", epoch=epoch, epochs=settings["epochs"])
        model.train()
        train_total = 0.0
        train_component_totals: dict[str, float] = {}
        for step, (images, targets) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                loss, components = _loss(model, images, targets, device)
            precision.backward_and_step(loss, optimizer)
            train_total += float(loss.detach().cpu())
            _add_components(train_component_totals, components, phase="train")
            if progress is not None:
                progress.emit_step_progress(
                    epoch=epoch,
                    epochs=settings["epochs"],
                    phase="train",
                    step=step,
                    total_steps=len(train_loader),
                )

        model.train()
        _freeze_batch_norm(model)
        validation_total = 0.0
        validation_component_totals: dict[str, float] = {}
        with torch.no_grad():
            for step, (images, targets) in enumerate(validation_loader, start=1):
                with precision.autocast():
                    loss, components = _loss(model, images, targets, device)
                validation_total += float(loss.detach().cpu())
                _add_components(
                    validation_component_totals,
                    components,
                    phase="validation",
                )
                if progress is not None:
                    progress.emit_step_progress(
                        epoch=epoch,
                        epochs=settings["epochs"],
                        phase="validation",
                        step=step,
                        total_steps=len(validation_loader),
                    )

        train_loss = train_total / len(train_loader)
        validation_loss = validation_total / len(validation_loader)
        train_loss_components = _average_components(
            train_component_totals, len(train_loader)
        )
        validation_loss_components = _average_components(
            validation_component_totals, len(validation_loader)
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_loss_components": train_loss_components,
            "validation_loss_components": validation_loss_components,
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
        should_stop = False
        if early_stopping is not None:
            min_delta = early_stopping["min_delta"]
            if validation_loss < early_reference_loss - min_delta:
                early_reference_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                should_stop = epochs_without_improvement >= early_stopping["patience"]
        if progress is not None:
            progress.emit(
                "epoch_completed",
                epoch=epoch,
                epochs=settings["epochs"],
                train_loss=train_loss,
                validation_loss=validation_loss,
                train_loss_components=train_loss_components,
                validation_loss_components=validation_loss_components,
                best_validation_loss=best_loss,
                best_epoch=best_epoch,
                is_best=is_best,
                epoch_seconds=round(time.perf_counter() - epoch_started_at, 3),
            )
        if should_stop:
            break

    if best_checkpoint is None:
        raise RuntimeError("training completed without a best checkpoint")
    last_checkpoint = {
        "epoch": history[-1]["epoch"],
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
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
