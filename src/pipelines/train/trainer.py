"""재현 가능한 torchvision detection 학습 loop입니다."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.common import S3Storage
from src.common.train_contract import OPTIMIZERS as SUPPORTED_OPTIMIZERS

from .errors import TrainError
from .progress import ProgressEmitter


RESUME_STATE_VERSION = 2


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

    def backward(self, loss: torch.Tensor) -> None:
        """gradient를 쌓습니다. fp16은 scale한 뒤 쌓습니다.

        갱신과 나눠 둔 이유는 gradient accumulation 때문입니다. microbatch마다
        backward는 하되 optimizer는 묶음이 끝날 때 한 번만 움직여야 합니다.
        """

        if self._scaler is None:
            loss.backward()
            return
        self._scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """쌓인 gradient로 가중치를 갱신합니다."""

        if self._scaler is None:
            optimizer.step()
            return
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self) -> dict[str, Any] | None:
        """fp16 GradScaler 상태를 checkpoint에 넣을 수 있게 복사합니다."""

        if self._scaler is None:
            return None
        return copy.deepcopy(self._scaler.state_dict())

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        """재개한 정밀도와 일치하는 GradScaler 상태를 되돌립니다."""

        if self._scaler is None:
            if state is not None:
                raise ValueError(
                    "resume checkpoint has GradScaler state but current precision does not"
                )
            return
        if not isinstance(state, Mapping):
            raise ValueError(
                "resume checkpoint is missing GradScaler state required by current precision"
            )
        self._scaler.load_state_dict(copy.deepcopy(dict(state)))


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


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: Mapping[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """정규화된 설정으로 learning rate schedule을 만듭니다.

    설정이 없으면 ``None``을 돌려주고 학습은 지금까지처럼 상수 learning rate로 돕니다.
    이것이 schedule을 몰랐던 옛 checkpoint를 그대로 이어서 학습할 수 있게 하는 조건이기도
    합니다.

    배율은 epoch이 아니라 **optimizer를 갱신할 때마다** 계산합니다. warmup을 epoch
    단위로만 셀 수 있으면 가장 짧은 warmup이 1 epoch인데, 이미지가 만 장이 넘는 지금은
    그것도 수천 번입니다.

    ``steps_per_epoch``도 batch 수가 아니라 **갱신 수**를 받습니다. gradient
    accumulation을 쓰면 batch마다 갱신하지 않으므로 둘이 달라지고, batch 수로 재면
    schedule이 그만큼 짧게 걸어 ``linear``와 ``cosine``이 설정한 최저 배율에 닿지
    못합니다. accumulation이 1이면 둘은 같습니다.
    """

    schedule = settings.get("lr_scheduler")
    if schedule is None:
        return None
    name = schedule["name"]
    warmup_steps = schedule["warmup_steps"]
    start_factor = schedule["warmup_start_factor"]
    # 전체 길이는 ``epochs``에서 옵니다. 이어서 학습하며 epochs를 늘리면 남은 곡선도
    # 그만큼 늘어납니다. epochs를 "남은 수가 아니라 전체 목표"로 읽는 규칙 그대로입니다.
    total_steps = max(1, settings["epochs"] * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return start_factor + (1.0 - start_factor) * (step / warmup_steps)
        if name == "step":
            # 이것만 epoch 단위입니다. "몇 epoch마다 줄인다"가 곧 설정이기 때문입니다.
            return schedule["gamma"] ** (
                (step // steps_per_epoch) // schedule["step_size"]
            )
        if name == "none":
            return 1.0
        minimum = schedule["min_lr_factor"]
        # 남은 step에서 1을 빼야 **마지막 갱신이** min_lr_factor를 실제로 씁니다.
        # 빼지 않으면 설정한 최저 learning rate는 한 번도 쓰이지 않고 끝납니다.
        progress = min(
            1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        )
        if name == "linear":
            return 1.0 + (minimum - 1.0) * progress
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(generator: torch.Generator) -> dict[str, Any]:
    """이어서 학습할 때 되돌릴 난수 상태를 모읍니다.

    checkpoint는 ``weights_only=True``로 다시 읽으므로 tensor와 기본 자료형만 담습니다.
    numpy 상태는 ndarray를 품고 있어서 평범한 정수 list로 바꿉니다.
    """

    python_version, python_keys, python_gauss = random.getstate()
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_gauss = (
        np.random.get_state()
    )
    return {
        "python": [python_version, [int(key) for key in python_keys], python_gauss],
        "numpy": [
            numpy_name,
            [int(key) for key in numpy_keys],
            int(numpy_position),
            int(numpy_has_gauss),
            float(numpy_gauss),
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader": generator.get_state(),
    }


def restore_rng(state: Mapping[str, Any], generator: torch.Generator) -> None:
    """``capture_rng``가 남긴 상태를 그대로 되돌립니다."""

    python_version, python_keys, python_gauss = state["python"]
    random.setstate((python_version, tuple(python_keys), python_gauss))
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_gauss = state["numpy"]
    np.random.set_state(
        (
            numpy_name,
            np.array(numpy_keys, dtype=np.uint32),
            numpy_position,
            numpy_has_gauss,
            numpy_gauss,
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = list(state["cuda"])
    # GPU가 없는 기계에서 이어서 하거나 GPU 개수가 다르면 되돌릴 수 없습니다.
    # 그때는 CPU 쪽 상태만 되돌리고 CUDA는 seed 그대로 둡니다.
    if (
        cuda_states
        and torch.cuda.is_available()
        and len(cuda_states) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(state["dataloader"].cpu())


def _load_schedule_state(
    schedule: torch.optim.lr_scheduler.LambdaLR | None,
    state: Mapping[str, Any] | None,
) -> None:
    """이어서 학습할 때 schedule이 어디까지 왔는지 되돌립니다.

    한쪽만 있는 상태로는 이어서 할 수 없습니다. 되돌리지 않고 시작하면 warmup을 처음부터
    다시 하면서, 이어붙인 실행이 끊기지 않은 실행과 조용히 달라집니다.
    """

    if schedule is None:
        if state is not None:
            raise ValueError(
                "resume checkpoint has learning rate schedule state but this run has no schedule"
            )
        return
    if not isinstance(state, Mapping):
        raise ValueError(
            "resume checkpoint is missing the learning rate schedule state this run needs"
        )
    schedule.load_state_dict(dict(state))


def _collate(batch: list[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return tuple(zip(*batch))


def give_worker_its_own_storage(worker_id: int) -> None:
    """fork로 만든 worker에게 자기 몫의 S3 연결을 새로 쥐어 줍니다.

    worker는 부모의 dataset을 그대로 물려받고, 그 안에는 부모가 이미 만들어 둔 S3
    client가 들어 있습니다. boto3 client는 열린 socket을 들고 있어서 process 사이에
    나눠 쓸 수 없습니다. 나눠 쓰면 부모의 checkpoint 업로드와 worker의 이미지
    다운로드가 같은 socket에서 엉켜 응답이 뒤바뀌거나 멈춥니다.

    미리 받기에서 놓친 이미지는 학습 도중 worker가 받으므로, 이 경로는 드물게가
    아니라 실제로 지나갑니다. 여기서 실패하면 학습이 죽으므로 조용히 넘어가지
    않습니다. bucket과 접속 설정은 그대로 두고 연결만 새로 엽니다.
    """

    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    storage = getattr(info.dataset, "storage", None)
    if not isinstance(storage, S3Storage):
        return
    info.dataset.storage = S3Storage(
        storage.bucket,
        prefix=storage.prefix,
        profile=storage.profile,
        region=storage.region,
    )


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


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_loss: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": _state_on_cpu(model),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "validation_loss": validation_loss,
    }


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
    *,
    resume: Mapping[str, Any] | None = None,
    on_checkpoint: (
        Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None] | None
    ) = None,
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
        worker_init_fn=give_worker_its_own_storage,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        collate_fn=_collate,
        worker_init_fn=give_worker_its_own_storage,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    optimizer = build_optimizer(parameters, settings)
    # schedule은 optimizer를 **갱신할 때마다** 한 걸음 갑니다. microbatch 수로 길이를
    # 재면 accumulation만큼 짧게 걸어 linear와 cosine이 최저 learning rate에 닿지
    # 못하고 step schedule의 epoch 경계도 늦어집니다.
    _accumulation = max(1, int(settings.get("gradient_accumulation_steps", 1) or 1))
    _updates_per_epoch = math.ceil(len(train_loader) / _accumulation)
    schedule = build_lr_scheduler(optimizer, settings, _updates_per_epoch)
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
    start_epoch = 1
    checkpoint_every = settings.get("checkpoint_every", 1)
    if resume is not None:
        # 모델과 optimizer를 평소대로 만든 뒤 되돌립니다. 순서가 바뀌면 optimizer가
        # 되돌린 parameter가 아니라 새로 만든 parameter를 가리키게 됩니다.
        checkpoint = resume["checkpoint"]
        model.load_state_dict(dict(checkpoint["model_state_dict"]))
        optimizer.load_state_dict(copy.deepcopy(dict(checkpoint["optimizer_state_dict"])))
        state = checkpoint["resume_state"]
        history = [dict(entry) for entry in state["history"]]
        # 이어붙이기 이전의 best 가중치는 옆에 있는 best_checkpoint.pt에서 옵니다.
        # 그래야 이어서 한 실행도 그때의 best를 그대로 다시 공개할 수 있습니다.
        best_source = resume["best"]
        best_loss = float(state["best"]["validation_loss"])
        best_epoch = int(state["best"]["epoch"])
        best_checkpoint = {
            "epoch": best_epoch,
            "model_state_dict": dict(best_source["model_state_dict"]),
            "optimizer_state_dict": copy.deepcopy(
                dict(best_source["optimizer_state_dict"])
            ),
            "validation_loss": best_loss,
        }
        stopping = state["early_stopping"]
        reference_loss = stopping["reference_loss"]
        early_reference_loss = (
            math.inf if reference_loss is None else float(reference_loss)
        )
        epochs_without_improvement = int(stopping["epochs_without_improvement"])
        precision.load_state_dict(state["grad_scaler_state_dict"])
        _load_schedule_state(schedule, state.get("scheduler_state_dict"))
        restore_rng(state["rng"], generator)
        start_epoch = int(state["completed_epoch"]) + 1
    # 몇 개의 microbatch를 모아 한 번 갱신할지입니다. 1이면 지금까지와 같습니다.
    # GPU 메모리가 모자라 batch_size를 못 올릴 때 같은 효과를 냅니다.
    accumulation = max(1, int(settings.get("gradient_accumulation_steps", 1) or 1))
    for epoch in range(start_epoch, settings["epochs"] + 1):
        epoch_started_at = time.perf_counter()
        if progress is not None:
            progress.emit("epoch_started", epoch=epoch, epochs=settings["epochs"])
        model.train()
        train_total = 0.0
        train_component_totals: dict[str, float] = {}
        # epoch이 끝난 뒤 남길 값입니다. 아래에서 batch마다 갱신합니다.
        learning_rate = optimizer.param_groups[0]["lr"]
        total_steps = len(train_loader)
        for step, (images, targets) in enumerate(train_loader, start=1):
            # 묶음의 첫 microbatch에서만 비웁니다. 매번 비우면 앞의 것이 사라져
            # 모으는 의미가 없어집니다.
            if (step - 1) % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                loss, components = _loss(model, images, targets, device)
            # 이 batch가 실제로 쓴 값이므로 schedule을 넘기기 전에 읽습니다.
            learning_rate = optimizer.param_groups[0]["lr"]
            # **그 묶음이 실제로 모은 수**로 나눕니다. 나누지 않으면 gradient가 그
            # 배로 커져 사실상 learning rate를 올린 것과 같아지고, 마지막 묶음까지
            # accumulation으로 나누면 반대로 그 몇 장의 gradient가 작아집니다.
            group_start = ((step - 1) // accumulation) * accumulation
            group_size = min(accumulation, total_steps - group_start)
            precision.backward(loss / group_size)
            # 마지막 묶음이 N개를 못 채우고 끝나면 그 gradient가 통째로 버려집니다.
            # 오류는 나지 않고 그 몇 장만 학습에서 빠지므로 알아채기 어렵습니다.
            if step % accumulation == 0 or step == total_steps:
                precision.step(optimizer)
                if schedule is not None:
                    schedule.step()
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
            # 이 epoch의 마지막 batch가 쓴 learning rate입니다.
            "learning_rate": learning_rate,
        }
        history.append(epoch_record)
        is_best = validation_loss < best_loss
        if is_best:
            best_loss = validation_loss
            best_epoch = epoch
            best_checkpoint = _checkpoint(
                model, optimizer, epoch=epoch, validation_loss=validation_loss
            )
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
                learning_rate=learning_rate,
            )
        # 마지막 epoch은 주기와 상관없이 남깁니다. 그래야 작업 폴더의 checkpoint가
        # 학습이 끝난 실제 상태와 같아지고, 그대로 공개할 수 있습니다.
        is_final_epoch = should_stop or epoch == settings["epochs"]
        if on_checkpoint is not None and (
            is_final_epoch or epoch % checkpoint_every == 0
        ):
            on_checkpoint(
                _checkpoint(
                    model, optimizer, epoch=epoch, validation_loss=validation_loss
                ),
                best_checkpoint,
                {
                    "version": RESUME_STATE_VERSION,
                    "completed_epoch": epoch,
                    "history": [dict(entry) for entry in history],
                    # 가중치는 담지 않습니다. best_checkpoint.pt가 바로 옆에 있고,
                    # 매 epoch 모델 하나를 통째로 더 쓰면 파일이 두 배가 됩니다.
                    "best": {"epoch": best_epoch, "validation_loss": best_loss},
                    "early_stopping": {
                        # math.inf는 JSON으로도 못 나가고 읽는 쪽도 헷갈립니다.
                        "reference_loss": (
                            None
                            if math.isinf(early_reference_loss)
                            else early_reference_loss
                        ),
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                    "grad_scaler_state_dict": precision.state_dict(),
                    # schedule을 쓰지 않는 실행은 None입니다. key 자체는 늘 있어야
                    # 읽는 쪽이 "없는 것"과 "못 읽은 것"을 구별할 수 있습니다.
                    "scheduler_state_dict": (
                        copy.deepcopy(schedule.state_dict())
                        if schedule is not None
                        else None
                    ),
                    "rng": capture_rng(generator),
                },
            )
        if should_stop:
            break

    if best_checkpoint is None:
        raise RuntimeError("training completed without a best checkpoint")
    last_checkpoint = _checkpoint(
        model,
        optimizer,
        epoch=history[-1]["epoch"],
        validation_loss=history[-1]["validation_loss"],
    )
    return best_checkpoint, last_checkpoint, history


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    settings: Mapping[str, Any],
    progress: ProgressEmitter | None = None,
    *,
    resume: Mapping[str, Any] | None = None,
    on_checkpoint: (
        Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None] | None
    ) = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Train deterministically without leaving the global algorithm mode changed."""
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        set_seed(settings["seed"])
        torch.use_deterministic_algorithms(True, warn_only=True)
        return _train_model(
            model,
            train_dataset,
            validation_dataset,
            settings,
            progress,
            resume=resume,
            on_checkpoint=on_checkpoint,
        )
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
