"""Train 설정 검증과 runtime config 생성.

이 module은 ``src/pipelines/train/pipeline.py``의 검증 규칙을 그대로 따라 합니다.
train을 import하지 않고 규칙만 복제하므로, GUI가 GPU 시간을 쓰기 전에 같은 이유로
같은 값을 거부합니다. 여기서 추가로 거부하는 것은 train보다 **먼저** 막는 경우뿐이고,
train이 거부하는 값을 여기서 통과시키지 않습니다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from src.common import create_storage
from src.common import train_contract as _contract

from .errors import (
    FieldError,
    JobNotFoundError,
    WebPathError,
    WebValidationError,
    collect,
    raise_if_any,
)
from . import state_sync
from .gpu import cuda_is_available, native_bf16_supported
from .masking import redact
from .paths import (
    CONFIG_DIRNAME,
    config_dir,
    normalize_relative_posix,
    resolve_within_repo,
)
from .train_capabilities import (
    ARCHITECTURE_BACKBONES,
    CUDA_ONLY_PRECISIONS,
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_ARCHITECTURE_BACKBONES,
    DEFAULT_INPUT_SIZE,
    MMDETECTION_ARCHITECTURES,
    MMDETECTION_REQUIRED,
    DEFAULT_AUGMENTATION,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_PRECISION,
    LEGACY_ARCHITECTURE,
    LEGACY_OPTIMIZER,
    NEW_EXPERIMENT_OPTIMIZER,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_AUGMENTATIONS,
    SUPPORTED_LR_SCHEDULERS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_PRECISIONS,
)


__all__ = [
    "DATA_ARTIFACT_KEYS",
    "LR_SCHEDULER_DEFAULTS",
    "LR_WARMUP_DEFAULTS",
    "OPTIONAL_DATA_ARTIFACT_KEYS",
    "OPTIMIZER_PROFILES",
    "build_runtime_config",
    "field_specs",
    "normalize_data_inputs",
    "normalize_train_settings",
    "read_runtime_config",
    "resume_checkpoint_exists",
    "validate_request",
    "write_runtime_config",
]


# 이름과 기본값은 train과 함께 쓰는 계약(`src/common/train_contract.py`)에서 옵니다.
RUN_ID_PATTERN = _contract.RUN_ID_PATTERN
_RUN_ID_MAX_LENGTH = 128

# 화면 표시 순서를 위해 tuple로 둡니다.
DATA_ARTIFACT_KEYS = _contract.DATA_ARTIFACT_KEYS

# Train은 읽지 않지만, 성공한 학습 뒤 Evaluate가 대회 submission을 만들 때 씁니다.
# 기존 데이터셋에는 없어도 되므로 필수 4개와 분리합니다.
OPTIONAL_DATA_ARTIFACT_KEYS = _contract.OPTIONAL_DATA_ARTIFACT_KEYS

# 모델마다 다른 "이쯤부터 보관하면 된다"는 값입니다. train이 기본값으로 쓰지는
# 않으므로, 이 화면이 켠 사람에게 채워 주는 값입니다.
EPOCH_ARCHIVE_START = _contract.EPOCH_ARCHIVE_START

DEFAULT_OUTPUT_DIR = _contract.SETTING_DEFAULTS["output_dir"]
DEFAULT_OUTPUT_PREFIX = _contract.SETTING_DEFAULTS["output_prefix"]

# train은 patience에 기본값이 없습니다(있으면 필수). 화면이 안내하는 출발값입니다.
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_EARLY_STOPPING_MIN_DELTA = _contract.DEFAULT_EARLY_STOPPING_MIN_DELTA
_EARLY_STOPPING_FIELDS = ("early_stopping_patience", "early_stopping_min_delta")

# (이름, 기본값, 최소값). 최소값은 화면이 먼저 막는 값이고, 기본값은 train의 것입니다.
# `num_workers`는 train이 device와 OS를 보고 정하므로 계약에 기본값이 없습니다. 화면은
# 0으로 안내하고 보내지 않습니다 — 그래야 train이 자기 규칙대로 고릅니다(제안 015).
_INTEGER_FIELDS = (
    ("seed", _contract.SETTING_DEFAULTS["seed"], 0),
    ("epochs", _contract.SETTING_DEFAULTS["epochs"], 1),
    ("batch_size", _contract.SETTING_DEFAULTS["batch_size"], 1),
    ("num_workers", 0, 0),
    ("checkpoint_every", _contract.SETTING_DEFAULTS["checkpoint_every"], 1),
)
# 기본값이 architecture에 따라 다른 정수 설정입니다. train과 같은 규칙입니다.
# MMDetection model은 batch 1로만 돌므로 그만큼 모아야 쓸 만한 유효 batch가 됩니다.
# 기존 모델은 지금까지처럼 1입니다.
_ACCUMULATION_FIELD = (
    "gradient_accumulation_steps",
    _contract.SETTING_DEFAULTS["gradient_accumulation_steps"],
    1,
)
# MMDetection architecture에만 쓰는 정수 설정입니다. 다른 architecture와 함께 오면
# train이 거부하므로 여기서 먼저 막고, 보내지도 않습니다.
_MMDETECTION_INTEGER_FIELDS = (("input_size", DEFAULT_INPUT_SIZE, 1),)

# 생략했을 때와 뜻이 같은 값입니다. 자동 실행 이름의 지문에서 뺍니다.
#
# 넣으면 실제 학습이 이 값을 쓰기 전과 똑같은데도 이름이 달라집니다. 그러면 같은
# 설정과 seed로 다시 돌렸을 때 예전 실행과 이름이 어긋나 중복 실험을 알아채지 못하고
# GPU 시간을 두 번 씁니다. 생략과 1은 같은 동작이므로 이름에 반영할 것은 1보다 큰
# 값뿐입니다.
_FINGERPRINT_SAME_AS_OMITTED = {
    "gradient_accumulation_steps": _contract.SETTING_DEFAULTS[
        "gradient_accumulation_steps"
    ]
}
# 이어서 학습할 checkpoint의 파일 이름과 작업 폴더 규칙입니다. train이 쓰는 자리를
# 그대로 찾아야 해서 계약에서 읽습니다.
RESUME_CHECKPOINT_NAME = _contract.RESUME_CHECKPOINT_NAME
WORKING_DIRECTORY_SUFFIX = _contract.WORKING_DIRECTORY_SUFFIX
RUNNING_PREFIX = _contract.RUNNING_PREFIX
LR_WARMUP_DEFAULTS = _contract.LR_WARMUP_DEFAULTS
LR_SCHEDULER_DEFAULTS = _contract.LR_SCHEDULER_DEFAULTS
# 화면의 평평한 칸 이름 -> train이 받는 nested object의 key.
_LR_FIELDS = {
    "lr_warmup_steps": "warmup_steps",
    "lr_warmup_start_factor": "warmup_start_factor",
    "lr_min_factor": "min_lr_factor",
    "lr_step_size": "step_size",
    "lr_gamma": "gamma",
}
OPTIMIZER_PROFILES = _contract.OPTIMIZER_PROFILES

_FIELD_LABELS = {
    "run_id": ("실행 이름", "실행 결과가 저장되는 directory 이름으로 그대로 쓰입니다."),
    "architecture": ("모델", "학습에 사용할 object detection architecture입니다."),
    "backbone": (
        "Backbone",
        "고른 모델의 특징 추출기입니다. 나머지 구조는 그대로 두고 이것만 바꿉니다."
        " resnet50이 지금까지 쓰던 것입니다. 1280px·batch 1·amp 기준 GPU 사용량은"
        " resnet50 3.0GB, swin_t 3.3GB, swin_b 3.8GB이고 swin_l은 11.2GB라 10GB"
        " 카드에서는 모자랍니다.",
    ),
    "optimizer": ("Optimizer", "가중치를 갱신할 optimizer와 관련 수치 항목을 선택합니다."),
    "augmentation": (
        "증강 preset",
        "학습 split에만 적용합니다. pill_basic은 뒤집기와 약한 색 변형이고,"
        " pill_geometric은 여기에 90° 회전·자르기·잡음을 더하고 색을 더 세게 흔듭니다."
        " 데이터가 적을수록 과적합을 줄여 줍니다.",
    ),
    "precision": (
        "연산 정밀도",
        "fp32 외에는 device가 cuda여야 합니다. amp는 GPU를 보고 bf16과 fp16 중 되는 쪽을"
        " 골라 줍니다. fp16은 어느 GPU에서나 되고, bf16은 Ampere(RTX 30 시리즈, A100) 이상만"
        " 됩니다. T4처럼 bf16이 없는 GPU라면 amp나 fp16을 쓰세요.",
    ),
    "seed": ("Random seed", "같은 seed와 같은 데이터면 같은 결과가 나옵니다."),
    "checkpoint_every": (
        "Checkpoint 주기",
        "몇 epoch마다 이어서 할 수 있는 checkpoint를 남길지 정합니다. 1이면 매 epoch입니다.",
    ),
    "archive_epochs": (
        "epoch 보관",
        "학습이 끝난 뒤 어느 epoch이 실제로 제일 잘 맞히는지 재 보려면 그 epoch들이"
        " 남아 있어야 합니다. 켜면 아래 epoch부터 매 checkpoint 시점에 평가용 사본을"
        " 하나씩 더 남깁니다. 모델 하나에 190MB쯤이니 20 epoch이면 4GB 가까이 씁니다.",
    ),
    "archive_epochs_from": (
        "보관 시작 epoch",
        "이 epoch부터 남깁니다. 수렴 전 epoch은 어차피 이기지 못하는데 자리는 똑같이"
        " 차지합니다. 비워 두면 모델에 맞는 값을 씁니다.",
    ),
    "epochs": ("Epochs", "전체 학습 데이터를 몇 번 반복할지 정합니다."),
    "batch_size": ("Batch size", "한 번에 처리할 이미지 수. GPU 메모리에 가장 큰 영향을 줍니다."),
    "num_workers": ("DataLoader workers", "이미지를 읽어 오는 보조 process 수. 0이면 주 process가 직접 읽습니다."),
    "gradient_accumulation_steps": (
        "Gradient accumulation",
        "몇 batch를 모아 한 번 갱신할지 정합니다. 1이면 batch마다 갱신합니다. GPU"
        " 메모리가 모자라 batch size를 못 올릴 때, 8을 주면 batch size 8과 비슷한"
        " 효과를 냅니다. 대신 그만큼 느려집니다.",
    ),
    "input_size": (
        "입력 크기",
        "비율을 유지해 긴 변을 이 크기에 맞춥니다. MMDetection 모델만 씁니다. 줄이면"
        " GPU 메모리를 덜 쓰지만 작은 알약을 놓치기 쉬워집니다.",
    ),
    "learning_rate": ("Learning rate", "한 번에 얼마나 크게 배울지 정합니다. 너무 크면 발산합니다."),
    "lr_scheduler": (
        "Learning rate schedule",
        "학습이 진행되면서 learning rate를 줄이는 방법입니다. none은 처음부터 끝까지 같은"
        " 값이고, cosine은 부드럽게, linear는 곧게, step은 정해진 epoch마다 계단처럼"
        " 줄입니다.",
    ),
    "lr_warmup_steps": (
        "Warmup steps",
        "처음 몇 번의 가중치 갱신 동안 learning rate를 조금씩 올릴지 정합니다. 0이면"
        " 쓰지 않습니다. 모아서 갱신하면 batch 수와 갱신 수가 달라지므로 batch가 아니라"
        " 갱신을 셉니다. 사전학습 가중치로 시작할 때 초반에 손실이 튀는 것을 막아 줍니다.",
    ),
    "lr_warmup_start_factor": (
        "Warmup 시작 배율",
        "warmup을 시작할 때의 learning rate 배율입니다. 0.001이면 위에 적은 값의"
        " 1/1000에서 출발해 warmup이 끝날 때 그 값에 닿습니다.",
    ),
    "lr_min_factor": (
        "최저 learning rate 배율",
        "학습이 끝날 때의 learning rate 배율입니다. 0.01이면 위에 적은 값의 1/100까지"
        " 내려갑니다. 0이면 마지막에 0이 됩니다.",
    ),
    "lr_step_size": ("줄이는 간격", "step schedule이 몇 epoch마다 learning rate를 줄일지 정합니다."),
    "lr_gamma": ("줄이는 배율", "step schedule이 한 번에 곱하는 값입니다. 0.1이면 10분의 1이 됩니다."),
    "momentum": ("Momentum", "SGD가 이전 방향을 얼마나 유지할지 정합니다."),
    "weight_decay": ("Weight decay", "과적합을 억제하는 정규화 강도입니다."),
    "beta1": ("Beta 1", "Adam 계열의 1차 모멘트 감쇠율입니다."),
    "beta2": ("Beta 2", "Adam 계열의 2차 모멘트 감쇠율입니다."),
    "epsilon": ("Epsilon", "Adam 계열 계산에서 0으로 나누는 것을 막는 값입니다."),
    "early_stopping": (
        "조기 종료",
        "검증 손실이 나아지지 않으면 남은 epoch를 채우지 않고 학습을 끝냅니다.",
    ),
    "early_stopping_patience": (
        "Patience",
        "몇 epoch 연속으로 나아지지 않으면 끝낼지 정합니다. 1 이상의 정수입니다.",
    ),
    "early_stopping_min_delta": (
        "Min delta",
        "이만큼은 낮아져야 나아진 것으로 봅니다. 0이면 조금이라도 낮아지면 됩니다.",
    ),
    "device": ("Device", "학습을 CPU에서 할지 CUDA GPU에서 할지 정합니다."),
    "pretrained": ("Pretrained 가중치", "COCO로 미리 학습된 가중치에서 시작할지 정합니다."),
    "output_dir": ("Local 출력 directory", "저장소 기준 상대 경로여야 합니다."),
    "output_prefix": ("S3 출력 prefix", "S3 backend를 쓸 때 checkpoint를 올릴 위치입니다."),
}

_DATA_LABELS = {
    "train_manifest_uri": "학습 manifest",
    "validation_manifest_uri": "검증 manifest",
    "class_map_uri": "클래스 맵",
    "dataset_summary_uri": "데이터셋 요약",
    "test_manifest_uri": "테스트 manifest",
}


def _utc_now() -> datetime:
    """테스트에서 고정할 수 있도록 현재 시각을 한 곳에서만 읽습니다."""

    return datetime.now(timezone.utc)


def generate_run_id() -> str:
    """시각으로만 만드는 run_id입니다. 설정을 모를 때 쓰는 마지막 수단입니다.

    train의 기본값은 ``train-`` 접두사를 쓰므로, CLI로 돌린 실행과 구분됩니다.
    """

    return _utc_now().strftime("web-%Y%m%dT%H%M%S%fZ")


def next_resume_run_id(original_run_id: str, taken: Iterable[str] = ()) -> str:
    """이어서 학습할 실행의 이름입니다. ``A`` 다음은 ``A.2``, 그다음은 ``A.3``입니다.

    이름은 계보를 읽으라고 있습니다. 시각을 붙이면 충돌은 확실히 피하지만 목록에서
    ``A``와 ``A-resume-20260817T...Z``가 이웃해 있어도 한쪽이 다른 쪽에서 나왔다는
    것이 보이지 않습니다.

    같은 이름을 다시 쓰면 train이 시작을 거부하므로, **이미 있는 번호는 건너뜁니다.**
    ``taken``은 이 서버가 아는 run_id 전부입니다. 저장소까지 뒤지지는 않습니다 —
    ``check_run_id_collision``도 같은 이유로 S3를 보지 않고, 그래도 남은 충돌은
    train이 첫 batch 전에 이름을 대며 거절합니다.
    """

    if not RUN_ID_PATTERN.fullmatch(original_run_id):
        return generate_run_id()
    used = set(taken)
    # 이어서 한 실행을 또 이어가면 A.2.2가 아니라 A.3입니다.
    stem, _, tail = original_run_id.rpartition(".")
    continued = bool(stem) and tail.isdigit()
    start = max(int(tail) + 1, 2) if continued else 2
    if not continued:
        stem = original_run_id
    for number in range(start, start + 1000):
        suffix = f".{number}"
        candidate = f"{stem[: _RUN_ID_MAX_LENGTH - len(suffix)]}{suffix}"
        if candidate not in used and RUN_ID_PATTERN.fullmatch(candidate):
            return candidate
    return generate_run_id()


# 표에서 한눈에 읽히도록 줄인 이름입니다. train에 모델이 늘면 아래 fallback이 받습니다.
_ARCHITECTURE_SHORT_NAMES = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": "mobile",
    "fasterrcnn_resnet50_fpn_v2": "frcnn",
    "retinanet_resnet50_fpn_v2": "retina",
}

# 이름은 설정을 읽으라고 있는 것이라, 경로와 이름 자체는 꼬리표 계산에서 뺍니다.
# `archive_epochs_from`도 같습니다 — 무엇을 배우는지가 아니라 무엇을 남기는지입니다.
# 여기 넣지 않으면 같은 학습을 보관만 켜고 다시 돌렸을 때 다른 이름이 붙어, 중복
# 실험이라는 사실이 이름에서 사라집니다.
_FINGERPRINT_IGNORED = frozenset(
    {"run_id", "output_dir", "output_prefix", "archive_epochs_from"}
)
# 어떤 값과도 같지 않은 표식입니다. `None`을 쓰면 값이 None인 설정을 잘못 빼 버립니다.
_MISSING = object()


def _short_architecture(name: Any) -> str:
    known = _ARCHITECTURE_SHORT_NAMES.get(name) if isinstance(name, str) else None
    if known is not None:
        return known
    # 모르는 이름은 첫 토큰만 씁니다. 이름을 못 만들어 저장이 막히면 안 됩니다.
    token = re.split(r"[^A-Za-z0-9]", str(name or ""))[0]
    return (token or "model").lower()[:8]


def _short_augmentation(name: Any) -> str:
    text = str(name or "none")
    # `pill_basic`은 이 프로젝트 전용 preset이라 앞머리가 모든 이름에서 같습니다.
    return text.split("_")[-1] if "_" in text else text


def _short_learning_rate(value: Any) -> str:
    """팀이 이미 쓰던 표기입니다: 0.006 -> 6e3, 0.0002 -> 2e4."""

    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return "na"
    mantissa, exponent = f"{float(value):.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    power = int(exponent)
    if power < 0:
        return f"{mantissa}e{-power}"
    # 1 이상인 learning rate는 사실상 없지만 이름은 언제나 만들어져야 합니다.
    return f"{float(value):g}".replace(".", "p")


def _settings_fingerprint(
    settings: Mapping[str, Any], data_inputs: Mapping[str, str] | None
) -> str:
    """설정과 데이터셋을 함께 요약한 4자입니다.

    같은 설정이면 언제나 같은 값이라 중복 실험을 이름만 보고 알아챌 수 있습니다.
    이름에 데이터셋이 들어가지 않으므로, **데이터셋 차이는 이 값이 맡습니다.**
    """

    material: dict[str, Any] = {
        key: value
        for key, value in settings.items()
        if key not in _FINGERPRINT_IGNORED
        and _FINGERPRINT_SAME_AS_OMITTED.get(key, _MISSING) != value
    }
    material["__data"] = dict(sorted((data_inputs or {}).items()))
    canonical = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:4]


def generate_settings_run_id(
    settings: Mapping[str, Any], data_inputs: Mapping[str, str] | None
) -> str:
    """설정을 그대로 읽을 수 있는 실행 이름을 만듭니다.

    ``retina-basic-e15-b4-lr6e3-s42-a7f3`` 처럼 모델·증강·epochs·batch·learning
    rate·seed와 꼬리표 4자로 이루어집니다. 이름이 사람마다 달라 목록에서 아무것도
    읽을 수 없던 것을 고치려는 것이라, 값이 바뀌면 이름도 함께 바뀝니다.

    seed를 넣는 것은 같은 설정을 다시 돌릴 때 이름이 겹쳐 train이 시작을 거부하는
    일을 피하려는 것입니다. 설정도 seed도 완전히 같으면 이름도 같은데, 그때는
    실제로 같은 실험이므로 겹쳤다고 알려 주는 편이 맞습니다.
    """

    parts = [
        _short_architecture(settings.get("architecture")),
        _short_augmentation(settings.get("augmentation")),
        f"e{settings.get('epochs')}",
        f"b{settings.get('batch_size')}",
        f"lr{_short_learning_rate(settings.get('learning_rate'))}",
        f"s{settings.get('seed')}",
        _settings_fingerprint(settings, data_inputs),
    ]
    candidate = "-".join(str(part) for part in parts)
    # 어떤 이유로든 규칙을 못 지키면 시각 이름으로 물러섭니다. 저장이 막히면 안 됩니다.
    return candidate if RUN_ID_PATTERN.fullmatch(candidate) else generate_run_id()


def field_specs() -> list[dict[str, Any]]:
    """새 실험 화면이 form을 그릴 때 쓰는 필드 정의입니다."""

    # GPU가 있는 컴퓨터에 맞춰 폼을 채웁니다. device와 precision은 짝이라 함께 정합니다.
    # amp는 device가 cuda일 때만 쓸 수 있으므로, 하나만 바꾸면 폼이 저장할 수 없는
    # 조합으로 시작합니다. CUDA가 없으면 둘 다 기존 값으로 둡니다.
    #
    # 여기서 정하는 것은 **폼의 출발값뿐**입니다. 아래 normalize_train_settings가 값을
    # 받지 못했을 때 쓰는 fallback은 train 기본값(cpu, fp32) 그대로 두어야 합니다.
    # 그쪽은 다른 소유 영역이고 test_web_train_contract.py가 두 값을 대조합니다.
    # 바로 아래 pretrained가 같은 구조입니다.
    has_cuda = cuda_is_available()
    form_device = "cuda" if has_cuda else "cpu"
    form_precision = "amp" if has_cuda else DEFAULT_PRECISION

    specs: list[dict[str, Any]] = []
    for name, default, choices in (
        ("architecture", LEGACY_ARCHITECTURE, SUPPORTED_ARCHITECTURES),
        ("optimizer", NEW_EXPERIMENT_OPTIMIZER, SUPPORTED_OPTIMIZERS),
        ("augmentation", DEFAULT_AUGMENTATION, SUPPORTED_AUGMENTATIONS),
        ("precision", form_precision, SUPPORTED_PRECISIONS),
        ("lr_scheduler", DEFAULT_LR_SCHEDULER, SUPPORTED_LR_SCHEDULERS),
    ):
        label, hint = _FIELD_LABELS[name]
        spec = {
            "name": name,
            "type": "enum",
            "default": default,
            "choices": list(choices),
            "label": label,
            "hint": hint,
        }
        # backbone만 다른 갈래를 묶은 표입니다. 화면은 이것으로 model 목록을 접고
        # backbone 칸을 하나 더 그린 뒤, 보내기 전에 architecture 이름 하나로
        # 합칩니다. **`choices`는 계약의 진짜 이름 그대로 둡니다** — 접힌 이름을
        # 실으면 그 목록을 그대로 보내는 다른 소비자가 서버에게 거절당합니다.
        # 보내는 값이 둘로 늘지 않는 것도 핵심입니다 — 둘이면 서로 어긋날 수 있습니다.
        if name == "architecture":
            backbone_label, backbone_hint = _FIELD_LABELS["backbone"]
            spec["backbones"] = {
                family: dict(table) for family, table in ARCHITECTURE_BACKBONES.items()
            }
            spec["backbone_defaults"] = dict(DEFAULT_ARCHITECTURE_BACKBONES)
            spec["backbone_label"] = backbone_label
            spec["backbone_hint"] = backbone_hint
        specs.append(spec)
    label, hint = _FIELD_LABELS["run_id"]
    specs.append(
        {
            "name": "run_id",
            "type": "string",
            "default": None,
            "label": label,
            "hint": hint,
            "pattern": RUN_ID_PATTERN.pattern,
            "placeholder": "비워 두면 자동으로 만듭니다",
        }
    )
    for name, default, minimum in (
        _INTEGER_FIELDS + (_ACCUMULATION_FIELD,) + _MMDETECTION_INTEGER_FIELDS
    ):
        label, hint = _FIELD_LABELS[name]
        spec: dict[str, Any] = {
            "name": name,
            "type": "integer",
            "default": default,
            "minimum": minimum,
            "label": label,
            "hint": hint,
        }
        # 기본값이 architecture마다 다른 칸입니다. 하나만 내려보내면 MMDetection을
        # 고르고 비워 둔 사람에게 1이라고 안내하면서 실제로는 8로 돕니다.
        if name == _ACCUMULATION_FIELD[0]:
            spec["defaults_by_architecture"] = {
                architecture: DEFAULT_ACCUMULATION_STEPS
                for architecture in MMDETECTION_ARCHITECTURES
            }
        # 그 모델을 고르지 않았으면 화면에서 감춥니다. 보이면 값을 정할 수 있는 것처럼
        # 읽히는데 이쪽은 거부합니다. 어느 모델이 쓰는지 화면이 옮겨 적지 않게 합니다.
        if name in {field[0] for field in _MMDETECTION_INTEGER_FIELDS}:
            spec["only_for_architectures"] = list(MMDETECTION_ARCHITECTURES)
        specs.append(spec)
    default_profile = OPTIMIZER_PROFILES[NEW_EXPERIMENT_OPTIMIZER]
    for name in (
        "learning_rate",
        "weight_decay",
        "momentum",
        "beta1",
        "beta2",
        "epsilon",
    ):
        fallback_profile = OPTIMIZER_PROFILES["SGD"] if name == "momentum" else default_profile
        default = fallback_profile.get(name)
        minimum = 1e-16 if name == "epsilon" else 0.0
        label, hint = _FIELD_LABELS[name]
        specs.append(
            {
                "name": name,
                "type": "number",
                "default": default,
                "defaults_by_optimizer": {
                    optimizer: profile[name]
                    for optimizer, profile in OPTIMIZER_PROFILES.items()
                    if name in profile
                },
                "minimum": minimum,
                "label": label,
                "hint": hint,
            }
        )
    label, hint = _FIELD_LABELS["device"]
    specs.append(
        {
            "name": "device",
            "type": "enum",
            "default": form_device,
            "choices": ["cpu", "cuda"],
            "label": label,
            "hint": hint,
        }
    )
    label, hint = _FIELD_LABELS["early_stopping"]
    specs.append(
        {"name": "early_stopping", "type": "boolean", "default": False, "label": label, "hint": hint}
    )
    label, hint = _FIELD_LABELS["early_stopping_patience"]
    specs.append(
        {
            "name": "early_stopping_patience",
            "type": "integer",
            "default": DEFAULT_EARLY_STOPPING_PATIENCE,
            "minimum": 1,
            "label": label,
            "hint": hint,
        }
    )
    label, hint = _FIELD_LABELS["early_stopping_min_delta"]
    specs.append(
        {
            "name": "early_stopping_min_delta",
            "type": "number",
            "default": DEFAULT_EARLY_STOPPING_MIN_DELTA,
            "minimum": 0.0,
            "label": label,
            "hint": hint,
        }
    )
    label, hint = _FIELD_LABELS["archive_epochs"]
    specs.append(
        {"name": "archive_epochs", "type": "boolean", "default": False, "label": label, "hint": hint}
    )
    label, hint = _FIELD_LABELS["archive_epochs_from"]
    specs.append(
        {
            "name": "archive_epochs_from",
            "type": "integer",
            # 모델을 고르기 전 화면에 보이는 값입니다. 아래 표가 고른 모델에 맞는 값으로
            # 덮어씁니다 — 하나만 내려보내면 DINO를 고른 사람에게 15라고 안내합니다.
            "default": EPOCH_ARCHIVE_START[LEGACY_ARCHITECTURE],
            "defaults_by_architecture": dict(EPOCH_ARCHIVE_START),
            "minimum": 1,
            "label": label,
            "hint": hint,
        }
    )
    for name, default, minimum, kind in (
        ("lr_warmup_steps", LR_WARMUP_DEFAULTS["warmup_steps"], 0, "integer"),
        ("lr_warmup_start_factor", LR_WARMUP_DEFAULTS["warmup_start_factor"], 0.0, "number"),
        ("lr_min_factor", LR_SCHEDULER_DEFAULTS["cosine"]["min_lr_factor"], 0.0, "number"),
        ("lr_step_size", LR_SCHEDULER_DEFAULTS["step"]["step_size"], 1, "integer"),
        ("lr_gamma", LR_SCHEDULER_DEFAULTS["step"]["gamma"], 0.0, "number"),
    ):
        label, hint = _FIELD_LABELS[name]
        specs.append(
            {
                "name": name,
                "type": kind,
                "default": default,
                "minimum": minimum,
                "label": label,
                "hint": hint,
            }
        )
    label, hint = _FIELD_LABELS["pretrained"]
    # 화면에서 시작하는 학습은 사전학습 가중치를 기본으로 씁니다. 아래
    # normalize_train_settings의 fallback은 train 기본값(False) 그대로 두어야 합니다.
    # 그쪽은 다른 소유 영역이고 test_web_train_contract.py가 두 값을 대조합니다.
    # 화면은 이 spec을 보고 pretrained를 명시적으로 실어 보냅니다.
    specs.append(
        {"name": "pretrained", "type": "boolean", "default": True, "label": label, "hint": hint}
    )
    for name, default in (
        ("output_dir", DEFAULT_OUTPUT_DIR),
        ("output_prefix", DEFAULT_OUTPUT_PREFIX),
    ):
        label, hint = _FIELD_LABELS[name]
        specs.append(
            {"name": name, "type": "string", "default": default, "label": label, "hint": hint}
        )
    return specs


def data_field_specs() -> list[dict[str, Any]]:
    """Data pipeline artifact 입력 칸 정의입니다."""

    return [
        {
            "name": key,
            "type": "uri",
            "label": _DATA_LABELS[key],
            "hint": "저장소 기준 상대 경로 또는 s3://bucket/key 형식입니다.",
            "required": True,
        }
        for key in DATA_ARTIFACT_KEYS
    ]


def _normalize_integer(
    raw: Any, name: str, default: int, minimum: int, errors: list[FieldError]
) -> int:
    value = raw.get(name, default)
    # train은 bool을 int 자리에서 명시적으로 거부합니다.
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        collect(errors, f"train.{name}", f"{minimum} 이상의 정수여야 합니다.")
        return default
    return value


def _normalize_float(
    raw: Any, name: str, default: float, minimum: float, errors: list[FieldError]
) -> float:
    value = raw.get(name, default)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        collect(errors, f"train.{name}", f"{minimum} 이상의 유한한 숫자여야 합니다.")
        return default
    return float(value)


def _normalize_probability(
    raw: Any, name: str, default: float, errors: list[FieldError]
) -> float:
    value = _normalize_float(raw, name, default, 0.0, errors)
    if value >= 1.0:
        collect(errors, f"train.{name}", "0 이상 1 미만의 유한한 숫자여야 합니다.")
        return default
    return value


def _normalize_ratio(
    raw: Any, name: str, default: float, errors: list[FieldError], *, above_zero: bool
) -> float:
    """0과 1 사이의 배율입니다. learning rate를 키우는 값은 schedule이 아닙니다."""

    value = _normalize_float(raw, name, default, 0.0, errors)
    if value > 1.0 or (above_zero and value <= 0.0):
        bound = "0 초과" if above_zero else "0 이상"
        collect(errors, f"train.{name}", f"{bound} 1 이하의 유한한 숫자여야 합니다.")
        return default
    return value


def _normalize_lr_scheduler(raw: Any, errors: list[FieldError]) -> dict[str, Any] | None:
    """화면의 평평한 칸을 train이 받는 object 하나로 접습니다.

    검증 규칙은 train의 ``_lr_scheduler``(pipeline.py)를 그대로 옮긴 것입니다.
    ``none``을 고르고 warmup도 쓰지 않으면 key 자체를 만들지 않아, schedule을 쓰지 않는
    사람의 config는 이 기능이 생기기 전과 한 글자도 달라지지 않습니다. 자동으로 짓는
    실행 이름이 설정에서 나오므로 그 이름까지 그대로입니다.
    """

    name = raw.get("lr_scheduler", DEFAULT_LR_SCHEDULER)
    if not isinstance(name, str) or name not in SUPPORTED_LR_SCHEDULERS:
        collect(
            errors,
            "train.lr_scheduler",
            f"{', '.join(SUPPORTED_LR_SCHEDULERS)} 중 하나여야 합니다.",
        )
        name = DEFAULT_LR_SCHEDULER

    used = {**LR_WARMUP_DEFAULTS, **LR_SCHEDULER_DEFAULTS[name]}
    # 고른 schedule이 쓰지 않는 값은 train이 거부합니다. optimizer의 momentum·beta를
    # 막는 것과 같은 이유입니다.
    for field, key in _LR_FIELDS.items():
        if key not in used and field in raw:
            collect(errors, f"train.{field}", f"{name} schedule에서 사용하지 않는 값입니다.")

    warmup_steps = _normalize_integer(
        raw, "lr_warmup_steps", used["warmup_steps"], 0, errors
    )
    settings: dict[str, Any] = {
        "name": name,
        "warmup_steps": warmup_steps,
        "warmup_start_factor": _normalize_ratio(
            raw,
            "lr_warmup_start_factor",
            used["warmup_start_factor"],
            errors,
            above_zero=True,
        ),
    }
    if "min_lr_factor" in used:
        settings["min_lr_factor"] = _normalize_ratio(
            raw, "lr_min_factor", used["min_lr_factor"], errors, above_zero=False
        )
    if "step_size" in used:
        settings["step_size"] = _normalize_integer(
            raw, "lr_step_size", used["step_size"], 1, errors
        )
        settings["gamma"] = _normalize_ratio(
            raw, "lr_gamma", used["gamma"], errors, above_zero=True
        )
    # warmup만 쓰는 것도 정상 조합입니다. 둘 다 안 쓸 때만 없던 일이 됩니다.
    if name == DEFAULT_LR_SCHEDULER and warmup_steps == 0:
        return None
    return settings


def _normalize_epoch_archive(
    raw: Any, architecture: str, errors: list[FieldError]
) -> int | None:
    """켜져 있으면 보관을 시작할 epoch, 꺼져 있으면 ``None``입니다.

    조기 종료와 같은 모양입니다. 꺼져 있으면 key 자체를 만들지 않아, 보관하지 않는
    사람의 config는 이 기능이 생기기 전과 한 글자도 달라지지 않습니다. 기본값이
    모델마다 다른 것은 무거운 model이 한 epoch에 더 비싸고 더 일찍 수렴하기 때문입니다.
    """

    enabled = raw.get("archive_epochs", False)
    if not isinstance(enabled, bool):
        collect(errors, "train.archive_epochs", "true 또는 false여야 합니다.")
        enabled = False
    if not enabled:
        if "archive_epochs_from" in raw:
            collect(
                errors,
                "train.archive_epochs_from",
                "epoch 보관을 켰을 때만 쓰는 값입니다.",
            )
        return None
    return _normalize_integer(
        raw, "archive_epochs_from", EPOCH_ARCHIVE_START[architecture], 1, errors
    )


def _normalize_early_stopping(
    raw: Any, errors: list[FieldError]
) -> dict[str, float | int] | None:
    """평평한 세 칸을 train이 받는 object 하나로 접습니다.

    검증 규칙은 train의 ``_early_stopping``(pipeline.py)을 그대로 옮긴 것입니다.
    꺼져 있으면 key 자체를 만들지 않아, 조기 종료를 쓰지 않는 사람의 config는
    이 기능이 생기기 전과 한 글자도 달라지지 않습니다.
    """

    enabled = raw.get("early_stopping", False)
    if not isinstance(enabled, bool):
        collect(errors, "train.early_stopping", "true 또는 false여야 합니다.")
        enabled = False

    if not enabled:
        # 끈 채로 값을 보내면 화면과 실제 학습이 달라 보입니다. optimizer에서 쓰지
        # 않는 값을 막는 것과 같은 이유입니다.
        for name in sorted(set(_EARLY_STOPPING_FIELDS) & set(raw)):
            collect(errors, f"train.{name}", "조기 종료를 사용할 때만 쓰는 값입니다.")
        return None

    # train은 patience를 필수로 받지만 그건 train이 받는 object의 규칙입니다. 다른 수치
    # 칸과 똑같이 비어 있으면 화면이 안내한 기본값을 채워 완성된 object를 보냅니다.
    patience = _normalize_integer(
        raw, "early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE, 1, errors
    )
    min_delta = _normalize_float(
        raw, "early_stopping_min_delta", DEFAULT_EARLY_STOPPING_MIN_DELTA, 0.0, errors
    )
    return {"patience": patience, "min_delta": min_delta}


def normalize_train_settings(
    raw: Any, data_inputs: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """``config["train"]`` 후보를 train과 같은 규칙으로 정규화합니다.

    문제를 하나 발견하면 멈추지 않고 모두 모아서 한 번에 보고합니다. 화면에서 여러
    칸의 오류를 동시에 보여줘야 하기 때문입니다.

    ``run_id``를 비우면 정규화가 끝난 설정으로 이름을 지어 줍니다. ``data_inputs``는
    그 이름의 꼬리표에만 쓰입니다. 데이터셋이 이름에 들어가지 않으므로, 주지 않으면
    데이터셋만 다른 두 실행이 같은 이름을 받습니다.
    """

    errors: list[FieldError] = []
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise WebValidationError([FieldError("train", "train 설정은 object여야 합니다.")])

    architecture = raw.get("architecture", LEGACY_ARCHITECTURE)
    if not isinstance(architecture, str) or architecture not in SUPPORTED_ARCHITECTURES:
        collect(
            errors,
            "train.architecture",
            f"{', '.join(SUPPORTED_ARCHITECTURES)} 중 하나여야 합니다.",
        )
        architecture = LEGACY_ARCHITECTURE

    augmentation = raw.get("augmentation", DEFAULT_AUGMENTATION)
    if not isinstance(augmentation, str) or augmentation not in SUPPORTED_AUGMENTATIONS:
        collect(
            errors,
            "train.augmentation",
            f"{', '.join(SUPPORTED_AUGMENTATIONS)} 중 하나여야 합니다.",
        )
        augmentation = DEFAULT_AUGMENTATION

    precision = raw.get("precision", DEFAULT_PRECISION)
    if not isinstance(precision, str) or precision not in SUPPORTED_PRECISIONS:
        collect(
            errors,
            "train.precision",
            f"{', '.join(SUPPORTED_PRECISIONS)} 중 하나여야 합니다.",
        )
        precision = DEFAULT_PRECISION

    optimizer = raw.get("optimizer", LEGACY_OPTIMIZER)
    if not isinstance(optimizer, str) or optimizer not in SUPPORTED_OPTIMIZERS:
        collect(
            errors,
            "train.optimizer",
            f"{', '.join(SUPPORTED_OPTIMIZERS)} 중 하나여야 합니다.",
        )
        optimizer = LEGACY_OPTIMIZER
    irrelevant = {"momentum"} if optimizer in {"AdamW", "Adam"} else {
        "beta1",
        "beta2",
        "epsilon",
    }
    for name in sorted(irrelevant & set(raw)):
        collect(errors, f"train.{name}", f"{optimizer}에서 사용하지 않는 값입니다.")

    device = raw.get("device", "cpu")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        collect(errors, "train.device", "'cpu' 또는 'cuda'여야 합니다.")
        device = "cpu"
    elif device == "cuda" and not cuda_is_available():
        collect(errors, "train.device", "CUDA를 사용할 수 없는 환경입니다.")

    if precision in CUDA_ONLY_PRECISIONS and device != "cuda":
        # train이 같은 조건을 거부합니다. 여기서 잡지 않으면 subprocess가 뜬 뒤에야
        # 알게 되고, 화면에는 어느 칸이 잘못됐는지 남지 않습니다.
        collect(
            errors,
            "train.precision",
            f"{precision} 정밀도는 device가 cuda일 때만 쓸 수 있습니다.",
        )
        precision = DEFAULT_PRECISION
    elif precision == "bf16" and native_bf16_supported() is False:
        # T4 같은 Turing GPU는 bf16을 emulation으로만 처리해 아주 느립니다. 확실히
        # 안 되는 것을 아는 경우에만 막고, 알 수 없으면(None) train이 판단하게 둡니다.
        collect(
            errors,
            "train.precision",
            "이 컴퓨터의 GPU는 bf16을 지원하지 않습니다. fp16이나 amp를 쓰세요.",
        )
        precision = DEFAULT_PRECISION

    pretrained = raw.get("pretrained", False)
    if not isinstance(pretrained, bool):
        collect(errors, "train.pretrained", "true 또는 false여야 합니다.")
        pretrained = False

    resume_from = _normalize_resume_from(raw, errors)

    # 비워 두면 정규화가 끝난 뒤 설정을 읽어 이름을 짓습니다. 여기서는 빈 채로 둡니다.
    run_id = raw.get("run_id") or ""
    if run_id and (not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id)):
        collect(
            errors,
            "train.run_id",
            "영문/숫자로 시작하고 영문·숫자·마침표·밑줄·붙임표만 쓸 수 있습니다(최대 128자).",
        )
        run_id = generate_run_id()

    output_dir = raw.get("output_dir", DEFAULT_OUTPUT_DIR)
    if not isinstance(output_dir, str) or not output_dir.strip():
        collect(errors, "train.output_dir", "비어 있지 않은 저장소 기준 상대 경로여야 합니다.")
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        try:
            output_dir = normalize_relative_posix(output_dir, label="출력 directory")
            resolve_within_repo(output_dir, label="출력 directory")
        except WebPathError as error:
            collect(errors, "train.output_dir", str(error))
            output_dir = DEFAULT_OUTPUT_DIR

    output_prefix = raw.get("output_prefix", DEFAULT_OUTPUT_PREFIX)
    if not isinstance(output_prefix, str) or not output_prefix.strip():
        collect(errors, "train.output_prefix", "비어 있지 않은 S3 prefix여야 합니다.")
        output_prefix = DEFAULT_OUTPUT_PREFIX

    settings: dict[str, Any] = {
        "run_id": run_id,
        "architecture": architecture,
        "optimizer": optimizer,
        "augmentation": augmentation,
        "precision": precision,
        "device": device,
        "pretrained": pretrained,
        "lr_scheduler": _normalize_lr_scheduler(raw, errors),
        "early_stopping": _normalize_early_stopping(raw, errors),
        "archive_epochs_from": _normalize_epoch_archive(raw, architecture, errors),
        "output_dir": output_dir,
        "output_prefix": output_prefix.strip("/"),
    }
    for name, default, minimum in _INTEGER_FIELDS:
        settings[name] = _normalize_integer(raw, name, default, minimum, errors)
    uses_mmdetection = architecture in MMDETECTION_ARCHITECTURES
    accumulation_name, accumulation_default, accumulation_minimum = _ACCUMULATION_FIELD
    settings[accumulation_name] = _normalize_integer(
        raw,
        accumulation_name,
        DEFAULT_ACCUMULATION_STEPS if uses_mmdetection else accumulation_default,
        accumulation_minimum,
        errors,
    )
    for name, default, minimum in _MMDETECTION_INTEGER_FIELDS:
        if uses_mmdetection:
            settings[name] = _normalize_integer(raw, name, default, minimum, errors)
        elif name in raw:
            collect(
                errors,
                f"train.{name}",
                f"{', '.join(MMDETECTION_ARCHITECTURES)}에서만 쓰는 값입니다.",
            )
    if uses_mmdetection:
        # 이 저장소가 지원하는 하나뿐인 조합입니다. train도 같은 이유로 거부하지만,
        # 여기서 먼저 막아야 어느 칸이 문제인지 화면에 남습니다.
        for name, required in MMDETECTION_REQUIRED.items():
            if settings[name] != required:
                collect(
                    errors,
                    f"train.{name}",
                    f"{architecture}는 {required!r}로만 돌릴 수 있습니다.",
                )
    profile = OPTIMIZER_PROFILES[optimizer]
    for name in ("learning_rate", "weight_decay"):
        settings[name] = _normalize_float(raw, name, profile[name], 0.0, errors)
    if optimizer == "SGD":
        settings["momentum"] = _normalize_float(
            raw, "momentum", profile["momentum"], 0.0, errors
        )
    else:
        settings["beta1"] = _normalize_probability(
            raw, "beta1", profile["beta1"], errors
        )
        settings["beta2"] = _normalize_probability(
            raw, "beta2", profile["beta2"], errors
        )
        settings["epsilon"] = _normalize_float(
            raw, "epsilon", profile["epsilon"], 1e-16, errors
        )

    raise_if_any(errors)
    # 이름을 안 준 실행은 여기서 짓습니다. 값이 모두 정규화된 뒤라야 같은 설정이
    # 언제나 같은 이름을 받습니다.
    if not settings["run_id"]:
        settings["run_id"] = generate_settings_run_id(settings, data_inputs)
    # train이 읽는 순서와 같게 정렬해 config를 읽기 쉽게 만듭니다.
    return {
        "run_id": settings["run_id"],
        "architecture": settings["architecture"],
        "optimizer": settings["optimizer"],
        # train은 preset key 하나만 든 object를 받고 다른 key가 있으면 거부합니다.
        "augmentation": {"preset": settings["augmentation"]},
        # 문자열 하나입니다. amp가 bf16이 될지 fp16이 될지는 train이 GPU를 보고 정합니다.
        "precision": settings["precision"],
        "seed": settings["seed"],
        "epochs": settings["epochs"],
        "checkpoint_every": settings["checkpoint_every"],
        # 보관하지 않는 실행은 key 자체를 넣지 않습니다. train은 없으면 지금과 완전히
        # 같게 동작합니다.
        **(
            {"archive_epochs_from": settings["archive_epochs_from"]}
            if settings["archive_epochs_from"] is not None
            else {}
        ),
        "batch_size": settings["batch_size"],
        "num_workers": settings["num_workers"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        # torchvision architecture와 함께 보내면 train이 거부합니다. 쓰지 않는 실행은
        # key 자체를 넣지 않습니다.
        **(
            {"input_size": settings["input_size"]}
            if "input_size" in settings
            else {}
        ),
        # 처음부터 학습하는 실행은 key 자체를 넣지 않습니다. train은 없으면 지금과
        # 완전히 같게 동작합니다.
        **({"resume_from": resume_from} if resume_from is not None else {}),
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        # 상수 learning rate로 돌리는 실행은 key 자체를 넣지 않습니다. train은 없으면
        # 지금과 완전히 같게 동작합니다.
        **(
            {"lr_scheduler": settings["lr_scheduler"]}
            if settings["lr_scheduler"] is not None
            else {}
        ),
        "device": settings["device"],
        "pretrained": settings["pretrained"],
        # 끄면 key 자체를 넣지 않습니다. train은 없으면 전체 epoch를 그대로 돕니다.
        **(
            {"early_stopping": settings["early_stopping"]}
            if settings["early_stopping"] is not None
            else {}
        ),
        "output_dir": settings["output_dir"],
        "output_prefix": settings["output_prefix"],
        **(
            {"momentum": settings["momentum"]}
            if optimizer == "SGD"
            else {
                "beta1": settings["beta1"],
                "beta2": settings["beta2"],
                "epsilon": settings["epsilon"],
            }
        ),
    }


def _normalize_resume_from(raw: dict, errors: list[FieldError]) -> str | None:
    """이어서 학습할 checkpoint 경로입니다. 없으면 처음부터 학습합니다.

    train은 저장소 안의 상대 경로와 ``s3://`` URI를 받습니다. 같은 규칙으로 미리
    거르지 않으면 subprocess가 뜬 뒤에야 알게 됩니다.
    """

    if "resume_from" not in raw:
        return None
    value = raw["resume_from"]
    if not isinstance(value, str) or not value.strip():
        collect(errors, "train.resume_from", "비어 있지 않은 문자열이어야 합니다.")
        return None

    text = value.strip()
    if text.lower().startswith("s3://"):
        split = urlsplit(text)
        if not split.netloc or not split.path.strip("/"):
            collect(errors, "train.resume_from", "s3://bucket/key 형식이어야 합니다.")
        return text
    if "://" in text:
        collect(
            errors,
            "train.resume_from",
            "저장소 기준 상대 경로 또는 s3:// URI만 쓸 수 있습니다.",
        )
        return text
    try:
        return normalize_relative_posix(text, label="이어서 학습할 checkpoint")
    except WebPathError as error:
        collect(errors, "train.resume_from", str(error))
        return text


def _normalize_data_uri(value: Any, key: str, errors: list[FieldError]) -> str:
    field = f"inputs.data.{key}"
    if not isinstance(value, str) or not value.strip():
        collect(errors, field, "비어 있지 않은 문자열이어야 합니다.")
        return ""

    text = value.strip()
    lowered = text.lower()
    if lowered.startswith("s3://"):
        split = urlsplit(text)
        if not split.netloc or not split.path.strip("/"):
            collect(errors, field, "s3://bucket/key 형식이어야 합니다.")
            return text
        if split.query or split.fragment:
            collect(errors, field, "s3 URI에 query나 fragment를 쓸 수 없습니다.")
        return text
    if "://" in text:
        collect(errors, field, "저장소 기준 상대 경로 또는 s3:// URI만 쓸 수 있습니다.")
        return text

    try:
        return normalize_relative_posix(text, label=_DATA_LABELS[key])
    except WebPathError as error:
        collect(errors, field, str(error))
        return text


def normalize_data_inputs(raw: Any) -> dict[str, str]:
    """필수 artifact 4개와 선택 test manifest URI를 검증합니다."""

    errors: list[FieldError] = []
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise WebValidationError([FieldError("inputs.data", "data 입력은 object여야 합니다.")])

    resolved: dict[str, str] = {}
    for key in DATA_ARTIFACT_KEYS:
        if key not in raw:
            collect(errors, f"inputs.data.{key}", f"{_DATA_LABELS[key]} 위치가 필요합니다.")
            continue
        resolved[key] = _normalize_data_uri(raw[key], key, errors)

    for key in OPTIONAL_DATA_ARTIFACT_KEYS:
        if key in raw:
            resolved[key] = _normalize_data_uri(raw[key], key, errors)

    raise_if_any(errors)
    ordered_keys = DATA_ARTIFACT_KEYS + OPTIONAL_DATA_ARTIFACT_KEYS
    return {key: resolved[key] for key in ordered_keys if key in resolved}


def uses_s3(data_inputs: dict[str, str]) -> bool:
    return any(value.lower().startswith("s3://") for value in data_inputs.values())


def run_id_output_path(settings: dict[str, Any]) -> Path:
    """Local backend에서 이 run이 결과를 쓸 directory입니다."""

    return resolve_within_repo(settings["output_dir"], label="출력 directory") / settings["run_id"]


def run_id_working_path(settings: dict[str, Any]) -> Path:
    """중단된 학습의 checkpoint가 남아 있는 자리입니다.

    train이 학습 중 checkpoint를 두는 폴더와 같은 이름이어야 합니다.
    """

    parent = resolve_within_repo(settings["output_dir"], label="출력 directory")
    return parent / f".{settings['run_id']}{WORKING_DIRECTORY_SUFFIX}"


def resume_checkpoint_uri(
    config: dict[str, Any], artifacts: Mapping[str, Any] | None = None
) -> str:
    """이 실행을 이어서 하려면 어느 checkpoint를 봐야 하는지 알려 줍니다.

    **끝까지 간 학습은 자기가 남긴 산출물을 씁니다.** 그때는 작업 폴더가 이미 공개
    자리로 옮겨졌거나 지워져서 아래 계산식이 가리키는 자리에 아무것도 없습니다. S3
    게시 경로에는 실행마다 다른 attempt id가 들어 있어 설정만으로는 만들 수도 없습니다.
    train이 돌려준 ``last_checkpoint_uri``에는 best 가중치까지 함께 들어 있어 그 파일
    하나로 이어집니다.
    """

    published = str((artifacts or {}).get("last_checkpoint_uri") or "").strip()
    if published:
        return published
    train = config["train"]
    backend = config.get("storage", {}).get("backend")
    if backend == "s3":
        bucket = os.environ.get("PILL_STORAGE_S3_BUCKET", "").strip()
        if not bucket:
            raise WebValidationError(
                [
                    FieldError(
                        "train.resume_from",
                        "S3에 올라간 학습을 이어서 하려면 PILL_STORAGE_S3_BUCKET "
                        "환경 변수가 필요합니다.",
                    )
                ]
            )
        storage = config.get("storage")
        s3 = storage.get("s3") if isinstance(storage, dict) else None
        configured_prefix = s3.get("prefix") if isinstance(s3, dict) else None
        common_prefix = next(
            (
                str(value).strip().strip("/")
                for value in (
                    os.environ.get("PILL_STORAGE_S3_PREFIX"),
                    configured_prefix,
                )
                if value is not None and str(value).strip()
            ),
            "",
        )
        key = "/".join(
            part
            for part in (
                common_prefix,
                train["output_prefix"].strip("/"),
                train["run_id"],
                RUNNING_PREFIX,
                RESUME_CHECKPOINT_NAME,
            )
            if part
        )
        return f"s3://{bucket}/{key}"
    directory = train["output_dir"].strip("/")
    return (
        f"{directory}/.{train['run_id']}{WORKING_DIRECTORY_SUFFIX}"
        f"/{RESUME_CHECKPOINT_NAME}"
    )


def resume_checkpoint_exists(
    config: dict[str, Any], artifacts: Mapping[str, Any] | None = None
) -> bool:
    """이 실행을 이어갈 checkpoint가 실제 저장소에 남아 있는지 확인합니다."""

    location = resume_checkpoint_uri(config, artifacts)
    if location.lower().startswith("s3://"):
        return create_storage(config).exists(location)
    return resolve_within_repo(location, label="이어서 학습할 checkpoint").is_file()


def build_resume_config(
    config: dict[str, Any],
    *,
    artifacts: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """앞선 실행의 설정을 그대로 두고 이어서 학습할 config를 만듭니다.

    이어서 하는 실행은 **새 이름**을 받습니다. 같은 이름을 다시 쓰면 train이 남아 있는
    작업 폴더를 보고 시작을 거부하고, 결과도 섞입니다.
    """

    resumed = copy.deepcopy(config)
    train = resumed["train"]
    train["resume_from"] = resume_checkpoint_uri(config, artifacts)
    new_run_id = run_id or next_resume_run_id(str(train.get("run_id") or ""))
    if not isinstance(new_run_id, str) or not RUN_ID_PATTERN.fullmatch(new_run_id):
        raise WebValidationError(
            [FieldError("train.run_id", "실행 이름에 쓸 수 없는 글자가 있습니다.")]
        )
    train["run_id"] = new_run_id
    if epochs is not None:
        # train은 epochs를 남은 수가 아니라 전체 목표로 읽고, 이어붙일 epoch보다
        # 크지 않으면 시작 전에 거부합니다.
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise WebValidationError(
                [FieldError("train.epochs", "1 이상의 정수여야 합니다.")]
            )
        train["epochs"] = epochs
    return resumed


def preflight_warnings(
    settings: dict[str, Any], data_inputs: dict[str, str]
) -> list[dict[str, str]]:
    """실행을 막지는 않지만 미리 알려 주면 좋은 것들을 모읍니다."""

    warnings: list[dict[str, str]] = []
    if not uses_s3(data_inputs):
        for key, value in data_inputs.items():
            try:
                path = resolve_within_repo(value, label=_DATA_LABELS[key])
            except WebPathError:
                continue
            if not path.exists():
                warnings.append(
                    {
                        "field": f"inputs.data.{key}",
                        "message": f"{_DATA_LABELS[key]} 파일이 아직 없습니다. 학습 시작 직후 실패할 수 있습니다.",
                    }
                )
    if settings["device"] == "cpu":
        warnings.append(
            {
                "field": "train.device",
                "message": "CPU로 학습하면 GPU보다 매우 오래 걸립니다.",
            }
        )
    if settings["num_workers"] > 0 and os.name == "nt":
        warnings.append(
            {
                "field": "train.num_workers",
                "message": "Windows에서는 worker를 늘려도 효과가 작고 메모리를 더 씁니다.",
            }
        )
    return warnings


def check_run_id_collision(settings: dict[str, Any], data_inputs: dict[str, str]) -> None:
    """이미 같은 run_id의 결과가 있으면 시작 전에 막습니다.

    train은 학습을 **끝낸 뒤** 저장 단계에서야 ``FileExistsError``를 냅니다
    (``train/pipeline.py:154``). 몇 시간을 버리지 않도록 여기서 먼저 확인합니다.
    """

    if uses_s3(data_inputs):
        return  # S3 존재 확인은 credential이 필요하므로 시도하지 않습니다.
    if run_id_output_path(settings).exists():
        raise WebValidationError(
            [
                FieldError(
                    "train.run_id",
                    f"'{settings['run_id']}' 결과가 이미 있습니다. 이름을 비워 두면 설정에서 "
                    "자동으로 짓는데, 설정과 seed가 똑같으면 이름도 같습니다. 같은 실험을 "
                    "다시 돌리려면 seed를 바꾸거나 이름을 직접 쓰세요.",
                )
            ]
        )
    working = run_id_working_path(settings)
    # train은 여기가 비어 있지 않으면 시작을 거부합니다. 중단된 학습의 유일한 사본이라
    # 덮어쓰지 않습니다.
    if working.is_dir() and any(working.iterdir()):
        raise WebValidationError(
            [
                FieldError(
                    "train.run_id",
                    f"'{settings['run_id']}' 학습이 중단된 채로 남아 있습니다. "
                    "이어서 학습하거나 다른 이름을 쓰세요.",
                )
            ]
        )


def validate_request(payload: Any) -> dict[str, Any]:
    """검증 결과를 화면이 그대로 쓸 수 있는 형태로 돌려줍니다. 아무것도 쓰지 않습니다."""

    if not isinstance(payload, dict):
        return {
            "valid": False,
            "errors": [{"field": "body", "message": "요청 본문은 object여야 합니다."}],
            "warnings": [],
            "normalized": None,
        }

    errors: list[dict[str, str]] = []
    settings: dict[str, Any] | None = None
    data_inputs: dict[str, str] | None = None

    # data 입력을 먼저 정규화합니다. 이름을 안 준 실행의 자동 이름이 데이터셋까지
    # 요약하기 때문입니다. 보고 순서는 지금까지처럼 train 오류가 앞입니다.
    inputs = payload.get("inputs")
    data_section = inputs.get("data") if isinstance(inputs, dict) else payload.get("data")
    data_error: WebValidationError | None = None
    try:
        data_inputs = normalize_data_inputs(data_section)
    except WebValidationError as error:
        data_error = error

    try:
        settings = normalize_train_settings(payload.get("train"), data_inputs)
    except WebValidationError as error:
        errors.extend(error.as_list())

    if data_error is not None:
        errors.extend(data_error.as_list())

    warnings: list[dict[str, str]] = []
    if settings is not None and data_inputs is not None and not errors:
        try:
            check_run_id_collision(settings, data_inputs)
        except WebValidationError as error:
            errors.extend(error.as_list())
        warnings = preflight_warnings(settings, data_inputs)

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "normalized": None}
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "normalized": build_runtime_config(settings, data_inputs),
    }


def build_runtime_config(
    settings: dict[str, Any], data_inputs: dict[str, str]
) -> dict[str, Any]:
    """``--only train``에 넘길 완결된 config를 만듭니다.

    ``src/common/config.py``의 ``load_config``는 파일 하나만 읽고 병합하지 않으므로
    이 결과는 그 자체로 완전해야 합니다.
    """

    if uses_s3(data_inputs):
        # bucket 이름은 환경 변수(PILL_STORAGE_S3_BUCKET)에서 오므로 config에 넣지 않습니다.
        storage: dict[str, Any] = {"backend": "s3", "s3": {"prefix": ""}}
    else:
        storage = {"backend": "local", "local": {"root": "artifacts"}}

    return {
        "project": {"name": "pill-object-detection"},
        # configs/base.json은 execution.mode가 "dummy"입니다. 그 값이 들어오면 train이
        # 학습을 건너뛰고 dummy 결과만 돌려주므로, 여기서 항상 명시적으로 덮어씁니다.
        "execution": {"mode": "real"},
        "storage": storage,
        "train": dict(settings),
        "inputs": {"data": dict(data_inputs)},
    }


def _config_path(config_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", config_id):
        raise JobNotFoundError("설정을 찾을 수 없습니다.")
    return config_dir() / f"{config_id}.json"


def write_runtime_config(config: dict[str, Any]) -> str:
    """Runtime config를 gitignore된 위치에 원자적으로 기록하고 id를 돌려줍니다."""

    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    config_id = uuid4().hex
    destination = directory / f"{config_id}.json"
    payload = json.dumps(config, ensure_ascii=False, indent=2) + "\n"

    handle, temporary_name = tempfile.mkstemp(
        dir=directory, prefix=f".{config_id}-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    # 이어서 학습은 job 기록만으로는 못 합니다. 그 기록이 가리키는 이 설정까지
    # 있어야 같은 dataset과 같은 값으로 다시 시작할 수 있습니다.
    state_sync.mirror_runtime_config(config_id, config)
    return config_id


def read_runtime_config(config_id: str) -> dict[str, Any]:
    """저장해 둔 runtime config를 읽습니다."""

    path = _config_path(config_id)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise JobNotFoundError("설정을 찾을 수 없습니다.") from error
    try:
        config = json.loads(content)
    except json.JSONDecodeError as error:
        raise JobNotFoundError("저장된 설정을 읽을 수 없습니다.") from error
    if not isinstance(config, dict):
        raise JobNotFoundError("저장된 설정 형식이 올바르지 않습니다.")
    return config


def stored_run_ids() -> set[str]:
    """저장된 runtime config가 붙잡고 있는 run_id 전부입니다.

    이름이 겹치는지 볼 때 **job 기록과 대기열만 세면 안 됩니다.** 대기열 항목은 시작할
    때 줄에서 빠지고, `JobRecord`는 그 뒤에 만들어집니다. 그 사이에는 어느 목록에도
    없는 이름이 생기고, 그 순간 들어온 요청이 같은 이름을 고릅니다.

    config 파일은 이름을 고른 **직후** 쓰이고 그 뒤로 지워지지 않으므로 그런 틈이
    없습니다. 지워진 기록의 이름까지 남아 번호를 하나 더 건너뛸 수는 있지만, 그쪽이
    안전한 방향입니다.

    읽지 못하는 파일은 건너뜁니다 — 깨진 파일 하나 때문에 이어 학습이 막히면 안 됩니다.
    """

    names: set[str] = set()
    directory = config_dir()
    if not directory.is_dir():
        return names
    for path in directory.glob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        # `train`이 object가 아닌 config 하나 때문에 이어 학습 전체가 500으로 막히면
        # 안 됩니다. `or {}`는 빈 값만 걸러 주고 list나 문자열은 그대로 통과시킵니다.
        train = document.get("train")
        run_id = train.get("run_id") if isinstance(train, dict) else None
        if isinstance(run_id, str) and run_id.strip():
            names.add(run_id.strip())
    return names


def config_relative_path(config_id: str) -> str:
    """subprocess ``--config`` 인자로 넘길 저장소 기준 상대 경로입니다."""

    _config_path(config_id)  # id 형식 검증
    return f"{CONFIG_DIRNAME}/{config_id}.json"


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """화면에 돌려줄 때 credential처럼 보이는 값을 가립니다."""

    return redact(config)
