"""학습 설정에서 train과 GUI가 **반드시 같아야 하는 값**들입니다.

Train이 정하고, GUI(web)는 그대로 사람에게 보여 주고 시작 전에 같은 이유로 거부합니다.
GUI는 train을 import할 수 없으므로(소유 경계) 이 값들을 통째로 복제해 두었고,
`test_web_train_contract.py`가 train의 source를 `ast`로 읽어 두 벌이 어긋나는지
감시했습니다. 감시가 필요했던 이유는 값이 두 곳에 있었기 때문입니다. 값을 여기 한
곳에 두면 어긋날 수가 없습니다.

**행동은 여기 없습니다.** 검증과 정규화는 train이, 화면 표시와 미리 거부하기는 web이
각자 합니다. 여기 있는 것은 둘이 같은 것을 말하기 위해 필요한 이름과 숫자뿐이라,
이 파일은 어떤 pipeline도 import하지 않고 torch도 부르지 않습니다.

값을 바꾸는 것은 **train 담당자**입니다. 여기서 이름을 하나 더하면 GUI는 그 순간
그것을 고를 수 있게 되므로, train이 실제로 받아들이기 전에 더하면 안 됩니다.
"""

from __future__ import annotations

import re
from typing import Any


__all__ = [
    "ARCHITECTURES",
    "AUGMENTATIONS",
    "CUDA_ONLY_PRECISIONS",
    "DATA_ARTIFACT_KEYS",
    "DEFAULT_ACCUMULATION_STEPS",
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_AUGMENTATION",
    "DEFAULT_EARLY_STOPPING_MIN_DELTA",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_LR_SCHEDULER",
    "DEFAULT_PRECISION",
    "DEVICES",
    "EARLY_STOPPING_KEYS",
    "LEGACY_OPTIMIZER",
    "LR_SCHEDULER_DEFAULTS",
    "LR_WARMUP_DEFAULTS",
    "MMDETECTION_ARCHITECTURES",
    "MMDETECTION_REQUIRED",
    "OPTIMIZERS",
    "OPTIMIZER_PROFILES",
    "OPTIONAL_DATA_ARTIFACT_KEYS",
    "PRECISIONS",
    "RUN_ID_PATTERN",
    "SETTING_DEFAULTS",
]


#: 학습을 시작하려면 있어야 하는 data artifact입니다.
DATA_ARTIFACT_KEYS = (
    "train_manifest_uri",
    "validation_manifest_uri",
    "class_map_uri",
    "dataset_summary_uri",
)
#: 있으면 evaluate가 쓰고, 없어도 학습은 됩니다.
OPTIONAL_DATA_ARTIFACT_KEYS = ("test_manifest_uri",)

#: run_id는 파일과 S3 key 이름이 되므로 경로가 될 수 있는 글자를 받지 않습니다.
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

DEVICES = ("cpu", "cuda")

#: 값이 없는 옛 설정이 뜻하던 model입니다. 새 실험의 기본값이기도 합니다.
DEFAULT_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
#: MMDetection으로 학습하는 model입니다. 아래 제약이 이 둘에만 걸립니다.
MMDETECTION_ARCHITECTURES = ("dino_r50_4scale", "cascade_rcnn_swin_t_fpn")
ARCHITECTURES = (
    DEFAULT_ARCHITECTURE,
    "fasterrcnn_resnet50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    *MMDETECTION_ARCHITECTURES,
)
#: MMDetection model이 8GB에서 도는 유일한 조합입니다. 다른 값이면 학습을 시작한 뒤
#: 메모리로 터지므로, GUI는 대기열에 넣기 전에 같은 이유로 막습니다.
MMDETECTION_REQUIRED = {
    "device": "cuda",
    "precision": "amp",
    "optimizer": "AdamW",
    "batch_size": 1,
}
#: MMDetection model만 쓰는 입력 크기입니다.
DEFAULT_INPUT_SIZE = 640
#: MMDetection model을 고르면 이만큼 모아 한 번 갱신합니다. batch 1로 도는 두 모델이
#: 쓸 만한 유효 batch를 갖게 하는 값입니다.
DEFAULT_ACCUMULATION_STEPS = 8

OPTIMIZERS = ("AdamW", "SGD", "Adam")
#: optimizer를 고르기 전에 학습한 실행이 쓰던 값입니다. 새 실험은 AdamW로 시작합니다.
LEGACY_OPTIMIZER = "SGD"
#: optimizer마다 자기가 쓰는 값과 그 기본값입니다.
OPTIMIZER_PROFILES: dict[str, dict[str, float]] = {
    "AdamW": {
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    },
    "SGD": {
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    },
    "Adam": {
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    },
}

#: 고를 수 있는 증강 preset의 이름입니다. 실제 변형 값은 train이 들고 있습니다.
AUGMENTATIONS = ("none", "pill_basic")
DEFAULT_AUGMENTATION = "none"

#: `amp`가 fp16을 쓸지 bf16을 쓸지는 train이 GPU와 architecture를 보고 정하고,
#: `fp16`·`bf16`은 고른 그대로 씁니다.
PRECISIONS = ("fp32", "amp", "fp16", "bf16")
#: 절반 정밀도는 CUDA에서만 됩니다.
CUDA_ONLY_PRECISIONS = ("amp", "fp16", "bf16")
DEFAULT_PRECISION = "fp32"

#: warmup은 아래 값으로 모든 schedule이 함께 씁니다.
LR_WARMUP_DEFAULTS = {"warmup_steps": 0, "warmup_start_factor": 0.001}
#: schedule마다 자기가 쓰는 값과 그 기본값입니다. 고를 수 있는 이름도 이 목록이
#: 정합니다. `none`은 상수 learning rate입니다.
LR_SCHEDULER_DEFAULTS: dict[str, dict[str, float | int]] = {
    "none": {},
    "cosine": {"min_lr_factor": 0.01},
    "step": {"step_size": 3, "gamma": 0.1},
    "linear": {"min_lr_factor": 0.01},
}
DEFAULT_LR_SCHEDULER = "none"

#: train은 모르는 key가 하나만 있어도 early_stopping object를 통째로 거부합니다.
EARLY_STOPPING_KEYS = ("patience", "min_delta")
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0

#: 비워 두고 보냈을 때 train이 쓰는 값입니다. GUI는 빈 칸에 이 값을 안내합니다.
#: `num_workers`는 여기 없습니다 — train이 device와 OS를 보고 정하므로 GUI가 미리
#: 말할 수 있는 값이 아닙니다. `gradient_accumulation_steps`는 MMDetection model에서만
#: `DEFAULT_ACCUMULATION_STEPS`가 됩니다.
SETTING_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "epochs": 1,
    "checkpoint_every": 1,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "device": "cpu",
    "pretrained": False,
    "output_dir": "artifacts/experiments/completed",
    "output_prefix": "experiments/completed",
}
