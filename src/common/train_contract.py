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
    "BEST_CHECKPOINT_NAME",
    "CUDA_ONLY_PRECISIONS",
    "DEFAULT_EMBEDDING_BACKBONE",
    "DEFAULT_TRAIN_TASK",
    "EMBEDDING_BACKBONES",
    "EMBEDDING_DATA_ARTIFACT_KEYS",
    "EMBEDDING_SETTING_DEFAULTS",
    "EMBEDDING_SETTING_KEYS",
    "TRAIN_TASKS",
    "TRAIN_TASK_KEY",
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
    "EPOCH_ARCHIVE_START",
    "LEGACY_OPTIMIZER",
    "LR_SCHEDULER_DEFAULTS",
    "LR_WARMUP_DEFAULTS",
    "MMDETECTION_ARCHITECTURES",
    "MMDETECTION_REQUIRED",
    "OPTIMIZERS",
    "OPTIMIZER_PROFILES",
    "OPTIONAL_DATA_ARTIFACT_KEYS",
    "PRECISIONS",
    "RESUME_CHECKPOINT_NAME",
    "RUNNING_PREFIX",
    "RUN_ID_PATTERN",
    "SETTING_DEFAULTS",
    "SETTING_KEYS",
    "WORKING_CHECKPOINT_NAMES",
    "WORKING_DIRECTORY_SUFFIX",
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
AUGMENTATIONS = ("none", "pill_basic", "pill_geometric")
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

#: 매 epoch checkpoint를 남기기 시작할 epoch입니다. **GUI가 새 실험 화면에 미리
#: 채워 주는 값이고, 기본값이 아닙니다** — 보내지 않으면 train은 아무것도 보관하지
#: 않습니다. 무거운 model은 한 epoch이 비싸고 일찍 수렴하므로 더 앞에서 시작합니다.
EPOCH_ARCHIVE_START: dict[str, int] = {
    name: (8 if name in MMDETECTION_ARCHITECTURES else 15) for name in ARCHITECTURES
}

#: train은 모르는 key가 하나만 있어도 early_stopping object를 통째로 거부합니다.
EARLY_STOPPING_KEYS = ("patience", "min_delta")
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0

#: ``config["train"]``에 담아 보낼 수 있는 칸 이름 전부입니다.
#:
#: 값이 같은지는 위의 표들이 지키지만, 그 값을 담아 보내는 **이름**은 지금까지 아무도
#: 지키지 않았습니다. web은 train을 import할 수 없어 이름을 옮겨 적을 뿐이고, 한쪽이
#: 이름을 바꾸며 자기 test까지 함께 고치면 양쪽 다 초록인 채로 그 값이 조용히 버려집니다.
#: train은 자기가 정말 이 칸들을 읽는지, web은 이 목록에 없는 칸을 보내지 않는지 각자
#: 확인합니다. 어느 쪽도 상대를 import하지 않고 여기만 봅니다.
SETTING_KEYS = (
    "architecture",
    "archive_epochs_from",
    "augmentation",
    "batch_size",
    "beta1",
    "beta2",
    "checkpoint_every",
    "device",
    "early_stopping",
    "epochs",
    "epsilon",
    "gradient_accumulation_steps",
    "input_size",
    "learning_rate",
    "lr_scheduler",
    "momentum",
    "num_workers",
    "optimizer",
    "output_dir",
    "output_prefix",
    "precision",
    "pretrained",
    "resume_from",
    "run_id",
    "seed",
    "weight_decay",
)

#: train이 할 수 있는 일입니다. `detector`는 상자를 찾는 기존 학습이고, `embedding`은
#: 잘라 낸 알약 crop 하나가 어떤 class인지 재는 자를 만듭니다. 둘은 입력도 설정도
#: 다르므로 위 `SETTING_KEYS`를 함께 쓰지 않습니다 — 한 목록에 넣으면 GUI가 detector
#: 화면에서 backbone을, 임베딩 화면에서 optimizer를 내놓게 됩니다.
TRAIN_TASKS = ("detector", "embedding")
DEFAULT_TRAIN_TASK = "detector"
TRAIN_TASK_KEY = "task"

#: 임베딩이 쓸 수 있는 backbone입니다. 구조가 서로 다를수록 틀리는 자리가 달라
#: 여럿을 함께 쓸 때 값이 있습니다.
EMBEDDING_BACKBONES = ("resnet18", "resnet34", "resnet50")
DEFAULT_EMBEDDING_BACKBONE = "resnet18"

#: 임베딩 학습을 시작하려면 있어야 하는 data artifact입니다. detector와 달리 manifest가
#: 아니라 **잘라 둔 crop 은행**을 읽습니다.
EMBEDDING_DATA_ARTIFACT_KEYS = ("crop_bank_uri", "class_map_uri")

#: ``config["train"]``에 담아 보낼 수 있는 칸 이름입니다(`task="embedding"`일 때).
#: detector 쪽과 같은 이유로 여기 이름만 보고 양쪽이 맞춥니다.
EMBEDDING_SETTING_KEYS = (
    TRAIN_TASK_KEY,
    "backbone",
    "batch_size",
    "checkpoint_every",
    "device",
    "epochs",
    "learning_rate",
    "num_workers",
    "output_dir",
    "output_prefix",
    "pretrained",
    "run_id",
    "seed",
    "weight_decay",
)

#: 비워 두고 보냈을 때 임베딩 학습이 쓰는 값입니다. 검증된 조합을 그대로 둡니다.
EMBEDDING_SETTING_DEFAULTS: dict[str, Any] = {
    "backbone": DEFAULT_EMBEDDING_BACKBONE,
    "batch_size": 32,
    "checkpoint_every": 1,
    "device": "cpu",
    "epochs": 30,
    "learning_rate": 3e-4,
    "pretrained": True,
    "seed": 42,
    "weight_decay": 1e-4,
    "output_dir": "artifacts/experiments/embeddings",
    "output_prefix": "experiments/embeddings",
}

#: 학습 중 작업 폴더에 두는 파일입니다. 마지막 것이 이어서 학습할 대상입니다.
WORKING_CHECKPOINT_NAMES = ("best_checkpoint.pt", "last_checkpoint.pt")
BEST_CHECKPOINT_NAME = WORKING_CHECKPOINT_NAMES[0]
RESUME_CHECKPOINT_NAME = WORKING_CHECKPOINT_NAMES[-1]

#: 끝나지 않은 학습이 어디 있는지 정하는 규칙입니다. train이 그 자리에 쓰고, GUI는
#: 이어서 학습 버튼을 올릴지 정하려고 같은 자리를 찾아봅니다. 한쪽만 바뀌면 GUI는
#: 있는 checkpoint를 없다고 하거나, 없는 것을 있다고 하고 train이 곧바로 거부합니다.
WORKING_DIRECTORY_SUFFIX = ".partial"
RUNNING_PREFIX = "running"

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
