"""GUI가 만든 학습 설정을 train이 실제로 그대로 읽는지 봅니다.

web은 train을 import할 수 없어서, 두 파이프라인이 같은 key 이름을 쓴다는 것은 지금껏
**아무도 확인하지 않았습니다.** 값이 같은지는 `src/common/train_contract.py`가 지키지만,
그 값을 담아 보내는 **칸 이름**은 계약에 없습니다. 한쪽이 `resume_from`을
`resume_uri`로 바꾸고 자기 test까지 함께 고치면 양쪽 다 초록인 채로, 이어서 학습이
조용히 처음부터 다시 돕니다.

두 파이프라인을 함께 부르는 test라 어느 쪽 소유도 아닌 여기(`tests/`)에 둡니다.
"""

from typing import Any

from src.pipelines.train.pipeline import _settings as train_settings
from src.pipelines.web.train_config import normalize_train_settings

RESUME = "artifacts/experiments/completed/.train-1.partial/last_checkpoint.pt"

#: 화면이 채워 보내는 설정 한 벌입니다. train이 CPU에서 받을 수 있는 값으로 고릅니다.
WEB_INPUT: dict[str, Any] = {
    "run_id": "train-handoff",
    "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "optimizer": "AdamW",
    "device": "cpu",
    "epochs": 3,
    "batch_size": 2,
    "checkpoint_every": 1,
    "seed": 42,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "augmentation": "pill_basic",
    "gradient_accumulation_steps": 2,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,
    "lr_min_factor": 0.01,
    "early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.001,
    "resume_from": RESUME,
}


def test_train_reads_every_setting_the_web_sends():
    """web이 실어 보내는 key를 train이 하나도 빠짐없이 알아본다."""

    sent = normalize_train_settings(WEB_INPUT)
    read = train_settings({"train": dict(sent)})

    unknown = set(sent) - set(read)
    assert unknown == set(), f"train이 알아보지 못하는 칸입니다: {sorted(unknown)}"


def test_the_checkpoint_the_web_points_at_is_the_one_train_resumes_from():
    """이어서 학습은 값까지 같아야 합니다. 이름만 맞고 값이 비면 처음부터 돕니다."""

    sent = normalize_train_settings(WEB_INPUT)
    read = train_settings({"train": dict(sent)})

    assert sent["resume_from"] == RESUME
    assert read["resume_from"] == RESUME

