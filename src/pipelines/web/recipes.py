"""화면이 한 번에 채워 주는, **실제로 점수를 받은** 학습 설정입니다.

새 실험 화면의 기본값은 가장 가벼운 model에 1 epoch입니다. 그 자리에서 최고 점수를
재현하려면 열 몇 칸을 손으로 정확히 고쳐야 하는데, 어느 칸이 결과를 바꿨는지는 저장소
어디에도 적혀 있지 않았습니다. 그래서 그 값을 여기 한 벌 둡니다.

**여기 있는 것은 기본값이 아니라 기록입니다.** 끝난 실행의 `config_snapshot.train`을
그대로 옮겨 적었고, 화면은 이 값을 칸에 채워 주기만 합니다. 검증과 거절은 지금까지처럼
`train_config.py`와 train이 합니다.

칸 이름은 화면이 쓰는 평평한 이름입니다(`lr_scheduler` 안쪽은 `lr_*`). `rerunSettings.ts`가
끝난 실행을 같은 모양으로 되돌리므로, 두 경로가 같은 자리에 같은 값을 놓습니다.

**값을 옮겨 적었으므로 안전장치는 하나뿐입니다** — `test_web_recipes.py`가 이 설정을
시작 전 검증(`validate_request`)에 그대로 통과시켜 봅니다. 이름이나 규칙이 train에서
바뀌면 그 자리에서 빨개집니다.
"""

from __future__ import annotations

from typing import Any


__all__ = ["RECIPES", "recipe_specs"]


#: 화면에 내주는 설정 한 벌입니다.
#:
#: `dino-basic-e12-b1-lr1e4-s42-4675`의 기록입니다. 단독 Kaggle 0.62437로 이 저장소의
#: 단일 model 최고점이고, 같은 계열 셋을 융합하고 crop 임베딩으로 재순위한 제출이
#: 0.63594로 전체 최고점입니다. 융합과 재순위는 학습이 아니라 evaluate가 하는 일이라
#: 이 표에 담기지 않습니다 — `docs/reproduce.md`가 그 절차를 적습니다.
RECIPES: tuple[dict[str, Any], ...] = (
    {
        "name": "best-detector",
        "label": "최고 점수 detector",
        "note": (
            "Kaggle 0.62437을 받은 dino-basic-e12-b1-lr1e4-s42-4675의 설정입니다. "
            "RTX 3080에서 12 epoch에 하루 가까이 걸립니다. 발표 시연이라면 epochs만 "
            "1이나 2로 낮추세요."
        ),
        "settings": {
            "architecture": "dino_r50_4scale",
            "pretrained": True,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08,
            "device": "cuda",
            "precision": "amp",
            "epochs": 12,
            "batch_size": 1,
            "num_workers": 0,
            "gradient_accumulation_steps": 4,
            "input_size": 1280,
            "checkpoint_every": 1,
            "seed": 42,
            "augmentation": "pill_basic",
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 1200,
            "lr_warmup_start_factor": 0.001,
            "lr_min_factor": 0.01,
        },
    },
)


def recipe_specs() -> list[dict[str, Any]]:
    """화면에 내줄 설정 목록입니다. 값을 복사해 돌려주므로 호출자가 고쳐도 안전합니다."""

    return [
        {
            "name": recipe["name"],
            "label": recipe["label"],
            "note": recipe["note"],
            "settings": dict(recipe["settings"]),
        }
        for recipe in RECIPES
    ]
