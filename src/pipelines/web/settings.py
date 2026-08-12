"""사람이 한 번 정해 두고 계속 쓰는 값입니다.

지금은 **평가를 언제 돌릴지** 하나뿐입니다. 학습이 성공하면 서버가 이어서 평가를
돌리는데, 그 시점을 GPU 사정에 맞게 고를 수 있어야 합니다.

- ``parallel`` — 다음 학습이 돌고 있어도 곧바로 평가합니다. 평가가 VRAM을 약 1.8GB
  더 쓰므로 8GB 카드에서 학습과 겹치면 out of memory로 둘 다 잃을 수 있습니다.
- ``serial`` — 도는 학습이 하나도 없을 때까지 기다렸다 평가합니다.

**고른 적이 없으면 ``None``이고, 그때는 자동 평가를 하지 않습니다.** 서버를 올렸다는
이유만으로 남의 GPU가 몇 시간씩 도는 일은 없어야 합니다. 사람이 설정 화면에서 한 번
고르고 저장해야 켜집니다.

값은 ``artifacts/web/settings.json`` 하나에 담깁니다. 읽지 못하면 고르지 않은 것으로
봅니다 — 설정 파일이 깨졌다고 학습이 멈추거나 저절로 돌면 안 됩니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import FieldError, WebValidationError
from .paths import web_state_dir


EVALUATION_MODES = ("parallel", "serial")

DEFAULTS: dict[str, Any] = {"evaluation_mode": None}


def _path() -> Path:
    return web_state_dir() / "settings.json"


def read_settings() -> dict[str, Any]:
    """저장된 설정입니다. 없거나 깨졌으면 "고른 적 없음"입니다."""

    try:
        stored = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULTS)
    if not isinstance(stored, dict):
        return dict(DEFAULTS)
    mode = stored.get("evaluation_mode")
    return {"evaluation_mode": mode if mode in EVALUATION_MODES else None}


def evaluation_mode() -> str | None:
    """고른 적이 없으면 ``None``. 부르는 쪽은 그때 아무것도 하지 않아야 합니다."""

    mode = read_settings()["evaluation_mode"]
    return str(mode) if mode in EVALUATION_MODES else None


def write_settings(values: Any) -> dict[str, Any]:
    """받은 값을 검사한 뒤 저장합니다. 모르는 key는 조용히 버립니다."""

    if not isinstance(values, dict):
        raise WebValidationError([FieldError("settings", "객체 형태로 보내야 합니다.")])
    mode = values.get("evaluation_mode")
    if mode not in EVALUATION_MODES:
        raise WebValidationError(
            [
                FieldError(
                    "evaluation_mode",
                    f"{' 또는 '.join(EVALUATION_MODES)} 중 하나여야 합니다.",
                )
            ]
        )
    saved = {"evaluation_mode": mode}
    directory = web_state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    _path().write_text(
        json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return saved
