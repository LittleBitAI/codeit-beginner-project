"""crop embedding 학습을 걸고, 재순위에 쓸 embedding을 고릅니다.

detector 학습과 **같은 대기열, 같은 log, 같은 취소**를 씁니다. embedding도
`--only train`으로 도는 학습이라, 자리를 따로 만들면 GPU를 쓰는 문이 둘이 되어
밤새 돌리는 목록이 서로를 밀어냅니다.

detector 설정 화면(`train_config.py`)과 칸을 섞지 않습니다. embedding이 받는
이름은 `EMBEDDING_SETTING_KEYS`뿐이고 detector가 받는 이름과 겹치는 것이 절반도
안 됩니다. 한 화면으로 묶으면 detector 폼에 backbone 선택이 뜹니다.

여기서 학습한 embedding은 앙상블 화면이 재순위에 씁니다. 어느 crop 은행으로
학습했는지를 함께 들고 다니는 것이 이 module의 두 번째 일입니다 — checkpoint와
참조 crop이 짝이 맞지 않으면 점수만 조용히 나빠집니다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.common import train_contract as _contract

from .errors import FieldError, WebError, collect, raise_if_any
from .train_config import RUN_ID_PATTERN, generate_run_id


__all__ = [
    "BACKBONES",
    "TASK",
    "build_config",
    "defaults",
    "list_runs",
    "rerank_settings",
]


TASK = _contract.TRAIN_TASK_KEY
TASK_VALUE = "embedding"
BACKBONES = _contract.EMBEDDING_BACKBONES
DEFAULTS = _contract.EMBEDDING_SETTING_DEFAULTS

#: 학습에 필요한 data artifact입니다. crop 은행은 data가 만들고, class map은
#: 어느 이름이 어느 class인지 알려 줍니다.
DATA_KEYS = _contract.EMBEDDING_DATA_ARTIFACT_KEYS

_DEVICES = ("cpu", "cuda")
#: 한 번에 고를 수 있는 embedding 수입니다. 재순위는 고른 수만큼 시험 crop을 다시
#: 훑으므로, 여덟 개를 고르면 여덟 배 걸립니다.
MAX_RERANK_MODELS = 8


def defaults() -> dict[str, Any]:
    """embedding 학습 폼이 그릴 값입니다. 계약의 기본값을 그대로 냅니다."""

    return {
        "backbones": list(BACKBONES),
        "devices": list(_DEVICES),
        "run_id_pattern": RUN_ID_PATTERN.pattern,
        "defaults": {
            "backbone": DEFAULTS["backbone"],
            "epochs": DEFAULTS["epochs"],
            "batch_size": DEFAULTS["batch_size"],
            "learning_rate": DEFAULTS["learning_rate"],
            "weight_decay": DEFAULTS["weight_decay"],
            "seed": DEFAULTS["seed"],
            "pretrained": DEFAULTS["pretrained"],
            "device": DEFAULTS["device"],
        },
    }


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _positive_integer(
    errors: list[FieldError], payload: Mapping[str, Any], field: str, label: str
) -> int:
    value = payload.get(field, DEFAULTS[field])
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        collect(errors, field, f"{label}은(는) 1 이상의 정수여야 합니다.")
        return int(DEFAULTS[field])
    return value


def _finite(value: int | float) -> bool:
    """유한한 숫자인지 봅니다. **검사가 스스로 터지지 않게** 합니다.

    python 정수는 크기 제한이 없어서 `10**400` 같은 값을 `float()`로 바꾸면
    `OverflowError`가 납니다. 거절하려던 값 때문에 검사가 그 자리에서 죽습니다.
    """

    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _number(
    errors: list[FieldError],
    payload: Mapping[str, Any],
    field: str,
    label: str,
    *,
    allow_zero: bool = False,
) -> float:
    value = payload.get(field, DEFAULTS[field])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        collect(errors, field, f"{label}은(는) 숫자여야 합니다.")
        return float(DEFAULTS[field])
    # 유한성을 먼저 봅니다. `nan`은 어느 비교에도 걸리지 않아, 크기만 재면
    # 그대로 통과해 learning rate가 `nan`인 학습이 대기열에 들어갑니다.
    if not _finite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = "0 이상" if allow_zero else "0보다 큰"
        collect(errors, field, f"{label}은(는) {bound} 유한한 숫자여야 합니다.")
        return float(DEFAULTS[field])
    return float(value)


def build_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """embedding 학습 설정을 검사하고 pipeline이 받을 config로 만듭니다.

    detector와 달리 manifest가 아니라 crop 은행 하나를 봅니다. 그 은행이 없으면
    학습할 그림이 없으므로 **여기서** 거절합니다. 뒤로 미루면 대기열에 들어간 뒤
    자기 차례가 와서야 실패하고, 그것을 밤에 발견합니다.
    """

    if not isinstance(payload, Mapping):
        raise WebError("설정은 object여야 합니다.")

    errors: list[FieldError] = []

    run_id = _text(payload.get("run_id")) or generate_run_id()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        collect(errors, "run_id", "영문·숫자로 시작하고 영문·숫자·`.`·`_`·`-`만 쓸 수 있습니다.")

    backbone = _text(payload.get("backbone")) or DEFAULTS["backbone"]
    if backbone not in BACKBONES:
        collect(errors, "backbone", f"{', '.join(BACKBONES)} 중 하나여야 합니다.")

    device = _text(payload.get("device")) or DEFAULTS["device"]
    if device not in _DEVICES:
        collect(errors, "device", f"{' 또는 '.join(_DEVICES)}여야 합니다.")

    pretrained = payload.get("pretrained", DEFAULTS["pretrained"])
    if not isinstance(pretrained, bool):
        collect(errors, "pretrained", "참 또는 거짓이어야 합니다.")
        pretrained = bool(DEFAULTS["pretrained"])

    seed = payload.get("seed", DEFAULTS["seed"])
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        collect(errors, "seed", "0 이상의 정수여야 합니다.")
        seed = int(DEFAULTS["seed"])

    epochs = _positive_integer(errors, payload, "epochs", "epoch 수")
    batch_size = _positive_integer(errors, payload, "batch_size", "batch 크기")
    learning_rate = _number(errors, payload, "learning_rate", "learning rate")
    weight_decay = _number(errors, payload, "weight_decay", "weight decay", allow_zero=True)

    data_inputs: dict[str, str] = {}
    for key in DATA_KEYS:
        value = _text(payload.get(key))
        if value is None:
            collect(errors, key, "필요한 입력입니다.")
            continue
        if ".." in value.replace("\\", "/").split("/"):
            collect(errors, key, "상위 폴더를 가리킬 수 없습니다.")
            continue
        data_inputs[key] = value

    raise_if_any(errors)

    from .ensemble import _storage_config, _storage_root, _uses_s3

    output_dir = (
        f"{_storage_root()}/experiments/embeddings/{run_id}"
        if _uses_s3()
        else f"artifacts/experiments/embeddings/{run_id}"
    )
    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "real"},
        "storage": _storage_config(),
        "inputs": {"data": data_inputs},
        "train": {
            TASK: TASK_VALUE,
            "run_id": run_id,
            "backbone": backbone,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pretrained": pretrained,
            "seed": seed,
            "device": device,
            "output_dir": output_dir,
        },
    }


def is_embedding_settings(settings: Mapping[str, Any] | None) -> bool:
    """그 학습이 embedding이었는지 봅니다. 기록에 남은 설정만 보고 판단합니다."""

    if not isinstance(settings, Mapping):
        return False
    return settings.get(TASK) == TASK_VALUE


def list_runs() -> list[dict[str, Any]]:
    """재순위에 쓸 수 있는 embedding 목록입니다. 최근 것부터 옵니다.

    이 서버가 돌린 학습 기록에서 찾습니다. registry 요약은 detector 설정만
    담아서(그쪽 `TRAINING_KEYS`) 어느 것이 embedding인지 말해 주지 않습니다.
    """

    from .jobs import get_manager

    runs: list[dict[str, Any]] = []
    for record in get_manager().list_jobs():
        if not is_embedding_settings(record.settings):
            continue
        checkpoint = _text(record.artifacts.get("best_checkpoint_uri"))
        runs.append(
            {
                "run_id": record.run_id,
                "job_id": record.job_id,
                "status": record.status,
                "backbone": record.settings.get("backbone"),
                "epochs": record.settings.get("epochs"),
                "checkpoint_uri": checkpoint,
                "crop_bank_uri": _text(record.data_inputs.get("crop_bank_uri")),
                "created_at": record.created_at,
                # 재순위에 넣을 수 있는가. 학습이 성공했고 checkpoint를 남겼을 때만입니다.
                "ready": record.status == "succeeded" and checkpoint is not None,
            }
        )
    runs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return runs


def rerank_settings(run_ids: Sequence[str]) -> dict[str, Any]:
    """고른 embedding들을 evaluate가 받는 재순위 설정으로 바꿉니다.

    **참조 crop은 하나여야 합니다.** evaluate는 은행 하나로 모든 checkpoint의
    margin을 재는데, 서로 다른 은행에서 학습한 model을 섞으면 한쪽은 자기가 본
    적 없는 crop과 견주게 됩니다. 그렇게 나온 값도 숫자이긴 해서, 막지 않으면
    조용히 나빠진 제출이 나옵니다.
    """

    wanted = [str(item).strip() for item in run_ids if str(item).strip()]
    if not wanted:
        return {}
    if len(set(wanted)) != len(wanted):
        raise WebError("같은 embedding을 두 번 골랐습니다.")
    if len(wanted) > MAX_RERANK_MODELS:
        raise WebError(f"재순위에 쓸 embedding은 {MAX_RERANK_MODELS}개까지 고를 수 있습니다.")

    by_run = {item["run_id"]: item for item in list_runs()}
    unknown = [name for name in wanted if name not in by_run]
    if unknown:
        raise WebError(f"기록에 없는 embedding입니다: {', '.join(unknown)}")
    selected = [by_run[name] for name in wanted]

    unready = [item["run_id"] for item in selected if not item["ready"]]
    if unready:
        raise WebError(f"아직 checkpoint가 없는 embedding입니다: {', '.join(unready)}")

    banks = {item["crop_bank_uri"] for item in selected}
    if None in banks:
        raise WebError("어느 crop 은행으로 학습했는지 모르는 embedding이 있습니다.")
    if len(banks) > 1:
        raise WebError(
            "서로 다른 crop 은행으로 학습한 embedding은 함께 쓸 수 없습니다: "
            f"{', '.join(sorted(str(bank) for bank in banks))}"
        )

    return {
        "rerank_checkpoint_uris": [str(item["checkpoint_uri"]) for item in selected],
        "rerank_crop_bank_uri": str(banks.pop()),
    }
