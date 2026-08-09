"""Job 기록과 설정을 S3의 자기 칸에 두어 런타임보다 오래 살게 합니다.

Colab은 세션이 끊기면 디스크가 함께 사라집니다. 학습 결과는 train이 epoch마다 S3에
올리므로 남지만, 화면의 "이어서 학습"은 ``artifacts/web``의 job 기록과 설정 파일을
읽습니다. 그 둘은 사라진 디스크에 있었으므로 새 런타임에는 누를 대상이 없습니다.

여기서 옮기는 것은 수 KB짜리 JSON 둘뿐입니다. log는 옮기지 않습니다. 이어서 학습에
필요한 것은 어떤 설정으로 어떤 이름의 학습이 돌고 있었는가이고, 그때까지의 학습
내용은 checkpoint에 있습니다.

``PILL_WEB_STATE_WORKSPACE``를 정한 사람만 씁니다. 그 이름이 곧 자기 칸이라 다른
팀원의 기록과 섞이지 않고, 정하지 않은 사람의 기록은 이 기계 밖으로 나가지
않습니다.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .errors import WebStateError
from .paths import config_dir, jobs_dir


__all__ = [
    "STATE_PREFIX",
    "WORKSPACE_VARIABLE",
    "mirror_job_record",
    "mirror_runtime_config",
    "restore",
    "workspace",
]


WORKSPACE_VARIABLE = "PILL_WEB_STATE_WORKSPACE"
BUCKET_VARIABLE = "PILL_STORAGE_S3_BUCKET"
STATE_PREFIX = "experiments/web-state"
RECORD_NAME = "record.json"

# 이 이름은 그대로 S3 key의 한 조각이 됩니다. `/`나 `..`가 들어가면 자기 칸을 벗어나
# 다른 팀원의 기록을 덮어쓸 수 있습니다.
_WORKSPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def workspace() -> str | None:
    """이 기계가 쓸 칸의 이름입니다. 정하지 않았으면 ``None``입니다."""

    value = (os.environ.get(WORKSPACE_VARIABLE) or "").strip()
    if not value:
        return None
    if not _WORKSPACE_PATTERN.fullmatch(value):
        raise WebStateError(
            f"{WORKSPACE_VARIABLE}는 영문·숫자로 시작하고 . _ - 만 쓸 수 있는 "
            f"64자 이내의 이름이어야 합니다: {value!r}"
        )
    return value


def _storage() -> Any | None:
    """자기 칸을 읽고 쓸 storage입니다. bucket이 없으면 ``None``입니다."""

    if not (os.environ.get(BUCKET_VARIABLE) or "").strip():
        return None
    from src.common import create_storage

    return create_storage({"storage": {"backend": "s3", "s3": {"prefix": ""}}})


def _prefix() -> str | None:
    name = workspace()
    return None if name is None else f"{STATE_PREFIX}/{name}"


def _mirror(kind: str, identifier: str, document: Any) -> None:
    """문서 하나를 자기 칸에 올립니다. 실패해도 학습을 멈추지 않습니다."""

    if not _ID_PATTERN.fullmatch(identifier):
        return
    prefix = _prefix()
    if prefix is None:
        return
    storage = _storage()
    if storage is None:
        return
    try:
        storage.write_json(f"{prefix}/{kind}/{identifier}.json", document, overwrite=True)
    except Exception:
        # 사본을 못 남긴다고 지금 도는 학습을 실패시키지는 않습니다. 이건 런타임이
        # 죽었을 때를 위한 보험이지 학습의 일부가 아닙니다.
        pass


def mirror_job_record(job_id: str, document: Any) -> None:
    """Job 기록 하나를 자기 칸에 올립니다."""

    _mirror("jobs", job_id, document)


def mirror_runtime_config(config_id: str, document: Any) -> None:
    """Runtime config 하나를 자기 칸에 올립니다."""

    _mirror("configs", config_id, document)


def _write_if_absent(destination: Path, document: Any) -> bool:
    """이미 있는 파일은 건드리지 않습니다.

    돌고 있는 server가 진실입니다. 오래된 사본이 끝난 학습을 대기 상태로 되돌리면
    그 자리에서 같은 학습이 한 번 더 시작됩니다.
    """

    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return True


def restore() -> set[str]:
    """자기 칸에 남아 있는 기록과 설정을 이 기계로 되살립니다.

    Server가 시작할 때 한 번 부릅니다. 여기서 되살린 job은 ``JobManager.load``가
    ``interrupted``로 바꾸고, 화면은 거기에 "이어서 학습"을 붙입니다.

    되살린 job의 id를 돌려줍니다. 그 job들은 이 기계에서 시작한 것이 아니므로
    화면에 다른 안내가 나가야 합니다.
    """

    restored: set[str] = set()
    prefix = _prefix()
    if prefix is None:
        return restored
    storage = _storage()
    if storage is None:
        return restored
    try:
        locations = storage.list(f"{prefix}/")
    except Exception:
        return restored

    for location in locations:
        name = str(location).split(f"{prefix}/", 1)[-1]
        kind, _, filename = name.partition("/")
        identifier = filename.removesuffix(".json")
        if kind not in {"jobs", "configs"} or not _ID_PATTERN.fullmatch(identifier):
            continue
        try:
            document = storage.read_json(location)
        except Exception:
            continue
        if not isinstance(document, dict):
            continue
        if kind == "configs":
            _write_if_absent(config_dir() / f"{identifier}.json", document)
            continue
        # 이 번호는 사라진 런타임의 것입니다. 새 VM에서는 전혀 다른 process가 그
        # 번호를 쓰고 있을 수 있어, 그대로 두면 죽은 학습이 살아 있다고 나옵니다.
        document = {**document, "process_id": None}
        if _write_if_absent(jobs_dir() / identifier / RECORD_NAME, document):
            restored.add(identifier)
    return restored
