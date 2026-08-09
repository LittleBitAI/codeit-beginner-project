"""Job 기록과 log를 gitignore된 위치에 보관합니다.

중앙 index 파일은 두지 않습니다. 여러 곳에서 같은 파일을 고치면 깨질 위험이 있어서,
job마다 자기 directory만 쓰고 목록은 시작할 때 훑어서 만듭니다.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from ..errors import JobNotFoundError
from ..masking import redact, sanitize_line
from ..paths import JOBS_DIRNAME, jobs_dir, repository_root, web_state_dir
from .model import JobRecord


__all__ = [
    "load_queue",
    "save_queue",
    "append_log",
    "delete_record",
    "job_directory",
    "load_all_records",
    "load_record",
    "read_logs",
    "save_record",
]


_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_RECORD_NAME = "record.json"
_LOG_NAME = "log.jsonl"


def _validate_id(job_id: str) -> str:
    """경로 조작을 막기 위해 디스크에 닿기 전에 형식을 확인합니다."""

    if not isinstance(job_id, str) or not _ID_PATTERN.fullmatch(job_id):
        raise JobNotFoundError("학습 기록을 찾을 수 없습니다.")
    return job_id


def job_directory(job_id: str) -> Path:
    return jobs_dir() / _validate_id(job_id)


def _write_atomic(destination: Path, payload: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_record(record: JobRecord) -> None:
    """기록을 저장합니다. 저장 전에 마스킹해 비밀이 디스크에도 닿지 않게 합니다."""

    payload = redact(record.to_dict())
    _write_atomic(
        job_directory(record.job_id) / _RECORD_NAME,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def load_record(job_id: str) -> JobRecord:
    path = job_directory(job_id) / _RECORD_NAME
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise JobNotFoundError("학습 기록을 찾을 수 없습니다.") from error
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise JobNotFoundError("학습 기록을 읽을 수 없습니다.") from error
    if not isinstance(payload, dict) or "job_id" not in payload:
        raise JobNotFoundError("학습 기록 형식이 올바르지 않습니다.")
    return JobRecord.from_dict(payload)


def load_all_records() -> list[JobRecord]:
    """저장된 모든 job을 최신순으로 읽습니다. 깨진 기록은 건너뜁니다."""

    root = jobs_dir()
    if not root.is_dir():
        return []

    records: list[JobRecord] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not _ID_PATTERN.fullmatch(entry.name):
            continue
        try:
            records.append(load_record(entry.name))
        except JobNotFoundError:
            continue  # 손상된 기록 하나가 목록 전체를 막지 않게 합니다.
    records.sort(key=lambda record: record.created_at, reverse=True)
    return records


def delete_record(job_id: str) -> None:
    """이 GUI가 만든 job directory 하나만 지웁니다.

    지우는 것은 ``artifacts/web/jobs/<job_id>/``의 ``record.json``과 ``log.jsonl``
    뿐입니다. checkpoint와 학습 결과 폴더는 train이 만든 산출물이고, registry
    기록과 팀에 공유된 기록도 이 화면의 것이 아닙니다. 설정 파일도 남깁니다.
    대기열 항목이나 이어서 학습이 아직 그 config를 가리킬 수 있기 때문입니다.

    ``job_directory``가 형식을 먼저 확인하므로 이름으로는 빠져나갈 수 없습니다.
    그것만으로는 부족합니다. ``artifacts/web/jobs`` 자체가 다른 곳을 가리키는
    link면 그 아래 job directory는 진짜 directory라서 ``rmtree``가 아무 의심 없이
    그쪽을 지웁니다(Windows junction은 권한 없이 만들 수 있습니다). 저장소 밖이면
    남의 파일이고, ``artifacts/experiments/completed``처럼 저장소 안이어도 train이
    만든 checkpoint입니다.

    그래서 **link를 따라간 실제 위치가 글자 그대로의 자리와 같은지**만 봅니다.
    jobs 루트를 함께 따라가서 비교하면(``jobs_dir().resolve()``) 그 link가 정확히
    상쇄돼 검사가 통과합니다. 기대 위치는 link를 따라가지 않고 만듭니다.
    """

    directory = job_directory(job_id)
    if not directory.is_dir():
        raise JobNotFoundError("학습 기록을 찾을 수 없습니다.")

    expected = repository_root().resolve() / JOBS_DIRNAME / job_id
    if directory.resolve() != expected:
        raise JobNotFoundError("학습 기록을 찾을 수 없습니다.")

    try:
        shutil.rmtree(directory)
    except OSError as error:
        raise JobNotFoundError("학습 기록을 지우지 못했습니다.") from error


def append_log(job_id: str, entries: Iterable[dict[str, Any]]) -> None:
    """Log 줄을 JSONL로 덧붙입니다."""

    materialized = list(entries)
    if not materialized:
        return
    directory = job_directory(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / _LOG_NAME).open("a", encoding="utf-8", newline="\n") as stream:
        for entry in materialized:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def read_logs(job_id: str, *, after: int = 0, limit: int = 500) -> dict[str, Any]:
    """``after``보다 큰 seq의 log를 최대 ``limit``개 읽습니다."""

    path = job_directory(job_id) / _LOG_NAME
    lines: list[dict[str, Any]] = []
    highest = after
    if not path.is_file():
        return {"lines": [], "next": after, "complete": True}

    try:
        with path.open("r", encoding="utf-8") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                sequence = entry.get("seq")
                if not isinstance(sequence, int) or sequence <= after:
                    continue
                highest = max(highest, sequence)
                if len(lines) < limit:
                    lines.append(entry)
    except OSError as error:
        raise JobNotFoundError("학습 log를 읽을 수 없습니다.") from error

    next_cursor = lines[-1]["seq"] if lines else after
    return {
        "lines": lines,
        "next": next_cursor,
        "complete": next_cursor >= highest,
    }


def make_log_entry(sequence: int, stream_name: str, level: str, text: str, timestamp: str) -> dict[str, Any]:
    """저장·전송 직전에 마스킹과 길이 제한을 적용한 log 항목을 만듭니다."""

    return {
        "seq": sequence,
        "stream": stream_name,
        "level": level,
        "text": sanitize_line(text),
        "ts": timestamp,
    }


_QUEUE_NAME = "queue.json"


def _queue_path() -> Path:
    return web_state_dir() / _QUEUE_NAME


def save_queue(state: dict[str, Any]) -> None:
    """대기열을 저장합니다. 서버가 다시 떠도 목록이 남아야 합니다.

    "돌려 놓고 자러 가기"가 이 기능의 용도라, 자는 동안 서버가 한 번 죽었다고
    아침에 아무것도 남아 있지 않으면 곤란합니다.
    """

    try:
        _write_atomic(
            _queue_path(),
            json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
    except OSError:
        # 대기열을 못 남긴다고 지금 도는 학습을 실패시키지는 않습니다.
        pass


def load_queue() -> dict[str, Any]:
    """저장된 대기열을 읽습니다. 없거나 깨졌으면 빈 대기열입니다."""

    try:
        value = json.loads(_queue_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"entries": [], "paused": True}
    if not isinstance(value, dict):
        return {"entries": [], "paused": True}
    entries = value.get("entries")
    return {
        "entries": [item for item in entries if isinstance(item, dict)]
        if isinstance(entries, list)
        else [],
        # 서버가 다시 떴을 때 GPU 학습이 저절로 시작되면 곤란하므로 멈춘 채로 읽습니다.
        "paused": True,
    }
