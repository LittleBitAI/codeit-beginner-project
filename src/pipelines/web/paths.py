"""Web pipeline이 쓰는 저장소 기준 경로와 경로 검증 helper."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .errors import WebPathError


# src/pipelines/web/paths.py -> web -> pipelines -> src -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# Web이 만드는 상태 파일은 모두 gitignore된 artifacts/ 아래에만 둡니다.
WEB_STATE_DIRNAME = "artifacts/web"
CONFIG_DIRNAME = f"{WEB_STATE_DIRNAME}/configs"
JOBS_DIRNAME = f"{WEB_STATE_DIRNAME}/jobs"

_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def repository_root() -> Path:
    """저장소 root를 호출 시점에 확인합니다.

    값으로 가져가면 import 시점에 고정돼 test가 root를 바꿔도 반영되지 않습니다.
    """

    return REPOSITORY_ROOT


def web_state_dir() -> Path:
    """Web이 만드는 상태 파일의 뿌리 경로입니다."""

    return REPOSITORY_ROOT / WEB_STATE_DIRNAME


def config_dir() -> Path:
    """Runtime config를 저장할 절대 경로를 반환합니다."""

    return REPOSITORY_ROOT / CONFIG_DIRNAME


def jobs_dir() -> Path:
    """Job 기록을 저장할 절대 경로를 반환합니다."""

    return REPOSITORY_ROOT / JOBS_DIRNAME


def _reject(message: str) -> None:
    raise WebPathError(message)


def normalize_relative_posix(value: object, *, label: str = "경로") -> str:
    """저장소 기준 상대 POSIX 경로로 정규화합니다.

    절대 경로, drive 문자, UNC, ``..``, NUL byte, Windows 예약 이름을 거부합니다.
    실제 file system 접근은 하지 않습니다. ``label``은 사람이 읽을 오류 메시지에만
    쓰이며, 기계용 field 이름은 부르는 쪽에서 따로 붙입니다.
    """

    if not isinstance(value, str) or not value.strip():
        _reject(f"{label}는 비어 있지 않은 문자열이어야 합니다.")

    text = value.strip()
    if "\x00" in text:
        _reject(f"{label}에 NUL 문자를 쓸 수 없습니다.")
    if text.startswith("\\\\") or text.startswith("//"):
        _reject(f"{label}에 UNC 경로를 쓸 수 없습니다.")

    unified = text.replace("\\", "/")
    if len(unified) >= 2 and unified[1] == ":":
        _reject(f"{label}에 drive 문자를 쓸 수 없습니다.")
    if unified.startswith("/"):
        _reject(f"{label}는 저장소 기준 상대 경로여야 합니다.")

    parts = [part for part in PurePosixPath(unified).parts if part not in ("", ".")]
    if not parts:
        _reject(f"{label}는 비어 있지 않은 문자열이어야 합니다.")
    for part in parts:
        if part == "..":
            _reject(f"{label}는 저장소 밖으로 나갈 수 없습니다.")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            _reject(f"{label}에 예약된 이름 '{part}'를 쓸 수 없습니다.")

    return PurePosixPath(*parts).as_posix()


def resolve_within_repo(value: object, *, label: str = "경로") -> Path:
    """저장소 안으로 확정되는 절대 경로를 만듭니다.

    ``Path.resolve()``가 symlink를 따라가므로 symlink를 통한 탈출도 막힙니다.
    """

    relative = normalize_relative_posix(value, label=label)
    resolved = (REPOSITORY_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        _reject(f"{label}는 저장소 밖으로 나갈 수 없습니다.")
    return resolved


def to_repo_relative_posix(path: Path) -> str:
    """저장소 안 절대 경로를 상대 POSIX 경로 문자열로 바꿉니다."""

    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
