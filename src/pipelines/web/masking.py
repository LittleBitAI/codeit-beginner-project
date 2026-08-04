"""화면과 log에 나가는 값에서 credential과 개인 절대 경로를 가립니다.

Registry pipeline에도 비슷한 ``redact()``가 있지만, 다른 pipeline의 내부 module을
import하면 소유 경계를 넘게 되므로 web 안에서 따로 구현합니다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .paths import REPOSITORY_ROOT


__all__ = ["mask_path", "mask_text", "redact", "sanitize_text", "truncate"]


REDACTED = "***"
HIDDEN_PATH = "<경로 숨김>"
REPOSITORY_LABEL = "<저장소>"
HOME_LABEL = "<홈>"

MAX_LINE_LENGTH = 4000

_SECRET_KEY_HINTS = (
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "access_key",
    "secret_key",
    "api_key",
    "apikey",
    "session",
    "authorization",
    "private",
)

_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # AWS access key ID (장기 credential과 임시 credential 모두)
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
    # Authorization: Bearer <token>
    (re.compile(r"\bBearer\s+[\w.\-~+/]+=*", re.IGNORECASE), f"Bearer {REDACTED}"),
    # aws_secret_access_key = ... / secret_key: ...
    (
        re.compile(
            r"\b(" + "|".join(_SECRET_KEY_HINTS) + r")[\w.\-]*\s*[=:]\s*\"?[^\s\"'&,;]+\"?",
            re.IGNORECASE,
        ),
        lambda match: f"{match.group(1)}={REDACTED}",
    ),
    # scheme://user:password@host -> scheme://***@host
    (re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://)[^/\s@]+@"), lambda m: f"{m.group(1)}{REDACTED}@"),
    # ?credential=... / &token=...
    (
        re.compile(
            r"([?&](?:" + "|".join(_SECRET_KEY_HINTS) + r")[\w.\-]*=)[^&\s\"']+",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group(1)}{REDACTED}",
    ),
)

_ABSOLUTE_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    # UNC 경로
    re.compile(r"\\\\[^\s\"'<>|]+"),
    # Windows drive 경로
    re.compile(r"\b[A-Za-z]:[\\/][^\s\"'<>|]*"),
    # 흔한 POSIX 절대 경로 (URL의 path 부분과 겹치지 않도록 앞부분을 한정합니다)
    re.compile(r"(?<![\w:/])/(?:home|Users|root|var|tmp|opt|usr|mnt|media)/[^\s\"'<>|]*"),
)


def _is_secret_key(key: object) -> bool:
    text = str(key).lower()
    return any(hint in text for hint in _SECRET_KEY_HINTS)


def redact(value: Any) -> Any:
    """Key 이름이 credential처럼 보이는 값을 가린 사본을 만듭니다.

    입력을 변경하지 않고 항상 새 값을 반환합니다.
    """

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = REDACTED if _is_secret_key(key_text) else redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def mask_text(text: str) -> str:
    """Key와 무관하게 값 자체가 credential로 보이는 부분을 가립니다."""

    if not isinstance(text, str):
        return text
    masked = text
    for pattern, replacement in _VALUE_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def _replace_literal_path(text: str, target: Path, label: str) -> str:
    raw = str(target)
    variants = {raw, raw.replace("\\", "/"), raw.replace("/", "\\")}
    masked = text
    for variant in variants:
        if not variant:
            continue
        masked = re.sub(re.escape(variant), label, masked, flags=re.IGNORECASE)
    return masked


def mask_path(text: str) -> str:
    """개인 컴퓨터의 절대 경로가 화면이나 log에 노출되지 않게 합니다."""

    if not isinstance(text, str):
        return text

    masked = _replace_literal_path(text, REPOSITORY_ROOT, REPOSITORY_LABEL)
    try:
        home = Path.home()
    except (OSError, RuntimeError):  # 홈 경로를 못 구해도 마스킹은 계속합니다.
        home = None
    if home is not None:
        masked = _replace_literal_path(masked, home, HOME_LABEL)

    for pattern in _ABSOLUTE_PATH_PATTERNS:
        masked = pattern.sub(HIDDEN_PATH, masked)
    return masked


def sanitize_text(text: str) -> str:
    """Credential과 절대 경로를 모두 가린 문자열을 반환합니다."""

    if not isinstance(text, str):
        return text
    return mask_path(mask_text(text))


def truncate(text: str, limit: int = MAX_LINE_LENGTH) -> str:
    """한 줄이 지나치게 길어 memory를 잡아먹지 않도록 자릅니다."""

    if not isinstance(text, str) or len(text) <= limit:
        return text
    return text[:limit] + "…(잘림)"


def sanitize_line(text: str, limit: int = MAX_LINE_LENGTH) -> str:
    """Log 한 줄에 마스킹과 길이 제한을 함께 적용합니다."""

    return truncate(sanitize_text(text), limit)


def sanitize_sequence(values: Sequence[str]) -> list[str]:
    return [sanitize_line(value) for value in values]
