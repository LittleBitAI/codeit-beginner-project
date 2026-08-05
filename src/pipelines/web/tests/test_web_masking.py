"""Credential과 개인 절대 경로가 화면·log로 새지 않는지 확인합니다.

여기 쓰인 값은 모두 형식만 흉내 낸 가짜입니다.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.pipelines.web.masking import mask_path, redact, sanitize_line, sanitize_text, truncate
from src.pipelines.web.paths import REPOSITORY_ROOT


@pytest.mark.parametrize(
    "key",
    (
        "secret",
        "aws_secret_access_key",
        "api_key",
        "apiKey",
        "TOKEN",
        "password",
        "credential",
        "session_token",
        "Authorization",
        "private_key",
    ),
)
def test_redacts_secret_like_keys(key):
    assert redact({key: "가짜-값"})[key] == "***"


def test_redacts_nested_structures():
    payload = {"a": [{"token": "가짜"}, {"safe": "ok"}], "b": {"c": {"password": "가짜"}}}

    result = redact(payload)

    assert result["a"][0]["token"] == "***"
    assert result["a"][1]["safe"] == "ok"
    assert result["b"]["c"]["password"] == "***"


def test_redact_does_not_mutate_input():
    payload = {"token": "가짜", "nested": {"list": [1, 2]}}
    before = deepcopy(payload)

    redact(payload)

    assert payload == before


@pytest.mark.parametrize("value", ("AKIAIOSFODNN7EXAMPLE", "ASIAIOSFODNN7EXAMPLE"))
def test_redacts_aws_access_key_values_in_free_text(value):
    masked = sanitize_text(f"실패: {value} 사용 중")

    assert value not in masked
    assert "***" in masked


def test_redacts_bearer_token():
    masked = sanitize_text("Authorization: Bearer abc.def.ghi")

    assert "abc.def.ghi" not in masked


def test_redacts_url_userinfo():
    masked = sanitize_text("s3://someone:hunter2@bucket/key.json")

    assert "hunter2" not in masked
    assert "***@bucket" in masked


def test_redacts_query_string_credentials():
    masked = sanitize_text("s3://bucket/record.json?credential=SENSITIVE_URI_VALUE")

    assert "SENSITIVE_URI_VALUE" not in masked


def test_masks_repository_absolute_path():
    text = str(REPOSITORY_ROOT / "artifacts" / "x.pt")

    masked = mask_path(text)

    assert str(REPOSITORY_ROOT) not in masked
    assert "<저장소>" in masked


def test_masks_home_directory_so_username_never_leaks():
    home = Path.home()
    text = f"열 수 없음: {home / 'private' / 'notes.txt'}"

    masked = mask_path(text)

    assert str(home) not in masked
    assert home.name not in masked


@pytest.mark.parametrize(
    "value",
    ("C:\\Users\\someone\\x.txt", "/home/someone/x.txt", "\\\\server\\share\\x.txt"),
)
def test_masks_other_absolute_paths(value):
    masked = mask_path(f"경로: {value}")

    assert "someone" not in masked
    assert "server" not in masked or "<경로 숨김>" in masked


def test_masking_does_not_alter_ordinary_text():
    text = "epoch 3/50 완료 · train 0.4312 · val 0.5109"

    assert sanitize_text(text) == text


def test_relative_artifact_uri_survives_masking():
    """artifact URI는 상대 경로라 그대로 보여야 합니다."""

    text = "artifacts/experiments/completed/web-1/best_checkpoint.pt"

    assert sanitize_text(text) == text


def test_truncate_limits_line_length():
    assert truncate("a" * 5000).endswith("…(잘림)")
    assert len(truncate("a" * 5000)) < 5000


def test_sanitize_line_applies_both_masking_and_truncation():
    home = Path.home()
    line = f"{home}/x " + "b" * 6000

    result = sanitize_line(line)

    assert str(home) not in result
    assert result.endswith("…(잘림)")
