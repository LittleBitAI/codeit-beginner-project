"""저장소 밖으로 나가는 경로를 막는지 확인합니다."""

from __future__ import annotations

import pytest

from src.pipelines.web.errors import WebPathError
from src.pipelines.web.paths import normalize_relative_posix, resolve_within_repo


def test_accepts_relative_posix_path():
    assert normalize_relative_posix("artifacts/data/x.json") == "artifacts/data/x.json"


def test_normalizes_backslashes_and_dot_segments():
    assert normalize_relative_posix("artifacts\\.\\data\\x.json") == "artifacts/data/x.json"


@pytest.mark.parametrize(
    "value",
    (
        "../outside",
        "artifacts/../../x",
        "/etc/passwd",
        "C:/Windows/system32",
        "c:\\Windows",
        "\\\\server\\share\\x",
        "//server/share/x",
        "",
        "   ",
        None,
        123,
    ),
)
def test_rejects_paths_that_leave_the_repository(value):
    with pytest.raises(WebPathError):
        normalize_relative_posix(value)


def test_rejects_null_byte():
    with pytest.raises(WebPathError):
        normalize_relative_posix("artifacts/x\x00.json")


@pytest.mark.parametrize("value", ("NUL", "artifacts/CON", "artifacts/COM1.txt", "AUX/x"))
def test_rejects_windows_reserved_names(value):
    with pytest.raises(WebPathError):
        normalize_relative_posix(value)


def test_resolve_within_repo_returns_absolute_path_inside_repo(isolated_repo):
    resolved = resolve_within_repo("artifacts/data/x.json")

    assert resolved == isolated_repo / "artifacts" / "data" / "x.json"
    assert resolved.is_absolute()


def test_resolve_within_repo_rejects_escape(isolated_repo):
    with pytest.raises(WebPathError):
        resolve_within_repo("../escape")
