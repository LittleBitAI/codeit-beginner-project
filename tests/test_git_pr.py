from subprocess import CompletedProcess

import pytest

from tools import git_pr


@pytest.mark.parametrize(
    "branch",
    (
        "pipeline/data/prepare-sample",
        "pipeline/web/onboarding-workflow",
        "pipeline/docs/instruction-role-split",
        "onboarding/octocat",
        "onboarding/LittleBitAI",
        f"onboarding/{'a' * 39}",
    ),
)
def test_allowed_pull_request_branch_names(branch):
    assert git_pr.BRANCH_PATTERN.fullmatch(branch)


@pytest.mark.parametrize(
    "branch",
    (
        "main",
        "pipeline/design/new-screen",
        "pipeline/docs/",
        "onboarding/",
        "onboarding/-octocat",
        "onboarding/octocat-",
        "onboarding/octo--cat",
        "onboarding/octocat/extra",
        f"onboarding/{'a' * 40}",
    ),
)
def test_rejected_pull_request_branch_names(branch):
    assert git_pr.BRANCH_PATTERN.fullmatch(branch) is None


# --- Pull Request 본문 자동 채우기 ------------------------------------------

TEMPLATE = """## 변경 요약

- 무엇을 변경했는지 간단히 적어 주세요.

## 변경 이유

- 이 변경이 필요한 이유를 적어 주세요.

## 검증

- 실행한 명령과 결과를 적어 주세요.

## 범위 확인

- [ ] 하나의 목적에 집중한 변경입니다.
- [ ] 관련 test와 check를 실행했습니다.
"""

SUMMARIES = ["서로 겹치던 문서 지침을 독자별 역할에 맞게 분리했습니다."]
REASONS = ["한 도구의 지침을 고치면 다른 도구의 지침까지 바뀌는 문제를 막기 위해서입니다."]
CHECKS = ["`python -m pytest -q` → 871 passed"]


def test_body_replaces_every_placeholder():
    body = git_pr.build_body(TEMPLATE, SUMMARIES, REASONS, CHECKS)

    assert "적어 주세요" not in body


def test_body_fills_explicit_summary_and_reason():
    body = git_pr.build_body(TEMPLATE, SUMMARIES, REASONS, CHECKS)

    assert f"- {SUMMARIES[0]}" in body
    assert f"- {REASONS[0]}" in body


def test_body_fills_verification_from_supplied_checks():
    body = git_pr.build_body(TEMPLATE, SUMMARIES, REASONS, CHECKS)

    assert "- `python -m pytest -q` → 871 passed" in body


def test_body_keeps_scope_checkboxes_unchecked():
    """범위 확인은 사람이 확인하는 항목이므로 자동으로 체크하지 않습니다."""

    body = git_pr.build_body(TEMPLATE, SUMMARIES, REASONS, CHECKS)

    assert "- [ ] 하나의 목적에 집중한 변경입니다." in body
    assert "- [x]" not in body


def test_body_keeps_section_order_and_headings():
    body = git_pr.build_body(TEMPLATE, SUMMARIES, REASONS, CHECKS)

    headings = [line for line in body.splitlines() if line.startswith("## ")]
    assert headings == ["## 변경 요약", "## 변경 이유", "## 검증", "## 범위 확인"]


@pytest.mark.parametrize(
    ("summaries", "reasons", "checks", "message"),
    (
        ([], REASONS, CHECKS, "변경 요약"),
        (SUMMARIES, [], CHECKS, "변경 이유"),
        (SUMMARIES, REASONS, [], "검증"),
    ),
)
def test_body_refuses_missing_explanation(summaries, reasons, checks, message):
    """실제 diff를 설명하지 않으면 PR을 만들지 않습니다."""

    with pytest.raises(RuntimeError, match=message):
        git_pr.build_body(TEMPLATE, summaries, reasons, checks)


def test_multiple_explanations_each_appear():
    summaries = ["설정 API에 인증 header를 추가했습니다.", "팀 학습 현황 화면을 추가했습니다."]
    reasons = ["학습 실행자를 확인하고 팀별 접근을 제한하기 위해서입니다."]

    body = git_pr.build_body(TEMPLATE, summaries, reasons, CHECKS)

    assert f"- {summaries[0]}" in body
    assert f"- {summaries[1]}" in body


def test_body_refuses_same_summary_and_reason():
    """제목을 요약과 이유에 반복한 #40 형태를 막습니다."""

    duplicated = ["web 팀 학습 실시간 동기화 추가"]

    with pytest.raises(RuntimeError, match="서로 다르게"):
        git_pr.build_body(TEMPLATE, duplicated, duplicated, CHECKS)


def test_body_requires_korean_explanation():
    with pytest.raises(RuntimeError, match="한국어"):
        git_pr.build_body(TEMPLATE, ["Add team sync"], REASONS, CHECKS)


def test_github_body_is_sent_as_utf8_without_bom(monkeypatch):
    captured = {}

    def fake_run(*command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return CompletedProcess(command, 0, stdout="https://example.test/pr/1\n", stderr="")

    monkeypatch.setattr(git_pr.subprocess, "run", fake_run)

    result = git_pr.capture_with_utf8_input(
        "한국어 PR 본문\n", "gh", "pr", "create", "--body-file", "-"
    )

    assert result == "https://example.test/pr/1"
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["input"].encode("utf-8").startswith(b"\xef\xbb\xbf") is False
    assert captured["command"][0][-2:] == ("--body-file", "-")


def test_console_output_is_configured_as_utf8_without_bom():
    class Stream:
        def __init__(self):
            self.options = None

        def reconfigure(self, **options):
            self.options = options

    stdout = Stream()
    stderr = Stream()

    git_pr.configure_utf8_console(stdout, stderr)

    assert stdout.options == {"encoding": "utf-8", "errors": "strict"}
    assert stderr.options == {"encoding": "utf-8", "errors": "strict"}
