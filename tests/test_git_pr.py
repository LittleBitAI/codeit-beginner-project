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

COMMITS = [("문서 역할 분리", "지침서가 서로 겹쳐서 정리했습니다.\n두 번째 줄입니다.")]
CHECKS = ["`python -m pytest -q` → 871 passed"]


def test_body_replaces_every_placeholder():
    """template의 안내 문구가 본문에 하나도 남지 않아야 합니다."""

    body = git_pr.build_body(TEMPLATE, COMMITS, CHECKS)

    assert "적어 주세요" not in body


def test_body_fills_summary_from_commit_subjects():
    body = git_pr.build_body(TEMPLATE, COMMITS, CHECKS)

    assert "- 문서 역할 분리" in body


def test_body_fills_reason_from_commit_body():
    body = git_pr.build_body(TEMPLATE, COMMITS, CHECKS)

    assert "지침서가 서로 겹쳐서 정리했습니다." in body
    assert "두 번째 줄입니다." in body


def test_body_fills_verification_from_supplied_checks():
    body = git_pr.build_body(TEMPLATE, COMMITS, CHECKS)

    assert "- `python -m pytest -q` → 871 passed" in body


def test_body_keeps_scope_checkboxes_unchecked():
    """범위 확인은 사람이 확인하는 항목이므로 자동으로 체크하지 않습니다."""

    body = git_pr.build_body(TEMPLATE, COMMITS, CHECKS)

    assert "- [ ] 하나의 목적에 집중한 변경입니다." in body
    assert "- [x]" not in body


def test_body_keeps_section_order_and_headings():
    body = git_pr.build_body(TEMPLATE, COMMITS, CHECKS)

    headings = [line for line in body.splitlines() if line.startswith("## ")]
    assert headings == ["## 변경 요약", "## 변경 이유", "## 검증", "## 범위 확인"]


def test_body_without_checks_is_refused():
    """검증 내용을 못 받으면 빈 절을 올리지 않고 중단합니다."""

    with pytest.raises(RuntimeError, match="검증"):
        git_pr.build_body(TEMPLATE, COMMITS, [])


def test_commit_with_no_body_still_explains_the_reason():
    body = git_pr.build_body(TEMPLATE, [("제목만 있는 commit", "")], CHECKS)

    assert "적어 주세요" not in body
    assert "제목만 있는 commit" in body


def test_multiple_commits_each_appear_in_the_summary():
    commits = [("첫 번째", "이유 하나"), ("두 번째", "이유 둘")]

    body = git_pr.build_body(TEMPLATE, commits, CHECKS)

    assert "- 첫 번째" in body
    assert "- 두 번째" in body


def test_reason_drops_git_trailers():
    """Co-Authored-By 같은 trailer는 변경 이유가 아니므로 본문에서 뺍니다."""

    commits = [("제목", "진짜 이유입니다.\n\nCo-Authored-By: 누군가 <a@b.c>")]

    body = git_pr.build_body(TEMPLATE, commits, CHECKS)

    assert "진짜 이유입니다." in body
    assert "Co-Authored-By" not in body


def test_reason_keeps_korean_lines_ending_with_colon():
    """trailer 제거가 한국어 본문 줄까지 지우면 안 됩니다."""

    commits = [("제목", "이유 첫 줄\n참고: 남아 있어야 합니다")]

    body = git_pr.build_body(TEMPLATE, commits, CHECKS)

    assert "참고: 남아 있어야 합니다" in body


def test_commit_log_args_carry_no_control_character():
    """회귀: 구분자 바이트를 인자에 직접 넣으면 Windows CreateProcess가 거부합니다.

    git이 출력에서 바꿔 주는 `%x1e` 표기를 인자로 보내야 합니다.
    """

    for argument in git_pr.commit_log_args():
        assert git_pr.COMMIT_SEPARATOR not in argument
        assert "\x00" not in argument


def test_parse_commit_log_splits_subject_and_body():
    raw = f"제목 A\n본문 A 첫 줄\n본문 A 둘째 줄\n{git_pr.COMMIT_SEPARATOR}제목 B\n\n{git_pr.COMMIT_SEPARATOR}"

    assert git_pr.parse_commit_log(raw) == [
        ("제목 A", "본문 A 첫 줄\n본문 A 둘째 줄"),
        ("제목 B", ""),
    ]
