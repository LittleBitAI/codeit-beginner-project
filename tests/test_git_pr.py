import pytest

from tools import git_pr


@pytest.mark.parametrize(
    "branch",
    (
        "pipeline/data/prepare-sample",
        "pipeline/web/onboarding-workflow",
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
