from unittest.mock import call, patch

import pytest

from tools import git_update_main


def test_update_main_switches_from_work_branch_and_fast_forwards(capsys):
    with (
        patch.object(git_update_main, "validate_tools"),
        patch.object(
            git_update_main,
            "capture",
            side_effect=[
                "/repository",
                "https://example.com/repository.git",
                "",
                "pipeline/data/example",
                "abc1234",
                "def5678",
                "## main...origin/main",
            ],
        ),
        patch.object(git_update_main, "execute") as execute,
    ):
        git_update_main.update_main()

    assert execute.call_args_list == [
        call("git", "switch", "main"),
        call("git", "pull", "--ff-only", "origin", "main"),
    ]
    output = capsys.readouterr().out
    assert "Updated main: abc1234 -> def5678" in output
    assert "Previous work branch was left intact: pipeline/data/example" in output


def test_update_main_stays_on_main_when_already_current(capsys):
    with (
        patch.object(git_update_main, "validate_tools"),
        patch.object(
            git_update_main,
            "capture",
            side_effect=[
                "/repository",
                "https://example.com/repository.git",
                "",
                "main",
                "abc1234",
                "abc1234",
                "## main...origin/main",
            ],
        ),
        patch.object(git_update_main, "execute") as execute,
    ):
        git_update_main.update_main()

    execute.assert_called_once_with("git", "pull", "--ff-only", "origin", "main")
    assert "main is already up to date at abc1234" in capsys.readouterr().out


def test_update_main_refuses_dirty_worktree():
    with (
        patch.object(git_update_main, "validate_tools"),
        patch.object(
            git_update_main,
            "capture",
            side_effect=[
                "/repository",
                "https://example.com/repository.git",
                " M README.md",
            ],
        ),
        patch.object(git_update_main, "execute") as execute,
    ):
        with pytest.raises(RuntimeError, match="Worktree is not clean"):
            git_update_main.update_main()

    execute.assert_not_called()


def test_install_alias_uses_repository_local_git_config():
    with (
        patch.object(git_update_main, "validate_tools"),
        patch.object(git_update_main, "capture", return_value="/repository"),
        patch.object(git_update_main, "execute") as execute,
    ):
        git_update_main.install_alias()

    execute.assert_called_once_with(
        "git",
        "config",
        "--local",
        "alias.update-main",
        "!python tools/git_update_main.py",
    )
