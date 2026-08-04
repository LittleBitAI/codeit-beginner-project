#!/usr/bin/env python3
"""Safely switch to main and fast-forward it from origin/main."""

import argparse
import shutil
import subprocess
import sys


ALIAS_VALUE = "!python tools/git_update_main.py"


def capture(*command: str) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(detail or f"Command failed: {' '.join(command)}")
    return result.stdout.strip()


def execute(*command: str) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def validate_tools() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("Git was not found.")


def install_alias() -> None:
    validate_tools()
    capture("git", "rev-parse", "--show-toplevel")
    execute("git", "config", "--local", "alias.update-main", ALIAS_VALUE)
    print("Installed: 'git update-main' is available in this repository.")


def update_main() -> None:
    """Update local main without stashing, resetting, or creating a merge commit."""
    validate_tools()
    capture("git", "rev-parse", "--show-toplevel")
    capture("git", "remote", "get-url", "origin")

    status = capture("git", "status", "--porcelain=v1")
    if status:
        raise RuntimeError(
            "Worktree is not clean. Commit or otherwise resolve the changes on "
            "the current work branch before updating main."
        )

    previous_branch = capture("git", "branch", "--show-current")
    if not previous_branch:
        raise RuntimeError("Cannot update main from a detached HEAD.")
    if previous_branch != "main":
        execute("git", "switch", "main")

    previous_commit = capture("git", "rev-parse", "--short", "HEAD")
    execute("git", "pull", "--ff-only", "origin", "main")
    current_commit = capture("git", "rev-parse", "--short", "HEAD")

    if previous_commit == current_commit:
        print(f"main is already up to date at {current_commit}.")
    else:
        print(f"Updated main: {previous_commit} -> {current_commit}")
    if previous_branch != "main":
        print(f"Previous work branch was left intact: {previous_branch}")
    print(capture("git", "status", "--short", "--branch"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely switch to main and fast-forward it from origin/main."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the git update-main alias in this repository.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.install:
            install_alias()
        else:
            update_main()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
