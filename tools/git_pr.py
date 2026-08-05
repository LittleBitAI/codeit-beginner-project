#!/usr/bin/env python3
"""Push the current work branch and create a draft pull request."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


BRANCH_PATTERN = re.compile(
    r"^(?:"
    r"pipeline/(data|train|evaluate|registry|web|docs)/[a-z0-9][a-z0-9-]*"
    r"|onboarding/(?!.*--)[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?"
    r")$"
)
ALIAS_VALUE = "!python tools/git_pr.py"


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


def repository_root() -> Path:
    return Path(capture("git", "rev-parse", "--show-toplevel"))


def install_alias() -> None:
    execute("git", "config", "--local", "alias.pr", ALIAS_VALUE)
    print("Installed: 'git pr' is available in this repository.")


def validate_tools() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("Git was not found.")
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI was not found. https://cli.github.com/")


def publish(dry_run: bool) -> None:
    validate_tools()
    root = repository_root()
    template = root / ".github" / "pull_request_template.md"
    if not template.is_file():
        raise RuntimeError("Pull request template was not found.")

    branch = capture("git", "branch", "--show-current")
    if not branch:
        raise RuntimeError("Cannot run from a detached HEAD.")
    if branch == "main":
        raise RuntimeError("Cannot create a pull request from main.")
    if BRANCH_PATTERN.fullmatch(branch) is None:
        raise RuntimeError(f"Branch name does not follow the repository rule: {branch}")

    capture("git", "remote", "get-url", "origin")
    status = capture("git", "status", "--porcelain=v1")

    if dry_run:
        ahead = capture("git", "rev-list", "--count", "origin/main..HEAD")
        print("DRY RUN: no branch will be pushed and no pull request will be created.")
        print(f"branch: {branch}")
        print(f"commits ahead of origin/main: {ahead}")
        if status:
            print("warning: actual 'git pr' will stop because the worktree is not clean.")
        if ahead == "0":
            print("warning: actual 'git pr' will stop because there is no new commit.")
        print(f"planned: git push -u origin {branch}")
        print(f"planned: create or update draft PR from {branch} to main")
        return

    if status:
        raise RuntimeError("Worktree is not clean. Commit the changes first.")

    execute("git", "fetch", "--quiet", "origin", "main")
    ahead = int(capture("git", "rev-list", "--count", "origin/main..HEAD"))
    if ahead == 0:
        raise RuntimeError("There is no commit ahead of origin/main.")

    execute("git", "push", "-u", "origin", branch)
    existing_url = capture(
        "gh",
        "pr",
        "list",
        "--head",
        branch,
        "--base",
        "main",
        "--state",
        "open",
        "--json",
        "url",
        "--jq",
        ".[0].url",
    )
    if existing_url:
        print(f"Existing pull request updated: {existing_url}")
        return

    title = capture("git", "log", "-1", "--pretty=%s")
    created_url = capture(
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch,
        "--draft",
        "--title",
        title,
        "--body-file",
        str(template),
    )
    print(f"Draft pull request created: {created_url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a work branch and create a draft pull request."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the git pr alias in the current repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the plan without making external changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.install:
            install_alias()
        else:
            publish(args.dry_run)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
