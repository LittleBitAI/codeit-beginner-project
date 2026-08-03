#!/usr/bin/env python3
"""현재 작업 branch를 push하고 draft Pull Request를 생성합니다."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


BRANCH_PATTERN = re.compile(
    r"^pipeline/(data|train|evaluate|registry|web)/[a-z0-9][a-z0-9-]*$"
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
        raise RuntimeError(detail or f"명령 실행 실패: {' '.join(command)}")
    return result.stdout.strip()


def execute(*command: str) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"명령 실행 실패: {' '.join(command)}")


def repository_root() -> Path:
    return Path(capture("git", "rev-parse", "--show-toplevel"))


def install_alias() -> None:
    execute("git", "config", "--local", "alias.pr", ALIAS_VALUE)
    print("설치 완료: 이 저장소에서 'git pr'을 사용할 수 있습니다.")


def validate_tools() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("Git을 찾을 수 없습니다.")
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI를 찾을 수 없습니다. https://cli.github.com/")


def publish(dry_run: bool) -> None:
    validate_tools()
    root = repository_root()
    template = root / ".github" / "pull_request_template.md"
    if not template.is_file():
        raise RuntimeError("Pull Request template을 찾을 수 없습니다.")

    branch = capture("git", "branch", "--show-current")
    if not branch:
        raise RuntimeError("detached HEAD에서는 실행할 수 없습니다.")
    if branch == "main":
        raise RuntimeError("main에서는 Pull Request를 만들 수 없습니다.")
    if BRANCH_PATTERN.fullmatch(branch) is None:
        raise RuntimeError(f"branch 이름이 공통 규칙과 다릅니다: {branch}")

    capture("git", "remote", "get-url", "origin")
    status = capture("git", "status", "--porcelain=v1")

    if dry_run:
        ahead = capture("git", "rev-list", "--count", "origin/main..HEAD")
        print("DRY RUN: push하거나 Pull Request를 만들지 않습니다.")
        print(f"branch: {branch}")
        print(f"origin/main보다 앞선 commit: {ahead}")
        if status:
            print("주의: 작업 트리가 clean하지 않아 실제 'git pr'은 중단됩니다.")
        if ahead == "0":
            print("주의: 새 commit이 없어 실제 'git pr'은 중단됩니다.")
        print(f"예정: git push -u origin {branch}")
        print(f"예정: {branch} -> main draft Pull Request 생성 또는 기존 PR 갱신")
        return

    if status:
        raise RuntimeError("작업 트리가 clean하지 않습니다. 먼저 변경을 commit하세요.")

    execute("git", "fetch", "--quiet", "origin", "main")
    ahead = int(capture("git", "rev-list", "--count", "origin/main..HEAD"))
    if ahead == 0:
        raise RuntimeError("origin/main보다 앞선 commit이 없습니다.")

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
        print(f"기존 Pull Request 갱신 완료: {existing_url}")
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
    print(f"Draft Pull Request 생성 완료: {created_url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="작업 branch를 push하고 draft Pull Request를 생성합니다."
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="현재 저장소에 git pr alias를 설치합니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="외부 변경 없이 실행 계획만 확인합니다.",
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
        print(f"오류: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
