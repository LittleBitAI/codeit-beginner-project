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

# Commit 하나가 끝나는 자리를 표시합니다. 본문에 줄바꿈이 들어가므로 줄 단위로는
# 나눌 수 없어서, commit message에 나올 수 없는 제어 문자를 구분자로 씁니다.
COMMIT_SEPARATOR = "\x1e"

SUMMARY_HEADING = "변경 요약"
REASON_HEADING = "변경 이유"
VERIFICATION_HEADING = "검증"


def parse_commit_log(raw: str) -> list[tuple[str, str]]:
    """`git log` 출력을 (제목, 본문) 목록으로 나눕니다. 순수 함수입니다."""

    commits: list[tuple[str, str]] = []
    for chunk in raw.split(COMMIT_SEPARATOR):
        text = chunk.strip("\n")
        if not text.strip():
            continue
        subject, _, body = text.partition("\n")
        commits.append((subject.strip(), body.strip()))
    return commits


def _split_sections(markdown: str) -> list[tuple[str, list[str]]]:
    """`## 제목` 기준으로 (제목, 본문 줄) 목록을 만듭니다. 머리말 제목은 빈 문자열입니다."""

    sections: list[tuple[str, list[str]]] = []
    heading = ""
    lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            sections.append((heading, lines))
            heading = line[3:].strip()
            lines = []
        else:
            lines.append(line)
    sections.append((heading, lines))
    return sections


# `Co-Authored-By: ...`처럼 끝에 붙는 git trailer입니다. ASCII 글자로 시작하는
# key만 봅니다. 한국어 본문 줄("참고: ...")은 걸리지 않습니다.
_TRAILER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z-]*:\s")


def _strip_trailers(body: str) -> str:
    """끝에 붙은 git trailer를 뺍니다. trailer는 변경 이유가 아닙니다."""

    lines = body.splitlines()
    while lines and (not lines[-1].strip() or _TRAILER_PATTERN.match(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


def _reason_lines(commits: list[tuple[str, str]]) -> list[str]:
    """Commit 본문을 변경 이유로 씁니다. 본문이 없으면 제목을 대신 씁니다."""

    blocks: list[str] = []
    for subject, body in commits:
        reason = _strip_trailers(body)
        blocks.append(reason if reason else f"- {subject}")
    return "\n\n".join(blocks).splitlines()


def build_body(
    template: str,
    commits: list[tuple[str, str]],
    checks: list[str],
) -> str:
    """Template의 안내 문구를 실제 내용으로 바꾼 Pull Request 본문을 만듭니다.

    `범위 확인` 항목은 사람이 확인하는 attestation이라 자동으로 체크하지
    않습니다. 검증 내용은 git에서 알아낼 수 없으므로 반드시 받아야 합니다.
    """

    if not commits:
        raise RuntimeError("origin/main보다 앞선 commit이 없어 본문을 만들 수 없습니다.")
    if not checks:
        raise RuntimeError(
            "검증 절을 채울 내용이 없습니다. 실행한 명령과 결과를 "
            "--check \"명령 → 결과\"로 전달하세요."
        )

    filled = {
        SUMMARY_HEADING: [f"- {subject}" for subject, _ in commits],
        REASON_HEADING: _reason_lines(commits),
        VERIFICATION_HEADING: [f"- {check}" for check in checks],
    }

    rendered: list[str] = []
    for heading, lines in _split_sections(template):
        if heading:
            rendered.append(f"## {heading}")
        body = filled.get(heading)
        if body is None:
            rendered.extend(lines)
        else:
            rendered.extend(["", *body, ""])
    return "\n".join(rendered).strip() + "\n"


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


def commit_log_args() -> list[str]:
    """`git log` 인자입니다.

    `%x1e`는 git이 **출력에서** 0x1E 바이트로 바꿔 주는 표기입니다. 구분자
    바이트를 인자 문자열에 직접 넣으면 안 됩니다. Windows `CreateProcess`가
    제어 문자가 들어간 인자를 거부해서 실행 자체가 실패합니다.
    """

    return [
        "git",
        "log",
        "origin/main..HEAD",
        "--reverse",
        "--pretty=format:%s%n%b%x1e",
    ]


def collect_commits() -> list[tuple[str, str]]:
    """origin/main 이후의 commit을 오래된 순서로 모읍니다."""

    return parse_commit_log(capture(*commit_log_args()))


def publish(dry_run: bool, checks: list[str], update_body: bool) -> None:
    validate_tools()
    root = repository_root()
    template_path = root / ".github" / "pull_request_template.md"
    if not template_path.is_file():
        raise RuntimeError("Pull request template was not found.")
    template = template_path.read_text(encoding="utf-8")

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
        try:
            body = build_body(template, collect_commits(), checks)
        except RuntimeError as error:
            print(f"warning: actual 'git pr' will stop: {error}")
        else:
            print("--- pull request body ---")
            print(body, end="")
        return

    if status:
        raise RuntimeError("Worktree is not clean. Commit the changes first.")

    execute("git", "fetch", "--quiet", "origin", "main")
    ahead = int(capture("git", "rev-list", "--count", "origin/main..HEAD"))
    if ahead == 0:
        raise RuntimeError("There is no commit ahead of origin/main.")

    # 본문을 push보다 먼저 만듭니다. 검증 내용이 없어서 중단될 때 branch만
    # 원격에 올라가 있는 상태를 남기지 않기 위해서입니다.
    body = build_body(template, collect_commits(), checks)

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
        if update_body:
            execute("gh", "pr", "edit", existing_url, "--body", body)
            print("Pull request body replaced.")
        else:
            print(
                "note: 본문은 그대로 두었습니다. 사람이 고친 내용을 덮어쓰지 "
                "않기 위해서입니다. 새 본문으로 바꾸려면 --update-body를 쓰세요."
            )
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
        "--body",
        body,
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
        help="Show the plan and the generated body without making external changes.",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "검증 절에 넣을 '실행한 명령 → 결과'. 여러 번 쓸 수 있습니다. "
            "git에서 알아낼 수 없는 내용이라 반드시 전달해야 합니다."
        ),
    )
    parser.add_argument(
        "--update-body",
        action="store_true",
        help="이미 열린 Pull Request의 본문을 새로 만든 내용으로 덮어씁니다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.install:
            install_alias()
        else:
            publish(args.dry_run, args.check, args.update_body)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
