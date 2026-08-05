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

SUMMARY_HEADING = "변경 요약"
REASON_HEADING = "변경 이유"
VERIFICATION_HEADING = "검증"


def configure_utf8_console(
    stdout: object | None = None, stderr: object | None = None
) -> None:
    """Windows에서도 한국어 안내를 UTF-8 without BOM으로 출력합니다."""

    for stream in (stdout or sys.stdout, stderr or sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")


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


def _require_korean(items: list[str], heading: str) -> list[str]:
    """한국어로 작성한 비어 있지 않은 PR 설명만 받습니다."""

    cleaned = [item.strip() for item in items if item.strip()]
    if not cleaned:
        raise RuntimeError(f"{heading} 절을 채울 내용이 없습니다.")
    if any(re.search(r"[가-힣]", item) is None for item in cleaned):
        raise RuntimeError(f"{heading} 절은 한국어로 작성하세요.")
    return cleaned


def build_body(
    template: str,
    summaries: list[str],
    reasons: list[str],
    checks: list[str],
) -> str:
    """Template의 안내 문구를 실제 내용으로 바꾼 Pull Request 본문을 만듭니다.

    `범위 확인` 항목은 사람이 확인하는 attestation이라 자동으로 체크하지
    않습니다. 요약과 이유는 작성자가 diff를 확인한 뒤 직접 전달해야 하며,
    commit 제목으로 대신하지 않습니다.
    """

    summaries = _require_korean(summaries, SUMMARY_HEADING)
    reasons = _require_korean(reasons, REASON_HEADING)
    if not checks:
        raise RuntimeError(
            "검증 절을 채울 내용이 없습니다. 실행한 명령과 결과를 "
            "--check \"명령 → 결과\"로 전달하세요."
        )

    if {item.casefold() for item in summaries} == {item.casefold() for item in reasons}:
        raise RuntimeError("변경 요약과 변경 이유는 서로 다르게 작성하세요.")

    filled = {
        SUMMARY_HEADING: [f"- {summary}" for summary in summaries],
        REASON_HEADING: [f"- {reason}" for reason in reasons],
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


def capture_with_utf8_input(input_text: str, *command: str) -> str:
    """Text를 UTF-8 without BOM 표준 입력으로 보내고 출력을 돌려줍니다."""

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=input_text,
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


def publish(
    dry_run: bool,
    summaries: list[str],
    reasons: list[str],
    checks: list[str],
    update_body: bool,
) -> None:
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
            body = build_body(template, summaries, reasons, checks)
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
    body = build_body(template, summaries, reasons, checks)

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
            capture_with_utf8_input(
                body, "gh", "pr", "edit", existing_url, "--body-file", "-"
            )
            print("Pull request body replaced.")
        else:
            print(
                "note: 본문은 그대로 두었습니다. 사람이 고친 내용을 덮어쓰지 "
                "않기 위해서입니다. 새 본문으로 바꾸려면 --update-body를 쓰세요."
            )
        return

    title = capture("git", "log", "-1", "--pretty=%s")
    created_url = capture_with_utf8_input(
        body,
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
        "-",
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
        "--summary",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "변경 요약에 넣을 한국어 설명. diff를 확인해 사용자 동작이나 구조가 "
            "어떻게 바뀌었는지 적으세요. 여러 번 쓸 수 있습니다."
        ),
    )
    parser.add_argument(
        "--reason",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "변경 이유에 넣을 한국어 설명. 기존 문제와 이 변경이 필요한 이유를 "
            "요약과 다르게 적으세요. 여러 번 쓸 수 있습니다."
        ),
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
            publish(
                args.dry_run,
                args.summary,
                args.reason,
                args.check,
                args.update_body,
            )
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    configure_utf8_console()
    raise SystemExit(main())
