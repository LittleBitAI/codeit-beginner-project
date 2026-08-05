#!/usr/bin/env python3
"""문서 규칙을 검사합니다.

검사하는 규칙은 다섯 가지입니다.

1. 유지하는 문서는 모두 5,000자 이내입니다.
2. 같은 directory의 `CLAUDE.md`와 `AGENTS.md`는 도구 이름을 뺀 내용이 같습니다.
3. `src/pipelines`의 각 pipeline은 두 지침서를 모두 가집니다. 하나만 있으면
   다른 도구로 그 directory에서 작업할 때 지침이 통째로 없습니다.
4. pipeline directory에는 `README.md`를 두지 않습니다. 낡기 때문입니다.
5. 도구 이름 바로 앞에 관사(`a`/`an`)를 쓰지 않습니다. 두 파일을 치환으로
   만들기 때문에 관사가 뒤 단어에 의존하면 한쪽이 반드시 문법에 어긋납니다.

사용법: `python scripts/check_docs.py [저장소 root]`
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


MAX_CHARACTERS = 5000
TOOL_DOCUMENTS = ("CLAUDE.md", "AGENTS.md")
PIPELINES_RELATIVE = Path("src") / "pipelines"

# 우리가 유지하지 않는 문서입니다. 외부에서 받은 자료와 내려받은 package,
# 도구가 만든 cache는 우리가 줄일 수 있는 대상이 아닙니다.
EXEMPT_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "node_modules",
        "design_handoff_pill_detect_platform",
    }
)

# 치환 도중 두 이름이 섞이지 않도록 잠시 넣어 두는 표시입니다.
_SENTINEL = "\x00"

# 관사 바로 뒤에 도구 이름이 오는 자리입니다. backtick은 있어도 되고 없어도
# 됩니다. `a twin AGENTS.md`처럼 사이에 다른 단어가 있으면 걸리지 않습니다.
_ARTICLE_PATTERN = re.compile(r"\b[Aa]n?\s+`?(?:CLAUDE|AGENTS)")


def swap_tool_name(text: str) -> str:
    """`CLAUDE`와 `AGENTS`를 서로 바꿉니다. 두 번 적용하면 원래대로 돌아옵니다."""

    return (
        text.replace("CLAUDE", _SENTINEL)
        .replace("AGENTS", "CLAUDE")
        .replace(_SENTINEL, "AGENTS")
    )


def find_article_errors(text: str) -> list[int]:
    """관사가 도구 이름 바로 앞에 오는 줄 번호를 1부터 세어 돌려줍니다."""

    return [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if _ARTICLE_PATTERN.search(line)
    ]


def _is_exempt(path: Path, root: Path) -> bool:
    return any(part in EXEMPT_PARTS for part in path.relative_to(root).parts)


def _git(root: Path, *arguments: str) -> str | None:
    """저장소 root에서 git 명령을 돌립니다. 실패하면 None을 돌려줍니다."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _git_documents(root: Path) -> list[Path] | None:
    """git이 관리하는 문서 목록입니다. 저장소 root가 아니면 None을 돌려줍니다.

    `artifacts/`처럼 gitignore된 생성물을 제외하기 위해 무시 목록을 직접
    유지하지 않고 git에게 묻습니다. 목록을 베껴 두면 `.gitignore`와 어긋납니다.
    아직 commit하지 않은 새 문서도 검사하도록 untracked 파일까지 포함합니다.
    """

    toplevel = _git(root, "rev-parse", "--show-toplevel")
    if toplevel is None:
        return None
    if Path(toplevel.strip()).resolve() != root.resolve():
        # root가 저장소 전체가 아니라 그 일부이면 git의 목록을 쓸 수 없습니다.
        return None

    listing = _git(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"
    )
    if listing is None:
        return None
    return [root / name for name in listing.split("\0") if name]


def _documents(root: Path) -> list[Path]:
    found = _git_documents(root)
    if found is None:
        found = list(root.rglob("*.md"))
    return sorted(
        path for path in found if path.is_file() and not _is_exempt(path, root)
    )


def _pipeline_directories(root: Path) -> list[Path]:
    base = root / PIPELINES_RELATIVE
    if not base.is_dir():
        return []
    return sorted(
        directory
        for directory in base.iterdir()
        if directory.is_dir() and not directory.name.startswith((".", "__"))
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def check_document(path: Path, root: Path) -> list[str]:
    """길이 규칙과 관사 규칙을 확인합니다."""

    name = _relative(path, root)
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []

    if len(text) > MAX_CHARACTERS:
        violations.append(
            f"{name}: {len(text)}자로 {MAX_CHARACTERS}자 제한을 넘었습니다."
        )

    if path.name in TOOL_DOCUMENTS:
        for line_number in find_article_errors(text):
            violations.append(
                f"{name}:{line_number}: 도구 이름 바로 앞에 관사가 있습니다. "
                "두 파일을 치환으로 만들기 때문에 한쪽이 문법에 어긋납니다."
            )
    return violations


def check_twins(directory: Path, root: Path, *, report_missing: bool = True) -> list[str]:
    """같은 directory의 두 지침서가 도구 이름만 빼고 같은지 확인합니다."""

    claude = directory / "CLAUDE.md"
    agents = directory / "AGENTS.md"
    if not claude.is_file() and not agents.is_file():
        return []

    if not agents.is_file():
        if not report_missing:
            return []
        return [f"{_relative(agents, root)}: CLAUDE.md만 있고 AGENTS.md가 없습니다."]
    if not claude.is_file():
        if not report_missing:
            return []
        return [f"{_relative(claude, root)}: AGENTS.md만 있고 CLAUDE.md가 없습니다."]

    expected = swap_tool_name(claude.read_text(encoding="utf-8"))
    if expected != agents.read_text(encoding="utf-8"):
        return [
            f"{_relative(agents, root)}: CLAUDE.md와 내용이 다릅니다. "
            "도구 이름을 뺀 나머지가 같아야 합니다."
        ]
    return []


def check_pipeline(directory: Path, root: Path) -> list[str]:
    """pipeline이 두 지침서를 갖고 README를 두지 않았는지 확인합니다."""

    violations: list[str] = []
    for name in TOOL_DOCUMENTS:
        if not (directory / name).is_file():
            violations.append(
                f"{_relative(directory / name, root)}: pipeline 지침서가 없습니다. "
                "각 pipeline은 두 파일을 모두 가져야 합니다."
            )

    readme = directory / "README.md"
    if readme.is_file():
        violations.append(
            f"{_relative(readme, root)}: pipeline에는 README를 두지 않습니다. "
            "필요한 내용은 지침서로 옮기고 삭제하세요."
        )
    return violations


def collect_violations(root: Path) -> list[str]:
    """저장소 전체를 확인해 위반 목록을 돌려줍니다. 비어 있으면 통과입니다."""

    root = Path(root)
    pipelines = _pipeline_directories(root)
    pipeline_set = set(pipelines)
    violations: list[str] = []

    for document in _documents(root):
        violations.extend(check_document(document, root))

    # pipeline은 지침서가 아예 없는 경우까지 잡아야 하므로 문서 목록이 아니라
    # directory 목록을 기준으로 확인합니다.
    for pipeline in pipelines:
        violations.extend(check_pipeline(pipeline, root))

    for directory in sorted({document.parent for document in _documents(root)}):
        violations.extend(
            check_twins(directory, root, report_missing=directory not in pipeline_set)
        )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="문서 규칙을 검사합니다.")
    parser.add_argument("root", nargs="?", default=".", help="저장소 root 경로")
    args = parser.parse_args(argv)

    violations = collect_violations(Path(args.root))
    if not violations:
        print("문서 규칙 검사를 통과했습니다.")
        return 0

    print(f"문서 규칙 위반 {len(violations)}건:")
    for violation in violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
