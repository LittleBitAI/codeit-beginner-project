"""문서 규칙 검사 script의 테스트.

각 규칙이 위반을 실제로 잡는지, 그리고 정상 문서를 잘못 잡지 않는지
확인합니다.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_docs


def build_repo(root: Path, *, pipelines: tuple[str, ...] = ("data",)) -> Path:
    """규칙을 모두 지키는 최소 저장소를 만듭니다."""

    (root / "README.md").write_text("팀 문서\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("Read `CLAUDE.md` only.\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("Read `AGENTS.md` only.\n", encoding="utf-8")
    for name in pipelines:
        directory = root / "src" / "pipelines" / name
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
        (directory / "CLAUDE.md").write_text(
            f"# {name}\n\nRead the root `CLAUDE.md` first.\n", encoding="utf-8"
        )
        (directory / "AGENTS.md").write_text(
            f"# {name}\n\nRead the root `AGENTS.md` first.\n", encoding="utf-8"
        )
    return root


def test_clean_repository_has_no_violation(tmp_path):
    assert check_docs.collect_violations(build_repo(tmp_path)) == []


# --- 8,000자 제한 -----------------------------------------------------------


def test_document_over_the_limit_is_reported(tmp_path):
    root = build_repo(tmp_path)
    over_limit = check_docs.MAX_CHARACTERS + 1
    (root / "README.md").write_text("가" * over_limit, encoding="utf-8")

    violations = check_docs.collect_violations(root)

    assert any(
        "README.md" in item and str(over_limit) in item for item in violations
    )


def test_document_exactly_at_the_limit_passes(tmp_path):
    root = build_repo(tmp_path)
    (root / "README.md").write_text("가" * check_docs.MAX_CHARACTERS, encoding="utf-8")

    assert check_docs.collect_violations(root) == []


def test_exempt_directory_is_not_length_checked(tmp_path):
    """외부에서 받은 자료는 우리가 줄일 수 있는 문서가 아닙니다."""

    root = build_repo(tmp_path)
    handoff = root / "design_handoff_pill_detect_platform"
    handoff.mkdir()
    (handoff / "README.md").write_text("x" * 40000, encoding="utf-8")

    assert check_docs.collect_violations(root) == []


def test_gitignored_directory_is_skipped(tmp_path):
    """회귀: 생성물 directory를 걸어 들어가면 자기 test fixture를 위반으로 신고합니다.

    무시 목록을 직접 유지하면 .gitignore를 베끼는 셈이라 낡습니다. git에게
    묻습니다.
    """

    root = build_repo(tmp_path)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    generated = root / "artifacts" / "pytest-tmp"
    generated.mkdir(parents=True)
    (generated / "CLAUDE.md").write_text("x" * 6000, encoding="utf-8")

    assert check_docs.collect_violations(root) == []


def test_node_modules_is_not_checked(tmp_path):
    root = build_repo(tmp_path)
    vendored = root / "src" / "pipelines" / "data" / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "README.md").write_text("y" * 9000, encoding="utf-8")

    assert check_docs.collect_violations(root) == []


# --- 두 파일이 같은지 -------------------------------------------------------


def test_twin_mismatch_is_reported(tmp_path):
    root = build_repo(tmp_path)
    (root / "AGENTS.md").write_text("Read `AGENTS.md` only. 다른 내용\n", encoding="utf-8")

    violations = check_docs.collect_violations(root)

    assert any("AGENTS.md" in item for item in violations)


def test_twin_that_differs_only_by_tool_name_passes(tmp_path):
    root = build_repo(tmp_path)
    (root / "CLAUDE.md").write_text(
        "Read `CLAUDE.md`, never `AGENTS.md`.\n", encoding="utf-8"
    )
    (root / "AGENTS.md").write_text(
        "Read `AGENTS.md`, never `CLAUDE.md`.\n", encoding="utf-8"
    )

    assert check_docs.collect_violations(root) == []


def test_missing_twin_is_reported(tmp_path):
    root = build_repo(tmp_path)
    (root / "AGENTS.md").unlink()

    violations = check_docs.collect_violations(root)

    assert any("AGENTS.md" in item for item in violations)


# --- pipeline 지침서 필수 / README 금지 -------------------------------------


@pytest.mark.parametrize("missing", ("CLAUDE.md", "AGENTS.md"))
def test_pipeline_without_both_instruction_files_is_reported(tmp_path, missing):
    root = build_repo(tmp_path)
    (root / "src" / "pipelines" / "data" / missing).unlink()

    violations = check_docs.collect_violations(root)

    assert any("data" in item and missing in item for item in violations)


def test_pipeline_readme_is_reported(tmp_path):
    """pipeline README는 낡아서 버리기로 한 문서입니다."""

    root = build_repo(tmp_path)
    (root / "src" / "pipelines" / "data" / "README.md").write_text("낡음\n", encoding="utf-8")

    violations = check_docs.collect_violations(root)

    assert any("README.md" in item and "data" in item for item in violations)


# --- 관사 검사 --------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    (
        "Every `CLAUDE.md` has an `AGENTS.md` twin.",
        "Create a `CLAUDE.md` for each pipeline.",
        "This is an AGENTS.md file.",
    ),
)
def test_article_before_the_tool_name_is_reported(tmp_path, sentence):
    """두 파일을 치환으로 만들기 때문에 관사가 도구 이름에 붙으면 한쪽이 틀립니다."""

    root = build_repo(tmp_path)
    (root / "CLAUDE.md").write_text(sentence + "\n", encoding="utf-8")
    (root / "AGENTS.md").write_text(
        check_docs.swap_tool_name(sentence) + "\n", encoding="utf-8"
    )

    violations = check_docs.collect_violations(root)

    assert any("관사" in item for item in violations)


@pytest.mark.parametrize(
    "sentence",
    (
        "Every `CLAUDE.md` has a twin `AGENTS.md`.",
        "Read `CLAUDE.md` only.",
        "A pipeline file never restates a root rule.",
    ),
)
def test_sentences_without_an_adjacent_article_pass(tmp_path, sentence):
    root = build_repo(tmp_path)
    (root / "CLAUDE.md").write_text(sentence + "\n", encoding="utf-8")
    (root / "AGENTS.md").write_text(
        check_docs.swap_tool_name(sentence) + "\n", encoding="utf-8"
    )

    assert check_docs.collect_violations(root) == []


def test_swap_tool_name_is_reversible():
    text = "Read `CLAUDE.md`, never `AGENTS.md`."

    assert check_docs.swap_tool_name(check_docs.swap_tool_name(text)) == text


# --- 종료 코드 --------------------------------------------------------------


def test_exit_code_is_zero_when_clean(tmp_path, capsys):
    assert check_docs.main([str(build_repo(tmp_path))]) == 0


def test_exit_code_is_one_and_every_violation_is_printed(tmp_path, capsys):
    root = build_repo(tmp_path)
    (root / "README.md").write_text(
        "가" * (check_docs.MAX_CHARACTERS + 1), encoding="utf-8"
    )
    (root / "src" / "pipelines" / "data" / "AGENTS.md").unlink()

    exit_code = check_docs.main([str(root)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "README.md" in output
    assert "AGENTS.md" in output
