import importlib
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PACKAGES = ("pytest",)
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "pytest.ini",
    "onboarding/docs/onboarding.md",
    "onboarding/docs/onboarding-status.md",
    "onboarding/scripts/verify_onboarding.py",
    "onboarding/tests/test_onboarding.py",
)


class VerificationError(RuntimeError):
    """온보딩 검증 실패를 나타냅니다."""


def check_python_version() -> None:
    if sys.version_info[:2] != (3, 11):
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise VerificationError(f"Python 3.11이 필요합니다. 현재 버전: {version}")


def check_required_packages() -> None:
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise VerificationError(
            "import할 수 없는 package: " + ", ".join(sorted(missing))
        )


def check_required_files() -> None:
    missing = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (REPOSITORY_ROOT / relative_path).is_file()
    ]
    if missing:
        raise VerificationError("누락된 repository file: " + ", ".join(missing))


def check_utf8_compatibility() -> None:
    issues = []
    for relative_path in REQUIRED_FILES:
        path = REPOSITORY_ROOT / relative_path
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            issues.append(f"{relative_path}: UTF-8 BOM")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"{relative_path}: invalid UTF-8")
        if b"\r" in data:
            issues.append(f"{relative_path}: CR/CRLF line ending")

    if issues:
        raise VerificationError("text compatibility failure: " + "; ".join(issues))


def run_onboarding_tests() -> None:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "onboarding",
        "onboarding/tests/test_onboarding.py",
        "-q",
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode != 0:
        raise VerificationError("onboarding test가 실패했습니다.")


def main() -> int:
    checks = (
        ("Python 3.11", check_python_version),
        ("required package imports", check_required_packages),
        ("required repository files", check_required_files),
        ("UTF-8 compatibility", check_utf8_compatibility),
        ("onboarding tests", run_onboarding_tests),
    )

    for label, check in checks:
        try:
            check()
        except VerificationError as error:
            print(f"[FAIL] {label}: {error}")
            print("ONBOARDING VERIFICATION FAILED")
            return 1
        print(f"[PASS] {label}")

    print("ONBOARDING VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
