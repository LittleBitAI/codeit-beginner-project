from pathlib import Path

import pytest

from onboarding.scripts import verify_onboarding


pytestmark = pytest.mark.onboarding


def test_local_profiles_require_python_311():
    assert verify_onboarding.supported_python_versions("cpu") == ((3, 11),)
    assert verify_onboarding.supported_python_versions("cuda") == ((3, 11),)


def test_colab_profile_supports_current_and_local_python():
    assert verify_onboarding.supported_python_versions("colab") == (
        (3, 11),
        (3, 12),
    )


def test_required_packages_are_importable_and_pinned():
    verify_onboarding.check_required_packages()


def test_requirements_match_expected_package_versions():
    requirements_path = Path(verify_onboarding.REPOSITORY_ROOT, "requirements.txt")
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    for distribution, (_, version) in verify_onboarding.EXPECTED_PACKAGES.items():
        assert f"{distribution}=={version}" in requirements


def test_requirements_use_cuda_126_wheel_index():
    requirements_path = Path(verify_onboarding.REPOSITORY_ROOT, "requirements.txt")
    requirements = requirements_path.read_text(encoding="utf-8").splitlines()
    assert (
        "--extra-index-url https://download.pytorch.org/whl/cu126" in requirements
    )


def test_pip_dependencies_are_consistent():
    verify_onboarding.check_dependency_consistency()


def test_deep_learning_libraries_work_together_on_cpu():
    verify_onboarding.check_deep_learning_compatibility("cpu")


def test_nvidia_driver_version_parser():
    assert verify_onboarding.parse_driver_version("591.86") == (591, 86)


def test_invalid_nvidia_driver_version_is_rejected():
    with pytest.raises(verify_onboarding.VerificationError):
        verify_onboarding.parse_driver_version("unknown")


def test_required_repository_files_exist():
    verify_onboarding.check_required_files()


def test_required_files_are_utf8_lf_without_bom():
    verify_onboarding.check_utf8_compatibility()
