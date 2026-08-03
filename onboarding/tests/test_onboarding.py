import importlib
import sys

import pytest

from onboarding.scripts import verify_onboarding


pytestmark = pytest.mark.onboarding


def test_python_version_is_311():
    assert sys.version_info[:2] == (3, 11)


def test_required_packages_are_importable():
    for package in verify_onboarding.REQUIRED_PACKAGES:
        assert importlib.import_module(package)


def test_required_repository_files_exist():
    verify_onboarding.check_required_files()


def test_required_files_are_utf8_lf_without_bom():
    verify_onboarding.check_utf8_compatibility()
