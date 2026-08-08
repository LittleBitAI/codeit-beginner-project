"""Web test가 주변 환경 변수와 무관하게 도는지 확인합니다.

이 pipeline은 storage backend와 팀 기록 여부를 환경 변수로 고릅니다. 그래서 개발자
기계에 실제 값이 설정되어 있으면 test가 팀의 진짜 S3 registry를 읽거나 로그인을
요구하게 됩니다. CI에는 그 값이 없어 통과하고 개발자 기계에서만 깨집니다.

여기서 지키는 것은 `conftest.py`의 fixture가 그 값을 test 동안 걷어 낸다는 사실입니다.
"""

from __future__ import annotations

import os

from src.pipelines.web.datasets import storage_environment
from src.pipelines.web.team_sync import TeamSyncConfig


def test_storage_backend_does_not_follow_the_ambient_environment():
    """주변 환경에 bucket이 설정돼 있어도 test는 local backend를 씁니다.

    `registry_config()`가 이 값으로 backend를 고르므로, s3가 되면 실험 목록 test가
    팀의 실제 registry를 읽는다.
    """

    environment = storage_environment()

    assert environment["default_backend"] == "local"
    assert environment["bucket_configured"] is False
    assert environment["forced_backend"] is None


def test_team_sync_is_off_unless_a_test_turns_it_on():
    """주변 환경에 팀 설정이 있어도 test에서는 꺼진 상태로 시작합니다.

    켜져 있으면 학습 시작 경로가 access token을 요구해 실패합니다.
    """

    assert TeamSyncConfig.from_environment().enabled is False


def test_no_pill_environment_variable_leaks_into_tests():
    """`PILL_`로 시작하는 값은 test가 직접 설정한 것만 남습니다."""

    leaked = sorted(name for name in os.environ if name.startswith("PILL_"))

    assert leaked == []
