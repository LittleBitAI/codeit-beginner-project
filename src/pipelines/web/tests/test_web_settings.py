"""설정과 자동 평가.

여기서 지키는 것은 하나입니다: **사람이 고르기 전에는 GPU가 저절로 돌지 않는다.**
서버를 올렸다는 이유만으로 밤새 평가가 도는 일이 없어야 합니다.
"""

from __future__ import annotations

import pytest

from src.pipelines.web import settings as web_settings
from src.pipelines.web.errors import WebValidationError


def test_고르기_전에는_평가_방식이_비어_있다(client):
    body = client.get("/api/settings").json()

    assert body == {"evaluation_mode": None}
    assert web_settings.evaluation_mode() is None


def test_고른_값이_저장되고_그대로_읽힌다(client):
    saved = client.put("/api/settings", json={"evaluation_mode": "parallel"})

    assert saved.status_code == 200
    assert saved.json() == {"evaluation_mode": "parallel"}
    assert client.get("/api/settings").json() == {"evaluation_mode": "parallel"}
    assert web_settings.evaluation_mode() == "parallel"


def test_모르는_값은_거절한다(client):
    response = client.put("/api/settings", json={"evaluation_mode": "가끔"})

    assert response.status_code == 422


def test_설정_파일이_깨져도_고른_적_없음으로_읽는다(client):
    client.put("/api/settings", json={"evaluation_mode": "serial"})
    (web_settings.web_state_dir() / "settings.json").write_text("{", encoding="utf-8")

    # 깨진 파일 때문에 학습이 멈추거나, 반대로 저절로 돌아서도 안 됩니다.
    assert web_settings.evaluation_mode() is None


def test_직접_부른_저장은_잘못된_값을_막는다():
    with pytest.raises(WebValidationError):
        web_settings.write_settings({"evaluation_mode": None})
    with pytest.raises(WebValidationError):
        web_settings.write_settings("serial")


def test_고르지_않았으면_자동_평가_thread를_띄우지_않는다(client):
    from src.pipelines.web.jobs import get_manager

    manager = get_manager()
    manager._evaluation_pending.append("없는-job")
    manager.wake_evaluation()

    # 줄에 있어도 설정이 비어 있으면 아무것도 시작하지 않습니다.
    assert client.get("/api/settings").json()["evaluation_mode"] is None


def test_줄이_비어_있으면_thread를_아예_만들지_않는다(client):
    from src.pipelines.web.jobs import get_manager

    manager = get_manager()
    manager._evaluation_pending.clear()
    manager.wake_evaluation()

    assert manager._evaluation_thread is None
