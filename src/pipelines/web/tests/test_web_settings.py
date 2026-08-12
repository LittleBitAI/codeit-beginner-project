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
    """할 일이 없는 thread를 띄워 두지 않습니다.

    깨어 있는 thread가 남으면 서버가 사는 내내 아무 일도 안 하면서 자리를
    차지하고, 나중에 설정을 저장할 때 어차피 다시 띄웁니다.
    """

    from src.pipelines.web.jobs import get_manager

    manager = get_manager()
    manager._evaluation_pending.append("없는-job")
    manager.wake_evaluation()

    assert manager._evaluation_thread is None


def test_줄이_비어_있으면_thread를_아예_만들지_않는다(client):
    from src.pipelines.web.jobs import get_manager

    manager = get_manager()
    manager._evaluation_pending.clear()
    manager.wake_evaluation()

    assert manager._evaluation_thread is None


def test_직렬이면_평가가_도는_동안_학습을_시작하지_않는다(client, valid_payload):
    """직렬을 고른 이유가 이것입니다 — 8GB에서 겹치면 둘 다 잃습니다.

    평가를 시작할 때만 학습을 확인하고 반대쪽을 비워 두면, 평가가 도는 사이에
    대기열의 다음 학습이나 사람이 누른 시작이 그대로 들어와 약속이 깨집니다.
    """

    from src.pipelines.web import evaluation
    from src.pipelines.web.errors import JobConflictError
    from src.pipelines.web.jobs import get_manager

    client.put("/api/settings", json={"evaluation_mode": "serial"})
    created = client.post("/api/train/configs", json=valid_payload).json()
    # 평가가 도는 중이라고 표시합니다.
    evaluation.get_evaluation_runner()._state = {"status": "running", "job_id": "다른-학습"}

    with pytest.raises(JobConflictError):
        get_manager().start(created["config_id"])


def test_확인과_시작이_한_문_안에서_일어난다(client, valid_payload, monkeypatch, fake_process_factory):
    """확인만 하고 문을 놓으면 두 thread가 서로를 못 보고 둘 다 출발합니다.

    학습이 "평가 없음"을 확인한 뒤 자리를 잡기 전에 평가가 끼어들 수 있으면,
    직렬로 두었는데도 둘이 겹쳐 돌아 8GB 카드에서 둘 다 잃습니다. 그래서 확인부터
    자리 잡기까지가 같은 잠금 안에 있어야 합니다.
    """

    import threading

    from src.pipelines.web import evaluation
    from src.pipelines.web.jobs import runner
    from src.pipelines.web.jobs import get_manager

    client.put("/api/settings", json={"evaluation_mode": "serial"})
    created = client.post("/api/train/configs", json=valid_payload).json()
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory())
    manager = get_manager()

    # 학습이 문을 쥐고 있는 동안 평가가 자리를 잡으려 하면 기다려야 합니다.
    grabbed = threading.Event()
    entered = threading.Event()

    def hold() -> None:
        with manager._gpu_gate:
            grabbed.set()
            entered.wait(timeout=2)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert grabbed.wait(timeout=2)

    # 문이 잡혀 있으니 평가 자리 잡기는 들어가지 못합니다.
    assert manager._gpu_gate.acquire(blocking=False) is False

    entered.set()
    holder.join(timeout=2)

    # 문이 풀리면 평소대로 시작합니다.
    evaluation.get_evaluation_runner()._state = {"status": "idle", "job_id": None}
    assert manager.start(created["config_id"]).job_id


def test_병렬이면_평가가_돌아도_학습을_막지_않는다(client, valid_payload, monkeypatch, fake_process_factory):
    from src.pipelines.web import evaluation
    from src.pipelines.web.jobs import runner
    from src.pipelines.web.jobs import get_manager

    client.put("/api/settings", json={"evaluation_mode": "parallel"})
    created = client.post("/api/train/configs", json=valid_payload).json()
    evaluation.get_evaluation_runner()._state = {"status": "running", "job_id": "다른-학습"}
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory())

    record = get_manager().start(created["config_id"])

    assert record.job_id


def test_시작하지_못한_학습은_줄_맨_앞으로_되돌린다(client):
    """사람이 화면에서 평가를 먼저 눌렀을 때의 경로입니다.

    되돌리지 않으면 꺼낸 순간 사라져서 그 학습은 영영 자동 평가되지 않습니다.
    """

    from src.pipelines.web.jobs import get_manager

    manager = get_manager()
    manager._evaluation_pending[:] = ["b"]
    manager._requeue_evaluation("a")

    assert manager._evaluation_pending == ["a", "b"]
