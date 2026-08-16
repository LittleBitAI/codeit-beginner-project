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

    assert body == {"evaluation_mode": None, "epoch_metrics": None}
    assert web_settings.evaluation_mode() is None


def test_고른_값이_저장되고_그대로_읽힌다(client):
    saved = client.put("/api/settings", json={"evaluation_mode": "parallel"})

    assert saved.status_code == 200
    assert saved.json() == {"evaluation_mode": "parallel", "epoch_metrics": None}
    assert client.get("/api/settings").json()["evaluation_mode"] == "parallel"
    assert web_settings.evaluation_mode() == "parallel"


# --- epoch 훑기가 무엇으로 순위를 매길지 ------------------------------------


def test_고르기_전에는_훑기_지표가_비어_있다(client):
    """무엇이 Kaggle 점수를 예측하는지 모르는 것이 이 기능을 만든 이유입니다.

    아무도 고르지 않은 기준으로 조용히 순위를 매기면 그 질문 자체가 사라집니다.
    """

    assert web_settings.epoch_metrics() is None


def test_고른_지표가_순서대로_저장된다(client):
    saved = client.put(
        "/api/settings",
        json={"evaluation_mode": "serial", "epoch_metrics": ["mAP50", "mAP", "recall50"]},
    )

    assert saved.status_code == 200
    # 순서가 곧 가중치입니다. 정렬해서 저장하면 1순위가 사라집니다.
    assert web_settings.epoch_metrics() == ["mAP50", "mAP", "recall50"]


@pytest.mark.parametrize(
    "metrics",
    [
        ["mAP", "mAP50"],
        ["mAP", "mAP50", "mAP75", "recall50"],
        ["mAP", "mAP", "mAP50"],
        ["mAP", "mAP50", "없는지표"],
    ],
)
def test_지표를_셋이_아니게_고르면_거절한다(client, metrics):
    response = client.put(
        "/api/settings", json={"evaluation_mode": "serial", "epoch_metrics": metrics}
    )

    assert response.status_code == 400
    assert web_settings.epoch_metrics() is None


def test_평가_방식만_고쳐도_고른_지표는_남는다(client):
    """평가 시점만 고치러 온 저장이 훑기 설정을 지우면 안 됩니다."""

    client.put(
        "/api/settings",
        json={"evaluation_mode": "serial", "epoch_metrics": ["mAP", "mAP50", "recall50"]},
    )

    client.put("/api/settings", json={"evaluation_mode": "parallel"})

    assert web_settings.epoch_metrics() == ["mAP", "mAP50", "recall50"]


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


def test_학습과_평가가_동시에_출발해도_직렬에서는_하나만_이긴다(
    client, valid_payload, monkeypatch, fake_process_factory
):
    """두 시작 경로를 실제로 맞붙여 봅니다.

    잠금 하나를 non-blocking으로 못 잡는지만 보는 테스트는, 어느 한쪽에서 문을
    빼 버려도 그대로 통과합니다. production의 `start()`와 `_start_one_evaluation()`을
    동시에 출발시켜 **둘 중 하나만** 자리를 잡는지를 봐야 회귀를 잡습니다.
    """

    import threading

    from src.pipelines.web import evaluation
    from src.pipelines.web.errors import JobConflictError
    from src.pipelines.web.jobs import get_manager, runner
    from src.pipelines.web.jobs import manager as manager_module
    from src.pipelines.web.jobs.model import JobRecord

    client.put("/api/settings", json={"evaluation_mode": "serial"})
    created = client.post("/api/train/configs", json=valid_payload).json()
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory())

    manager = get_manager()
    evaluation_runner = evaluation.get_evaluation_runner()

    # 평가 대상 record는 **직접** 만듭니다. manager.start()로 만들면 background
    # thread가 남아, 늦게 도착한 _finalize()가 이 record를 failed로 되돌립니다.
    # 그러면 평가가 대상을 못 찾아 시작하지 않고, 자리 다툼을 보려던 테스트가
    # 아무것도 검사하지 못한 채 통과합니다.
    done = JobRecord(
        job_id="평가-대상",
        config_id=created["config_id"],
        run_id="이미-끝난-학습",
        status="succeeded",
    )
    manager._records[done.job_id] = done
    manager._evaluation_pending.append(done.job_id)

    started_evaluation = threading.Event()

    def fake_start(record, options):  # noqa: ANN001, ARG001
        with evaluation_runner._lock:
            evaluation_runner._state = {"status": "running", "job_id": record.job_id}
        started_evaluation.set()
        return evaluation_runner.status()

    monkeypatch.setattr(evaluation_runner, "start", fake_start)

    # 학습이 "평가 없음"을 확인한 **직후** 멈춰 세웁니다. 여기가 위험한 창입니다.
    # 이 사이에 평가가 자리를 잡아 버리면 둘이 겹칩니다.
    checked = threading.Event()
    resume = threading.Event()
    real_read = manager_module.read_runtime_config

    def blocking_read(config_id: str):  # noqa: ANN202
        checked.set()
        resume.wait(timeout=5)
        return real_read(config_id)

    monkeypatch.setattr(manager_module, "read_runtime_config", blocking_read)

    outcome: dict[str, object] = {}

    def begin_training() -> None:
        try:
            outcome["train"] = manager.start(created["config_id"]).job_id
        except JobConflictError as error:
            outcome["train"] = error

    trainer = threading.Thread(target=begin_training, daemon=True)
    trainer.start()
    assert checked.wait(timeout=5), "학습이 평가 확인 지점에 닿지 못했습니다"

    # 창이 열린 그 순간 평가가 끼어들려 합니다. 문이 있으면 학습이 자리를 잡을
    # 때까지 기다리고, 없으면 그대로 통과해 둘이 겹칩니다.
    evaluator = threading.Thread(
        target=lambda: manager._start_one_evaluation(
            evaluation_runner, JobConflictError, serial=True
        ),
        daemon=True,
    )
    evaluator.start()
    # 평가가 끼어들 시간을 충분히 줍니다. 문이 있으면 여기서 막혀 있습니다.
    started_evaluation.wait(timeout=0.5)

    resume.set()
    trainer.join(timeout=10)
    evaluator.join(timeout=10)

    trained = not isinstance(outcome.get("train"), JobConflictError)
    evaluated = started_evaluation.is_set()
    # 둘 다 자리를 잡으면 8GB 카드에서 둘 다 out of memory로 잃습니다.
    assert not (trained and evaluated), "학습과 평가가 동시에 GPU를 잡았습니다"
    assert trained or evaluated, "둘 다 시작하지 못했습니다"


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
