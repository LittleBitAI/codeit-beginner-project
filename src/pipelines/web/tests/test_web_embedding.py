"""crop embedding 학습 걸기와, 재순위에 쓸 embedding 고르기 test입니다.

두 가지가 조용히 틀리면 제출이 나빠지므로 그것부터 잽니다. detector 학습으로
오인해 GPU를 잡는 것, 그리고 서로 다른 crop 은행으로 학습한 embedding을 함께
쓰는 것입니다. 둘 다 오류로 멈추지 않고 낮아진 점수로만 드러납니다.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.common import train_contract
from src.pipelines.web import embedding, ensemble
from src.pipelines.web.errors import WebError, WebValidationError
from src.pipelines.web.jobs.model import JobRecord


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "crop_bank_uri": "datasets/pill_detection/processed/v5-8020-group/crop_bank.tar",
        "class_map_uri": "datasets/pill_detection/processed/v5-8020-group/class_map.json",
        "run_id": "emb-r18",
    }
    payload.update(overrides)
    return payload


def _record(
    run_id: str,
    *,
    task: str | None = "embedding",
    status: str = "succeeded",
    checkpoint: str | None = "artifacts/embeddings/best.pt",
    crop_bank: str | None = "datasets/v5/crop_bank.tar",
    backbone: str = "resnet18",
) -> JobRecord:
    settings: dict[str, Any] = {"run_id": run_id, "backbone": backbone}
    if task is not None:
        settings["task"] = task
    return JobRecord(
        job_id=f"job-{run_id}",
        config_id=f"config-{run_id}",
        run_id=run_id,
        status=status,
        settings=settings,
        artifacts={"best_checkpoint_uri": checkpoint} if checkpoint else {},
        data_inputs={"crop_bank_uri": crop_bank} if crop_bank else {},
    )


@pytest.fixture
def fake_jobs(monkeypatch: pytest.MonkeyPatch):
    """이 서버가 아는 학습 기록을 흉내 냅니다."""

    records: list[JobRecord] = []

    class _Manager:
        def list_jobs(self) -> list[JobRecord]:
            return list(records)

    monkeypatch.setattr("src.pipelines.web.jobs.get_manager", lambda: _Manager())
    return records


def test_embedding_defaults_come_from_the_shared_contract():
    """기본값을 여기서 다시 적으면 화면과 train이 서로 다른 값을 씁니다."""

    shown = embedding.defaults()

    assert shown["backbones"] == list(train_contract.EMBEDDING_BACKBONES)
    for name, value in shown["defaults"].items():
        assert value == train_contract.EMBEDDING_SETTING_DEFAULTS[name]


def test_embedding_config_asks_train_for_the_embedding_task(isolated_repo):
    """detector가 아니라 embedding을 학습하라고 말해야 합니다."""

    config = embedding.build_config(_payload(backbone="resnet34", epochs=12))

    assert config["train"]["task"] == "embedding"
    assert config["train"]["backbone"] == "resnet34"
    assert config["train"]["epochs"] == 12
    assert config["train"]["run_id"] == "emb-r18"
    assert config["inputs"]["data"]["crop_bank_uri"].endswith("crop_bank.tar")
    assert config["execution"]["mode"] == "real"


def test_embedding_config_needs_reference_crops(isolated_repo):
    """은행이 없으면 학습할 그림이 없습니다. 대기열에 넣기 전에 거절합니다."""

    with pytest.raises(WebValidationError) as error:
        embedding.build_config(_payload(crop_bank_uri="   "))

    assert [item["field"] for item in error.value.as_list()] == ["crop_bank_uri"]


@pytest.mark.parametrize(
    "learning_rate",
    [
        # `nan`은 어느 비교에도 걸리지 않습니다. 크기만 재면 그대로 통과해,
        # loss가 처음부터 `nan`인 학습이 밤새 GPU를 잡습니다.
        float("nan"),
        float("inf"),
        # python 정수는 크기 제한이 없어, `float()`로 바꾸는 자리에서 검사가
        # 스스로 터집니다.
        10**400,
    ],
    ids=["nan", "inf", "overflow"],
)
def test_embedding_config_refuses_a_learning_rate_that_is_not_a_number(
    isolated_repo, learning_rate
):
    """숫자처럼 보이지만 학습을 못 하는 값입니다. 대기열에 넣기 전에 거절합니다."""

    with pytest.raises(WebValidationError) as error:
        embedding.build_config(_payload(learning_rate=learning_rate))

    assert [item["field"] for item in error.value.as_list()] == ["learning_rate"]


def test_embedding_config_refuses_a_backbone_train_cannot_build(isolated_repo):
    with pytest.raises(WebValidationError) as error:
        embedding.build_config(_payload(backbone="resnet101"))

    assert [item["field"] for item in error.value.as_list()] == ["backbone"]


def test_embedding_list_leaves_out_detector_training(fake_jobs):
    """detector 학습이 재순위 후보로 뜨면 고른 사람이 알 방법이 없습니다."""

    fake_jobs.append(_record("dino-e12", task=None))
    fake_jobs.append(_record("emb-r18"))
    fake_jobs.append(_record("emb-r34", status="running", checkpoint=None))

    runs = {item["run_id"]: item for item in embedding.list_runs()}

    assert set(runs) == {"emb-r18", "emb-r34"}
    assert runs["emb-r18"]["ready"] is True
    assert runs["emb-r34"]["ready"] is False


def test_rerank_settings_name_every_chosen_checkpoint(fake_jobs):
    fake_jobs.append(_record("emb-r18", checkpoint="s3://b/r18.pt"))
    fake_jobs.append(_record("emb-r34", checkpoint="s3://b/r34.pt", backbone="resnet34"))

    settings = embedding.rerank_settings(["emb-r18", "emb-r34"])

    assert settings["rerank_checkpoint_uris"] == ["s3://b/r18.pt", "s3://b/r34.pt"]
    assert settings["rerank_crop_bank_uri"] == "datasets/v5/crop_bank.tar"


def test_rerank_settings_refuse_two_different_crop_banks(fake_jobs):
    """참조가 갈리면 한쪽은 자기가 본 적 없는 crop과 견주게 됩니다."""

    fake_jobs.append(_record("emb-r18", crop_bank="datasets/v5/crop_bank.tar"))
    fake_jobs.append(_record("emb-r34", crop_bank="datasets/v4/crop_bank.tar"))

    with pytest.raises(WebError) as error:
        embedding.rerank_settings(["emb-r18", "emb-r34"])

    assert "서로 다른 crop 은행" in str(error.value)


def test_rerank_settings_refuse_an_embedding_without_a_checkpoint(fake_jobs):
    fake_jobs.append(_record("emb-r18"))
    fake_jobs.append(_record("emb-r34", status="failed", checkpoint=None))

    with pytest.raises(WebError) as error:
        embedding.rerank_settings(["emb-r18", "emb-r34"])

    assert "아직 checkpoint가 없는" in str(error.value)


def test_nothing_chosen_changes_no_setting(fake_jobs):
    """고르지 않으면 지금까지와 똑같은 제출이 나와야 합니다."""

    assert embedding.rerank_settings([]) == {}


def test_fusion_config_carries_the_chosen_embeddings(monkeypatch, fake_jobs):
    """화면에서 고른 embedding이 evaluate 설정까지 그대로 가야 합니다."""

    fake_jobs.append(_record("emb-r18", checkpoint="s3://bucket/r18.pt"))
    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "bucket")
    candidates = [
        {
            "run_id": name,
            "checkpoint_uri": f"s3://bucket/{name}/best.pt",
            "test_predictions_uri": f"s3://bucket/{name}/test_predictions.json",
            "ready": True,
            "dataset_label": "v5",
            "kaggle_score": 0.6,
            "created_at": None,
            "data_inputs": {
                "validation_manifest_uri": "s3://bucket/v/validation_manifest.json",
                "test_manifest_uri": "s3://bucket/v/test_manifest.json",
                "class_map_uri": "s3://bucket/v/class_map.json",
            },
            "predictions_uri": None,
        }
        for name in ("dino-a", "dino-b")
    ]
    monkeypatch.setattr(ensemble, "list_candidates", lambda: list(candidates))

    config = ensemble.build_fusion_config(
        ["dino-a", "dino-b"], run_id="fused", embedding_run_ids=["emb-r18"]
    )

    assert config["evaluate"]["rerank_checkpoint_uris"] == ["s3://bucket/r18.pt"]
    assert config["evaluate"]["rerank_crop_bank_uri"] == "datasets/v5/crop_bank.tar"


def test_automatic_evaluation_walks_past_an_embedding_run(manager, fake_jobs):
    """embedding을 평가에 넘기면 GPU를 잡고 나서 checkpoint를 못 읽어 실패합니다."""

    detector = _record("dino-e12", task=None)
    embedded = _record("emb-r18")
    manager._records = {detector.job_id: detector, embedded.job_id: embedded}
    manager._evaluation_pending = [embedded.job_id, detector.job_id]

    assert manager._next_unevaluated() is detector
