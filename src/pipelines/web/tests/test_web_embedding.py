"""crop embedding 학습 걸기와, 재순위에 쓸 embedding 고르기 test입니다.

두 가지가 조용히 틀리면 제출이 나빠지므로 그것부터 잽니다. detector 학습으로
오인해 GPU를 잡는 것, 그리고 서로 다른 crop 은행으로 학습한 embedding을 함께
쓰는 것입니다. 둘 다 오류로 멈추지 않고 낮아진 점수로만 드러납니다.
"""

from __future__ import annotations

from types import SimpleNamespace
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


@pytest.mark.parametrize(
    "uri",
    [
        "/etc/passwd",
        "C:/Windows/crop_bank.tar",
        "C:\\Windows\\crop_bank.tar",
        # UNC입니다. `..`도 없고 드라이브 글자도 없어 두 검사를 모두 지나갑니다.
        "//server/share/crop_bank.tar",
        "\\\\server\\share\\crop_bank.tar",
        # 드라이브 글자 `C:`와 이어지는 `//`가 합쳐져 **scheme처럼 보입니다.**
        # 글자 어디에나 `://`가 있으면 통과시키면 이것이 지나갑니다.
        "C://Windows/crop_bank.tar",
        "//server://share/crop_bank.tar",
    ],
)
def test_embedding_config_refuses_a_path_outside_the_repository(isolated_repo, uri):
    """S3에서는 앞의 `/`가 떨어져 **다른 key**가 됩니다.

    고른 적 없는 은행으로 조용히 학습하게 되고, 절대 경로가 응답에도 실립니다.
    """

    with pytest.raises(WebValidationError) as error:
        embedding.build_config(_payload(crop_bank_uri=uri))

    assert [item["field"] for item in error.value.as_list()] == ["crop_bank_uri"]


def test_embedding_config_still_takes_a_storage_address(isolated_repo):
    """`s3://`는 저장 계층이 판단합니다. 여기서 막으면 팀 저장소를 못 씁니다."""

    config = embedding.build_config(
        _payload(
            crop_bank_uri="s3://bucket/v5/crop_bank.tar",
            class_map_uri="s3://bucket/v5/class_map.json",
        )
    )

    assert config["inputs"]["data"]["crop_bank_uri"] == "s3://bucket/v5/crop_bank.tar"


def test_embedding_config_refuses_a_seed_numpy_cannot_use(isolated_repo):
    """`numpy.random.seed()`는 2**32 이상을 거절합니다.

    여기서 막지 않으면 학습이 시작된 **뒤에** 죽습니다. 그때는 GPU를 잡은 뒤입니다.
    """

    with pytest.raises(WebValidationError) as error:
        embedding.build_config(_payload(seed=embedding.MAX_SEED + 1))

    assert [item["field"] for item in error.value.as_list()] == ["seed"]


def test_embedding_config_refuses_a_weight_decay_train_will_refuse(isolated_repo):
    """train의 `_positive_number()`는 0을 거절합니다. 화면이 먼저 거절해야 합니다."""

    with pytest.raises(WebValidationError) as error:
        embedding.build_config(_payload(weight_decay=0))

    assert [item["field"] for item in error.value.as_list()] == ["weight_decay"]


def test_embedding_list_leaves_out_detector_training(fake_jobs):
    """detector 학습이 재순위 후보로 뜨면 고른 사람이 알 방법이 없습니다."""

    fake_jobs.append(_record("dino-e12", task=None))
    fake_jobs.append(_record("emb-r18"))
    fake_jobs.append(_record("emb-r34", status="running", checkpoint=None))

    runs = {item["run_id"]: item for item in embedding.list_runs()}

    assert set(runs) == {"emb-r18", "emb-r34"}
    assert runs["emb-r18"]["ready"] is True
    assert runs["emb-r34"]["ready"] is False


def test_rerank_settings_name_every_chosen_checkpoint(monkeypatch, fake_jobs):
    # bucket을 함께 세웁니다. 이 주소를 읽을 수 있는 설정이라야 실제로 있을 수 있는
    # 조합이고, 못 읽는 주소는 이제 재순위를 걸기 전에 거절합니다.
    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "b")
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


def test_rerank_settings_read_the_same_bank_written_two_ways(monkeypatch, fake_jobs):
    """같은 S3 객체를 상대 key와 `s3://` 주소로 적으면 글자만 다릅니다.

    글자로 견주면 함께 쓸 수 있는 것을 거절합니다. 저장 계층에 맡깁니다.
    """

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "bucket")
    fake_jobs.append(_record("emb-r18", crop_bank="v5/crop_bank.tar"))
    fake_jobs.append(
        _record("emb-r34", crop_bank="s3://bucket/v5/crop_bank.tar", backbone="resnet34")
    )

    settings = embedding.rerank_settings(["emb-r18", "emb-r34"])

    assert settings["rerank_crop_bank_uri"] == "v5/crop_bank.tar"


def test_rerank_settings_refuse_a_checkpoint_the_storage_cannot_read(monkeypatch, fake_jobs):
    """은행만 보고 넘기면 못 읽는 checkpoint가 재순위 직전까지 살아남습니다.

    그러면 evaluate가 GPU를 쓴 **뒤에** 같은 주소를 거절합니다. 은행에 거는 것과
    같은 계층에 checkpoint도 물어봅니다.
    """

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "bucket")
    fake_jobs.append(_record("emb-r18", checkpoint="s3://bucket/r18.pt"))
    fake_jobs.append(
        _record("emb-r34", checkpoint="s3://other/r34.pt", backbone="resnet34")
    )

    with pytest.raises(WebError) as error:
        embedding.rerank_settings(["emb-r18", "emb-r34"])

    assert "s3://other/r34.pt" in str(error.value)


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
        ["dino-a", "dino-b"],
        run_id="fused",
        rerank=embedding.rerank_settings(["emb-r18"]),
    )

    assert config["evaluate"]["rerank_checkpoint_uris"] == ["s3://bucket/r18.pt"]
    assert config["evaluate"]["rerank_crop_bank_uri"] == "datasets/v5/crop_bank.tar"


def test_fusion_does_not_look_the_embeddings_up_again(monkeypatch, fake_jobs):
    """합치는 자리에서 이름으로 다시 찾으면 **늦게** 실패합니다.

    detector 예측을 만드는 몇 분 사이에 그 기록을 지운 사람이 있으면, checkpoint는
    멀쩡한데 "기록에 없는 embedding"으로 끝납니다. 그때는 추론이 이미 끝난 뒤라
    그 시간이 통째로 버려집니다. 그래서 걸 때 한 번 풀어 들고 갑니다.
    """

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
    # 기록은 하나도 남아 있지 않습니다. 이름으로 찾는다면 여기서 실패합니다.
    assert embedding.list_runs() == []

    config = ensemble.build_fusion_config(
        ["dino-a", "dino-b"],
        run_id="fused",
        rerank={
            "rerank_checkpoint_uris": ["s3://bucket/r18.pt"],
            "rerank_crop_bank_uri": "datasets/v5/crop_bank.tar",
        },
    )

    assert config["evaluate"]["rerank_checkpoint_uris"] == ["s3://bucket/r18.pt"]


def test_the_runner_reads_the_chosen_embeddings_exactly_once(monkeypatch, fake_jobs):
    """두 번 찾으면 **검사한 것과 실제로 쓰는 것이 갈립니다.**

    그 사이에 기록을 지운 사람이 있으면, 통과한 검사와 다른 결과로 끝납니다.
    여기서는 목록이 한 번만 답하고 그다음에는 비도록 해서 그것을 잽니다.
    """

    from src.pipelines.web import ensemble_jobs

    fake_jobs.append(_record("emb-r18", checkpoint="s3://bucket/r18.pt"))
    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "bucket")
    answered: list[int] = []
    real = embedding.list_runs

    def once() -> list[dict[str, Any]]:
        answered.append(1)
        return real() if len(answered) == 1 else []

    monkeypatch.setattr(embedding, "list_runs", once)
    monkeypatch.setattr(ensemble, "pending_runs", lambda names: [])
    monkeypatch.setattr(ensemble, "check_selection", lambda *args, **kwargs: [])
    # 일꾼 thread는 띄우지 않습니다. 재는 것은 **거는 자리에서 몇 번 찾는가**이고,
    # 살아남은 thread는 뒤따르는 test의 대기열을 건드립니다.
    monkeypatch.setattr(
        ensemble_jobs.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None),
    )

    state = ensemble_jobs.EnsembleRunner().start(
        ["dino-a", "dino-b"], run_id="fused", embedding_run_ids=["emb-r18"]
    )

    assert state["status"] == "running"
    assert answered == [1]


def test_automatic_evaluation_walks_past_an_embedding_run(manager, fake_jobs):
    """embedding을 평가에 넘기면 GPU를 잡고 나서 checkpoint를 못 읽어 실패합니다."""

    detector = _record("dino-e12", task=None)
    embedded = _record("emb-r18")
    manager._records = {detector.job_id: detector, embedded.job_id: embedded}
    manager._evaluation_pending = [embedded.job_id, detector.job_id]

    assert manager._next_unevaluated() is detector
