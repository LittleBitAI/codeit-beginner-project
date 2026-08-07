"""Registry index 기반 Web 실험 목록과 선택 비교 경로."""

from __future__ import annotations

from src.pipelines.web import experiments


def registry_summary(run_id: str) -> dict:
    return {
        "summary_version": "1",
        "run_id": run_id,
        "created_at": "2026-08-06T00:00:00+00:00",
        "seed": 42,
        "schema_version": "1.2",
        "experiment_record_uri": f"artifacts/registry/{run_id}/experiment_record.json",
        "metrics": {"mAP": 0.31, "mAP50": 0.55},
        "metrics_source": "metrics_file",
        "artifacts": {
            "train_manifest_uri": "artifacts/data/train.json",
            "validation_manifest_uri": "artifacts/data/validation.json",
            "class_map_uri": "artifacts/data/classes.json",
            "dataset_summary_uri": "artifacts/data/summary.json",
            "submission_uri": None,
        },
    }


def registry_training() -> dict:
    return {
        "architecture": "retinanet_resnet50_fpn_v2",
        "pretrained": True,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "momentum": None,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "device": "cuda",
        "epochs": 50,
        "batch_size": 4,
        "num_workers": 0,
    }


def experiment_record(run_id: str) -> dict:
    return {
        "schema_version": "1.2",
        "run_id": run_id,
        "config_snapshot": {
            "train": {
                "architecture": "retinanet_resnet50_fpn_v2",
                "optimizer": "AdamW",
                "learning_rate": 0.0002,
                "weight_decay": 0.02,
                "beta1": 0.8,
                "beta2": 0.98,
                "epsilon": 1e-7,
                "device": "cuda",
                "epochs": 4,
                "batch_size": 2,
                "num_workers": 1,
                "seed": 42,
                "pretrained": True,
            }
        },
    }


def test_experiment_api_lists_registry_index_only(client, manager, monkeypatch):
    called = []
    monkeypatch.setattr(
        experiments,
        "list_experiment_summaries",
        lambda config: (called.append(config), [registry_summary("run-1")])[1],
    )

    response = client.get("/api/train/experiments")

    assert response.status_code == 200
    payload = response.json()["experiments"]
    assert [item["run_id"] for item in payload] == ["run-1"]
    assert payload[0]["metrics"]["map"] == 0.31
    assert "experiment_record_uri" not in response.text
    assert called


def test_compare_reads_only_selected_exact_records(client, monkeypatch):
    summaries = [registry_summary("run-1"), registry_summary("run-2")]
    read = []
    listed = []

    def list_summaries(config, **_kwargs):
        listed.append(config)
        return summaries

    # 두 이름을 함께 바꿔야 index를 몇 번 읽는지 셀 수 있습니다. web이 부르는
    # 이름과, 공용 registry 안에서 다시 부르는 이름이 다른 객체이기 때문입니다.
    monkeypatch.setattr(experiments, "list_experiment_summaries", list_summaries)
    monkeypatch.setattr(
        "src.common.experiment_registry.list_experiment_summaries", list_summaries
    )

    def read_record(uri, config, *, expected_run_id=None):
        read.append((uri, expected_run_id))
        return experiment_record(expected_run_id)

    monkeypatch.setattr(experiments, "read_experiment_record", read_record)

    response = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["run-2", "run-1"]}
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["run_id"] for item in payload["experiments"]] == ["run-2", "run-1"]
    optimizer = payload["experiments"][0]["optimizer"]
    assert optimizer == {
        "name": "AdamW",
        "source": "record",
        "learning_rate": 0.0002,
        "momentum": None,
        "weight_decay": 0.02,
        "beta1": 0.8,
        "beta2": 0.98,
        "epsilon": 1e-7,
    }
    # record는 함께 읽으므로 읽는 순서는 정해져 있지 않습니다. 고른 것만 읽었는지와
    # 응답 순서가 요청 순서와 같은지가 지켜야 할 약속입니다.
    assert {run_id for _, run_id in read} == {"run-1", "run-2"}
    # Index는 한 번만 읽습니다. 실험 하나를 골라도 전부를 훑으면 화면이 그만큼 느립니다.
    assert len(listed) == 1
    assert "experiment_record_uri" not in response.text


def test_compare_uses_sgd_for_legacy_record_without_optimizer(client, monkeypatch):
    summary = registry_summary("legacy")
    legacy = experiment_record("legacy")
    legacy["config_snapshot"]["train"] = {}
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])
    monkeypatch.setattr(experiments, "read_experiment_record", lambda *a, **k: legacy)

    response = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["legacy"]}
    )

    optimizer = response.json()["experiments"][0]["optimizer"]
    assert optimizer["name"] == "SGD"
    assert optimizer["learning_rate"] == 0.005
    assert optimizer["momentum"] == 0.9
    assert optimizer["beta1"] is None
    assert response.json()["experiments"][0]["model"] == {
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained": None,
        "source": "legacy_fallback",
    }


def test_compare_preserves_explicit_zero_values(client, monkeypatch):
    summary = registry_summary("zero-values")
    summary["seed"] = 7
    record = experiment_record("zero-values")
    record["config_snapshot"]["train"].update({"learning_rate": 0.0, "seed": 0})
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])
    monkeypatch.setattr(experiments, "read_experiment_record", lambda *a, **k: record)

    response = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["zero-values"]}
    )

    compared = response.json()["experiments"][0]
    assert compared["optimizer"]["learning_rate"] == 0.0
    assert compared["training"]["seed"] == 0


def test_experiment_list_fills_training_from_index_summary(client, monkeypatch):
    summary = registry_summary("run-1")
    summary["training"] = registry_training()
    summary["training_source"] = "config_snapshot"
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])

    listed = client.get("/api/train/experiments").json()["experiments"][0]

    assert listed["model"] == {
        "architecture": "retinanet_resnet50_fpn_v2",
        "pretrained": True,
        "source": "record",
    }
    assert listed["optimizer"] == {
        "name": "AdamW",
        "source": "record",
        "learning_rate": 0.0001,
        "momentum": None,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    }
    # registry는 training 안에 seed를 넣지 않으므로 summary 최상위 seed를 그대로 씁니다.
    assert listed["training"] == {
        "device": "cuda",
        "epochs": 50,
        "batch_size": 4,
        "num_workers": 0,
        "seed": 42,
    }


def test_experiment_list_shows_nothing_for_index_without_training_key(client, monkeypatch):
    """이 기능 이전의 옛 index는 값을 모르므로 기본값을 지어내지 않습니다."""

    summary = registry_summary("old-index")
    assert "training" not in summary
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])

    listed = client.get("/api/train/experiments").json()["experiments"][0]

    assert listed["model"]["architecture"] is None
    assert listed["optimizer"]["name"] is None
    assert listed["optimizer"]["learning_rate"] is None
    assert listed["training"] == {
        "device": None,
        "epochs": None,
        "batch_size": None,
        "num_workers": None,
        "seed": 42,
    }


def test_experiment_list_masks_paths_like_compare_does(client, monkeypatch):
    """목록도 비교와 같은 redact를 거쳐야 한쪽으로만 경로가 새지 않습니다."""

    summary = registry_summary("leaky")
    summary["training"] = {**registry_training(), "device": "cuda /home/someone/keys"}
    summary["training_source"] = "config_snapshot"
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])

    response = client.get("/api/train/experiments")

    assert "someone" not in response.text
    assert response.json()["experiments"][0]["training"]["device"] != summary["training"]["device"]


def test_experiment_list_matches_compare_for_unavailable_training(client, monkeypatch):
    """registry가 판단해 값이 빈 index는 비교 화면과 같은 호환 기본값을 보여야 합니다."""

    summary = registry_summary("legacy")
    summary["training"] = {key: None for key in registry_training()}
    summary["training_source"] = "unavailable"
    record = experiment_record("legacy")
    record["config_snapshot"]["train"] = {}
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])
    monkeypatch.setattr(experiments, "read_experiment_record", lambda *a, **k: record)

    listed = client.get("/api/train/experiments").json()["experiments"][0]
    compared = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["legacy"]}
    ).json()["experiments"][0]

    assert listed["model"]["source"] == "legacy_fallback"
    assert listed["optimizer"]["name"] == "SGD"
    for block in ("model", "optimizer", "training"):
        assert listed[block] == compared[block]
