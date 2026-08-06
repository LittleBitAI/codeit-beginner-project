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
    monkeypatch.setattr(
        experiments,
        "compare_experiment_summaries",
        lambda run_ids, config: {
            "run_ids": list(run_ids),
            "fields": {
                "experiment_record_uri": {
                    "values": {
                        run_id: f"artifacts/registry/{run_id}/experiment_record.json"
                        for run_id in run_ids
                    },
                    "differs": True,
                }
            },
            "missing": [],
        },
    )
    monkeypatch.setattr(
        experiments,
        "list_experiment_summaries",
        lambda config: summaries,
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
    assert [run_id for _, run_id in read] == ["run-2", "run-1"]
    assert "experiment_record_uri" not in response.text


def test_compare_uses_sgd_for_legacy_record_without_optimizer(client, monkeypatch):
    summary = registry_summary("legacy")
    legacy = experiment_record("legacy")
    legacy["config_snapshot"]["train"] = {}
    monkeypatch.setattr(
        experiments,
        "compare_experiment_summaries",
        lambda run_ids, config: {
            "run_ids": ["legacy"],
            "fields": {"experiment_record_uri": {"values": {"legacy": summary["experiment_record_uri"]}}},
            "missing": [],
        },
    )
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
    monkeypatch.setattr(
        experiments,
        "compare_experiment_summaries",
        lambda run_ids, config: {
            "run_ids": ["zero-values"],
            "fields": {
                "experiment_record_uri": {
                    "values": {"zero-values": summary["experiment_record_uri"]}
                }
            },
            "missing": [],
        },
    )
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])
    monkeypatch.setattr(experiments, "read_experiment_record", lambda *a, **k: record)

    response = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["zero-values"]}
    )

    compared = response.json()["experiments"][0]
    assert compared["optimizer"]["learning_rate"] == 0.0
    assert compared["training"]["seed"] == 0
