"""JobRecord 기반 experiment 비교 API와 adapter 검증."""

from __future__ import annotations

from copy import deepcopy

from src.pipelines.web.experiments import experiment_summary
from src.pipelines.web.jobs import store
from src.pipelines.web.jobs.model import JobRecord
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


def make_record(*, job_id: str = "a" * 32, complete_data: bool = True) -> JobRecord:
    data_inputs = {
        key: f"artifacts/datasets/pills/{key}.json" for key in DATA_ARTIFACT_KEYS
    }
    if not complete_data:
        data_inputs.pop("dataset_summary_uri")
    return JobRecord(
        job_id=job_id,
        config_id="c" * 32,
        run_id=f"run-{job_id[0]}",
        status="succeeded",
        created_at="2026-08-05T01:00:00Z",
        started_at="2026-08-05T01:00:10Z",
        finished_at="2026-08-05T01:02:10Z",
        settings={
            "device": "cuda",
            "epochs": 10,
            "batch_size": 2,
            "num_workers": 4,
            "seed": 42,
            "pretrained": True,
            "learning_rate": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
        },
        data_inputs=data_inputs,
        summary={
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "best_epoch": 7,
            "best_validation_loss": 0.314,
            "metrics": {"mAP": 0.41, "mAP50": 0.63},
        },
    )


def test_adapter_selects_comparison_fields_without_mutating_record():
    record = make_record()
    before = deepcopy(record.to_dict())

    summary = experiment_summary(record)

    assert summary["experiment_id"] == record.job_id
    assert summary["run_id"] == "run-a"
    assert summary["elapsed_seconds"] == 120.0
    assert summary["model"] == {
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "pretrained": True,
    }
    assert summary["optimizer"] == {
        "name": None,
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
    }
    assert summary["training"] == {
        "device": "cuda",
        "epochs": 10,
        "batch_size": 2,
        "num_workers": 4,
        "seed": 42,
    }
    assert summary["metrics"] == {
        "best_epoch": 7,
        "best_validation_loss": 0.314,
        "map": 0.41,
        "map50": 0.63,
    }
    assert record.to_dict() == before


def test_dataset_identity_matches_for_the_same_complete_artifact_set():
    first = make_record(job_id="a" * 32)
    second = make_record(job_id="b" * 32)
    second.data_inputs = dict(reversed(list(second.data_inputs.items())))

    first_dataset = experiment_summary(first)["dataset"]
    second_dataset = experiment_summary(second)["dataset"]

    assert first_dataset["identity"] == second_dataset["identity"]
    assert first_dataset["identity_source"] == "artifact_set"
    assert first_dataset["artifacts_complete"] is True


def test_dataset_identity_is_unknown_when_any_artifact_is_missing():
    dataset = experiment_summary(make_record(complete_data=False))["dataset"]

    assert dataset == {
        "identity": None,
        "identity_source": "unknown",
        "artifacts_complete": False,
    }


def test_experiment_api_is_newest_first_and_does_not_expose_artifact_uris(
    client, manager
):
    older = make_record(job_id="a" * 32)
    newer = make_record(job_id="b" * 32)
    older.created_at = "2026-08-05T01:00:00Z"
    newer.created_at = "2026-08-05T02:00:00Z"
    newer.data_inputs["access_token"] = "do-not-return-this"
    store.save_record(older)
    store.save_record(newer)
    manager._loaded = False

    response = client.get("/api/train/experiments")

    assert response.status_code == 200
    payload = response.json()
    assert [item["experiment_id"] for item in payload["experiments"]] == [
        newer.job_id,
        older.job_id,
    ]
    serialized = response.text
    assert "train_manifest_uri" not in serialized
    assert "artifacts/datasets/pills" not in serialized
    assert "do-not-return-this" not in serialized
