"""Registry 기반 experiment API의 정보 노출 방지 검증."""

from __future__ import annotations

from src.pipelines.web import experiments
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


def test_experiment_api_does_not_expose_registry_artifact_uris(client, monkeypatch):
    artifacts = {
        key: f"artifacts/datasets/pills/{key}.json" for key in DATA_ARTIFACT_KEYS
    }
    monkeypatch.setattr(
        experiments,
        "list_experiment_summaries",
        lambda config: [
            {
                "run_id": "newer",
                "created_at": "2026-08-05T02:00:00Z",
                "seed": 42,
                "metrics": {"mAP": 0.4},
                "experiment_record_uri": (
                    "artifacts/registry/newer/experiment_record.json"
                ),
                "artifacts": {**artifacts, "access_token": "do-not-return-this"},
            }
        ],
    )

    response = client.get("/api/train/experiments")

    assert response.status_code == 200
    payload = response.json()
    assert [item["run_id"] for item in payload["experiments"]] == ["newer"]
    serialized = response.text
    assert "experiment_record_uri" not in serialized
    assert "train_manifest_uri" not in serialized
    assert "artifacts/datasets/pills" not in serialized
    assert "do-not-return-this" not in serialized
