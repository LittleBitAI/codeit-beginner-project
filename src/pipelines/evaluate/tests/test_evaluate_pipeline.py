"""공개 함수 run(config)의 성공·실패 계약 test입니다."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.pipelines.evaluate import run as public_run
from src.pipelines.evaluate.pipeline import resolve_settings, run

from conftest import IMAGE_RECORDS, write_json, write_jsonl


RESULT_KEYS = {"status", "artifacts", "summary", "message"}
ARTIFACT_KEYS = {"run_id", "metrics_uri", "predictions_uri"}


def _read_json(repository_root: Path, uri: str) -> dict:
    return json.loads((repository_root / uri).read_text(encoding="utf-8"))


def test_public_run_is_the_pipeline_entry_point():
    assert public_run is run


def test_successful_run_returns_contract_keys(base_config: dict, repository_root: Path):
    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    assert set(result) == RESULT_KEYS
    assert set(result["artifacts"]) == ARTIFACT_KEYS
    assert result["artifacts"]["run_id"] == "evaluate-0001"
    assert result["summary"]["pipeline"] == "evaluate"
    assert result["summary"]["prediction_source"] == "predictions_file"


def test_successful_run_writes_repository_relative_artifacts(
    base_config: dict, repository_root: Path
):
    result = run(base_config)

    metrics_uri = result["artifacts"]["metrics_uri"]
    predictions_uri = result["artifacts"]["predictions_uri"]

    assert metrics_uri == "artifacts/evaluate/evaluate-0001/metrics.json"
    assert predictions_uri == "artifacts/evaluate/evaluate-0001/predictions.json"
    assert not Path(metrics_uri).is_absolute()
    assert (repository_root / metrics_uri).is_file()
    assert (repository_root / predictions_uri).is_file()


def test_metrics_document_contains_expected_scores(base_config: dict, repository_root: Path):
    result = run(base_config)
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])

    assert metrics["run_id"] == "evaluate-0001"
    assert metrics["metrics"]["mAP"] == pytest.approx(1.0)
    assert metrics["metrics"]["mAP50"] == pytest.approx(1.0)
    assert metrics["image_count"] == 2
    assert metrics["annotation_count"] == 3
    assert len(metrics["iou_thresholds"]) == 10
    assert {entry["name"] for entry in metrics["per_class"]} == {"tylenol", "aspirin"}


def test_predictions_document_keeps_evaluated_detections(
    base_config: dict, repository_root: Path
):
    result = run(base_config)
    document = _read_json(repository_root, result["artifacts"]["predictions_uri"])

    assert document["bbox_format"] == "xywh"
    assert document["prediction_count"] == 3
    assert {entry["image_id"] for entry in document["predictions"]} == {"img-1", "img-2"}
    assert all(set(entry) == {"image_id", "category_id", "bbox", "score"} for entry in document["predictions"])


def test_run_does_not_modify_inputs(base_config: dict, repository_root: Path):
    before = copy.deepcopy(base_config["inputs"])

    run(base_config)

    assert base_config["inputs"] == before


def test_run_is_deterministic_for_the_same_seed(base_config: dict, repository_root: Path):
    first = run(base_config)
    second_config = copy.deepcopy(base_config)
    second_config["evaluate"]["run_id"] = "evaluate-0002"
    second = run(second_config)

    assert first["summary"]["metrics"] == second["summary"]["metrics"]


def test_score_threshold_and_top_k_are_applied(base_config: dict, repository_root: Path):
    write_json(
        repository_root / "data/val/predictions.json",
        [
            {"image_id": "img-1", "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.95},
            {"image_id": "img-1", "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.10},
            {"image_id": "img-2", "category_id": 1, "bbox": [30, 30, 10, 10], "score": 0.80},
        ],
    )
    base_config["evaluate"]["score_threshold"] = 0.5

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    assert result["summary"]["prediction_count"] == 2
    assert result["summary"]["max_detections_per_image"] == 4


def test_missing_manifest_returns_error(base_config: dict, repository_root: Path):
    base_config["inputs"]["data"].pop("validation_manifest_uri")

    result = run(base_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "validation manifest" in result["message"]


def test_missing_prediction_source_returns_error(base_config: dict, repository_root: Path):
    base_config["evaluate"].pop("predictions_input_uri")
    base_config["inputs"]["train"].pop("best_checkpoint_uri")

    result = run(base_config)

    assert result["status"] == "error"
    assert "예측을 만들 수 없습니다" in result["message"]


def test_missing_manifest_file_returns_error(base_config: dict, repository_root: Path):
    base_config["evaluate"]["validation_manifest_uri"] = "data/val/does-not-exist.jsonl"

    result = run(base_config)

    assert result["status"] == "error"
    assert "artifact가 없습니다" in result["message"]


def test_broken_manifest_returns_error_without_writing_artifacts(
    base_config: dict, repository_root: Path
):
    (repository_root / "data/val/manifest.jsonl").write_text(
        '{"image_id": "img-1"\n', encoding="utf-8", newline="\n"
    )

    result = run(base_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert not (repository_root / "artifacts/evaluate/evaluate-0001").exists()


def test_prediction_for_unknown_image_returns_error(base_config: dict, repository_root: Path):
    write_json(
        repository_root / "data/val/predictions.json",
        [{"image_id": "img-999", "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.5}],
    )

    result = run(base_config)

    assert result["status"] == "error"
    assert "manifest에 없는 image_id" in result["message"]


def test_existing_artifact_is_not_overwritten_by_default(
    base_config: dict, repository_root: Path
):
    first = run(base_config)
    assert first["status"] == "ok", first["message"]

    second = run(base_config)

    assert second["status"] == "error"
    assert "이미 있습니다" in second["message"]
    metrics = _read_json(repository_root, first["artifacts"]["metrics_uri"])
    assert metrics["run_id"] == "evaluate-0001"


def test_overwrite_option_allows_rerun(base_config: dict, repository_root: Path):
    assert run(base_config)["status"] == "ok"
    base_config["evaluate"]["overwrite"] = True

    result = run(base_config)

    assert result["status"] == "ok", result["message"]


def test_manifest_without_class_map_uses_category_id_as_name(
    base_config: dict, repository_root: Path
):
    base_config["inputs"]["data"].pop("class_map_uri")

    result = run(base_config)
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])

    assert result["status"] == "ok", result["message"]
    assert {entry["name"] for entry in metrics["per_class"]} == {"1", "2"}


def test_manifest_with_zero_annotations_is_rejected(base_config: dict, repository_root: Path):
    write_jsonl(
        repository_root / "data/val/manifest.jsonl",
        [dict(record, annotations=[]) for record in IMAGE_RECORDS],
    )
    write_json(repository_root / "data/val/predictions.json", [])

    result = run(base_config)
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])

    assert result["status"] == "ok", result["message"]
    assert metrics["metrics"]["mAP"] == 0.0
    assert metrics["evaluated_class_count"] == 0


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"iou_thresholds": []}, "iou_thresholds"),
        ({"iou_thresholds": [1.5]}, "iou_thresholds"),
        ({"score_threshold": 2}, "score_threshold"),
        ({"max_detections_per_image": 0}, "max_detections_per_image"),
        ({"seed": "7"}, "seed"),
        ({"overwrite": "yes"}, "overwrite"),
        ({"device": ""}, "device"),
    ],
)
def test_invalid_settings_return_error(
    base_config: dict, repository_root: Path, settings: dict, message: str
):
    base_config["evaluate"].update(settings)

    result = run(base_config)

    assert result["status"] == "error"
    assert message in result["message"]


def test_non_object_evaluate_settings_return_error(repository_root: Path):
    result = run({"evaluate": "invalid"})

    assert result["status"] == "error"
    assert "object여야 합니다" in result["message"]


def test_dummy_execution_mode_keeps_previous_behaviour():
    result = run({"execution": {"mode": "dummy"}})

    assert result["status"] == "ok"
    assert result["summary"] == {"pipeline": "evaluate", "mode": "dummy"}
    assert result["artifacts"] == {}


def test_settings_prefer_own_config_over_inputs(base_config: dict, repository_root: Path):
    base_config["evaluate"]["validation_manifest_uri"] = "data/val/manifest.jsonl"
    base_config["inputs"]["data"]["validation_manifest_uri"] = "data/val/other.jsonl"

    settings = resolve_settings(base_config)

    assert settings.validation_manifest_uri == "data/val/manifest.jsonl"
    assert settings.checkpoint_uri == "checkpoints/best.pt"


def test_run_id_falls_back_to_train_run_id(base_config: dict, repository_root: Path):
    base_config["evaluate"].pop("run_id")

    settings = resolve_settings(base_config)

    assert settings.run_id == "train-0001"
    assert settings.metrics_uri == "artifacts/evaluate/train-0001/metrics.json"
