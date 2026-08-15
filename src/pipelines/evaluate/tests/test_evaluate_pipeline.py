"""공개 함수 run(config)의 성공·실패 계약 test입니다."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.common import validate_pipeline_result
from src.pipelines.evaluate import run as public_run
from src.pipelines.evaluate.pipeline import resolve_settings, run

from conftest import IMAGE_RECORDS, write_json, write_jsonl


RESULT_KEYS = {"status", "artifacts", "summary", "message"}
ARTIFACT_KEYS = {"run_id", "metrics_uri", "predictions_uri"}
COMPETITION_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]


def _read_json(repository_root: Path, uri: str) -> dict:
    return json.loads((repository_root / uri).read_text(encoding="utf-8"))


def _add_test_manifest(base_config: dict, repository_root: Path) -> None:
    write_json(
        repository_root / "data/test/instances.json",
        {
            "images": [
                {"id": 20, "file_name": "0020.jpg", "width": 100, "height": 100},
                {"id": 10, "file_name": "0010.jpg", "width": 100, "height": 100},
            ],
            "annotations": [],
            "categories": [{"id": 3, "name": "pill-a"}, {"id": 7, "name": "pill-b"}],
        },
    )
    base_config["inputs"]["data"]["test_manifest_uri"] = "data/test/instances.json"


def _normalised_validation_predictions() -> list[dict]:
    raw_predictions = [
        {"image_id": "img-1", "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.95},
        {"image_id": "img-1", "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.90},
        {"image_id": "img-2", "category_id": 1, "bbox": [30, 30, 10, 10], "score": 0.80},
    ]
    return [
        dict(prediction, image_key=str(prediction["image_id"]))
        for prediction in raw_predictions
    ]


def _test_prediction(
    image_id: int,
    category_id: int,
    bbox_x: float,
    score: float,
) -> dict:
    return {
        "image_id": image_id,
        "image_key": str(image_id),
        "category_id": category_id,
        "bbox": [bbox_x, 2.0, 3.0, 4.0],
        "score": score,
    }


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
    # 메인 지표는 mAP@[0.75:0.95] 5점이고, COCOeval에는 10점을 넘깁니다.
    assert metrics["iou_thresholds"] == [0.75, 0.8, 0.85, 0.9, 0.95]
    assert len(metrics["iou_thresholds_all"]) == 10
    assert {entry["name"] for entry in metrics["per_class"]} == {"tylenol", "aspirin"}
    assert metrics["analysis"]["score_threshold"] == 0.5
    assert set(metrics["analysis"]["by_iou"]) == {"0.50", "0.75"}
    assert metrics["metrics"]["precision50"] == pytest.approx(1.0)
    assert metrics["metrics"]["recall50"] == pytest.approx(1.0)


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


def test_competition_run_reuses_one_checkpoint_and_writes_submission(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"].pop("predictions_input_uri")
    calls = []

    def fake_predict_groups(
        store, record_groups, *, checkpoint_uri, device, seed, on_progress=None
    ):
        calls.append([[record["image_id"] for record in records] for records in record_groups])
        return [
            _normalised_validation_predictions(),
            [
                _test_prediction(20, 7, 0.123456789012345, 0.9),
                _test_prediction(10, 7, 2.0, 0.8),
                _test_prediction(10, 3, 1.0, 0.8),
                _test_prediction(10, 7, 3.0, 0.95),
                _test_prediction(10, 3, 4.0, 0.7),
                _test_prediction(10, 3, 5.0, 0.6),
                _test_prediction(20, 3, 6.0, 0.0),
            ],
        ]

    monkeypatch.setattr(pipeline, "predict_record_groups_with_checkpoint", fake_predict_groups)

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    assert calls == [
        [
            ["img-1", "img-2"],
            [20, 10],
        ]
    ]
    assert set(result["artifacts"]) == ARTIFACT_KEYS | {
        "submission_uri",
        "test_predictions_uri",
    }
    assert result["artifacts"]["submission_uri"] == (
        "submissions/evaluate-0001/submission.csv"
    )
    rows = (repository_root / result["artifacts"]["submission_uri"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert rows == [
        "annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score",
        "1,10,7,3.0,2.0,3.0,4.0,0.95",
        "2,10,3,1.0,2.0,3.0,4.0,0.8",
        "3,10,7,2.0,2.0,3.0,4.0,0.8",
        "4,10,3,4.0,2.0,3.0,4.0,0.7",
        "5,20,7,0.123456789012345,2.0,3.0,4.0,0.9",
        "6,20,3,6.0,2.0,3.0,4.0,0.0",
    ]
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])
    assert metrics["iou_thresholds"] == COMPETITION_THRESHOLDS
    # mAP@0.50은 보조 지표이므로 대회 실행에서도 항상 계산합니다.
    # (이전에는 threshold 목록에 0.5가 없어 null이었습니다.)
    assert metrics["metrics"]["mAP50"] is not None
    assert result["summary"]["score_threshold"] == 0.0
    assert result["summary"]["max_detections_per_image"] == 4
    assert "test_metrics" not in result["summary"]


def test_competition_run_saves_the_submitted_test_predictions(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """제출한 것과 똑같은 test 예측을 파일로 남깁니다.

    CSV는 사람이 올리는 형식이라 앙상블처럼 예측을 다시 읽어 쓰는 쪽에서는 상자를
    문자열로 되돌려 읽어야 합니다. 같은 내용을 JSON으로도 남겨 그 왕복을 없앱니다.
    """
    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"].pop("predictions_input_uri")

    def fake_predict_groups(
        store, record_groups, *, checkpoint_uri, device, seed, on_progress=None
    ):
        return [
            _normalised_validation_predictions(),
            [
                _test_prediction(10, 7, 1.0, 0.95),
                _test_prediction(10, 3, 2.0, 0.80),
                _test_prediction(10, 7, 3.0, 0.70),
                _test_prediction(10, 3, 4.0, 0.60),
                # 이미지당 상한(4) 밖이라 제출에 없고, 이 파일에도 없어야 합니다.
                _test_prediction(10, 3, 5.0, 0.50),
                _test_prediction(20, 7, 6.0, 0.90),
            ],
        ]

    monkeypatch.setattr(pipeline, "predict_record_groups_with_checkpoint", fake_predict_groups)

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    # artifact key 자체는 위 계약 test가 지킵니다. 여기서는 자리와 내용만 봅니다.
    assert result["artifacts"]["test_predictions_uri"] == (
        "artifacts/evaluate/evaluate-0001/test_predictions.json"
    )

    document = _read_json(repository_root, result["artifacts"]["test_predictions_uri"])
    assert document["bbox_format"] == "xywh"
    # 어느 test set을 본 예측인지 없으면 다른 dataset 판끼리 섞여도 모릅니다.
    assert document["test_manifest_uri"] == "data/test/instances.json"
    # 뺀 class가 있는지 없는지를 적어 두지 않으면, 읽는 쪽이 빠진 class를 모델이 못
    # 맞힌 것으로 읽습니다.
    assert document["submission_excluded_category_ids"] == []
    assert document["prediction_count"] == 5

    # 상자 네 값을 전부 담고 순서까지 세어 비교합니다. 일부만 보면 y·너비·높이가
    # 망가지거나 같은 줄이 두 번 들어가도 지나갑니다.
    saved = sorted(
        (item["image_id"], item["category_id"], *item["bbox"], item["score"])
        for item in document["predictions"]
    )
    assert saved == sorted(
        [
            (10, 7, 1.0, 2.0, 3.0, 4.0, 0.95),
            (10, 3, 2.0, 2.0, 3.0, 4.0, 0.80),
            (10, 7, 3.0, 2.0, 3.0, 4.0, 0.70),
            (10, 3, 4.0, 2.0, 3.0, 4.0, 0.60),
            (20, 7, 6.0, 2.0, 3.0, 4.0, 0.90),
        ]
    )

    # 제출 CSV와 같은 줄을 담고 있어야 합니다. 둘이 어긋나면 앙상블이 올리지 않은
    # 예측으로 융합하게 됩니다.
    rows = (
        (repository_root / result["artifacts"]["submission_uri"])
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    )
    from_csv = sorted(
        (
            int(parts[1]),
            int(parts[2]),
            *(float(value) for value in parts[3:7]),
            float(parts[7]),
        )
        for parts in (row.split(",") for row in rows)
    )
    assert from_csv == saved


def test_empty_test_predictions_still_leave_a_file(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """예측이 하나도 없어도 파일은 남깁니다.

    제출 CSV는 header만 남는데 JSON만 사라지면, 읽는 쪽이 "평가를 안 돌렸다"와
    "돌렸는데 아무것도 못 찾았다"를 구분할 수 없습니다.
    """
    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"].pop("predictions_input_uri")

    monkeypatch.setattr(
        pipeline,
        "predict_record_groups_with_checkpoint",
        lambda *args, **kwargs: [_normalised_validation_predictions(), []],
    )

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    document = _read_json(repository_root, result["artifacts"]["test_predictions_uri"])
    assert document["predictions"] == []
    assert document["prediction_count"] == 0


def test_existing_test_predictions_stop_before_inference(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    existing = repository_root / "artifacts/evaluate/evaluate-0001/test_predictions.json"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("keep\n", encoding="utf-8", newline="\n")

    def unexpected_inference(*args, **kwargs):
        pytest.fail("preflight 뒤에는 inference가 호출되면 안 됩니다.")

    monkeypatch.setattr(pipeline, "predict_record_groups_with_checkpoint", unexpected_inference)

    result = run(base_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert existing.read_text(encoding="utf-8") == "keep\n"


def test_existing_submission_stops_before_inference(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    submission_path = repository_root / "submissions/evaluate-0001/submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text("keep\n", encoding="utf-8", newline="\n")

    def unexpected_inference(*args, **kwargs):
        pytest.fail("preflight 뒤에는 inference가 호출되면 안 됩니다.")

    monkeypatch.setattr(pipeline, "predict_record_groups_with_checkpoint", unexpected_inference)

    result = run(base_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert submission_path.read_text(encoding="utf-8") == "keep\n"


def test_failed_metrics_write_removes_new_submission(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines.evaluate import pipeline, storage_io

    _add_test_manifest(base_config, repository_root)
    monkeypatch.setattr(
        pipeline,
        "predict_record_groups_with_checkpoint",
        lambda *args, **kwargs: [[]],
    )
    original_write_json = storage_io.ArtifactStore.write_json

    def fail_metrics(self, uri, value, *, overwrite=False):
        if uri.endswith("metrics.json"):
            raise storage_io.ArtifactWriteError("의도한 저장 실패")
        return original_write_json(self, uri, value, overwrite=overwrite)

    monkeypatch.setattr(storage_io.ArtifactStore, "write_json", fail_metrics)

    result = run(base_config)

    assert result["status"] == "error"
    assert not (repository_root / "submissions/evaluate-0001/submission.csv").exists()
    assert not (repository_root / "artifacts/evaluate/evaluate-0001/predictions.json").exists()
    assert not (
        repository_root / "artifacts/evaluate/evaluate-0001/test_predictions.json"
    ).exists()


def test_test_manifest_requires_checkpoint_even_with_validation_predictions(
    base_config: dict, repository_root: Path
):
    _add_test_manifest(base_config, repository_root)
    base_config["inputs"]["train"].pop("best_checkpoint_uri")

    result = run(base_config)

    assert result["status"] == "error"
    assert "test image 추론에는 checkpoint" in result["message"]


def test_competition_thresholds_cannot_drift(
    base_config: dict, repository_root: Path
):
    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"]["iou_thresholds"] = [0.5]

    result = run(base_config)

    assert result["status"] == "error"
    assert "competition iou_thresholds" in result["message"]


def test_submission_path_outside_repository_is_rejected(
    base_config: dict, repository_root: Path
):
    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"]["submission_uri"] = "../outside/submission.csv"

    result = run(base_config)

    assert result["status"] == "error"
    assert "저장소 root 밖의 local 경로" in result["message"]
    assert not (repository_root.parent / "outside/submission.csv").exists()


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


def test_successful_run_accepts_coco_validation_manifest(
    base_config: dict, repository_root: Path
):
    write_json(
        repository_root / "data/val/manifest.jsonl",
        {
            "images": [
                {"id": 1, "file_name": "img-1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "img-2.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20]},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [30, 30, 10, 10]},
            ],
            "categories": [
                {"id": 1, "name": "tylenol"},
                {"id": 2, "name": "aspirin"},
            ],
        },
    )
    write_json(
        repository_root / "data/val/predictions.json",
        [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.95},
            {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.90},
            {"image_id": 2, "category_id": 1, "bbox": [30, 30, 10, 10], "score": 0.80},
        ],
    )

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    assert set(result["artifacts"]) == ARTIFACT_KEYS
    assert result["summary"]["image_count"] == 2
    assert result["summary"]["annotation_count"] == 3
    assert result["summary"]["metrics"]["mAP"] == pytest.approx(1.0)


def test_coco_manifest_with_unknown_reference_returns_error_without_artifacts(
    base_config: dict, repository_root: Path
):
    write_json(
        repository_root / "data/val/manifest.jsonl",
        {
            "images": [
                {"id": 1, "file_name": "img-1.jpg", "width": 100, "height": 100}
            ],
            "annotations": [
                {"id": 1, "image_id": 999, "category_id": 1, "bbox": [10, 10, 20, 20]}
            ],
            "categories": [{"id": 1, "name": "tylenol"}],
        },
    )

    result = run(base_config)

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert "존재하지 않는 image_id" in result["message"]
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
    # ground truth가 하나도 없으면 평균 낼 대상이 없으므로 0.0이 아니라 null입니다.
    assert metrics["metrics"]["mAP"] is None
    assert metrics["evaluated_class_count"] == 0


def test_excluded_categories_leave_the_submission_but_stay_in_validation(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """제외 목록은 제출 CSV에만 적용하고 validation 예측은 건드리지 않습니다.

    `기타 알약`처럼 대회에 없는 보조 class는 채점될 수 없으므로 제출에서 빼야 하지만,
    학습에 쓰는 실제 class이므로 validation 지표에는 남아 있어야 합니다.
    """

    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"].pop("predictions_input_uri")
    # 1은 validation이 예측하는 category, 3은 test가 예측하는 category입니다.
    base_config["evaluate"]["submission_excluded_category_ids"] = [1, 3]

    def fake_predict_groups(
        store, record_groups, *, checkpoint_uri, device, seed, on_progress=None
    ):
        return [
            _normalised_validation_predictions(),
            [
                _test_prediction(10, 7, 3.0, 0.95),
                _test_prediction(10, 3, 1.0, 0.80),
                _test_prediction(20, 3, 6.0, 0.70),
                _test_prediction(20, 7, 0.5, 0.60),
            ],
        ]

    monkeypatch.setattr(pipeline, "predict_record_groups_with_checkpoint", fake_predict_groups)

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    rows = (
        (repository_root / result["artifacts"]["submission_uri"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    # category 3은 모두 빠지고 남은 행의 annotation_id는 1부터 끊기지 않습니다.
    assert rows[1:] == [
        "1,10,7,3.0,2.0,3.0,4.0,0.95",
        "2,20,7,0.5,2.0,3.0,4.0,0.6",
    ]
    assert not any(",3," in row for row in rows[1:])

    # validation 경로는 그대로입니다. category 1을 뺐지만 예측 3개가 모두 남습니다.
    document = _read_json(repository_root, result["artifacts"]["predictions_uri"])
    assert document["prediction_count"] == 3
    assert 1 in {entry["category_id"] for entry in document["predictions"]}
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])
    assert metrics["metrics"]["mAP"] is not None
    # 무엇을 뺐는지 실행 기록에 남습니다. JSON으로 직렬화할 수 있는 list여야 합니다.
    assert result["summary"]["submission_excluded_category_ids"] == [1, 3]

    # test 예측 파일도 제출과 같아야 합니다. 뺀 class가 여기 다시 들어오면 앙상블이
    # 올리지 않은 예측으로 융합하고, 뺀 목록을 안 적으면 읽는 쪽이 그 class를 모델이
    # 못 맞힌 것으로 읽습니다.
    test_predictions = _read_json(
        repository_root, result["artifacts"]["test_predictions_uri"]
    )
    assert test_predictions["submission_excluded_category_ids"] == [1, 3]
    assert {entry["category_id"] for entry in test_predictions["predictions"]} == {7}
    assert test_predictions["prediction_count"] == 2
    assert validate_pipeline_result(result, pipeline_name="evaluate") is result


def test_summary_stays_unchanged_when_no_category_is_excluded(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """설정하지 않은 실행의 summary에는 제외 key가 생기지 않습니다."""

    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"].pop("predictions_input_uri")

    def fake_predict_groups(
        store, record_groups, *, checkpoint_uri, device, seed, on_progress=None
    ):
        return [_normalised_validation_predictions(), [_test_prediction(10, 7, 3.0, 0.95)]]

    monkeypatch.setattr(pipeline, "predict_record_groups_with_checkpoint", fake_predict_groups)

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    assert "submission_excluded_category_ids" not in result["summary"]


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
        ({"submission_excluded_category_ids": "3"}, "submission_excluded_category_ids"),
        ({"submission_excluded_category_ids": [-1]}, "submission_excluded_category_ids"),
        # list가 아닌 값은 순회하다 TypeError가 run() 밖으로 새지 않도록 먼저 막습니다.
        ({"submission_excluded_category_ids": 3}, "submission_excluded_category_ids"),
        ({"submission_excluded_category_ids": {"3": 1}}, "submission_excluded_category_ids"),
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


def test_same_filename_for_both_artifacts_is_rejected(base_config: dict, repository_root: Path):
    """한 산출물이 다른 산출물을 덮어쓰지 않도록 실행 전에 막습니다."""
    base_config["evaluate"]["metrics_filename"] = "result.json"
    base_config["evaluate"]["predictions_filename"] = "result.json"

    result = run(base_config)

    assert result["status"] == "error"
    assert "같은 위치에 저장할 수 없습니다" in result["message"]
    assert not (repository_root / "artifacts/evaluate/evaluate-0001").exists()


def test_failed_metrics_write_keeps_pre_existing_predictions(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """overwrite 실행이 중간에 실패해도 실행 전부터 있던 file은 지우지 않습니다."""
    from src.pipelines.evaluate import storage_io

    predictions_path = repository_root / "artifacts/evaluate/evaluate-0001/predictions.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text('{"from": "previous run"}\n', encoding="utf-8", newline="\n")
    base_config["evaluate"]["overwrite"] = True

    original_write_json = storage_io.ArtifactStore.write_json

    def write_json(self, uri, value, *, overwrite=False):
        if uri.endswith("metrics.json"):
            raise storage_io.ArtifactWriteError("의도한 저장 실패")
        return original_write_json(self, uri, value, overwrite=overwrite)

    monkeypatch.setattr(storage_io.ArtifactStore, "write_json", write_json)

    result = run(base_config)

    assert result["status"] == "error"
    assert result["message"] == "의도한 저장 실패"
    assert predictions_path.is_file()


def test_failed_metrics_write_removes_newly_created_predictions(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines.evaluate import storage_io

    original_write_json = storage_io.ArtifactStore.write_json

    def write_json(self, uri, value, *, overwrite=False):
        if uri.endswith("metrics.json"):
            raise storage_io.ArtifactWriteError("의도한 저장 실패")
        return original_write_json(self, uri, value, overwrite=overwrite)

    monkeypatch.setattr(storage_io.ArtifactStore, "write_json", write_json)

    result = run(base_config)

    assert result["status"] == "error"
    assert not (repository_root / "artifacts/evaluate/evaluate-0001/predictions.json").exists()


def test_saved_predictions_reproduce_the_same_metrics(
    base_config: dict, repository_root: Path
):
    """저장된 predictions로 다시 평가해도 같은 metric이 나와야 합니다."""
    write_json(
        repository_root / "data/val/predictions.json",
        [
            {
                "image_id": "img-1",
                "category_id": 1,
                "bbox": [10.000123456789, 10.98765432101, 20.111111111, 20.999999999],
                "score": 0.123456789012345,
            },
            {"image_id": "img-2", "category_id": 1, "bbox": [30, 30, 10, 10], "score": 0.8},
        ],
    )
    first = run(base_config)
    assert first["status"] == "ok", first["message"]

    rerun_config = copy.deepcopy(base_config)
    rerun_config["evaluate"]["run_id"] = "evaluate-0002"
    rerun_config["evaluate"]["predictions_input_uri"] = first["artifacts"]["predictions_uri"]
    second = run(rerun_config)

    assert second["status"] == "ok", second["message"]
    assert second["summary"]["metrics"] == first["summary"]["metrics"]


def test_repository_outside_paths_are_rejected(base_config: dict, repository_root: Path):
    base_config["evaluate"]["validation_manifest_uri"] = "../outside/manifest.jsonl"

    result = run(base_config)

    assert result["status"] == "error"
    assert "저장소 root 밖의 local 경로" in result["message"]


def test_output_dir_outside_the_repository_is_rejected(base_config: dict, repository_root: Path):
    base_config["evaluate"]["output_dir"] = "../outside/evaluate"

    result = run(base_config)

    assert result["status"] == "error"
    assert "저장소 root 밖의 local 경로" in result["message"]
    assert not (repository_root.parent / "outside").exists()


def test_custom_iou_thresholds_are_rejected(base_config: dict, repository_root: Path):
    """메인 지표 구간은 고정입니다.

    설정 key가 있는데 조용히 무시되면 다른 기준으로 잰 값을 메인 지표로 읽게
    되므로, 다른 값이 오면 명시적으로 거절합니다.
    """
    base_config["evaluate"]["iou_thresholds"] = [0.6, 0.7]

    result = run(base_config)

    assert result["status"] == "error"
    assert "iou_thresholds" in result["message"]
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


def test_metrics_exclusion_drops_a_category_from_validation_scoring(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """채점되지 않는 class를 validation 지표에서도 뺄 수 있어야 합니다.

    `기타 알약`은 제출 CSV에서 빠지므로 대회 점수에 들어가지 않는데, 지금은 로컬
    mAP에만 포함돼 두 숫자가 서로 다른 class 집합을 평균합니다. 그러면 로컬이
    올랐는지 내렸는지로 대회 점수를 짐작할 수 없습니다.
    """

    from src.pipelines.evaluate import pipeline

    base_config["evaluate"]["metrics_excluded_category_ids"] = [1]

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])
    # 뺀 class는 per_class에도 지표에도 남지 않습니다.
    assert 1 not in {entry["category_id"] for entry in metrics["per_class"]}
    assert metrics["metrics_excluded_category_ids"] == [1]
    assert result["summary"]["metrics_excluded_category_ids"] == [1]
    assert validate_pipeline_result(result, pipeline_name="evaluate") is result


def test_metrics_exclusion_keeps_the_predictions_file_whole(
    base_config: dict, repository_root: Path
):
    """지표에서 뺀다고 예측까지 지우면 안 됩니다.

    저장된 예측은 다시 채점할 수 있어야 하는 원본입니다. 여기서 지우면 나중에 다른
    class 집합으로 재계산할 수 없습니다.
    """

    base_config["evaluate"]["metrics_excluded_category_ids"] = [1]

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    document = _read_json(repository_root, result["artifacts"]["predictions_uri"])
    assert 1 in {entry["category_id"] for entry in document["predictions"]}


def test_summary_has_no_metrics_exclusion_key_when_unset(
    base_config: dict, repository_root: Path
):
    """설정하지 않은 실행의 결과 모양은 지금과 완전히 같아야 합니다."""

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    assert "metrics_excluded_category_ids" not in result["summary"]
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])
    assert "metrics_excluded_category_ids" not in metrics


def test_metrics_exclusion_applies_before_the_per_image_cap(
    base_config: dict, repository_root: Path
):
    """제외를 상한 뒤에 적용하면 채점될 예측이 상한 밖에서 이미 버려집니다.

    한 이미지에 제외 class의 고득점 예측이 상한 안을 차지하고 채점 대상 예측이 그
    뒤로 밀려 있으면, 상한을 먼저 적용한 뒤 제외하는 순서에서는 그 예측이 사라집니다.
    제출 CSV는 제외를 상한보다 먼저 적용하므로(`006`) 로컬 지표가 대회가 실제로
    채점하는 목록과 달라집니다.
    """

    # img-1의 정답 두 개 중 category 2는 상한 밖으로 밀려 있습니다. 제외 class 1이
    # 앞자리 세 개를 차지하기 때문입니다.
    crowded = [
        {"image_id": "img-1", "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": "img-1", "category_id": 1, "bbox": [11, 11, 20, 20], "score": 0.98},
        {"image_id": "img-1", "category_id": 1, "bbox": [12, 12, 20, 20], "score": 0.97},
        {"image_id": "img-1", "category_id": 1, "bbox": [13, 13, 20, 20], "score": 0.96},
        {"image_id": "img-1", "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.10},
        {"image_id": "img-2", "category_id": 1, "bbox": [30, 30, 10, 10], "score": 0.80},
    ]
    write_json(repository_root / "data/val/crowded.json", crowded)
    base_config["evaluate"]["predictions_input_uri"] = "data/val/crowded.json"
    base_config["evaluate"]["metrics_excluded_category_ids"] = [1]

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    metrics = _read_json(repository_root, result["artifacts"]["metrics_uri"])
    scored = {entry["category_id"] for entry in metrics["per_class"]}
    # category 2를 채점할 수 있어야 합니다. 상한을 먼저 적용했다면 예측이 없어
    # per_class에서 사라지거나 AP가 0이 됩니다.
    assert scored == {2}
    [entry] = metrics["per_class"]
    assert entry["prediction_count"] == 1
    assert entry["ap"] is not None and entry["ap"] > 0.0


@pytest.mark.parametrize(
    "metrics_filename",
    [
        pytest.param("test_predictions.json", id="같은-이름"),
        pytest.param("./test_predictions.json", id="표기만-다른-같은-파일"),
    ],
)
def test_test_predictions_cannot_share_a_file_with_another_output(
    base_config: dict, repository_root: Path, metrics_filename: str
):
    """새 출력이 다른 산출물과 같은 파일이면 실행 전에 막습니다.

    출력이 하나 늘면 겹칠 수 있는 짝도 늘어납니다. 이름 자체는 설정으로 열지 않지만,
    다른 출력의 이름을 이쪽으로 맞추면 여전히 겹칩니다. `overwrite=true`면 덮고도
    성공을 보고하므로 실행 전에 거절해야 합니다.
    """
    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"]["metrics_filename"] = metrics_filename
    base_config["evaluate"]["overwrite"] = True

    result = run(base_config)

    assert result["status"] == "error"
    assert "같은 위치에 저장할 수 없습니다" in result["message"]


def test_test_predictions_record_the_checkpoint_as_their_source(
    base_config: dict, repository_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """validation이 저장된 예측을 읽어도 test 예측의 출처는 checkpoint입니다.

    실행 전체의 `prediction_source`를 그대로 베끼면, 이 파일이 "저장된 예측에서
    왔다"고 잘못 말합니다. 나중에 합치는 쪽이 어느 모델이 만든 것인지 잘못 읽습니다.
    """
    from src.pipelines.evaluate import pipeline

    _add_test_manifest(base_config, repository_root)
    # `predictions_input_uri`를 그대로 둡니다 — validation은 파일에서 읽고 test만
    # 추론합니다. 대회 실행의 정상 경로입니다.
    monkeypatch.setattr(
        pipeline,
        "predict_record_groups_with_checkpoint",
        lambda *args, **kwargs: [[_test_prediction(10, 7, 1.0, 0.9)]],
    )

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    # 실행 전체는 저장된 예측을 읽었다고 말합니다.
    assert result["summary"]["prediction_source"] == "predictions_file"
    # 그러나 test 예측 파일은 checkpoint에서 왔습니다.
    document = _read_json(repository_root, result["artifacts"]["test_predictions_uri"])
    assert document["prediction_source"] == "checkpoint"
    assert document["predictions_input_uri"] is None


def test_outputs_that_are_hard_links_of_each_other_are_rejected(
    base_config: dict, repository_root: Path
):
    """이미 있는 두 출력이 서로 hard link면 겹친 것으로 봅니다.

    덮어쓰기를 켠 실행은 이미 있는 파일에 씁니다. 표기로만 판정하면 이름이 달라
    통과한 뒤 차례로 덮고도 성공을 보고합니다. 저장 계층에 물어봐야 걸립니다.
    """
    import os

    _add_test_manifest(base_config, repository_root)
    base_config["evaluate"]["overwrite"] = True

    output_dir = repository_root / "artifacts/evaluate/evaluate-0001"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8", newline="\n")
    try:
        os.link(output_dir / "metrics.json", output_dir / "test_predictions.json")
    except (OSError, NotImplementedError, AttributeError) as error:
        pytest.skip(f"이 filesystem은 hard link를 만들 수 없습니다: {error}")

    result = run(base_config)

    assert result["status"] == "error"
    assert "같은 위치에 저장할 수 없습니다" in result["message"]


def test_fusing_saved_predictions_makes_a_submission_without_a_checkpoint(
    base_config: dict, repository_root: Path
):
    """합칠 예측을 주면 추론 없이 제출을 만듭니다.

    앙상블의 목적입니다. 이미 점수가 달린 체크포인트들의 예측을 다시 읽어 합치는
    것이라 재학습도 재추론도 필요 없습니다.
    """
    _add_test_manifest(base_config, repository_root)
    base_config["inputs"]["train"].pop("best_checkpoint_uri")

    for name, bbox_x in (("a", 3.0), ("b", 3.4)):
        write_json(
            repository_root / f"data/test/{name}.json",
            {
                "test_manifest_uri": "data/test/instances.json",
                "checkpoint_uri": f"checkpoints/{name}.pt",
                "predictions": [
                    {"image_id": 10, "category_id": 7, "bbox": [bbox_x, 2.0, 3.0, 4.0], "score": 0.9},
                    {"image_id": 20, "category_id": 3, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.5},
                ],
            },
        )
    base_config["evaluate"]["test_predictions_input_uris"] = [
        "data/test/a.json",
        "data/test/b.json",
    ]

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    rows = (
        (repository_root / result["artifacts"]["submission_uri"])
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    )
    # 두 실행이 거의 같은 자리를 가리켰으므로 image 10은 한 줄로 합쳐집니다.
    assert len(rows) == 2
    fused = [row for row in rows if row.split(",")[1] == "10"][0].split(",")
    # 상자는 두 값 사이에 놓입니다.
    assert 3.0 < float(fused[3]) < 3.4
    # 둘 다 찾았으므로 확신도가 그대로입니다.
    assert float(fused[7]) == pytest.approx(0.9)


def test_fusion_refuses_predictions_from_another_test_set(
    base_config: dict, repository_root: Path
):
    """다른 시험지를 본 예측을 섞지 않습니다.

    image_id만 보고 섞으면 조용히 틀린 제출이 나옵니다.
    """
    _add_test_manifest(base_config, repository_root)
    write_json(
        repository_root / "data/test/other.json",
        {
            "test_manifest_uri": "data/test/instances.json",
            "checkpoint_uri": "checkpoints/other.pt",
            "predictions": [
                {"image_id": 999, "category_id": 7, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9}
            ],
        },
    )
    base_config["evaluate"]["test_predictions_input_uris"] = ["data/test/other.json"]

    result = run(base_config)

    assert result["status"] == "error"
    assert "manifest에 없는 image_id" in result["message"]


def _write_fusion_input(
    repository_root: Path,
    name: str,
    image_ids: list[int],
    *,
    checkpoint: str | None = None,
    manifest_uri: str = "data/test/instances.json",
) -> str:
    """수집한 `test_predictions.json`과 같은 모양의 입력을 만듭니다.

    `test_manifest_uri`와 `checkpoint_uri`는 합쳐도 되는지 판정하는 데 쓰이므로
    실제 파일처럼 담아 둡니다.
    """
    write_json(
        repository_root / f"data/test/{name}.json",
        {
            "test_manifest_uri": manifest_uri,
            "checkpoint_uri": checkpoint or f"checkpoints/{name}.pt",
            "run_id": f"harvest-{name}",
            "predictions": [
                {"image_id": image_id, "category_id": 7,
                 "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9}
                for image_id in image_ids
            ],
        },
    )
    return f"data/test/{name}.json"


@pytest.mark.parametrize(
    "second",
    [
        pytest.param("same-path", id="같은-파일을-두-번"),
        pytest.param("copy", id="같은-실행을-다른-경로로"),
    ],
)
def test_the_same_run_cannot_be_fused_with_itself(
    base_config: dict, repository_root: Path, second: str
):
    """같은 실행의 예측을 두 번 주면 막습니다.

    그대로 두면 서로 다른 실행으로 세어 자기 자신과 동의하고, 확신도가 근거 없이
    올라갑니다. 저장 위치만 보면 **복사본**을 갈라내지 못하므로 그 예측을 만든
    checkpoint로 봅니다.
    """
    _add_test_manifest(base_config, repository_root)
    first = _write_fusion_input(repository_root, "one", [10, 20], checkpoint="ckpt/a.pt")
    other = (
        f"./{first}"
        if second == "same-path"
        else _write_fusion_input(repository_root, "copy", [10, 20], checkpoint="ckpt/a.pt")
    )
    base_config["evaluate"]["test_predictions_input_uris"] = [first, other]

    result = run(base_config)

    assert result["status"] == "error"
    assert "같은 실행의 예측이 두 번" in result["message"]


def test_fusion_refuses_a_prediction_from_another_test_set(
    base_config: dict, repository_root: Path
):
    """다른 시험지를 본 예측은 합치지 않습니다.

    예측에 나온 id만 보면 안 됩니다. 같은 id를 쓰는 다른 시험지가 그대로 지나갑니다.
    각 파일이 적어 둔 manifest를 실제로 읽어 견줍니다.
    """
    _add_test_manifest(base_config, repository_root)
    write_json(
        repository_root / "data/test/other_instances.json",
        {
            "images": [{"id": 10, "file_name": "10.png", "width": 100, "height": 100}],
            "annotations": [],
            "categories": [{"id": 7, "name": "pill-b"}],
        },
    )
    base_config["evaluate"]["test_predictions_input_uris"] = [
        _write_fusion_input(repository_root, "ours", [10, 20]),
        _write_fusion_input(
            repository_root, "theirs", [10], manifest_uri="data/test/other_instances.json"
        ),
    ]

    result = run(base_config)

    assert result["status"] == "error"
    assert "다른 시험지" in result["message"]


def test_fusion_accepts_a_run_that_found_nothing_in_some_images(
    base_config: dict, repository_root: Path
):
    """어떤 이미지에서 아무것도 못 찾은 실행도 합칩니다.

    검출이 0건인 이미지는 예측 목록에 없는 것이 **정상**입니다. 예측에 나온 id로
    시험지를 판정하면 멀쩡한 입력을 거절합니다.
    """
    _add_test_manifest(base_config, repository_root)
    base_config["inputs"]["train"].pop("best_checkpoint_uri")
    base_config["evaluate"]["test_predictions_input_uris"] = [
        _write_fusion_input(repository_root, "both", [10, 20]),
        _write_fusion_input(repository_root, "one_only", [10]),
    ]

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    rows = (
        (repository_root / result["artifacts"]["submission_uri"])
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    )
    assert {row.split(",")[1] for row in rows} == {"10", "20"}


def test_a_fused_file_records_what_it_was_fused_from(
    base_config: dict, repository_root: Path
):
    """합친 결과는 무엇을 합쳤는지 남깁니다.

    `checkpoint`라고 적으면 checkpoint 없이 만든 실행과 모순되고, 어느 실행들의
    앙상블인지 되짚을 수 없습니다.
    """
    _add_test_manifest(base_config, repository_root)
    base_config["inputs"]["train"].pop("best_checkpoint_uri")
    uris = [
        _write_fusion_input(repository_root, "left", [10, 20]),
        _write_fusion_input(repository_root, "right", [10, 20]),
    ]
    base_config["evaluate"]["test_predictions_input_uris"] = uris

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    document = _read_json(repository_root, result["artifacts"]["test_predictions_uri"])
    assert document["prediction_source"] == "fusion"
    # 어느 실행을 합쳤는지 되짚을 수 있어야 합니다. 경로는 덮어써질 수 있으므로
    # 그 예측을 만든 checkpoint까지 남깁니다.
    assert [entry["uri"] for entry in document["fused_from"]] == uris
    assert [entry["checkpoint_uri"] for entry in document["fused_from"]] == [
        "checkpoints/left.pt",
        "checkpoints/right.pt",
    ]
    assert document["fusion_iou_threshold"] is not None


def test_fusion_refuses_a_prediction_that_does_not_say_which_test_set_it_saw(
    base_config: dict, repository_root: Path
):
    """어느 시험지를 봤는지 안 적힌 예측은 합치지 않습니다.

    `test_predictions_uri`가 생기기 전에 만든 파일이 그렇습니다. 확인할 수 없는 것을
    통과시키면 시험지 검사가 있으나 마나입니다.
    """
    _add_test_manifest(base_config, repository_root)
    write_json(
        repository_root / "data/test/nameless.json",
        {
            "checkpoint_uri": "checkpoints/nameless.pt",
            "predictions": [
                {"image_id": 10, "category_id": 7, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9}
            ],
        },
    )
    base_config["evaluate"]["test_predictions_input_uris"] = ["data/test/nameless.json"]

    result = run(base_config)

    assert result["status"] == "error"
    assert "어느 test manifest를 본 예측인지" in result["message"]
