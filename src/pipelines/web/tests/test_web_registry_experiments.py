"""Registry index 기반 Web 실험 목록과 선택 비교 경로."""

from __future__ import annotations

import pytest

from src.common import ExperimentNameError
from src.pipelines.web import experiment_detail, experiments
from src.pipelines.web.errors import JobNotFoundError


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


def use_index(monkeypatch, summaries: list[dict]) -> None:
    """registry index에 이 실험들이 있는 상태로 둡니다.

    목록은 prefix 아래를 훑고, 이름을 아는 조회(비교·상세·점수 기록)는 그 이름의
    항목 하나만 읽습니다. 두 길이 같은 index를 보게 해야 한쪽만 바꿔 놓고 통과하는
    test가 생기지 않습니다.
    """

    by_run = {item["run_id"]: item for item in summaries}
    monkeypatch.setattr(
        experiments, "list_experiment_summaries", lambda config: list(summaries)
    )
    monkeypatch.setattr(
        experiments,
        "read_experiment_summary",
        lambda run_id, config: by_run.get(run_id),
    )


def refuse_listing(monkeypatch) -> None:
    """index 전체를 훑으면 실패하게 둡니다.

    이름을 아는 조회가 다시 목록으로 돌아가도 `use_index`는 그것까지 성공시키므로,
    이 PR의 핵심인 "전체 index를 훑지 않는다"가 test에서 보호되지 않습니다. 두 이름을
    함께 막는 것은 web이 부르는 이름과 공용 registry 안에서 다시 부르는 이름이 다른
    객체이기 때문입니다.
    """

    def boom(config, **_kwargs):
        raise AssertionError("이름을 아는 조회는 index 전체를 훑으면 안 됩니다.")

    monkeypatch.setattr(experiments, "list_experiment_summaries", boom)
    monkeypatch.setattr("src.common.experiment_registry.list_experiment_summaries", boom)


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
    read = []

    def refuse_listing(config, **_kwargs):
        raise AssertionError("비교는 index 전체를 훑으면 안 됩니다.")

    # 두 이름을 함께 막아야 합니다. web이 부르는 이름과, 공용 registry 안에서 다시
    # 부르는 이름이 다른 객체이기 때문입니다.
    monkeypatch.setattr(experiments, "list_experiment_summaries", refuse_listing)
    monkeypatch.setattr(
        "src.common.experiment_registry.list_experiment_summaries", refuse_listing
    )
    monkeypatch.setattr(
        experiments,
        "read_experiment_summary",
        lambda run_id, config: registry_summary(run_id),
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
    assert "experiment_record_uri" not in response.text


def test_compare_uses_sgd_for_legacy_record_without_optimizer(client, monkeypatch):
    summary = registry_summary("legacy")
    legacy = experiment_record("legacy")
    legacy["config_snapshot"]["train"] = {}
    use_index(monkeypatch, [summary])
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
    use_index(monkeypatch, [summary])
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
        "gradient_accumulation_steps": None,
        "input_size": None,
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
        "gradient_accumulation_steps": None,
        "input_size": None,
        "seed": 42,
    }


def test_experiment_list_exports_every_metric_and_loss(client, monkeypatch):
    """새 index summary가 채워 준 지표 5개와 loss 4개를 그대로 내보냅니다."""

    summary = registry_summary("full")
    summary["metrics"] = {
        "mAP": 0.31,
        "mAP50": 0.55,
        "mAP75": 0.28,
        "precision50": 0.61,
        "recall50": 0.47,
    }
    summary["losses"] = {
        "best_epoch": 3,
        "best_validation_loss": 0.42,
        "final_train_loss": 0.31,
        "final_validation_loss": 0.45,
    }
    summary["losses_source"] = "training_history"
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])

    listed = client.get("/api/train/experiments").json()["experiments"][0]

    assert listed["metrics"] == {
        "best_epoch": 3,
        "best_validation_loss": 0.42,
        "final_train_loss": 0.31,
        "final_validation_loss": 0.45,
        "map": 0.31,
        "map50": 0.55,
        "map75": 0.28,
        "precision50": 0.61,
        "recall50": 0.47,
        "kaggle_score": None,
    }


def test_experiment_metrics_stay_empty_without_losses_block(client, monkeypatch):
    """losses가 없는 옛 summary는 값을 지어내지 않고 목록과 비교가 똑같이 비웁니다."""

    summary = registry_summary("old-index")
    assert "losses" not in summary
    record = experiment_record("old-index")
    use_index(monkeypatch, [summary])
    monkeypatch.setattr(experiments, "read_experiment_record", lambda *a, **k: record)

    listed = client.get("/api/train/experiments").json()["experiments"][0]
    compared = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["old-index"]}
    ).json()["experiments"][0]

    assert listed["metrics"] == compared["metrics"]
    for key in (
        "best_epoch",
        "best_validation_loss",
        "final_train_loss",
        "final_validation_loss",
        "map75",
        "precision50",
        "recall50",
    ):
        assert listed["metrics"][key] is None
    # 옛 summary에도 있던 지표는 그대로 남습니다.
    assert listed["metrics"]["map"] == 0.31


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
    use_index(monkeypatch, [summary])
    monkeypatch.setattr(experiments, "read_experiment_record", lambda *a, **k: record)

    listed = client.get("/api/train/experiments").json()["experiments"][0]
    compared = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["legacy"]}
    ).json()["experiments"][0]

    assert listed["model"]["source"] == "legacy_fallback"
    assert listed["optimizer"]["name"] == "SGD"
    for block in ("model", "optimizer", "training"):
        assert listed[block] == compared[block]


# --- 완료 단계와 저장소 범위 ---------------------------------------------------


def submitted_summary(run_id: str) -> dict:
    """평가와 제출까지 끝난 실험 하나입니다."""

    summary = registry_summary(run_id)
    summary["artifacts"]["submission_uri"] = f"artifacts/evaluate/{run_id}/submission.csv"
    summary["submission_check"] = {
        "checked": True,
        "row_count": 2942,
        "image_count": 500,
        "max_detections_per_image": 100,
        "skipped_reason": None,
    }
    return summary


def test_experiment_list_reports_how_far_each_run_got(client, monkeypatch):
    """CSV 생성은 실제 Kaggle 제출 완료로 오해하지 않습니다."""

    monkeypatch.setattr(
        experiments,
        "list_experiment_summaries",
        lambda config: [submitted_summary("done"), registry_summary("trained-only")],
    )

    payload = client.get("/api/train/experiments").json()["experiments"]
    by_run = {item["run_id"]: item["completion"] for item in payload}

    assert by_run["done"] == {
        "evaluated": True,
        "submission_generated": True,
        "submitted": False,
        "submission_checked": True,
        "submission_rows": 2942,
    }
    assert by_run["trained-only"]["submission_generated"] is False
    assert by_run["trained-only"]["submitted"] is False
    assert by_run["trained-only"]["submission_rows"] is None


def test_kaggle_score_marks_a_generated_submission_as_actually_submitted(
    client, monkeypatch
):
    """직접 입력한 실제 점수만 Kaggle 제출을 확인하는 증거입니다."""

    use_index(monkeypatch, [submitted_summary("done")])

    saved = client.put(
        "/api/train/experiments/done/kaggle-score", json={"score": 0.8734}
    )
    listed = client.get("/api/train/experiments").json()["experiments"][0]

    assert saved.status_code == 200
    assert saved.json() == {"run_id": "done", "kaggle_score": 0.8734}
    assert listed["metrics"]["kaggle_score"] == 0.8734
    assert listed["completion"]["submission_generated"] is True
    assert listed["completion"]["submitted"] is True


def test_kaggle_score_rejects_values_outside_the_metric_range(client, monkeypatch):
    use_index(monkeypatch, [submitted_summary("done")])

    response = client.put(
        "/api/train/experiments/done/kaggle-score", json={"score": 1.01}
    )

    assert response.status_code == 422


def test_kaggle_score_does_not_overwrite_an_existing_record(client, monkeypatch):
    """수정을 요청하지 않은 저장은 기존 실제 점수를 그대로 둡니다."""

    use_index(monkeypatch, [submitted_summary("done")])

    first = client.put(
        "/api/train/experiments/done/kaggle-score", json={"score": 0.8123}
    )
    second = client.put(
        "/api/train/experiments/done/kaggle-score", json={"score": 0.9999}
    )
    listed = client.get("/api/train/experiments").json()["experiments"][0]

    assert first.status_code == 200
    assert second.status_code == 400
    assert listed["metrics"]["kaggle_score"] == 0.8123


def test_kaggle_score_is_corrected_only_when_the_request_asks_to_overwrite(
    client, monkeypatch
):
    """잘못 적은 점수는 화면에서 수정 버튼을 켠 요청에서만 바뀝니다."""

    use_index(monkeypatch, [submitted_summary("done")])

    client.put("/api/train/experiments/done/kaggle-score", json={"score": 0.8123})
    corrected = client.put(
        "/api/train/experiments/done/kaggle-score",
        json={"score": 0.9012, "overwrite": True},
    )
    listed = client.get("/api/train/experiments").json()["experiments"][0]

    assert corrected.status_code == 200
    assert corrected.json() == {"run_id": "done", "kaggle_score": 0.9012}
    assert listed["metrics"]["kaggle_score"] == 0.9012


def test_experiment_list_says_whether_the_registry_is_shared(client, monkeypatch):
    """local backend면 이 PC 기록만 보입니다. 화면이 그렇게 말할 수 있어야 합니다."""

    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [])

    assert client.get("/api/train/experiments").json()["scope"] == {
        "backend": "local",
        "shared": False,
    }

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "pill-team")
    assert client.get("/api/train/experiments").json()["scope"] == {
        "backend": "s3",
        "shared": True,
    }


def test_experiment_list_names_the_dataset_folder(client, monkeypatch):
    """표에는 100자짜리 URI 대신 팀이 실제로 부르는 폴더 이름을 둡니다."""

    summary = registry_summary("run-1")
    summary["artifacts"]["train_manifest_uri"] = (
        "s3://bucket/datasets/pill_detection/processed/v3-seed42-8020-group/train_manifest.json"
    )
    monkeypatch.setattr(experiments, "list_experiment_summaries", lambda config: [summary])

    dataset = client.get("/api/train/experiments").json()["experiments"][0]["dataset"]

    assert dataset["label"] == "v3-seed42-8020-group"


# --- 실험 하나 상세 -------------------------------------------------------------


def metrics_document() -> dict:
    """evaluate가 쓰는 metrics.json에서 상세 화면이 읽는 부분만 담았습니다."""

    return {
        "image_count": 500,
        "annotation_count": 1200,
        "prediction_count": 1500,
        "evaluated_class_count": 57,
        "max_detections_per_image": 100,
        "metrics": {
            "mAP": 0.9664,
            "mAP50_95": 0.81,
            "mAP75_95": 0.9664,
            "mAP50": 0.9891,
            "mAP75": 0.982,
            "precision50": 0.95,
            "recall50": 0.93,
            "precision75": 0.91,
            "recall75": 0.89,
        },
        "analysis": {
            "score_threshold": 0.5,
            "by_iou": {"0.50": {"tp": 1, "fp": 2, "fn": 3}},
            # 화면에 보내면 안 되는 큰 블록들입니다.
            "confusion_matrix": {"0.50": [[0] * 58 for _ in range(58)]},
            "per_image": {"0.50": [{"image_id": index} for index in range(2100)]},
            "score_sweep": {
                "0.50": {
                    "0.10": {"precision": 0.8, "recall": 0.99, "f1": 0.88},
                    "0.05": {"precision": 0.7, "recall": 1.0, "f1": 0.82},
                    "0.50": {"precision": 0.95, "recall": 0.93, "f1": 0.94},
                }
            },
            "best_f1": {
                "0.50": {"threshold": 0.5, "precision": 0.95, "recall": 0.93, "f1": 0.94}
            },
            "per_class_summary": {
                "min_truth_count": 10,
                "top_n": 5,
                "counts": {"weak": 3, "sparse": 2, "unmeasured": 0},
                "weak": [{"category_id": 1, "name": "약1", "ap": 0.2}],
                "sparse": [],
                "unmeasured": [],
            },
        },
        "per_class": [{"category_id": index} for index in range(57)],
    }


def training_history() -> list[dict]:
    return [
        {"epoch": 2, "train_loss": 0.3, "validation_loss": 0.35, "is_best": False},
        # 이 기능 이전에 학습한 실행에는 learning_rate가 없습니다.
        {"epoch": 1, "train_loss": 0.9, "validation_loss": 0.8, "is_best": False},
        {
            "epoch": 3,
            "train_loss": 0.1,
            "validation_loss": 0.0575,
            "is_best": True,
            "learning_rate": 0.00025,
        },
    ]


def record_with_artifacts(run_id: str) -> dict:
    record = experiment_record(run_id)
    record["pipelines"] = {
        "evaluate": {"metrics_uri": {"uri": f"artifacts/evaluate/{run_id}/metrics.json"}},
        "train": {
            "training_history_uri": {
                "uri": f"artifacts/experiments/{run_id}/training_history.json"
            }
        },
    }
    return record


def test_compare_carries_the_loss_curve_so_the_canvas_asks_once(client, monkeypatch):
    """곡선까지 비교 응답에 함께 옵니다.

    예전에는 화면이 비교를 부른 뒤 고른 실행마다 상세를 또 불렀습니다. 상세는 그때
    마다 index 전체를 훑고, 곡선에 쓰지도 않는 650KB짜리 평가 파일까지 읽었습니다.
    실행 3개를 견주면 그것만 네 번이라 화면이 눈에 띄게 느렸습니다.
    """

    use_index(monkeypatch, [registry_summary("run-1"), registry_summary("run-2")])
    monkeypatch.setattr(
        experiments,
        "read_experiment_record",
        lambda uri, config, *, expected_run_id=None: record_with_artifacts(
            expected_run_id
        ),
    )
    read = []

    def read_document(uri, storage_config):
        read.append(uri)
        return training_history()

    monkeypatch.setattr(experiment_detail, "_read_document", read_document)

    payload = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["run-1", "없는-실험"]}
    ).json()

    curve = payload["curves"]["run-1"]
    assert [item["epoch"] for item in curve["epochs"]] == [1, 2, 3]
    assert curve["available"] is True
    # 등록되지 않은 이름은 곡선도 없고 `missing`으로 알립니다.
    assert payload["missing"] == ["없는-실험"]
    assert "없는-실험" not in payload["curves"]
    # 평가 파일(metrics.json)은 곡선에 쓰이지 않으므로 읽지 않습니다.
    assert read == ["artifacts/experiments/run-1/training_history.json"]


def test_compare_says_why_a_curve_is_missing_instead_of_sending_an_empty_one(
    client, monkeypatch
):
    """학습 기록을 못 읽은 것과 아직 한 epoch도 안 끝난 것은 다릅니다.

    epoch 목록만 보내면 둘 다 빈 배열이라, 화면은 "epoch이 하나도 끝나지 않았다"고
    단정합니다. S3가 잠깐 흔들린 것도 그렇게 읽히므로 이유를 함께 보냅니다.
    """

    use_index(monkeypatch, [registry_summary("run-1")])
    monkeypatch.setattr(
        experiments,
        "read_experiment_record",
        lambda uri, config, *, expected_run_id=None: record_with_artifacts("run-1"),
    )
    # 파일을 못 읽는 상황입니다.
    monkeypatch.setattr(experiment_detail, "_read_document", lambda uri, config: None)

    curve = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["run-1"]}
    ).json()["curves"]["run-1"]

    assert curve["available"] is False
    assert curve["reason"]
    assert curve["epochs"] == []


def stub_detail_sources(monkeypatch, run_id: str, documents: dict) -> None:
    use_index(monkeypatch, [submitted_summary(run_id)])
    monkeypatch.setattr(
        experiments,
        "read_experiment_record",
        lambda uri, config, *, expected_run_id=None: record_with_artifacts(run_id),
    )
    monkeypatch.setattr(
        experiment_detail,
        "_read_document",
        lambda uri, storage_config: documents.get(uri),
    )


def test_a_name_that_cannot_be_an_experiment_is_404_not_500(client):
    """주소를 잘못 친 사람에게 "서버 고장"이라고 말하지 않습니다.

    index를 훑던 예전 경로는 이런 이름에 "그런 실험 없음"(404)으로 답했습니다. 이름
    한 건만 읽게 되면서 그것이 500이 됐는데, 500은 서버가 고장났다는 뜻이라 사실이
    아닙니다. 저장소를 읽다 실패한 것은 지금도 500입니다 — S3 장애를 "실험이
    사라졌다"로 말하면 그것도 거짓말이기 때문입니다.

    이름은 `%00`으로 보냅니다. `..%2F` 같은 값은 라우터가 먼저 404를 내서 이 코드에
    닿지도 않으므로, 그것으로 재면 매핑을 지워도 초록인 test가 됩니다.
    """

    assert client.get("/api/train/experiments/%00").status_code == 404
    assert (
        client.put(
            "/api/train/experiments/%00/kaggle-score", json={"score": 0.5}
        ).status_code
        == 404
    )


@pytest.mark.parametrize("run_id", ["\x00", "../바깥", "..\\바깥"])
def test_a_name_outside_the_index_reads_as_not_found(client, run_id):
    """경로 구분자가 든 이름은 URL로 보낼 수 없으므로 함수를 직접 부릅니다.

    화면에서 그런 이름이 오는 길은 URL만이 아닙니다 — 비교 요청 본문과 점수 기록에도
    같은 값이 실릴 수 있어, 판정은 함수 쪽에 있어야 합니다.
    """

    with pytest.raises(JobNotFoundError):
        experiments.read_registry_experiment(run_id)
    with pytest.raises(JobNotFoundError):
        experiments.save_kaggle_score(run_id, 0.5)


def test_a_name_that_cannot_be_an_experiment_only_drops_that_column(client, monkeypatch):
    """비교에서는 그 실험만 빠집니다. 하나 때문에 나머지 표가 사라지면 안 됩니다."""

    use_index(monkeypatch, [registry_summary("run-1")])
    monkeypatch.setattr(
        experiments,
        "read_experiment_summary",
        lambda run_id, config: (_ for _ in ()).throw(ExperimentNameError("나쁜 이름"))
        if run_id != "run-1"
        else registry_summary("run-1"),
    )
    monkeypatch.setattr(
        experiments, "read_experiment_record", lambda *a, **k: experiment_record("run-1")
    )

    payload = client.post(
        "/api/train/experiments/compare", json={"run_ids": ["run-1", "../바깥"]}
    ).json()

    assert [item["run_id"] for item in payload["experiments"]] == ["run-1"]
    assert payload["missing"] == ["../바깥"]


def test_detail_reads_one_index_entry_instead_of_scanning(client, monkeypatch):
    """상세도 이름으로 한 건만 읽습니다.

    비교에만 이 검사가 있었습니다. 그래서 상세를 목록 훑기로 되돌려도 test가 조용해,
    이 PR이 없애려던 느림이 그 길로 그대로 돌아올 수 있었습니다.
    """

    refuse_listing(monkeypatch)
    monkeypatch.setattr(
        experiments, "read_experiment_summary", lambda run_id, config: submitted_summary(run_id)
    )
    monkeypatch.setattr(
        experiments,
        "read_experiment_record",
        lambda uri, config, *, expected_run_id=None: record_with_artifacts("done"),
    )
    monkeypatch.setattr(experiment_detail, "_read_document", lambda uri, config: None)

    response = client.get("/api/train/experiments/done")

    assert response.status_code == 200
    assert response.json()["experiment"]["run_id"] == "done"


def test_kaggle_score_reads_one_index_entry_instead_of_scanning(client, monkeypatch):
    """점수 기록도 이름으로 한 건만 읽습니다. 이유는 상세와 같습니다."""

    refuse_listing(monkeypatch)
    monkeypatch.setattr(
        experiments, "read_experiment_summary", lambda run_id, config: submitted_summary(run_id)
    )

    saved = client.put(
        "/api/train/experiments/done/kaggle-score", json={"score": 0.8734}
    )

    assert saved.status_code == 200


def test_detail_carries_all_nine_metrics_and_the_loss_curve(client, monkeypatch):
    stub_detail_sources(
        monkeypatch,
        "done",
        {
            "artifacts/evaluate/done/metrics.json": metrics_document(),
            "artifacts/experiments/done/training_history.json": training_history(),
        },
    )

    payload = client.get("/api/train/experiments/done").json()

    assert payload["experiment"]["run_id"] == "done"
    assert payload["evaluation"]["metrics"]["precision75"] == 0.91
    assert len(payload["evaluation"]["metrics"]) == 9
    assert payload["evaluation"]["counts"]["image_count"] == 500
    assert payload["evaluation"]["best_f1"]["0.50"]["f1"] == 0.94
    assert payload["evaluation"]["per_class_summary"]["counts"]["weak"] == 3
    # 곡선을 그리려면 순서가 있어야 합니다. 파일 순서를 믿지 않습니다.
    assert [item["epoch"] for item in payload["history"]["epochs"]] == [1, 2, 3]
    assert payload["history"]["epochs"][2]["validation_loss"] == 0.0575
    # 옛 실행에는 learning_rate가 없습니다. 없는 값을 지어내지 않습니다.
    assert [item["learning_rate"] for item in payload["history"]["epochs"]] == [
        None,
        None,
        0.00025,
    ]


def test_detail_never_sends_the_huge_blocks(client, monkeypatch):
    """metrics.json은 confusion matrix와 per_image까지 들어 650KB가 넘습니다."""

    stub_detail_sources(
        monkeypatch,
        "done",
        {"artifacts/evaluate/done/metrics.json": metrics_document()},
    )

    body = client.get("/api/train/experiments/done").text

    # key 이름으로 봅니다. `max_detections_per_image`가 부분 문자열로 걸립니다.
    assert '"confusion_matrix"' not in body
    assert '"per_image"' not in body
    assert '"per_class"' not in body
    assert len(body) < 20_000


def test_detail_sorts_the_score_sweep_by_threshold(client, monkeypatch):
    stub_detail_sources(
        monkeypatch,
        "done",
        {"artifacts/evaluate/done/metrics.json": metrics_document()},
    )

    sweep = client.get("/api/train/experiments/done").json()["evaluation"]["score_sweep"]

    assert [row["threshold"] for row in sweep["0.50"]] == [0.05, 0.1, 0.5]


def test_detail_shows_settings_even_when_the_result_files_are_gone(client, monkeypatch):
    """평가 파일을 못 읽는다고 설정까지 안 보이면 화면이 쓸모없어집니다."""

    stub_detail_sources(monkeypatch, "done", {})

    payload = client.get("/api/train/experiments/done").json()

    assert payload["experiment"]["training"]["epochs"] == 4
    assert payload["evaluation"]["available"] is False
    assert payload["evaluation"]["reason"]
    assert payload["history"]["available"] is False


def test_detail_reports_404_for_an_unregistered_run(client, monkeypatch):
    use_index(monkeypatch, [])

    assert client.get("/api/train/experiments/nope").status_code == 404


def test_detail_keeps_the_same_keys_when_the_files_cannot_be_read(client, monkeypatch):
    """못 읽었을 때도 성공했을 때와 같은 key를 채워 보냅니다.

    key를 통째로 빼면 화면이 available을 확인하기 전에 값을 만지는 순간 죽습니다.
    실제로 학습만 하고 평가를 돌리지 않은 기록에서 상세 화면이 흰 채로 멈췄습니다.
    """

    stub_detail_sources(monkeypatch, "done", {})
    payload = client.get("/api/train/experiments/done").json()

    good = experiment_detail.evaluation_block(
        "artifacts/evaluate/done/metrics.json", {"backend": "local"}
    )
    assert payload["evaluation"]["available"] is False
    assert set(payload["evaluation"]) == set(good)
    assert payload["evaluation"]["score_sweep"] == {}
    assert payload["evaluation"]["best_f1"] == {}
    assert payload["evaluation"]["per_class_summary"] is None
    assert payload["history"]["epochs"] == []
