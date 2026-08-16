"""앙상블 진단이 **낭비를 실제로 잡는지** 봅니다.

이 화면의 값어치는 고르는 편의가 아니라 **합치기 전에 멈춰 세우는 것**입니다. 그래서
test도 "경고가 떴는가"를 잽니다. 경고가 안 뜨면 사람이 제출을 한 번 버립니다.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.pipelines.web import ensemble


def _prediction_document(
    *,
    checkpoint: str,
    boxes: dict[tuple[int, int], list[float]],
    manifest: str = "s3://bucket/datasets/processed/v5-seed42-8020-group/test_manifest.json",
    prediction_source: str | None = None,
    fused_from: Any = None,
) -> dict[str, Any]:
    return {
        "test_manifest_uri": manifest,
        "checkpoint_uri": checkpoint,
        "prediction_source": prediction_source,
        "fused_from": fused_from,
        "predictions": [
            {"image_id": image_id, "category_id": category_id, "bbox": box, "score": 0.9}
            for (image_id, category_id), box in boxes.items()
        ],
    }


@pytest.fixture
def fake_runs(monkeypatch: pytest.MonkeyPatch):
    """후보 목록과 예측 파일을 화면이 보는 모양 그대로 흉내 냅니다."""

    documents: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []

    def add(
        run_id: str,
        score: float | None,
        document: dict[str, Any] | None = None,
    ) -> None:
        """`document`가 None이면 아직 예측이 없는 실행입니다 — 추론이 먼저입니다."""

        uri = f"s3://bucket/{run_id}/test_predictions.json" if document else None
        if document is not None:
            documents[uri] = document
        candidates.append(
            {
                "run_id": run_id,
                "checkpoint_uri": f"s3://bucket/{run_id}/best_checkpoint.pt",
                "test_predictions_uri": uri,
                "ready": document is not None,
                "dataset_label": "v5-seed42-8020-group",
                "kaggle_score": score,
                "created_at": None,
                "data_inputs": {
                    "validation_manifest_uri": "s3://bucket/v/validation_manifest.json",
                    "test_manifest_uri": "s3://bucket/v/test_manifest.json",
                    "class_map_uri": "s3://bucket/v/class_map.json",
                },
                "predictions_uri": f"s3://bucket/{run_id}/predictions.json",
            }
        )

    monkeypatch.setattr(ensemble, "list_candidates", lambda: list(candidates))
    monkeypatch.setattr(ensemble, "_read_predictions", lambda uri: _load(documents[uri], uri))
    # 저장해 둔 값을 읽거나 쓰는 것은 이 test의 대상이 아닙니다.
    monkeypatch.setattr(ensemble, "_load_stored_pair", lambda key: None)
    monkeypatch.setattr(ensemble, "_store_pair", lambda key, value: None)
    monkeypatch.setattr(ensemble, "_PAIR_CACHE", {})
    return add


def _load(document: dict[str, Any], uri: str) -> dict[str, Any]:
    """`_read_predictions`가 하는 정리를 그대로 합니다."""

    boxes = {
        (row["image_id"], row["category_id"]): tuple(float(v) for v in row["bbox"])
        for row in document["predictions"]
    }
    return {
        "uri": uri,
        "test_manifest_uri": document["test_manifest_uri"],
        "checkpoint_uri": document["checkpoint_uri"],
        "prediction_source": document.get("prediction_source"),
        "fused_from": document.get("fused_from"),
        "boxes": boxes,
    }


def _check(result: dict[str, Any], identifier: str) -> dict[str, str]:
    return next(item for item in result["checks"] if item["id"] == identifier)


def test_almost_identical_predictions_are_warned(fake_runs) -> None:
    """거의 같게 예측하는 둘을 합치면 이득이 없습니다 — 제출 한 번이 사라집니다.

    실제로 이 저장소에서 97.7% 일치 조합의 이득이 +0.002였고, 96.9%짜리를 넣어 볼
    뻔했습니다.
    """

    same = {(1, 7): [10.0, 10.0, 20.0, 20.0], (2, 3): [30.0, 30.0, 20.0, 20.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=same))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=same))

    result = ensemble.diagnose(["a", "b"])

    assert _check(result, "diversity")["level"] == "warn"
    assert result["diversity"]["agreement"] == pytest.approx(1.0)


def test_predictions_that_differ_are_not_warned(fake_runs) -> None:
    """서로 다르게 예측하면 합칠 여지가 있습니다. 그때까지 막으면 쓸 수가 없습니다."""

    fake_runs(
        "a",
        0.62,
        _prediction_document(
            checkpoint="ckpt/a.pt",
            boxes={(index, 7): [0.0, 0.0, 10.0, 10.0] for index in range(10)},
        ),
    )
    fake_runs(
        "b",
        0.61,
        _prediction_document(
            checkpoint="ckpt/b.pt",
            boxes={(index, 3): [0.0, 0.0, 10.0, 10.0] for index in range(10)},
        ),
    )

    result = ensemble.diagnose(["a", "b"])

    assert _check(result, "diversity")["level"] == "ok"


def test_a_much_weaker_run_is_warned(fake_runs) -> None:
    """약한 실행은 결과를 pool 평균 쪽으로 끌어내립니다.

    일곱 개를 합쳤을 때 0.57485짜리 때문에 단독 최고보다 낮게 나왔습니다. 그 일을
    다시 겪지 않으려고 잽니다.
    """

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("strong", 0.624, _prediction_document(checkpoint="ckpt/s.pt", boxes=boxes))
    fake_runs("weak", 0.575, _prediction_document(checkpoint="ckpt/w.pt", boxes=boxes))

    result = ensemble.diagnose(["strong", "weak"])

    dilution = _check(result, "dilution")
    assert dilution["level"] == "warn"
    assert "weak" in dilution["detail"]


def test_a_different_test_set_is_warned(fake_runs) -> None:
    """다른 시험지를 본 예측을 섞으면 evaluate가 거부해 실행 자체가 실패합니다."""

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs(
        "b",
        0.61,
        _prediction_document(
            checkpoint="ckpt/b.pt",
            boxes=boxes,
            manifest="s3://bucket/datasets/processed/v6-seed42-8020-group-angle/test_manifest.json",
        ),
    )

    result = ensemble.diagnose(["a", "b"])

    assert _check(result, "test_set")["level"] == "warn"


def test_a_fusion_result_among_the_inputs_is_warned(fake_runs) -> None:
    """합친 결과는 다시 합칠 수 없습니다. evaluate가 거부합니다."""

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs(
        "fused",
        0.63,
        _prediction_document(
            checkpoint="ckpt/f.pt", boxes=boxes, prediction_source="fusion"
        ),
    )

    result = ensemble.diagnose(["a", "fused"])

    assert _check(result, "reuse")["level"] == "warn"


def test_the_same_run_twice_is_warned(fake_runs) -> None:
    """같은 checkpoint가 두 번 들어오면 한 실행이 두 표를 갖습니다."""

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/same.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/same.pt", boxes=boxes))

    result = ensemble.diagnose(["a", "b"])

    assert _check(result, "reuse")["level"] == "warn"


def test_four_or_more_warns_that_classes_drop(fake_runs) -> None:
    """합치는 수가 늘수록 고유 class가 줄었습니다(84 → 82 → 81 → 78 → 77)."""

    boxes = {(index, 7): [float(index), 0.0, 10.0, 10.0] for index in range(5)}
    for name in ("a", "b", "c", "d"):
        fake_runs(name, 0.62, _prediction_document(checkpoint=f"ckpt/{name}.pt", boxes=boxes))

    result = ensemble.diagnose(["a", "b", "c", "d"])

    assert _check(result, "pool_size")["level"] == "warn"


def test_warnings_never_block(fake_runs) -> None:
    """경고는 막지 않습니다. 예측이 틀릴 때가 있고, 막으면 반증할 길까지 막힙니다."""

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("weak", 0.50, _prediction_document(checkpoint="ckpt/w.pt", boxes=boxes))

    result = ensemble.diagnose(["a", "weak"])

    assert any(item["level"] == "warn" for item in result["checks"])
    assert result["blocking"] is False


def test_fusion_config_uses_the_inputs_the_runs_declared(fake_runs) -> None:
    """데이터 입력을 새로 정하면 학습이 본 것과 다른 시험지를 채점하게 됩니다."""

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))

    config = ensemble.build_fusion_config(["a", "b"], run_id="fusion-two")
    settings = config["evaluate"]

    assert settings["test_predictions_input_uris"] == [
        "s3://bucket/a/test_predictions.json",
        "s3://bucket/b/test_predictions.json",
    ]
    assert settings["class_map_uri"] == "s3://bucket/v/class_map.json"
    # 점수가 가장 높은 실행의 검증 예측을 그대로 씁니다. 융합은 test만 바꿉니다.
    assert settings["predictions_input_uri"] == "s3://bucket/a/predictions.json"
    # GPU를 잡지 않습니다. 학습이 도는 중에도 합칠 수 있어야 합니다.
    assert settings["device"] == "cpu"


def test_one_run_is_not_a_fusion(fake_runs) -> None:
    """하나만 주면 합칠 것이 없는데도 겹치는 상자를 묶어 원본과 다른 예측이 나옵니다."""

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))

    with pytest.raises(Exception) as error:
        ensemble.build_fusion_config(["a"], run_id="fusion-one")
    assert "둘 이상" in str(error.value)


def test_a_run_without_predictions_is_still_a_candidate(fake_runs) -> None:
    """체크포인트가 있으면 후보입니다.

    처음에는 `test_predictions.json`이 있는 실행만 후보로 삼았는데, 그 key가 계약에
    늦게 들어와서 47개 중 2개만 목록에 떴습니다. **가장 점수가 높은 실행들이 전부
    빠졌습니다.** 예측이 없는 것은 만들면 됩니다.
    """

    boxes = {(1, 7): [10.0, 10.0, 20.0, 20.0]}
    fake_runs("ready", 0.62, _prediction_document(checkpoint="ckpt/r.pt", boxes=boxes))
    fake_runs("fresh", 0.61, None)

    result = ensemble.diagnose(["ready", "fresh"])

    readiness = _check(result, "readiness")
    assert readiness["level"] == "warn"
    assert "fresh" in readiness["detail"]
    # 예측이 없다고 진단을 통째로 접지 않습니다. 나머지는 그대로 쓸모가 있습니다.
    assert _check(result, "dilution")["level"] == "ok"


def test_pending_runs_keeps_the_order_they_were_picked(fake_runs) -> None:
    """추론 순서는 고른 순서입니다. 화면이 몇 번째인지 세어 보여 줍니다."""

    fake_runs("a", 0.62, None)
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes={(1, 7): [0.0, 0.0, 5.0, 5.0]}))
    fake_runs("c", 0.60, None)

    pending = ensemble.pending_runs(["c", "b", "a"])

    assert [item["run_id"] for item in pending] == ["c", "a"]


def test_fusing_before_the_predictions_exist_is_refused(fake_runs) -> None:
    """추론이 먼저입니다. 실행기가 그것을 끝낸 뒤 다시 부릅니다."""

    fake_runs("ready", 0.62, _prediction_document(checkpoint="ckpt/r.pt", boxes={(1, 7): [0.0, 0.0, 5.0, 5.0]}))
    fake_runs("fresh", 0.61, None)

    with pytest.raises(Exception) as error:
        ensemble.build_fusion_config(["ready", "fresh"], run_id="fusion-two")
    assert "fresh" in str(error.value)


def test_harvest_writes_beside_web_state_not_over_the_run(fake_runs, monkeypatch) -> None:
    """남이 만든 artifact를 덮어쓰지 않습니다.

    그 실행의 `experiments/completed/.../evaluate/`는 그대로 두고, web이 소유한 자리에
    만듭니다. 그리고 제출이 남기는 4개가 아니라 스무 개를 저장합니다 — 합칠 때 고를
    것이 있어야 하기 때문입니다.
    """

    monkeypatch.setattr(
        ensemble, "storage_environment", lambda: {"default_backend": "s3", "bucket": "bucket"}
    )
    fake_runs("fresh", 0.61, None)
    candidate = next(item for item in ensemble.list_candidates() if item["run_id"] == "fresh")

    config = ensemble.build_harvest_config(candidate, device="cpu")
    settings = config["evaluate"]

    assert settings["output_dir"] == "s3://bucket/experiments/web-state/ensemble-candidates/fresh"
    assert "experiments/completed" not in settings["output_dir"]
    assert settings["max_detections_per_image"] == 20
    assert settings["checkpoint_uri"] == "s3://bucket/fresh/best_checkpoint.pt"
    # 제출할 수 없는 CSV라 submissions/ 아래에 두지 않습니다.
    assert "/submissions/" not in settings["submission_uri"]


#: fixture가 덮어쓰기 전의 진짜 함수입니다.
_real_harvested_runs = ensemble._harvested_runs


def _summary(run_id: str, *, checkpoint: bool = True, predictions: bool = False) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "train_manifest_uri": "s3://bucket/datasets/processed/v5-seed42-8020-group/train_manifest.json",
        "validation_manifest_uri": "s3://bucket/v/validation_manifest.json",
        "test_manifest_uri": "s3://bucket/v/test_manifest.json",
        "class_map_uri": "s3://bucket/v/class_map.json",
        "predictions_uri": f"s3://bucket/{run_id}/predictions.json",
    }
    if checkpoint:
        artifacts["best_checkpoint_uri"] = f"s3://bucket/{run_id}/best_checkpoint.pt"
    if predictions:
        artifacts["test_predictions_uri"] = f"s3://bucket/{run_id}/evaluate/test_predictions.json"
    return {"run_id": run_id, "created_at": "2026-08-16T00:00:00Z", "artifacts": artifacts}


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch):
    """`list_candidates`를 **진짜로** 돌립니다.

    다른 test는 후보 목록을 통째로 대신 세워 두는데, 그러면 "누가 후보인가"를 정하는
    바로 그 코드가 빠집니다. mutation으로 예측 없는 실행을 후보에서 빼 봤을 때
    아무 test도 빨개지지 않아서 알았습니다.
    """

    summaries: list[dict[str, Any]] = []
    monkeypatch.setattr(ensemble, "list_experiment_summaries", lambda config: list(summaries))
    monkeypatch.setattr(ensemble, "registry_config", lambda: {})
    monkeypatch.setattr(ensemble.kaggle_scores, "load_scores", lambda: {"low": 0.4, "high": 0.6})
    monkeypatch.setattr(
        ensemble, "storage_environment", lambda: {"default_backend": "s3", "bucket": "bucket"}
    )
    monkeypatch.setattr(ensemble, "_harvested_runs", set)
    return summaries


def test_every_run_with_a_checkpoint_is_a_candidate(fake_registry) -> None:
    """예측이 없어도 후보입니다. 이것이 이 화면이 쓸모 있어지는 조건입니다.

    예측이 있는 실행만 세면 이 저장소에서 47개 중 2개만 남고, 점수가 가장 높은
    실행들이 통째로 빠집니다.
    """

    fake_registry.append(_summary("high", predictions=True))
    fake_registry.append(_summary("low", predictions=False))

    candidates = ensemble.list_candidates()

    assert {item["run_id"] for item in candidates} == {"high", "low"}
    ready = {item["run_id"]: item["ready"] for item in candidates}
    assert ready == {"high": True, "low": False}


def test_a_run_without_a_checkpoint_is_not_a_candidate(fake_registry) -> None:
    """체크포인트가 없으면 예측을 만들 길이 없으므로 고르게 두면 안 됩니다."""

    fake_registry.append(_summary("high", checkpoint=False))

    assert ensemble.list_candidates() == []


def test_higher_kaggle_scores_come_first(fake_registry) -> None:
    """보는 순서가 곧 pool 품질 순서여야 약한 실행을 무심코 넣지 않습니다."""

    fake_registry.append(_summary("low"))
    fake_registry.append(_summary("high"))

    assert [item["run_id"] for item in ensemble.list_candidates()] == ["high", "low"]


def test_predictions_this_screen_made_count_as_ready(fake_registry, monkeypatch) -> None:
    """기록이 아직 안 가리켜도, 이 화면이 만들어 둔 예측이 있으면 바로 합칠 수 있습니다."""

    fake_registry.append(_summary("low", predictions=False))
    monkeypatch.setattr(ensemble, "_harvested_runs", lambda: {"low"})

    candidate = ensemble.list_candidates()[0]

    assert candidate["ready"] is True
    assert candidate["test_predictions_uri"] == (
        "s3://bucket/experiments/web-state/ensemble-candidates/low/test_predictions.json"
    )


def test_agreement_does_not_depend_on_which_run_came_first(fake_runs) -> None:
    """한쪽 수로 나누면 고른 순서가 답을 바꿉니다.

    94개가 100개에 통째로 들어 있으면 순서에 따라 100%(경고)와 94%(통과)로 갈립니다.
    저장해 두는 값의 key는 순서를 가리지 않아서, 먼저 잰 쪽이 영원히 남습니다.
    """

    small = {(index, 7): [float(index), 0.0, 5.0, 5.0] for index in range(4)}
    large = {**small, **{(index, 7): [float(index), 0.0, 5.0, 5.0] for index in range(4, 10)}}
    fake_runs("small", 0.62, _prediction_document(checkpoint="ckpt/s.pt", boxes=small))
    fake_runs("large", 0.61, _prediction_document(checkpoint="ckpt/l.pt", boxes=large))

    forward = ensemble.diagnose(["small", "large"])["diversity"]["agreement"]
    ensemble._PAIR_CACHE.clear()
    backward = ensemble.diagnose(["large", "small"])["diversity"]["agreement"]

    assert forward == pytest.approx(backward)
    # Dice: 2 * 4 / (4 + 10). 부분집합이라고 "완전히 같다"로 읽으면 안 됩니다.
    assert forward == pytest.approx(2 * 4 / 14)


def test_the_warning_threshold_still_catches_what_it_used_to(fake_runs) -> None:
    """공식을 바꾸면서 **눈금이 함께 바뀌면** 경고가 조용히 사라집니다.

    예전에는 `겹침 / 왼쪽 수`라서 96.9% 조합이 임계값 0.95를 넘어 경고했습니다.
    합집합으로 나누면(Jaccard) 같은 조합이 94.0%가 되어 빠져나갑니다. Dice는 두 실행의
    예측 수가 같을 때 예전과 같은 값을 내므로 임계값을 다시 맞출 필요가 없습니다.
    """

    total, overlap = 1000, 969
    left = {(index, 7): [float(index), 0.0, 5.0, 5.0] for index in range(total)}
    right = {
        (index if index < overlap else index + total, 7): [float(index), 0.0, 5.0, 5.0]
        for index in range(total)
    }
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=left))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=right))

    result = ensemble.diagnose(["a", "b"])

    assert result["diversity"]["agreement"] == pytest.approx(0.969)
    assert _check(result, "diversity")["level"] == "warn"


def test_a_single_run_is_refused_before_any_inference(fake_runs) -> None:
    """검증을 뒤로 미루면 GPU가 9분 돌고 나서야 "둘 이상 필요"로 실패합니다."""

    fake_runs("only", 0.62, None)

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["only"], run_id="fusion-one")
    assert "둘 이상" in str(error.value)


def test_a_result_name_cannot_escape_its_own_folder(fake_runs) -> None:
    """길이만 재면 local backend에서 `../train/name`이 다른 pipeline 자리로 풀립니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))

    for bad in ("../train/name", "a/b", ".hidden", ""):
        with pytest.raises(Exception):
            ensemble.check_selection(["a", "b"], run_id=bad)
    # 멀쩡한 이름은 통과해야 합니다. 막기만 하면 쓸 수가 없습니다.
    assert len(ensemble.check_selection(["a", "b"], run_id="fusion-top3")) == 2


def test_the_same_run_twice_is_refused_up_front(fake_runs) -> None:
    """한 실행이 두 표를 가지면 확신도가 부풀려집니다. 추론 전에 막습니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["a", "a"], run_id="fusion-dup")
    assert "두 번" in str(error.value)


def test_harvest_works_without_an_s3_bucket(fake_registry, monkeypatch) -> None:
    """S3만 지원하면 기본 local 환경에서는 목록에 후보가 보이는데 눌러도 늘 실패합니다."""

    monkeypatch.setattr(
        ensemble, "storage_environment", lambda: {"default_backend": "local", "bucket": None}
    )
    fake_registry.append(_summary("low", predictions=False))
    candidate = ensemble.list_candidates()[0]

    config = ensemble.build_harvest_config(candidate, device="cpu")

    assert config["storage"]["backend"] == "local"
    assert config["evaluate"]["output_dir"] == "artifacts/web/ensemble-candidates/low"


def test_the_same_folder_name_in_another_place_is_not_the_same_test_set(fake_runs) -> None:
    """폴더 이름만 보면 bucket이 달라도 같은 판으로 읽히고, 거부는 evaluate에 가서야 납니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs(
        "b",
        0.61,
        _prediction_document(
            checkpoint="ckpt/b.pt",
            boxes=boxes,
            manifest="s3://other-bucket/datasets/processed/v5-seed42-8020-group/test_manifest.json",
        ),
    )

    result = ensemble.diagnose(["a", "b"])

    assert _check(result, "test_set")["level"] == "warn"


def test_the_runner_refuses_before_it_spends_any_gpu(fake_runs, monkeypatch) -> None:
    """검증이 `check_selection`에 있어도, **실행기가 그것을 부르지 않으면 소용없습니다.**

    부르지 않으면 예측 없는 후보 하나만 보내도 체크포인트당 9분씩 추론한 뒤에야
    "둘 이상 필요"로 실패합니다. 그 시간은 되돌릴 수 없습니다.
    """

    from src.pipelines.web import ensemble_jobs

    fake_runs("only", 0.62, None)
    calls: list[Any] = []
    monkeypatch.setattr(ensemble_jobs, "run_evaluation", lambda *a, **k: calls.append(a) or {"ok": True})

    runner = ensemble_jobs.EnsembleRunner()
    with pytest.raises(Exception) as error:
        runner.start(["only"], run_id="fusion-one")

    assert "둘 이상" in str(error.value)
    # 추론이 **한 번도** 시작되지 않아야 합니다.
    assert calls == []
    assert runner.status()["status"] == "idle"


def test_the_runner_refuses_a_name_that_escapes_its_folder(fake_runs, monkeypatch) -> None:
    """이름 검사도 실행기가 불러야 뜻이 있습니다."""

    from src.pipelines.web import ensemble_jobs

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))
    calls: list[Any] = []
    monkeypatch.setattr(ensemble_jobs, "run_evaluation", lambda *a, **k: calls.append(a) or {"ok": True})

    runner = ensemble_jobs.EnsembleRunner()
    with pytest.raises(Exception):
        runner.start(["a", "b"], run_id="../train/name")

    assert calls == []


def test_local_harvest_results_are_recognised(fake_registry, monkeypatch, tmp_path) -> None:
    """local 저장소는 **절대 OS 경로**를 돌려줍니다.

    앞자리를 `root` 길이로 잘라 내면 엉뚱한 곳이 잘리고, Windows에서는 구분자가
    역슬래시라 아예 걸리지도 않습니다. 그러면 수확에 성공해도 영원히 `ready`가 되지
    않고 융합 단계에서 실패합니다.
    """

    monkeypatch.setattr(
        ensemble, "storage_environment", lambda: {"default_backend": "local", "bucket": None}
    )
    monkeypatch.setattr(ensemble, "repository_root", lambda: tmp_path)
    # fixture가 비워 둔 것을 여기서만 되살립니다 — 이 test가 재려는 것이 그 함수입니다.
    monkeypatch.setattr(ensemble, "_harvested_runs", _real_harvested_runs)
    made = tmp_path / "artifacts/web/ensemble-candidates/low"
    made.mkdir(parents=True)
    (made / "test_predictions.json").write_text("{}", encoding="utf-8")
    fake_registry.append(_summary("low", predictions=False))

    candidate = ensemble.list_candidates()[0]

    assert candidate["ready"] is True
    assert candidate["test_predictions_uri"] == "artifacts/web/ensemble-candidates/low/test_predictions.json"


def test_a_run_missing_its_data_inputs_is_refused_before_inference(fake_runs) -> None:
    """순서대로 확인하면 세 번째의 결함을 앞의 둘을 18분 추론한 뒤에 알게 됩니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))
    fake_runs("broken", 0.60, None)
    # 마지막 후보의 입력을 지웁니다. fixture가 세운 목록을 그대로 씁니다.
    broken = next(item for item in ensemble.list_candidates() if item["run_id"] == "broken")
    broken["data_inputs"]["class_map_uri"] = None

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["a", "b", "broken"], run_id="fusion-three")
    assert "class_map_uri" in str(error.value)


def test_different_test_manifests_are_refused_before_inference(fake_runs) -> None:
    """**예측 파일이 선언한** 시험지가 다르면 evaluate가 거부합니다.

    확인 없이 추론부터 돌리면 그 시간이 전부 버려집니다.
    """

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs(
        "b",
        0.61,
        _prediction_document(
            checkpoint="ckpt/b.pt",
            boxes=boxes,
            manifest="s3://bucket/datasets/processed/v6-seed42-8020-group-angle/test_manifest.json",
        ),
    )

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["a", "b"], run_id="fusion-two")
    assert "test manifest" in str(error.value)

    # 사람이 같은 사진임을 확인했다고 말하면 통과합니다. 막기만 하면 쓸 수 없습니다.
    assert len(
        ensemble.check_selection(["a", "b"], run_id="fusion-two", allow_copied_images=True)
    ) == 2


def test_values_measured_by_the_old_formula_are_not_reused(monkeypatch) -> None:
    """값만 보고는 어느 공식으로 쟀는지 알 수 없습니다. 자리로 갈라야 합니다."""

    path = ensemble._stored_pair_path("prefix", ("a", "b"))
    monkeypatch.setattr(ensemble, "_AGREEMENT_FORMULA", "another-v9")

    assert ensemble._stored_pair_path("prefix", ("a", "b")) != path


def test_a_neighbouring_folder_is_not_mistaken_for_a_candidate(
    fake_registry, monkeypatch, tmp_path
) -> None:
    """구분자 없는 앞자리로 거르면 `ensemble-candidates-old/…`까지 걸립니다."""

    monkeypatch.setattr(
        ensemble, "storage_environment", lambda: {"default_backend": "local", "bucket": None}
    )
    monkeypatch.setattr(ensemble, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(ensemble, "_harvested_runs", _real_harvested_runs)
    for folder in ("artifacts/web/ensemble-candidates-old/low", "artifacts/web/ensemble-candidates/low/nested"):
        made = tmp_path / folder
        made.mkdir(parents=True)
        (made / "test_predictions.json").write_text("{}", encoding="utf-8")
    fake_registry.append(_summary("low", predictions=False))

    # 이웃 폴더의 것도, 한 칸 더 깊은 것도 준비 완료가 아닙니다.
    assert ensemble.list_candidates()[0]["ready"] is False


def test_a_deeper_path_is_not_taken_as_a_run_name(monkeypatch, tmp_path) -> None:
    """파일이 든 폴더 이름만 보면 `<root>/a/b/…`의 `b`가 실행 이름으로 잡힙니다."""

    monkeypatch.setattr(
        ensemble, "storage_environment", lambda: {"default_backend": "local", "bucket": None}
    )
    monkeypatch.setattr(ensemble, "repository_root", lambda: tmp_path)
    shallow = tmp_path / "artifacts/web/ensemble-candidates/good"
    deep = tmp_path / "artifacts/web/ensemble-candidates/outer/inner"
    for made in (shallow, deep):
        made.mkdir(parents=True)
        (made / "test_predictions.json").write_text("{}", encoding="utf-8")

    assert _real_harvested_runs() == {"good"}


def test_two_names_for_the_same_checkpoint_are_refused_before_inference(fake_runs) -> None:
    """이름이 달라도 **예측 파일이 같은 checkpoint를 선언하면** 한 실행이 두 표를 갖습니다.

    기록이 아니라 파일 안의 값을 evaluate가 읽습니다. 기록으로만 견주면 여기서
    통과하고 추론을 다 마친 뒤에 거절됩니다.
    """

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("first", 0.62, _prediction_document(checkpoint="ckpt/shared.pt", boxes=boxes))
    fake_runs("second", 0.61, _prediction_document(checkpoint="ckpt/shared.pt", boxes=boxes))

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["first", "second"], run_id="fusion-two")
    assert "같은 checkpoint" in str(error.value)


def test_an_existing_result_name_is_refused_before_inference(fake_runs, monkeypatch) -> None:
    """덮어쓰지 않는 것이 기본이라, 이름이 겹치면 추론 뒤 출력 충돌로 실패합니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))
    monkeypatch.setattr(ensemble, "_existing_output", lambda name: f"artifacts/ensemble/{name}/submission.csv")

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["a", "b"], run_id="fusion-two")
    assert "이미 있습니다" in str(error.value)

    # 덮어쓰기를 켜면 통과합니다. 막기만 하면 다시 돌릴 길이 없습니다.
    assert len(ensemble.check_selection(["a", "b"], run_id="fusion-two", overwrite=True)) == 2


def test_the_route_passes_overwrite_through(fake_runs, monkeypatch) -> None:
    """요청이 받은 값을 실행기에 안 넘기면, true를 명시해도 늘 false로 돕니다."""

    from src.pipelines.web.api import routes_ensemble
    from src.pipelines.web import ensemble_jobs

    seen: dict[str, Any] = {}

    class Recorder:
        def start(self, run_ids, **kwargs):
            seen.update(kwargs)
            return {"status": "running"}

    monkeypatch.setattr(routes_ensemble, "get_ensemble_runner", Recorder)
    routes_ensemble.start(
        routes_ensemble.StartRequest(run_ids=["a", "b"], run_id="fusion-two", overwrite=True)
    )

    assert seen["overwrite"] is True
    assert ensemble_jobs  # import가 살아 있는지 확인합니다.


def test_a_forced_local_backend_with_a_bucket_stays_local(monkeypatch) -> None:
    """bucket이 있는지만 보면 저장은 local인데 경로만 s3://가 되어 실행이 실패합니다."""

    monkeypatch.setattr(
        ensemble,
        "storage_environment",
        lambda: {"default_backend": "local", "bucket": "some-bucket"},
    )

    assert ensemble._harvest_root() == "artifacts/web/ensemble-candidates"
    assert ensemble._storage_config()["backend"] == "local"
    assert not ensemble._submission_uri("fusion-two").startswith("s3://")


def test_two_spellings_of_one_checkpoint_are_refused(fake_runs, monkeypatch, tmp_path) -> None:
    """`s3://bucket/a.pt`와 `a.pt`는 글자로만 다르고 같은 파일입니다.

    helper를 흉내 내면 진짜 구현이 글자를 돌려주도록 망가져도 이 test가 통과합니다.
    그래서 **실제 저장 계층에 물어보게** 두고, 같은 파일을 두 표기로 가리킵니다.
    """

    monkeypatch.setattr(
        ensemble, "storage_environment", lambda: {"default_backend": "local", "bucket": None}
    )
    monkeypatch.setattr(ensemble, "repository_root", lambda: tmp_path)
    shared = tmp_path / "ckpt/shared.pt"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"weights")

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("first", 0.62, _prediction_document(checkpoint="ckpt/shared.pt", boxes=boxes))
    fake_runs("second", 0.61, _prediction_document(checkpoint="./ckpt/shared.pt", boxes=boxes))

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["first", "second"], run_id="fusion-two")
    assert "같은 checkpoint" in str(error.value)


def test_an_already_fused_input_is_refused_before_inference(fake_runs) -> None:
    """합친 것을 다시 합칠 수 없습니다. 추론 뒤가 아니라 지금 압니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("plain", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs(
        "already",
        0.63,
        _prediction_document(checkpoint="ckpt/f.pt", boxes=boxes, prediction_source="fusion"),
    )

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["plain", "already"], run_id="fusion-two")
    assert "이미 합친 결과" in str(error.value)


def test_failing_to_check_the_output_is_reported_not_guessed(fake_runs, monkeypatch) -> None:
    """없다고 넘기면 추론 뒤 충돌하고, 있다고 넘기면 멀쩡한 이름이 막힙니다.

    저장소를 못 읽는 상태면 뒤 단계도 안전하게 끝나지 않으므로 지금 알립니다.
    """

    from src.common import StorageError

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))

    class Broken:
        def identity(self, uri):
            return uri

        def exists(self, uri):
            raise StorageError("접근 거부")

    monkeypatch.setattr(ensemble, "create_storage", lambda config: Broken())

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["a", "b"], run_id="fusion-two")
    assert "확인하지 못했습니다" in str(error.value)


@pytest.mark.parametrize(
    ("field", "phrase"),
    [
        pytest.param("checkpoint_uri", "checkpoint의 증거", id="checkpoint를-안-적음"),
        pytest.param("test_manifest_uri", "시험지를 본 것인지", id="시험지를-안-적음"),
    ],
)
def test_a_prediction_file_that_does_not_say_what_it_is_is_refused(
    fake_runs, field: str, phrase: str
) -> None:
    """기록이 아니라 **예측 파일 안의 값**을 evaluate가 읽습니다.

    파일이 스스로 무엇의 증거인지 말하지 않으면 evaluate가 거절하는데, 추론을 다
    마친 뒤에 알게 됩니다.
    """

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    document = _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes)
    document[field] = None
    fake_runs("b", 0.61, document)

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["a", "b"], run_id="fusion-two")
    assert phrase in str(error.value)


def test_the_document_wins_over_the_record_for_checkpoints(fake_runs) -> None:
    """기록은 서로 다른데 **파일이 같은 checkpoint를 선언한** 경우입니다.

    기록으로만 견주면 통과하고, evaluate가 파일을 읽어 추론 뒤에 거절합니다.
    """

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("first", 0.62, _prediction_document(checkpoint="ckpt/same.pt", boxes=boxes))
    fake_runs("second", 0.61, _prediction_document(checkpoint="ckpt/same.pt", boxes=boxes))
    # 기록에는 서로 다른 checkpoint가 적혀 있습니다.
    for item in ensemble.list_candidates():
        item["checkpoint_uri"] = f"s3://bucket/{item['run_id']}/best_checkpoint.pt"

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["first", "second"], run_id="fusion-two")
    assert "같은 checkpoint" in str(error.value)


def test_the_document_wins_over_the_record_for_manifests(fake_runs) -> None:
    """기록은 같은데 **파일이 다른 시험지를 선언한** 경우입니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("ready", 0.62, _prediction_document(
        checkpoint="ckpt/a.pt",
        boxes=boxes,
        manifest="s3://bucket/datasets/processed/v6-seed42-8020-group-angle/test_manifest.json",
    ))
    fake_runs("pending", 0.61, None)
    # 기록에는 둘 다 같은 시험지가 적혀 있습니다.
    for item in ensemble.list_candidates():
        item["data_inputs"]["test_manifest_uri"] = "s3://bucket/v/test_manifest.json"

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["ready", "pending"], run_id="fusion-two")
    assert "test manifest" in str(error.value)


def test_failing_to_read_a_checkpoint_identity_is_reported(fake_runs, monkeypatch) -> None:
    """글자로 되돌리면 **같은 파일을 두 번 넣은 것을 놓칩니다.**

    S3는 원격에 닿지 않고 경로만 풀고, local은 파일이 없어도 정상 신원을 냅니다.
    그래도 오류가 났다면 저장 설정 자체가 잘못된 것입니다.
    """

    from src.common import StorageError

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))

    class Broken:
        def __init__(self, *args, **kwargs):
            pass

        def identity(self, uri):
            raise StorageError("설정 오류")

    monkeypatch.setattr(ensemble, "LocalStorage", Broken)
    monkeypatch.setattr(ensemble, "S3Storage", Broken)

    with pytest.raises(Exception) as error:
        ensemble.check_selection(["a", "b"], run_id="fusion-two")
    assert "저장 신원을 얻지 못했습니다" in str(error.value)


def test_scope_and_paths_agree_on_the_backend(monkeypatch) -> None:
    """같은 판단을 두 곳에 적으면 다시 갈립니다. 한 곳만 봅니다."""

    monkeypatch.setattr(
        ensemble,
        "storage_environment",
        lambda: {"default_backend": "local", "bucket": "some-bucket"},
    )

    assert ensemble._scope()[0]["storage"]["backend"] == "local"
    assert ensemble._storage_config()["backend"] == "local"
    assert not ensemble._harvest_root().startswith("s3://")


def test_the_s3_identity_matches_the_shared_storage_layer() -> None:
    """규칙을 새로 적지 않고 공용 계층에 맡깁니다.

    직접 풀었더니 계층과 반대로 움직였습니다 — S3에서 `a//x.pt`와 `a/x.pt`는 서로
    **다른** key인데 같다고 보았습니다. 그래서 답을 손으로 적지 않고 **계층이 내는
    값과 견줍니다.** 계층이 바뀌면 이 test가 함께 따라갑니다.
    """

    from src.common import S3Storage

    for uri in ("s3://b/a/x.pt", "s3://b//a//x.pt", "s3://b/a%2Fx.pt"):
        expected = str(S3Storage(bucket="b").identity(uri))
        assert ensemble._checkpoint_identity(uri) == expected


def test_s3_keys_that_differ_by_a_slash_are_different_objects() -> None:
    """`a//x.pt`와 `a/x.pt`는 S3에서 다른 key입니다. 같다고 보면 멀쩡한 조합을 막습니다."""

    assert ensemble._checkpoint_identity("s3://b/a/x.pt") != ensemble._checkpoint_identity(
        "s3://b//a//x.pt"
    )


def test_an_s3_identity_does_not_need_a_configured_bucket() -> None:
    """bucket이 없어도 물어볼 수 있어야 합니다. 못 물으면 합치기가 통째로 막힙니다.

    `identity()`는 원격에 닿지 않으므로 credential도, 설정된 bucket도 필요 없습니다.
    """

    identity = ensemble._checkpoint_identity("s3://any/where.pt")

    assert "any" in identity and "where.pt" in identity
    assert identity != ensemble._checkpoint_identity("s3://other/where.pt")
