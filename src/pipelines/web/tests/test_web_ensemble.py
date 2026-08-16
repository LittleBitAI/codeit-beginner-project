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
    """시험지가 다르면 evaluate가 거부합니다. 확인 없이 추론부터 돌리면 전부 버려집니다."""

    boxes = {(1, 7): [0.0, 0.0, 5.0, 5.0]}
    fake_runs("a", 0.62, _prediction_document(checkpoint="ckpt/a.pt", boxes=boxes))
    fake_runs("b", 0.61, _prediction_document(checkpoint="ckpt/b.pt", boxes=boxes))
    other = next(item for item in ensemble.list_candidates() if item["run_id"] == "b")
    other["data_inputs"]["test_manifest_uri"] = "s3://bucket/v6/test_manifest.json"

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
