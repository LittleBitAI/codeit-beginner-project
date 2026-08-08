"""평가 결과 파일에서 class별 요약만 꺼내 오는 경로.

evaluate를 import하지 않고 evaluate가 공개한 artifact(metrics_uri)만 읽습니다.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.web import evaluate_metrics


def state(**overrides):
    base = {
        "status": "succeeded",
        "artifacts": {"metrics_uri": "artifacts/evaluate/run-1/metrics.json"},
        "storage": {"backend": "local", "local": {"root": "artifacts"}},
    }
    base.update(overrides)
    return base


def document(**analysis):
    return {"metrics": {"mAP": 0.5}, "analysis": analysis}


class FakeStorage:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.error = error
        self.reads: list[str] = []

    def read_json(self, source):
        self.reads.append(str(source))
        if self.error is not None:
            raise self.error
        return self.payload


@pytest.fixture
def storage(monkeypatch):
    """`create_storage`를 가로채 fake를 돌려주고, 그 fake를 test에 넘깁니다."""

    holder: dict[str, FakeStorage] = {}

    def install(payload, error=None):
        fake = FakeStorage(payload, error)
        holder["fake"] = fake
        monkeypatch.setattr(evaluate_metrics, "create_storage", lambda config: fake)
        return fake

    return install


SUMMARY = {
    "min_truth_count": 4,
    "top_n": 5,
    "counts": {"weak": 2, "sparse": 1, "unmeasured": 0},
    "weak": [
        {
            "category_id": 19552,
            "name": "트루비타정 60mg/병",
            "ap": 0.6506,
            "ap50": 0.6733,
            "ap75": 0.66,
            "truth_count": 87,
            "prediction_count": 59,
        }
    ],
    "sparse": [],
    "unmeasured": [],
}


def test_reads_the_summary_block_out_of_the_metrics_artifact(storage):
    fake = storage(document(per_class_summary=SUMMARY))

    result = evaluate_metrics.read_per_class_summary(state())

    assert result == SUMMARY
    assert fake.reads == ["artifacts/evaluate/run-1/metrics.json"]


def test_older_results_without_the_block_report_nothing_instead_of_guessing(storage):
    """이 계약 이전 평가에는 요약이 없습니다. 빈 표를 지어내면 안 됩니다."""

    storage(document())

    assert evaluate_metrics.read_per_class_summary(state()) is None


@pytest.mark.parametrize(
    "broken",
    [
        {"artifacts": {}},  # metrics_uri 없음
        {"artifacts": {"metrics_uri": "   "}},
        {"storage": {}},  # 어디서 읽을지 모름
        {"status": "running"},  # 아직 결과가 없음
    ],
)
def test_incomplete_state_is_reported_as_nothing(broken, storage):
    storage(document(per_class_summary=SUMMARY))

    assert evaluate_metrics.read_per_class_summary(state(**broken)) is None


@pytest.mark.parametrize("error", [OSError("disk"), ValueError("bad json"), RuntimeError("x")])
def test_a_failed_read_never_breaks_the_screen(error, storage):
    """이 표는 부가 정보입니다. 못 읽는다고 평가 화면이 깨지면 안 됩니다."""

    storage(None, error=error)

    assert evaluate_metrics.read_per_class_summary(state()) is None


def test_a_summary_of_the_wrong_shape_is_refused(storage):
    storage(document(per_class_summary=[1, 2, 3]))

    assert evaluate_metrics.read_per_class_summary(state()) is None


def test_the_whole_document_is_not_handed_to_the_browser(storage):
    """metrics.json은 confusion matrix까지 들어 650KB입니다. 요약만 내보냅니다."""

    fake_document = document(
        per_class_summary=SUMMARY,
        confusion_matrix={"0.50": {"matrix": [[0] * 58 for _ in range(58)]}},
        per_image=[{"image_id": index} for index in range(2100)],
    )
    storage(fake_document)

    result = evaluate_metrics.read_per_class_summary(state())

    assert result is not None
    assert set(result) == set(SUMMARY)
    assert len(json.dumps(result)) < 4000
