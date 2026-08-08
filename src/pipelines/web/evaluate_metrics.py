"""평가 결과 파일에서 화면이 쓸 class별 요약만 꺼냅니다.

evaluate는 `metrics.json`의 `analysis.per_class_summary`에 "표본이 충분한데 약한
class"를 정렬해 둡니다. 그런데 그 블록은 evaluate의 `run(config)` 반환 summary에는
없어서, 화면은 파일을 직접 읽어야 알 수 있습니다.

evaluate를 import하지는 않습니다. evaluate가 artifact로 공개한 `metrics_uri`를
`src/common`의 storage로 읽을 뿐입니다.

**이 module의 함수는 어떤 이유로도 예외를 던지지 않습니다.** 이 표는 평가 화면의
부가 정보라서, 못 읽는다고 평가 결과 전체가 안 보이면 안 됩니다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.common import create_storage


__all__ = ["SUMMARY_KEYS", "read_per_class_summary"]


# evaluate의 `summarize_per_class`가 만드는 key입니다. 이 모양이 아니면 버립니다.
SUMMARY_KEYS = frozenset(
    {"min_truth_count", "top_n", "counts", "weak", "sparse", "unmeasured"}
)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def read_per_class_summary(evaluation: Any) -> dict[str, Any] | None:
    """평가 결과에서 `analysis.per_class_summary`만 돌려줍니다. 없으면 `None`입니다.

    **전체 문서를 그대로 넘기지 않습니다.** `metrics.json`은 58x58 confusion matrix와
    이미지 2100장의 per_image까지 들어 650KB가 넘습니다. 화면이 필요한 것은 그중
    2KB 남짓한 요약뿐입니다.
    """

    try:
        if not isinstance(evaluation, Mapping) or evaluation.get("status") != "succeeded":
            # 아직 결과가 없는 평가에서 지어낼 값이 없습니다.
            return None
        artifacts = evaluation.get("artifacts")
        storage_config = evaluation.get("storage")
        if not isinstance(artifacts, Mapping) or not isinstance(storage_config, Mapping):
            return None
        if not storage_config:
            # 어디서 읽을지 모르는 상태입니다. 기본값을 가정하면 남의 파일을 읽습니다.
            return None
        uri = _text(artifacts.get("metrics_uri"))
        if uri is None:
            return None

        document = create_storage({"storage": dict(storage_config)}).read_json(uri)
        if not isinstance(document, Mapping):
            return None
        analysis = document.get("analysis")
        if not isinstance(analysis, Mapping):
            return None
        summary = analysis.get("per_class_summary")
        if not isinstance(summary, Mapping) or set(summary) != SUMMARY_KEYS:
            # 이 계약 이전 평가에는 블록 자체가 없습니다. 빈 표를 지어내지 않습니다.
            return None
        return dict(summary)
    except Exception:
        # storage 오류, 깨진 JSON, 예상 못 한 형태 무엇이든 여기서 멈춥니다.
        return None
