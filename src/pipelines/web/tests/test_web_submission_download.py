"""만들어 둔 제출 CSV를 화면에서 그대로 받아 갈 수 있는지 확인합니다.

경로만 알려 주면, 저장소를 clone만 해 본 사람은 그 자리를 찾는 것부터 한 단계입니다.
"""

from __future__ import annotations

import pytest


CSV = "annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score\n1,1,250,1,2,3,4,0.9\n"


def _write(root, relative: str) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CSV, encoding="utf-8", newline="\n")
    return relative


def test_the_submission_comes_back_named_after_its_run(client, isolated_repo):
    uri = _write(isolated_repo, "artifacts/ensemble/fusion-top3/submission.csv")

    response = client.get("/api/submission", params={"uri": uri})

    assert response.status_code == 200
    assert response.text == CSV
    # 여러 실행의 제출을 받으면 전부 같은 이름이 되어 어느 것인지 모릅니다.
    assert "fusion-top3-submission.csv" in response.headers["content-disposition"]


def test_a_missing_submission_is_not_found(client, isolated_repo):
    response = client.get(
        "/api/submission", params={"uri": "artifacts/ensemble/없는것/submission.csv"}
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "uri",
    (
        "../outside/submission.csv",
        "artifacts/web/settings.json",
        "artifacts/ensemble/run/metrics.json",
    ),
)
def test_only_a_submission_inside_the_repository_is_served(client, isolated_repo, uri):
    """이 route로 저장소 안 아무 파일이나 읽어 가지 못해야 합니다."""

    _write(isolated_repo, "artifacts/web/settings.json")
    _write(isolated_repo, "artifacts/ensemble/run/metrics.json")

    response = client.get("/api/submission", params={"uri": uri})

    assert response.status_code in (400, 404)
    assert CSV not in response.text
