"""Submission CSV 규격 검사 테스트.

`contracts/proposals/001-competition-artifact-contract.md`가 정한 제출 CSV 규격을
registry가 실제로 막아 내는지 확인합니다. 작은 local 파일만 쓰므로 CPU에서 바로
동작합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipelines import registry
from src.pipelines.registry.record import InvalidSubmissionError
from src.pipelines.registry.submission import SUBMISSION_HEADER, check_submission_csv


FIXTURE_DIR = Path(__file__).parent / "fixtures"

HEADER_LINE = ",".join(SUBMISSION_HEADER)


def load_fixture(name: str) -> dict:
    """계약 형식의 artifact fixture를 읽습니다."""

    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def materialize_local_artifacts(inputs: dict, repo_root: Path) -> None:
    """fixture가 가리키는 local artifact file을 임시 저장소 안에 만듭니다."""

    for pipeline, artifacts in inputs.items():
        for key, uri in artifacts.items():
            if not key.endswith("_uri"):
                continue
            path = repo_root / uri
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"pipeline": pipeline, "artifact": key}, ensure_ascii=False),
                encoding="utf-8",
                newline="\n",
            )


def make_config(repo_root: Path, inputs: dict, **registry_options) -> dict:
    """저장소 밖을 건드리지 않는 local storage config를 만듭니다."""

    registry_config = {"repo_root": str(repo_root)}
    registry_config.update(registry_options)
    return {
        "project": {"name": "pill-object-detection"},
        "seed": 42,
        "storage": {
            "backend": "local",
            "local": {"root": str(repo_root / "artifacts")},
        },
        "registry": registry_config,
        "inputs": inputs,
    }

# image 10에 2행, image 20에 1행. 계약이 정한 정렬을 그대로 지킨 최소 예시입니다.
VALID_ROWS = (
    "1,10,3,1.0,2.0,3.0,4.0,0.9",
    "2,10,5,1.0,2.0,3.0,4.0,0.5",
    "3,20,1,0.0,0.0,1.0,1.0,0.7",
)


def write_csv(path: Path, *lines: str) -> Path:
    """BOM 없는 UTF-8, LF 줄바꿈으로 CSV를 씁니다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def write_valid_csv(path: Path) -> Path:
    return write_csv(path, HEADER_LINE, *VALID_ROWS)


# --- 성공 경로 -------------------------------------------------------------


def test_valid_submission_csv_is_accepted(tmp_path: Path):
    result = check_submission_csv(write_valid_csv(tmp_path / "submission.csv"))

    assert result == {
        "row_count": 3,
        "image_count": 2,
        "max_detections_per_image": 2,
    }


def test_header_only_csv_is_valid(tmp_path: Path):
    """계약이 '검출 0개여도 header만 있는 CSV는 유효하다'고 명시합니다."""

    result = check_submission_csv(write_csv(tmp_path / "submission.csv", HEADER_LINE))

    assert result == {
        "row_count": 0,
        "image_count": 0,
        "max_detections_per_image": 0,
    }


# --- 실패 경로 -------------------------------------------------------------


def test_wrong_header_is_rejected(tmp_path: Path):
    path = write_csv(
        tmp_path / "submission.csv",
        "image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score",
        "10,3,1.0,2.0,3.0,4.0,0.9",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_row_with_wrong_field_count_is_rejected(tmp_path: Path):
    path = write_csv(tmp_path / "submission.csv", HEADER_LINE, "1,10,3,1.0,2.0,3.0,4.0")

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_annotation_id_must_start_at_one_and_increase(tmp_path: Path):
    path = write_csv(
        tmp_path / "submission.csv",
        HEADER_LINE,
        "7,10,3,1.0,2.0,3.0,4.0,0.9",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_more_than_four_detections_for_one_image_is_rejected(tmp_path: Path):
    path = write_csv(
        tmp_path / "submission.csv",
        HEADER_LINE,
        "1,10,1,0.0,0.0,1.0,1.0,0.9",
        "2,10,1,0.0,0.0,1.0,1.0,0.8",
        "3,10,1,0.0,0.0,1.0,1.0,0.7",
        "4,10,1,0.0,0.0,1.0,1.0,0.6",
        "5,10,1,0.0,0.0,1.0,1.0,0.5",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_image_id_must_be_ascending(tmp_path: Path):
    path = write_csv(
        tmp_path / "submission.csv",
        HEADER_LINE,
        "1,20,1,0.0,0.0,1.0,1.0,0.9",
        "2,10,1,0.0,0.0,1.0,1.0,0.8",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_score_must_descend_within_one_image(tmp_path: Path):
    path = write_csv(
        tmp_path / "submission.csv",
        HEADER_LINE,
        "1,10,1,0.0,0.0,1.0,1.0,0.4",
        "2,10,1,0.0,0.0,1.0,1.0,0.9",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_non_finite_score_is_rejected(tmp_path: Path):
    path = write_csv(
        tmp_path / "submission.csv",
        HEADER_LINE,
        "1,10,1,0.0,0.0,1.0,1.0,NaN",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_non_integer_image_id_is_rejected(tmp_path: Path):
    path = write_csv(
        tmp_path / "submission.csv",
        HEADER_LINE,
        "1,10.5,1,0.0,0.0,1.0,1.0,0.9",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


def test_bom_is_rejected(tmp_path: Path):
    """저장소 전체가 BOM 없는 UTF-8을 요구합니다."""

    path = tmp_path / "submission.csv"
    path.write_text(
        "\n".join((HEADER_LINE, *VALID_ROWS)) + "\n",
        encoding="utf-8-sig",
        newline="\n",
    )

    with pytest.raises(InvalidSubmissionError):
        check_submission_csv(path)


# --- run() 경계 ------------------------------------------------------------


def prepare_submission_run(tmp_path: Path) -> tuple[Path, dict, Path]:
    """submission_uri까지 갖춘 local 실행 입력을 만듭니다."""

    inputs = load_fixture("inputs_local.json")
    inputs["evaluate"]["submission_uri"] = "submissions/exp-0001/submission.csv"
    materialize_local_artifacts(inputs, tmp_path)
    submission_path = tmp_path / inputs["evaluate"]["submission_uri"]
    return tmp_path, inputs, submission_path


def test_run_accepts_a_valid_submission(tmp_path: Path):
    repo_root, inputs, submission_path = prepare_submission_run(tmp_path)
    write_valid_csv(submission_path)

    result = registry.run(make_config(repo_root, inputs))

    assert result["status"] == "ok"
    record = json.loads(
        (repo_root / result["artifacts"]["experiment_record_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert record["submission_check"] == {
        "checked": True,
        "row_count": 3,
        "image_count": 2,
        "max_detections_per_image": 2,
        "skipped_reason": None,
    }


def test_run_rejects_a_malformed_submission_and_writes_no_record(tmp_path: Path):
    repo_root, inputs, submission_path = prepare_submission_run(tmp_path)
    write_csv(submission_path, HEADER_LINE, "1,10,1,0.0,0.0,1.0,1.0")

    result = registry.run(make_config(repo_root, inputs))

    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert result["summary"]["error"] == "InvalidSubmissionError"
    assert not (repo_root / "artifacts" / "registry").exists()


def test_verify_artifacts_false_skips_content_check_but_keeps_uri_safety(
    tmp_path: Path,
):
    repo_root, inputs, submission_path = prepare_submission_run(tmp_path)
    write_csv(submission_path, HEADER_LINE, "1,10,1,0.0,0.0,1.0,1.0")

    result = registry.run(make_config(repo_root, inputs, verify_artifacts=False))

    assert result["status"] == "ok"
    record = json.loads(
        (repo_root / result["artifacts"]["experiment_record_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert record["submission_check"]["checked"] is False
    assert record["submission_check"]["skipped_reason"] is not None

    # 안전 장치는 verify_artifacts와 무관하게 살아 있어야 합니다.
    escaping = load_fixture("inputs_local.json")
    escaping["evaluate"]["submission_uri"] = "../바깥/submission.csv"
    failed = registry.run(make_config(repo_root, escaping, verify_artifacts=False))
    assert failed["status"] == "error"
    assert failed["summary"]["error"] == "InvalidSchemaError"


def test_remote_submission_is_not_content_checked(tmp_path: Path):
    """원격 artifact는 AWS 접근 없이 참조만 기록한다는 기존 정책을 따릅니다."""

    inputs = load_fixture("inputs_local.json")
    materialize_local_artifacts(inputs, tmp_path)
    # 원격 URI는 실제 파일을 만들 수 없으므로 materialize 뒤에 넣습니다.
    inputs["evaluate"]["submission_uri"] = "s3://example-bucket/submissions/x.csv"

    result = registry.run(make_config(tmp_path, inputs))

    assert result["status"] == "ok"
    record = json.loads(
        (tmp_path / result["artifacts"]["experiment_record_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert record["submission_check"]["checked"] is False


def test_run_without_submission_reports_it_was_not_checked(tmp_path: Path):
    inputs = load_fixture("inputs_local.json")
    materialize_local_artifacts(inputs, tmp_path)

    result = registry.run(make_config(tmp_path, inputs))

    record = json.loads(
        (tmp_path / result["artifacts"]["experiment_record_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert record["submission_check"]["checked"] is False
    assert record["submission_check"]["row_count"] is None
