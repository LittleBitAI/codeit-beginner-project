"""전처리 결과 폴더에서 artifact 4개를 찾아내는 기능.

이 저장소에는 data artifact의 표준 file 이름이 없어서 내용으로 판별합니다.
그래서 이름이 달라도 찾아야 하고, 형태가 같아 헷갈리는 경우는 사실대로 알려야 합니다.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.web import datasets
from src.pipelines.web.errors import WebValidationError
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS


MANIFEST = {
    "images": [{"id": 1, "file_name": "a.png", "width": 16, "height": 12}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [2, 3, 5, 4], "iscrowd": 0}],
    "categories": [{"id": 7, "name": "pill"}],
}
CLASS_MAP = {"pill": 1}
SUMMARY = {"train_images": 1, "validation_images": 1}


def write(directory, name, document) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8", newline="\n"
    )


@pytest.fixture
def dataset_dir(isolated_repo):
    directory = isolated_repo / "artifacts" / "demo"
    write(directory, "train.json", MANIFEST)
    write(directory, "validation.json", MANIFEST)
    write(directory, "class_map.json", CLASS_MAP)
    write(directory, "summary.json", SUMMARY)
    return "artifacts/demo"


# --- 내용으로 판별 ----------------------------------------------------------


def test_classify_recognises_each_shape():
    assert datasets.classify_document(MANIFEST) == "manifest"
    assert datasets.classify_document(CLASS_MAP) == "class_map"
    assert datasets.classify_document({"anything": "else"}) == "summary"
    assert datasets.classify_document([]) == "unknown"
    assert datasets.classify_document("문자열") == "unknown"


def test_classify_accepts_id_to_name_class_map():
    """train의 load_class_map은 두 방향을 모두 받습니다."""

    assert datasets.classify_document({"7": "pill"}) == "class_map"


def test_finds_all_four_artifacts(dataset_dir):
    result = datasets.inspect_directory(dataset_dir)

    assert result["complete"] is True
    assert result["missing"] == []
    assert result["problems"] == []
    assert set(result["data"]) == set(DATA_ARTIFACT_KEYS)
    assert result["data"]["train_manifest_uri"] == "artifacts/demo/train.json"
    assert result["data"]["class_map_uri"] == "artifacts/demo/class_map.json"
    assert result["data"]["dataset_summary_uri"] == "artifacts/demo/summary.json"


def test_finds_artifacts_with_unconventional_names(isolated_repo):
    """file 이름 규약에 기대지 않습니다."""

    directory = isolated_repo / "artifacts" / "odd"
    write(directory, "coco_train_split.json", MANIFEST)
    write(directory, "coco_val_split.json", MANIFEST)
    write(directory, "labels.json", CLASS_MAP)
    write(directory, "dataset_stats.json", SUMMARY)

    result = datasets.inspect_directory("artifacts/odd")

    assert result["complete"] is True
    assert result["matched"]["train_manifest_uri"]["name"] == "coco_train_split.json"
    assert result["matched"]["validation_manifest_uri"]["name"] == "coco_val_split.json"
    assert result["matched"]["class_map_uri"]["name"] == "labels.json"
    assert result["matched"]["dataset_summary_uri"]["name"] == "dataset_stats.json"


def test_validation_hint_wins_over_train_hint(isolated_repo):
    """'validation'에는 'val'과 'trn'이 섞일 수 있어 순서가 중요합니다."""

    directory = isolated_repo / "artifacts" / "hints"
    write(directory, "train.json", MANIFEST)
    write(directory, "validation.json", MANIFEST)
    write(directory, "class_map.json", CLASS_MAP)
    write(directory, "summary.json", SUMMARY)

    result = datasets.inspect_directory("artifacts/hints")

    assert result["matched"]["validation_manifest_uri"]["name"] == "validation.json"


def test_class_map_and_summary_are_separated_by_name(isolated_repo):
    """{"pill": 1}과 {"train_images": 1}은 형태가 완전히 같습니다."""

    directory = isolated_repo / "artifacts" / "ambiguous"
    write(directory, "train.json", MANIFEST)
    write(directory, "validation.json", MANIFEST)
    write(directory, "class_map.json", {"pill": 1, "tablet": 2})
    write(directory, "summary.json", {"train_images": 10, "validation_images": 3})

    result = datasets.inspect_directory("artifacts/ambiguous")

    assert result["complete"] is True
    assert result["matched"]["class_map_uri"]["name"] == "class_map.json"
    assert result["matched"]["dataset_summary_uri"]["name"] == "summary.json"


# --- 못 찾거나 헷갈리는 경우 ------------------------------------------------


def test_reports_missing_artifacts_instead_of_guessing(isolated_repo):
    directory = isolated_repo / "artifacts" / "partial"
    write(directory, "train.json", MANIFEST)

    result = datasets.inspect_directory("artifacts/partial")

    assert result["complete"] is False
    assert "validation_manifest_uri" in result["missing"]
    assert "class_map_uri" in result["missing"]


def test_reports_when_manifests_cannot_be_told_apart(isolated_repo):
    directory = isolated_repo / "artifacts" / "nohints"
    write(directory, "a.json", MANIFEST)
    write(directory, "b.json", MANIFEST)
    write(directory, "class_map.json", CLASS_MAP)
    write(directory, "summary.json", SUMMARY)

    result = datasets.inspect_directory("artifacts/nohints")

    assert result["complete"] is False
    assert any("가릴 수 없습니다" in problem for problem in result["problems"])


def test_empty_directory_is_reported(isolated_repo):
    (isolated_repo / "artifacts" / "empty").mkdir(parents=True)

    result = datasets.inspect_directory("artifacts/empty")

    assert result["complete"] is False
    assert any("JSON 파일이 없습니다" in problem for problem in result["problems"])


def test_broken_json_is_reported_not_raised(isolated_repo):
    directory = isolated_repo / "artifacts" / "broken"
    directory.mkdir(parents=True)
    (directory / "bad.json").write_text("{깨짐", encoding="utf-8", newline="\n")

    result = datasets.inspect_directory("artifacts/broken")

    entry = result["examined"][0]
    assert entry["kind"] == "unknown"
    assert entry["problem"] == "올바른 JSON이 아닙니다."


def test_missing_directory_is_rejected(isolated_repo):
    with pytest.raises(WebValidationError) as error:
        datasets.inspect_directory("artifacts/none")

    assert "폴더가 없습니다" in error.value.errors[0].message


@pytest.mark.parametrize(
    "bad", ("../outside", "/etc", "C:/Windows", "\\\\server\\share", "", "   ", None, 123)
)
def test_directory_cannot_leave_the_repository(isolated_repo, bad):
    with pytest.raises(WebValidationError):
        datasets.inspect_directory(bad)


def test_non_json_files_are_ignored(isolated_repo, dataset_dir):
    (isolated_repo / "artifacts" / "demo" / "train.png").write_bytes(b"\x89PNG")

    result = datasets.inspect_directory(dataset_dir)

    assert all(entry["name"].endswith(".json") for entry in result["examined"])
    assert result["complete"] is True


def test_large_file_is_recognised_from_its_head(isolated_repo):
    """manifest는 이미지가 많으면 수십 MB가 됩니다. 앞부분만 봅니다."""

    directory = isolated_repo / "artifacts" / "big"
    directory.mkdir(parents=True)
    padded = dict(MANIFEST)
    padded["images"] = [
        {"id": index, "file_name": f"{index}.png", "width": 16, "height": 12}
        for index in range(1, 40000)
    ]
    (directory / "train.json").write_text(json.dumps(padded), encoding="utf-8", newline="\n")

    result = datasets.inspect_directory("artifacts/big")

    assert (directory / "train.json").stat().st_size > datasets.FULL_PARSE_LIMIT
    assert result["examined"][0]["kind"] == "manifest"


# --- 선택 저장 --------------------------------------------------------------


def test_selection_round_trips(isolated_repo, dataset_dir):
    assert datasets.load_selection() is None

    saved = datasets.save_selection(dataset_dir)
    assert saved["complete"] is True
    assert saved["selected_at"]

    loaded = datasets.load_selection()
    assert loaded is not None
    assert loaded["directory"] == "artifacts/demo"
    assert loaded["available"] is True
    assert loaded["data"] == saved["data"]


def test_incomplete_directory_cannot_be_selected(isolated_repo):
    directory = isolated_repo / "artifacts" / "partial"
    write(directory, "train.json", MANIFEST)

    with pytest.raises(WebValidationError) as error:
        datasets.save_selection("artifacts/partial")

    assert "찾지 못했습니다" in error.value.errors[0].message
    assert datasets.load_selection() is None


def test_selection_reports_when_folder_disappears(isolated_repo, dataset_dir):
    import shutil

    datasets.save_selection(dataset_dir)
    shutil.rmtree(isolated_repo / "artifacts" / "demo")

    loaded = datasets.load_selection()

    # 선택은 남기되 지금 쓸 수 없다는 사실을 그대로 알립니다.
    assert loaded is not None
    assert loaded["available"] is False
    assert loaded["complete"] is False
    assert loaded["problems"]


def test_selection_reflects_folder_changes(isolated_repo, dataset_dir):
    datasets.save_selection(dataset_dir)
    (isolated_repo / "artifacts" / "demo" / "class_map.json").unlink()

    loaded = datasets.load_selection()

    assert loaded["complete"] is False
    assert "class_map_uri" in loaded["missing"]


def test_clear_selection(isolated_repo, dataset_dir):
    datasets.save_selection(dataset_dir)

    datasets.clear_selection()

    assert datasets.load_selection() is None


# --- data pipeline으로 검증 -------------------------------------------------


def test_data_config_never_uses_dummy_mode():
    """execution.mode가 dummy면 data가 검증을 건너뛰고 dummy 결과만 돌려줍니다."""

    config = datasets.build_data_config({key: f"artifacts/{key}.json" for key in DATA_ARTIFACT_KEYS})

    assert config["execution"] == {"mode": "real"}
    assert set(config["inputs"]["data"]) == set(DATA_ARTIFACT_KEYS)


def test_data_config_switches_to_s3_backend():
    config = datasets.build_data_config({key: f"s3://bucket/{key}.json" for key in DATA_ARTIFACT_KEYS})

    assert config["storage"]["backend"] == "s3"


def test_verify_calls_the_public_cli_with_only_data(isolated_repo, monkeypatch):
    """train과 같은 방식으로 공개 CLI만 부릅니다."""

    import subprocess
    import sys

    captured = {}

    def fake_run_stage(config_relative_path, stage, *, cwd, timeout):
        captured["path"] = config_relative_path
        captured["stage"] = stage
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "ok",
                    "artifacts": {"data": {key: "x" for key in DATA_ARTIFACT_KEYS}},
                    "summary": {"data": {"pipeline": "data", "mode": "integration"}},
                    "message": "pipeline 실행을 완료했습니다.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            stderr="",
        )

    monkeypatch.setattr(datasets.runner, "run_stage", fake_run_stage)

    result = datasets.verify_with_pipeline({key: f"artifacts/{key}.json" for key in DATA_ARTIFACT_KEYS})

    assert captured["stage"] == "data"
    assert captured["path"].startswith("artifacts/web/configs/")
    assert result["ok"] is True
    # stage 이름으로 감싼 한 겹을 벗겨야 합니다.
    assert set(result["artifacts"]) == set(DATA_ARTIFACT_KEYS)
    assert result["summary"]["mode"] == "integration"
    assert sys.executable  # argv 구성은 runner test가 확인합니다


def test_verify_removes_the_temporary_config(isolated_repo, monkeypatch):
    import subprocess

    monkeypatch.setattr(
        datasets.runner,
        "run_stage",
        lambda *a, **k: subprocess.CompletedProcess([], 0, '{"status":"ok"}', ""),
    )

    datasets.verify_with_pipeline({key: f"artifacts/{key}.json" for key in DATA_ARTIFACT_KEYS})

    leftovers = list((isolated_repo / "artifacts" / "web" / "configs").glob("*.json"))
    assert leftovers == []


def test_verify_reports_pipeline_failure(isolated_repo, monkeypatch):
    import subprocess

    monkeypatch.setattr(
        datasets.runner,
        "run_stage",
        lambda *a, **k: subprocess.CompletedProcess(
            [],
            1,
            json.dumps({"status": "error", "artifacts": {}, "summary": {}, "message": "data: 실패"}),
            "",
        ),
    )

    result = datasets.verify_with_pipeline({key: "x" for key in DATA_ARTIFACT_KEYS})

    assert result["ok"] is False
    assert result["exit_code"] == 1
    assert "실패" in result["message"]


def test_verify_handles_timeout(isolated_repo, monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(datasets.runner, "run_stage", timeout)

    result = datasets.verify_with_pipeline({key: "x" for key in DATA_ARTIFACT_KEYS})

    assert result["ok"] is False
    assert "시간 안에 끝나지 않았습니다" in result["message"]


def test_verify_handles_unparsable_output_without_leaking_paths(isolated_repo, monkeypatch):
    import subprocess

    from src.pipelines.web.paths import REPOSITORY_ROOT

    monkeypatch.setattr(
        datasets.runner,
        "run_stage",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "깨진 출력", f"실패 {REPOSITORY_ROOT}/x"),
    )

    result = datasets.verify_with_pipeline({key: "x" for key in DATA_ARTIFACT_KEYS})

    assert result["ok"] is False
    assert str(REPOSITORY_ROOT) not in result["message"]


# --- 원본에서 준비 실행 -------------------------------------------------------


@pytest.mark.parametrize("ratio", ("8:2", "9:1"))
def test_prepare_config_carries_the_chosen_ratio(ratio):
    config = datasets.build_prepare_config(ratio, seed=7, overwrite=True)

    assert config["data"] == {
        "prepare": True,
        "split_ratio": ratio,
        "seed": 7,
        "overwrite": True,
    }
    # dummy면 data가 준비를 건너뛰고 dummy 결과만 돌려줍니다.
    assert config["execution"] == {"mode": "real"}


@pytest.mark.parametrize("bad", ("7:3", "80:20", "0.2", "", "8:2 ", None, 0.2, True))
def test_prepare_config_rejects_other_ratios(bad):
    with pytest.raises(WebValidationError) as error:
        datasets.build_prepare_config(bad)

    assert error.value.errors[0].field == "split_ratio"


@pytest.mark.parametrize("bad", (-1, 2**32, "42", True, 1.5))
def test_prepare_config_rejects_bad_seed(bad):
    with pytest.raises(WebValidationError) as error:
        datasets.build_prepare_config("8:2", seed=bad)

    assert error.value.errors[0].field == "seed"


def completed_prepare(status: str = "ok", mode: str = "prepare", returncode: int = 0):
    import subprocess

    return subprocess.CompletedProcess(
        [],
        returncode,
        json.dumps(
            {
                "status": status,
                "artifacts": {"data": {key: f"artifacts/p/{key}.json" for key in DATA_ARTIFACT_KEYS}},
                "summary": {
                    "data": {
                        "pipeline": "data",
                        "mode": mode,
                        "split_ratio": "8:2",
                        "train_images": 8,
                        "validation_images": 2,
                    }
                },
                "message": "준비 완료",
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
    )


def test_prepare_runs_only_data_stage(isolated_repo, monkeypatch):
    captured = {}

    def fake_run_stage(path, stage, *, cwd, timeout):
        captured["stage"] = stage
        captured["timeout"] = timeout
        return completed_prepare()

    monkeypatch.setattr(datasets.runner, "run_stage", fake_run_stage)

    result = datasets.prepare_dataset(datasets.build_prepare_config("8:2"))

    assert captured["stage"] == "data"
    assert captured["timeout"] == datasets.PREPARE_TIMEOUT_SECONDS
    assert result["ok"] is True
    assert result["supported"] is True
    assert set(result["artifacts"]) == set(DATA_ARTIFACT_KEYS)
    assert result["summary"]["train_images"] == 8


def test_prepare_detects_a_pipeline_without_the_feature(isolated_repo, monkeypatch):
    """준비를 요청했는데 mode가 prepare가 아니면 그 pipeline은 이 기능을 모릅니다."""

    monkeypatch.setattr(
        datasets.runner,
        "run_stage",
        lambda *a, **k: completed_prepare(status="error", mode="integration", returncode=1),
    )

    result = datasets.prepare_dataset(datasets.build_prepare_config("8:2"))

    assert result["ok"] is False
    assert result["supported"] is False
    assert "지원하지 않습니다" in result["message"]


def test_prepare_removes_the_temporary_config(isolated_repo, monkeypatch):
    monkeypatch.setattr(datasets.runner, "run_stage", lambda *a, **k: completed_prepare())

    datasets.prepare_dataset(datasets.build_prepare_config("9:1"))

    assert list((isolated_repo / "artifacts" / "web" / "configs").glob("*.json")) == []


def test_prepare_handles_timeout(isolated_repo, monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(datasets.runner, "run_stage", timeout)

    result = datasets.prepare_dataset(datasets.build_prepare_config("8:2"))

    assert result["ok"] is False
    assert "시간 안에 끝나지 않았습니다" in result["message"]


# --- 준비 결과를 데이터셋으로 고르기 -----------------------------------------


def test_prepared_selection_round_trips(isolated_repo):
    data = {key: f"s3://bucket/processed/{key}.json" for key in DATA_ARTIFACT_KEYS}

    saved = datasets.save_prepared_selection(data, {"split_ratio": "9:1", "processed_prefix": "p/"})

    assert saved["origin"] == "prepared"
    assert saved["complete"] is True
    assert saved["data"] == data

    loaded = datasets.load_selection()
    assert loaded["origin"] == "prepared"
    assert loaded["data"] == data
    # s3 산출물이라 폴더를 훑을 수 없습니다. 기록해 둔 값을 그대로 씁니다.
    assert loaded["available"] is True
    assert loaded["preparation"]["split_ratio"] == "9:1"


def test_prepared_selection_rejects_incomplete_data(isolated_repo):
    with pytest.raises(WebValidationError):
        datasets.save_prepared_selection({"train_manifest_uri": "a"})


def test_folder_selection_still_reports_its_origin(isolated_repo, dataset_dir):
    saved = datasets.save_selection(dataset_dir)

    assert saved["origin"] == "folder"
    assert datasets.load_selection()["origin"] == "folder"


def test_corrupt_selection_file_is_ignored(isolated_repo):
    path = isolated_repo / "artifacts" / "web" / "data_source.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{깨짐", encoding="utf-8", newline="\n")

    assert datasets.load_selection() is None
