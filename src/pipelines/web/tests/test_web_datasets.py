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
TEST_MANIFEST = {
    "images": [{"id": 2, "file_name": "2.png", "width": 16, "height": 12}],
    "annotations": [],
    "categories": [{"id": 7, "name": "pill", "supercategory": "pill"}],
}


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


def test_finds_optional_test_manifest_without_making_it_a_train_requirement(isolated_repo):
    directory = isolated_repo / "artifacts" / "competition"
    write(directory, "train.json", MANIFEST)
    write(directory, "validation.json", MANIFEST)
    write(directory, "test_manifest.json", TEST_MANIFEST)
    write(directory, "class_map.json", CLASS_MAP)
    write(directory, "summary.json", SUMMARY)

    result = datasets.inspect_directory("artifacts/competition")

    assert result["complete"] is True
    assert result["missing"] == []
    assert result["data"]["test_manifest_uri"] == "artifacts/competition/test_manifest.json"
    assert result["matched"]["test_manifest_uri"]["name"] == "test_manifest.json"


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


# --- S3에 이미 있는 산출물 고르기 ---------------------------------------------


class FakeS3Storage:
    """s3:// URI를 돌려주는 최소 storage 대역입니다."""

    def __init__(self, documents: dict[str, object]) -> None:
        self.documents = documents

    def list(self, prefix):
        return sorted(uri for uri in self.documents if uri.startswith(str(prefix)))

    def read_json(self, location):
        return self.documents[str(location)]


@pytest.fixture
def fake_s3(monkeypatch):
    base = "s3://bucket/datasets/pill_detection/processed/v1-seed42/"
    documents = {
        base + "train_manifest.json": MANIFEST,
        base + "validation_manifest.json": MANIFEST,
        base + "class_map.json": CLASS_MAP,
        base + "dataset_summary.json": SUMMARY,
        base + "notes.txt": "무시되어야 합니다",
    }
    import src.common as common

    monkeypatch.setattr(common, "create_storage", lambda config: FakeS3Storage(documents))
    return base


def test_finds_artifacts_already_prepared_in_s3(fake_s3):
    """이미 S3에 준비된 산출물을 그대로 쓸 수 있어야 합니다."""

    result = datasets.inspect_directory(fake_s3)

    assert result["complete"] is True
    assert result["directory"] == fake_s3
    assert result["data"]["class_map_uri"] == fake_s3 + "class_map.json"
    # artifact URI가 s3:// 이면 train도 registry도 그대로 받습니다.
    assert all(uri.startswith("s3://") for uri in result["data"].values())


def test_s3_prefix_without_a_trailing_slash_is_accepted(fake_s3):
    result = datasets.inspect_directory(fake_s3.rstrip("/"))

    assert result["complete"] is True


def test_s3_prefix_ignores_non_json_objects(fake_s3):
    result = datasets.inspect_directory(fake_s3)

    assert all(entry["name"].endswith(".json") for entry in result["examined"])


@pytest.mark.parametrize("bad", ("s3://", "s3:///key", "s3://bucket/p/?x=1", "s3://bucket/p/#f"))
def test_bad_s3_locations_are_rejected(bad):
    with pytest.raises(WebValidationError) as error:
        datasets.inspect_directory(bad)

    assert error.value.errors[0].field == "directory"


def test_s3_access_failure_does_not_leak_details(monkeypatch):
    from src.common import StorageError
    import src.common as common

    class Failing:
        def list(self, prefix):
            raise StorageError("token=SENSITIVE 접근 거부")

    monkeypatch.setattr(common, "create_storage", lambda config: Failing())

    with pytest.raises(WebValidationError) as error:
        datasets.inspect_directory("s3://bucket/p/")

    message = error.value.errors[0].message
    assert "SENSITIVE" not in message
    assert "StorageError" in message


def test_s3_selection_round_trips(isolated_repo, fake_s3):
    saved = datasets.save_selection(fake_s3)

    assert saved["origin"] == "folder"
    assert saved["complete"] is True

    loaded = datasets.load_selection()
    assert loaded["directory"] == fake_s3
    assert loaded["data"]["train_manifest_uri"].startswith("s3://")


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


def test_prepare_uses_s3_when_a_bucket_is_configured(monkeypatch):
    """AWS 데이터를 쓰려면 backend가 s3여야 합니다. local로 고정하면 안 됩니다."""

    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "some-bucket")
    monkeypatch.delenv("PILL_STORAGE_BACKEND", raising=False)

    config = datasets.build_prepare_config("8:2", backend="auto")

    assert config["storage"]["backend"] == "s3"
    # bucket 이름은 환경 변수에서 오므로 config 파일에 적지 않습니다.
    assert "some-bucket" not in json.dumps(config)


def test_prepare_falls_back_to_local_without_a_bucket(monkeypatch):
    monkeypatch.delenv("PILL_STORAGE_S3_BUCKET", raising=False)
    monkeypatch.delenv("PILL_STORAGE_BACKEND", raising=False)

    assert datasets.build_prepare_config("8:2", backend="auto")["storage"]["backend"] == "local"


def test_prepare_can_force_local_even_with_a_bucket(monkeypatch):
    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "some-bucket")

    assert datasets.build_prepare_config("8:2", backend="local")["storage"]["backend"] == "local"


def test_prepare_rejects_s3_without_a_bucket(monkeypatch):
    monkeypatch.delenv("PILL_STORAGE_S3_BUCKET", raising=False)

    with pytest.raises(WebValidationError) as error:
        datasets.build_prepare_config("8:2", backend="s3")

    assert error.value.errors[0].field == "backend"


@pytest.mark.parametrize("bad", ("aws", "S3", "", None, 1))
def test_prepare_rejects_unknown_backend(bad):
    with pytest.raises(WebValidationError):
        datasets.build_prepare_config("8:2", backend=bad)


def test_storage_environment_does_not_expose_credentials(monkeypatch):
    monkeypatch.setenv("PILL_STORAGE_S3_BUCKET", "some-bucket")
    monkeypatch.setenv("AWS_PROFILE", "team")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "가짜-비밀-값")

    environment = datasets.storage_environment()

    assert environment["bucket_configured"] is True
    assert environment["profile_configured"] is True
    # profile 이름과 credential 자체는 돌려주지 않습니다.
    serialized = json.dumps(environment, ensure_ascii=False)
    assert "team" not in serialized
    assert "가짜-비밀-값" not in serialized


@pytest.mark.parametrize("bad", (-1, 2**32, "42", True, 1.5))
def test_prepare_config_rejects_bad_seed(bad):
    with pytest.raises(WebValidationError) as error:
        datasets.build_prepare_config("8:2", seed=bad)

    assert error.value.errors[0].field == "seed"


def prepare_stdout(status: str = "ok", mode: str = "prepare") -> str:
    return json.dumps(
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
    )


def spawned(fake_process_factory, monkeypatch, **kwargs):
    """``runner.spawn``이 돌려줄 가짜 process를 심고, 넘어간 argv를 담아 둡니다."""

    captured: dict = {}
    process = fake_process_factory(**kwargs)

    def fake_spawn(argv, *, cwd, env):
        captured["argv"] = argv
        captured["cwd"] = cwd
        return process

    monkeypatch.setattr(datasets.runner, "spawn", fake_spawn)
    return captured, process


def test_prepare_runs_only_data_stage(isolated_repo, monkeypatch, fake_process_factory):
    captured, _ = spawned(fake_process_factory, monkeypatch, stdout=prepare_stdout())

    result = datasets.prepare_dataset(datasets.build_prepare_config("8:2"))

    assert captured["argv"][-2:] == ["--only", "data"]
    assert result["ok"] is True
    assert result["supported"] is True
    assert set(result["artifacts"]) == set(DATA_ARTIFACT_KEYS)
    assert result["summary"]["train_images"] == 8


def test_prepare_feeds_stderr_lines_to_the_progress_callback(
    isolated_repo, monkeypatch, fake_process_factory
):
    """진행 로그는 stderr로만 옵니다. stdout은 결과 JSON 문서 하나입니다."""

    progress_line = json.dumps(
        {"schema": "data.progress/1", "event": "step_started", "step": "split"}
    )
    spawned(
        fake_process_factory,
        monkeypatch,
        stdout=prepare_stdout(),
        stderr=f"{progress_line}\nbotocore 경고\n",
    )
    seen: list[str] = []

    result = datasets.prepare_dataset(
        datasets.build_prepare_config("8:2"), on_progress_line=seen.append
    )

    assert [item.strip() for item in seen] == [progress_line, "botocore 경고"]
    assert result["ok"] is True  # stdout은 그대로 파싱됩니다


def test_prepare_survives_a_progress_callback_that_raises(
    isolated_repo, monkeypatch, fake_process_factory
):
    """진행 로그를 못 읽는다고 준비가 실패하면 안 됩니다."""

    spawned(fake_process_factory, monkeypatch, stdout=prepare_stdout(), stderr="한 줄\n")

    def explode(line):
        raise RuntimeError("진행 로그 처리 실패")

    result = datasets.prepare_dataset(
        datasets.build_prepare_config("8:2"), on_progress_line=explode
    )

    assert result["ok"] is True


def test_prepare_detects_a_pipeline_without_the_feature(
    isolated_repo, monkeypatch, fake_process_factory
):
    """준비를 요청했는데 mode가 prepare가 아니면 그 pipeline은 이 기능을 모릅니다."""

    spawned(
        fake_process_factory,
        monkeypatch,
        stdout=prepare_stdout(status="error", mode="integration"),
        exit_code=1,
    )

    result = datasets.prepare_dataset(datasets.build_prepare_config("8:2"))

    assert result["ok"] is False
    assert result["supported"] is False
    assert "지원하지 않습니다" in result["message"]


def test_prepare_removes_the_temporary_config(isolated_repo, monkeypatch, fake_process_factory):
    spawned(fake_process_factory, monkeypatch, stdout=prepare_stdout())

    datasets.prepare_dataset(datasets.build_prepare_config("9:1"))

    assert list((isolated_repo / "artifacts" / "web" / "configs").glob("*.json")) == []


def test_prepare_handles_timeout(isolated_repo, monkeypatch, fake_process_factory):
    _, process = spawned(fake_process_factory, monkeypatch, block_until_signalled=True)
    monkeypatch.setattr(datasets, "PREPARE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(datasets.runner, "terminate_tree", lambda proc: proc.terminate())

    result = datasets.prepare_dataset(datasets.build_prepare_config("8:2"))

    assert result["ok"] is False
    assert result["exit_code"] is None
    assert "시간 안에 끝나지 않았습니다" in result["message"]
    assert process.terminate_calls == 1  # 시간이 지나면 그냥 두지 않고 종료합니다


def test_prepare_reports_a_process_that_never_started(isolated_repo, monkeypatch):
    def refuse(argv, *, cwd, env):
        raise OSError("실행할 수 없습니다")

    monkeypatch.setattr(datasets.runner, "spawn", refuse)

    result = datasets.prepare_dataset(datasets.build_prepare_config("8:2"))

    assert result["ok"] is False
    assert result["exit_code"] is None
    assert "실행하지 못했습니다" in result["message"]
    assert list((isolated_repo / "artifacts" / "web" / "configs").glob("*.json")) == []


def test_prepare_does_not_deadlock_on_large_output(isolated_repo, monkeypatch):
    """양쪽 pipe를 동시에 읽지 않으면 OS buffer가 차면서 교착합니다.

    실제 subprocess를 써야만 증명되는 성질이라 여기서만 진짜 process를 띄웁니다.
    준비는 하지 않습니다.
    """

    import sys

    script = (
        "import sys\n"
        "for i in range(4000):\n"
        "    sys.stdout.write('x' * 80 + '\\n')\n"
        "    sys.stderr.write('y' * 80 + '\\n')\n"
    )
    monkeypatch.setattr(datasets.runner, "build_argv", lambda path, stage: [sys.executable, "-c", script])
    seen: list[str] = []

    result = datasets.prepare_dataset(
        datasets.build_prepare_config("8:2"), on_progress_line=seen.append
    )

    assert len(seen) == 4000  # stderr를 한 줄도 잃지 않았습니다
    # stdout은 JSON이 아니므로 준비는 실패로 보고됩니다. 교착하지 않는 것이 요점입니다.
    assert result["ok"] is False
    assert result["exit_code"] == 0


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


def test_prepared_selection_preserves_optional_test_manifest(isolated_repo):
    data = {key: f"s3://bucket/processed/{key}.json" for key in DATA_ARTIFACT_KEYS}
    data["test_manifest_uri"] = "s3://bucket/processed/test_manifest.json"

    saved = datasets.save_prepared_selection(
        data,
        {"test_manifest_images": 842, "test_images_used": 0},
    )

    assert saved["complete"] is True
    assert saved["data"]["test_manifest_uri"] == data["test_manifest_uri"]
    assert saved["matched"]["test_manifest_uri"]["name"] == "test_manifest.json"
    assert datasets.load_selection()["data"]["test_manifest_uri"] == data["test_manifest_uri"]


def test_prepared_selection_directory_comes_from_the_artifact_uris(isolated_repo):
    """pipeline이 알려 준 prefix에는 s3://bucket/ 이 빠져 있습니다.

    그 값을 그대로 쓰면 화면에 반쪽짜리 위치가 나오고 다시 조회할 수도 없습니다.
    """

    base = "s3://bucket/datasets/pill_detection/processed/v1-seed42-8020/"
    data = {key: base + f"{key}.json" for key in DATA_ARTIFACT_KEYS}

    saved = datasets.save_prepared_selection(
        data, {"processed_prefix": "datasets/pill_detection/processed/v1-seed42-8020/"}
    )

    assert saved["directory"] == base
    assert datasets.load_selection()["directory"] == base


def test_prepared_selection_keeps_reported_prefix_when_uris_differ(isolated_repo):
    data = {key: f"s3://bucket/somewhere/{key}/{key}.json" for key in DATA_ARTIFACT_KEYS}

    saved = datasets.save_prepared_selection(data, {"processed_prefix": "reported/"})

    # 네 개가 서로 다른 directory에 있으면 공통 위치를 만들 수 없습니다.
    assert saved["directory"] == "reported/"


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


def test_prepare_config_carries_the_source_the_screen_chose():
    """원본을 고를 칸이 없으면 기본 원본 하나만 준비할 수 있습니다."""

    assert "raw_prefix" not in datasets.build_prepare_config("8:2")["data"]
    assert (
        datasets.build_prepare_config("8:2", raw_prefix="datasets/pill_detection/raw/v90/")["data"][
            "raw_prefix"
        ]
        == "datasets/pill_detection/raw/v90/"
    )


@pytest.mark.parametrize(
    "bad", ("../밖/원본/", "/절대/경로/", "C:/원본/", "\\서버\공유\\", "  ")
)
def test_prepare_config_refuses_a_source_path_that_leaves_the_repository(bad: str):
    """잘못된 경로는 202를 주고 subprocess에서 죽는 대신 여기서 거절합니다."""

    with pytest.raises(WebValidationError):
        datasets.build_prepare_config("8:2", raw_prefix=bad)


def test_prepare_config_keeps_the_trailing_slash_data_expects():
    section = datasets.build_prepare_config(
        "8:2", raw_prefix="datasets/pill_detection/raw/v90"
    )["data"]

    assert section["raw_prefix"] == "datasets/pill_detection/raw/v90/"
