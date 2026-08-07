"""Train 설정 검증이 train pipeline과 똑같이 동작하는지 확인합니다."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.pipelines.web import train_config
from src.pipelines.web.errors import WebValidationError
from src.pipelines.web.train_config import (
    DATA_ARTIFACT_KEYS,
    build_runtime_config,
    normalize_data_inputs,
    normalize_train_settings,
    read_runtime_config,
    validate_request,
    write_runtime_config,
)


def fields_of(error: WebValidationError) -> set[str]:
    return {item.field for item in error.errors}


# --- 기본값 -----------------------------------------------------------------


def test_defaults_match_train_pipeline_defaults():
    """src/pipelines/train/pipeline.py:79-92의 기본값과 정확히 같아야 합니다.

    train이 기본값을 바꾸면 이 test가 깨져서 알아차릴 수 있습니다.
    """

    settings = normalize_train_settings({})

    assert settings["seed"] == 42
    assert settings["epochs"] == 1
    assert settings["batch_size"] == 1
    assert settings["num_workers"] == 0
    assert settings["learning_rate"] == 0.005
    assert settings["momentum"] == 0.9
    assert settings["weight_decay"] == 0.0005
    assert settings["device"] == "cpu"
    assert settings["pretrained"] is False
    assert settings["output_dir"] == "artifacts/experiments/completed"
    assert settings["output_prefix"] == "experiments/completed"


def test_form_starts_with_pretrained_turned_on():
    """화면에서 시작하는 학습은 COCO 사전학습 가중치를 기본으로 씁니다.

    train 파이프라인의 기본값은 그대로 False입니다. 그쪽은 다른 소유 영역이고
    test_web_train_contract.py가 두 값을 대조합니다. 화면은 이 spec을 보고 폼을
    채우고 pretrained를 명시적으로 실어 보내므로, GUI 학습만 기본이 바뀝니다.
    """

    spec = next(item for item in train_config.field_specs() if item["name"] == "pretrained")

    assert spec["default"] is True


def test_generated_run_id_matches_train_pattern(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(
        train_config,
        "_utc_now",
        lambda: datetime(2026, 8, 5, 1, 2, 3, 456789, tzinfo=timezone.utc),
    )
    settings = normalize_train_settings({})

    assert settings["run_id"] == "web-20260805T010203456789Z"
    assert train_config.RUN_ID_PATTERN.fullmatch(settings["run_id"])


def test_output_prefix_slashes_are_stripped():
    assert normalize_train_settings({"output_prefix": "/a/b/"})["output_prefix"] == "a/b"


# --- 잘못된 값 --------------------------------------------------------------


@pytest.mark.parametrize("name", ("seed", "epochs", "batch_size", "num_workers"))
def test_rejects_bool_for_integer_fields(name):
    """train은 bool을 정수 자리에서 명시적으로 거부합니다(pipeline.py:43)."""

    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({name: True})

    assert f"train.{name}" in fields_of(error.value)


@pytest.mark.parametrize("name", ("learning_rate", "momentum", "weight_decay"))
def test_rejects_bool_for_float_fields(name):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({name: True})

    assert f"train.{name}" in fields_of(error.value)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), -0.1))
def test_rejects_non_finite_or_negative_floats(value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"learning_rate": value})

    assert "train.learning_rate" in fields_of(error.value)


@pytest.mark.parametrize(
    ("name", "value"),
    (("epochs", 0), ("batch_size", 0), ("num_workers", -1), ("seed", -1), ("epochs", "3")),
)
def test_rejects_out_of_range_integers(name, value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({name: value})

    assert f"train.{name}" in fields_of(error.value)


@pytest.mark.parametrize("value", ("tpu", "GPU", "", 1, None))
def test_rejects_unknown_device(value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"device": value})

    assert "train.device" in fields_of(error.value)


def test_rejects_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(train_config, "cuda_is_available", lambda: False)

    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"device": "cuda"})

    assert "train.device" in fields_of(error.value)


def test_accepts_cuda_when_available(monkeypatch):
    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    assert normalize_train_settings({"device": "cuda"})["device"] == "cuda"


@pytest.mark.parametrize("value", ("true", 1, 0, None, "yes"))
def test_rejects_non_boolean_pretrained(value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"pretrained": value})

    assert "train.pretrained" in fields_of(error.value)


@pytest.mark.parametrize("value", ("   ", "-leading", "a b", "a" * 129, "한글", "run;rm -rf /"))
def test_rejects_run_id_with_unsupported_characters(value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"run_id": value})

    assert "train.run_id" in fields_of(error.value)


@pytest.mark.parametrize("value", ("", None))
def test_empty_run_id_is_generated_like_train(value):
    """train도 ``raw.get("run_id") or <생성>``이라 빈 값은 자동 생성입니다."""

    settings = normalize_train_settings({"run_id": value})

    assert settings["run_id"].startswith("web-")
    assert train_config.RUN_ID_PATTERN.fullmatch(settings["run_id"])


@pytest.mark.parametrize(
    "value", ("", "   ", "C:\\tmp", "/tmp/out", "\\\\server\\share", "../outside", "artifacts/../../x")
)
def test_rejects_bad_output_dir(value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"output_dir": value})

    assert "train.output_dir" in fields_of(error.value)


@pytest.mark.parametrize("value", ("artifacts/NUL", "artifacts/CON/x", "artifacts/COM1.txt"))
def test_rejects_windows_reserved_output_dir_names(value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"output_dir": value})

    assert "train.output_dir" in fields_of(error.value)


def test_all_problems_are_reported_together():
    """화면에서 여러 칸의 오류를 한 번에 보여줘야 합니다."""

    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"epochs": 0, "seed": -1, "device": "tpu", "pretrained": "yes"})

    assert fields_of(error.value) == {
        "train.epochs",
        "train.seed",
        "train.device",
        "train.pretrained",
    }


# --- data 입력 --------------------------------------------------------------


@pytest.mark.parametrize("missing", DATA_ARTIFACT_KEYS)
def test_requires_all_four_data_artifacts(data_inputs, missing):
    payload = {key: value for key, value in data_inputs.items() if key != missing}

    with pytest.raises(WebValidationError) as error:
        normalize_data_inputs(payload)

    assert f"inputs.data.{missing}" in fields_of(error.value)


@pytest.mark.parametrize("value", ("", "   ", None, 123, True))
def test_rejects_blank_data_artifact_uri(data_inputs, value):
    payload = dict(data_inputs)
    payload["class_map_uri"] = value

    with pytest.raises(WebValidationError) as error:
        normalize_data_inputs(payload)

    assert "inputs.data.class_map_uri" in fields_of(error.value)


@pytest.mark.parametrize(
    "value",
    (
        "../../etc/passwd",
        "file:///etc/passwd",
        "http://example.com/x.json",
        "C:/secrets/x.json",
        "/etc/passwd",
        "\\\\server\\share\\x.json",
    ),
)
def test_rejects_traversal_and_scheme_in_data_uri(data_inputs, value):
    payload = dict(data_inputs)
    payload["train_manifest_uri"] = value

    with pytest.raises(WebValidationError) as error:
        normalize_data_inputs(payload)

    assert "inputs.data.train_manifest_uri" in fields_of(error.value)


def test_accepts_s3_data_uri():
    payload = {key: f"s3://bucket/data/{key}.json" for key in DATA_ARTIFACT_KEYS}

    resolved = normalize_data_inputs(payload)

    assert resolved["class_map_uri"] == "s3://bucket/data/class_map_uri.json"


def test_optional_test_manifest_is_preserved_in_runtime_input(data_inputs):
    """Train에는 필수가 아니지만 이후 submission 생성까지 전달돼야 합니다."""

    payload = {
        **data_inputs,
        "test_manifest_uri": "artifacts/data/test_manifest.json",
    }

    resolved = normalize_data_inputs(payload)
    config = build_runtime_config(normalize_train_settings({}), resolved)

    assert resolved["test_manifest_uri"] == "artifacts/data/test_manifest.json"
    assert config["inputs"]["data"]["test_manifest_uri"] == resolved["test_manifest_uri"]


def test_optional_test_manifest_uses_the_same_path_safety_rules(data_inputs):
    payload = {**data_inputs, "test_manifest_uri": "../../test_manifest.json"}

    with pytest.raises(WebValidationError) as error:
        normalize_data_inputs(payload)

    assert "inputs.data.test_manifest_uri" in fields_of(error.value)


@pytest.mark.parametrize("value", ("s3://bucket", "s3:///key.json", "s3://bucket/key?x=1"))
def test_rejects_malformed_s3_uri(data_inputs, value):
    payload = dict(data_inputs)
    payload["class_map_uri"] = value

    with pytest.raises(WebValidationError) as error:
        normalize_data_inputs(payload)

    assert "inputs.data.class_map_uri" in fields_of(error.value)


# --- runtime config ---------------------------------------------------------


def test_runtime_config_never_sets_dummy_execution(data_inputs):
    """configs/base.json은 execution.mode가 "dummy"입니다.

    그 값이 그대로 들어가면 train이 학습을 건너뛰고 dummy 결과만 돌려줍니다.
    """

    settings = normalize_train_settings({"epochs": 3})

    config = build_runtime_config(settings, normalize_data_inputs(data_inputs))

    assert config["execution"] == {"mode": "real"}
    assert config["execution"]["mode"] != "dummy"


def test_runtime_config_is_self_contained(data_inputs):
    """load_config는 파일 하나만 읽고 병합하지 않습니다."""

    config = build_runtime_config(
        normalize_train_settings({}), normalize_data_inputs(data_inputs)
    )

    assert set(config) == {"project", "execution", "storage", "train", "inputs"}
    assert set(config["inputs"]["data"]) == set(DATA_ARTIFACT_KEYS)
    assert config["storage"]["backend"] == "local"


def test_runtime_config_uses_s3_backend_for_s3_inputs():
    payload = {key: f"s3://bucket/{key}.json" for key in DATA_ARTIFACT_KEYS}

    config = build_runtime_config(normalize_train_settings({}), normalize_data_inputs(payload))

    assert config["storage"]["backend"] == "s3"
    assert "bucket" not in json.dumps(config["storage"])  # bucket 이름은 환경 변수에서 옵니다.


def test_runtime_config_contains_no_credentials(data_inputs):
    config = build_runtime_config(
        normalize_train_settings({}), normalize_data_inputs(data_inputs)
    )

    serialized = json.dumps(config).lower()
    for hint in ("secret", "token", "password", "access_key", "credential"):
        assert hint not in serialized


def test_builder_does_not_mutate_input(data_inputs):
    raw = {"epochs": 2, "run_id": "keep-me"}
    before_raw = deepcopy(raw)
    before_data = deepcopy(data_inputs)

    build_runtime_config(normalize_train_settings(raw), normalize_data_inputs(data_inputs))

    assert raw == before_raw
    assert data_inputs == before_data


def test_write_runtime_config_uses_uuid_filename_and_lf(isolated_repo, data_inputs):
    config = build_runtime_config(
        normalize_train_settings({}), normalize_data_inputs(data_inputs)
    )

    config_id = write_runtime_config(config)

    assert train_config.re.fullmatch(r"[0-9a-f]{32}", config_id)
    path = isolated_repo / "artifacts" / "web" / "configs" / f"{config_id}.json"
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")  # BOM 없음
    assert b"\r\n" not in raw  # LF
    assert raw.endswith(b"\n")
    assert read_runtime_config(config_id) == config


@pytest.mark.parametrize("bad", ("../../etc/passwd", "zz", "", "A" * 32, "0" * 31))
def test_read_runtime_config_rejects_bad_id(isolated_repo, bad):
    from src.pipelines.web.errors import JobNotFoundError

    with pytest.raises(JobNotFoundError):
        read_runtime_config(bad)


def test_run_id_collision_is_rejected(isolated_repo, data_inputs):
    """train은 학습을 끝낸 뒤에야 FileExistsError를 냅니다(pipeline.py:154)."""

    settings = normalize_train_settings({"run_id": "already-there"})
    existing = isolated_repo / settings["output_dir"] / "already-there"
    existing.mkdir(parents=True)

    result = validate_request({"train": {"run_id": "already-there"}, "inputs": {"data": data_inputs}})

    assert result["valid"] is False
    assert any(item["field"] == "train.run_id" for item in result["errors"])


def test_validate_request_reports_missing_data_files_as_warning(isolated_repo, valid_payload):
    result = validate_request(valid_payload)

    assert result["valid"] is True
    warned = {item["field"] for item in result["warnings"]}
    assert "inputs.data.class_map_uri" in warned


def test_validate_request_rejects_non_object_body():
    result = validate_request("not an object")

    assert result["valid"] is False
    assert result["normalized"] is None
