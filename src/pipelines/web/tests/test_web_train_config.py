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


def _device_spec() -> dict:
    return next(item for item in train_config.field_specs() if item["name"] == "device")


def test_device_form_default_is_cuda_when_this_computer_has_one(monkeypatch):
    """GPU가 있는데 폼이 cpu로 시작하면 매번 사람이 바꿔 줘야 합니다.

    이 화면은 GPU가 달린 컴퓨터에서 학습을 돌리려고 만든 것입니다. 바꾸는 것을
    잊으면 몇 분이면 끝날 학습이 몇 시간짜리가 됩니다.
    """

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    assert _device_spec()["default"] == "cuda"


def test_device_form_default_falls_back_to_cpu_without_cuda(monkeypatch):
    """CUDA가 없는 컴퓨터에서 cuda로 채우면 저장 자체가 막힙니다.

    device 검증이 'CUDA를 사용할 수 없는 환경입니다'로 거부하므로, 폼이 처음부터
    저장할 수 없는 값을 들고 시작하게 됩니다.
    """

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: False)

    assert _device_spec()["default"] == "cpu"


def _precision_spec() -> dict:
    return next(item for item in train_config.field_specs() if item["name"] == "precision")


def test_precision_form_default_is_amp_when_this_computer_has_a_gpu(monkeypatch):
    """GPU가 있으면 절반 정밀도로 시작합니다.

    같은 GPU에서 더 빠르고 메모리를 덜 씁니다. Device 기본값과 짝을 이룹니다.
    amp는 device가 cuda여야 하므로 둘이 따로 놀면 폼이 저장할 수 없는 조합으로
    시작하게 됩니다.
    """

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    assert _precision_spec()["default"] == "amp"


def test_precision_form_default_falls_back_to_fp32_without_cuda(monkeypatch):
    """CUDA가 없으면 amp는 저장 자체가 막힙니다. fp32로 시작해야 합니다."""

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: False)

    assert _precision_spec()["default"] == "fp32"


def test_train_precision_default_stays_fp32(monkeypatch):
    """폼 기본값만 바꿉니다. 값을 주지 않았을 때의 fallback은 train과 같아야 합니다."""

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    assert normalize_train_settings({})["precision"] == "fp32"


def test_train_device_default_stays_cpu(monkeypatch):
    """폼 기본값만 바꿉니다. train의 기본값은 건드리지 않습니다.

    train은 GPU가 없는 곳에서도 돌아야 하고, test_web_train_contract.py가 web의
    정규화 기본값과 train source를 대조합니다. pretrained와 같은 구조입니다.
    """

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    assert normalize_train_settings({"device": "cuda"})["device"] == "cuda"
    # 값을 주지 않았을 때의 fallback은 train과 같아야 합니다.
    monkeypatch.setattr(train_config, "cuda_is_available", lambda: False)
    assert normalize_train_settings({})["device"] == "cpu"


def test_time_based_run_id_matches_train_pattern(monkeypatch):
    """설정을 모를 때 쓰는 마지막 수단입니다. 이어서 학습이 이 이름을 씁니다."""

    from datetime import datetime, timezone

    monkeypatch.setattr(
        train_config,
        "_utc_now",
        lambda: datetime(2026, 8, 5, 1, 2, 3, 456789, tzinfo=timezone.utc),
    )

    assert train_config.generate_run_id() == "web-20260805T010203456789Z"
    assert train_config.RUN_ID_PATTERN.fullmatch(train_config.generate_run_id())


def test_output_prefix_slashes_are_stripped():
    assert normalize_train_settings({"output_prefix": "/a/b/"})["output_prefix"] == "a/b"


# --- 조기 종료 --------------------------------------------------------------


def test_early_stopping_is_off_by_default_and_leaves_the_config_untouched():
    """끄면 key 자체가 없어야 train이 예전과 똑같이 전체 epoch를 돕니다."""

    assert "early_stopping" not in normalize_train_settings({})


def test_early_stopping_reaches_train_as_the_object_it_expects():
    settings = normalize_train_settings(
        {"early_stopping": True, "early_stopping_patience": 5, "early_stopping_min_delta": 0.01}
    )

    assert settings["early_stopping"] == {"patience": 5, "min_delta": 0.01}


def test_early_stopping_patience_falls_back_to_the_default_the_form_shows():
    """칸을 비우면 화면이 "기본값 5"라고 안내한 그 값이 그대로 쓰여야 합니다.

    train은 patience를 필수로 받지만 그것은 train이 받는 object의 규칙입니다.
    다른 수치 칸과 마찬가지로 web이 자기 기본값을 채워 완성된 object를 보냅니다.
    여기서 오류를 내면 스위치를 켜자마자 저장이 막혀 아무것도 할 수 없습니다.
    """

    spec = next(
        item for item in train_config.field_specs() if item["name"] == "early_stopping_patience"
    )
    settings = normalize_train_settings({"early_stopping": True})

    assert settings["early_stopping"]["patience"] == spec["default"]


@pytest.mark.parametrize("patience", (0, -1, True, "5", 2.5, None))
def test_rejects_bad_patience(patience):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"early_stopping": True, "early_stopping_patience": patience})

    assert "train.early_stopping_patience" in fields_of(error.value)


@pytest.mark.parametrize("min_delta", (-0.1, True, "0.1", float("inf"), float("nan")))
def test_rejects_bad_min_delta(min_delta):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings(
            {
                "early_stopping": True,
                "early_stopping_patience": 5,
                "early_stopping_min_delta": min_delta,
            }
        )

    assert "train.early_stopping_min_delta" in fields_of(error.value)


@pytest.mark.parametrize("name", ("early_stopping_patience", "early_stopping_min_delta"))
def test_rejects_settings_that_do_nothing_while_early_stopping_is_off(name):
    """끈 채로 값을 보내면 화면과 실제 학습이 달라 보입니다. SGD의 beta처럼 막습니다."""

    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({name: 5})

    assert f"train.{name}" in fields_of(error.value)


def test_rejects_a_non_boolean_early_stopping_switch():
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"early_stopping": "yes"})

    assert "train.early_stopping" in fields_of(error.value)


# --- learning rate schedule --------------------------------------------------


@pytest.mark.parametrize("raw", ({}, {"lr_scheduler": "none"}))
def test_no_schedule_leaves_the_config_untouched(raw):
    """고르지 않으면 key 자체가 없어야 train이 예전과 똑같은 상수 learning rate로 돕니다.

    화면은 고르지 않아도 enum 기본값 ``none``을 실어 보냅니다. 그래도 config가 예전과
    한 글자도 달라지지 않아야 자동으로 짓는 실행 이름까지 그대로입니다.
    """

    assert "lr_scheduler" not in normalize_train_settings(raw)


def test_schedule_reaches_train_as_the_object_it_expects():
    settings = normalize_train_settings(
        {
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 500,
            "lr_warmup_start_factor": 0.01,
            "lr_min_factor": 0.05,
        }
    )

    assert settings["lr_scheduler"] == {
        "name": "cosine",
        "warmup_steps": 500,
        "warmup_start_factor": 0.01,
        "min_lr_factor": 0.05,
    }


def test_warmup_alone_is_a_real_choice():
    """decay 없이 warmup만 쓰는 것은 정상 조합입니다. 그때도 설정이 train까지 가야 합니다."""

    settings = normalize_train_settings({"lr_scheduler": "none", "lr_warmup_steps": 300})

    assert settings["lr_scheduler"] == {
        "name": "none",
        "warmup_steps": 300,
        "warmup_start_factor": 0.001,
    }


@pytest.mark.parametrize(
    ("name", "field"),
    [("cosine", "lr_step_size"), ("step", "lr_min_factor")],
)
def test_rejects_settings_the_chosen_schedule_does_not_use(name, field):
    """train이 같은 조건을 거부합니다. 여기서 막지 않으면 subprocess가 뜬 뒤에야 압니다."""

    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"lr_scheduler": name, field: 2})

    assert f"train.{field}" in fields_of(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr_warmup_steps", -1),
        ("lr_warmup_start_factor", 0.0),
        ("lr_warmup_start_factor", 1.5),
        ("lr_min_factor", 1.5),
    ],
)
def test_rejects_values_outside_what_train_accepts(field, value):
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"lr_scheduler": "cosine", field: value})

    assert f"train.{field}" in fields_of(error.value)


def test_rejects_an_unknown_schedule():
    with pytest.raises(WebValidationError) as error:
        normalize_train_settings({"lr_scheduler": "sqrt"})

    assert "train.lr_scheduler" in fields_of(error.value)


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
def test_empty_run_id_is_generated_from_the_settings(value):
    """train도 ``raw.get("run_id") or <생성>``이라 빈 값은 자동 생성입니다."""

    settings = normalize_train_settings({"run_id": value, "epochs": 7, "batch_size": 2})

    assert settings["run_id"].startswith("mobile-none-e7-b2-")
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


# --- 자동 실행 이름 ------------------------------------------------------------


def data_uris(dataset: str = "v3-seed42-8020") -> dict[str, str]:
    return {key: f"artifacts/data/{dataset}/{key}.json" for key in DATA_ARTIFACT_KEYS}


def test_generated_run_id_reads_like_the_settings_it_came_from():
    """이름만 보고 무엇으로 돌린 학습인지 알 수 있어야 합니다."""

    settings = normalize_train_settings(
        {
            "architecture": "retinanet_resnet50_fpn_v2",
            "augmentation": "pill_basic",
            "optimizer": "AdamW",
            "epochs": 15,
            "batch_size": 4,
            "learning_rate": 0.006,
            "seed": 42,
        },
        data_uris(),
    )

    name = settings["run_id"]

    assert name.startswith("retina-basic-e15-b4-lr6e3-s42-")
    assert train_config.RUN_ID_PATTERN.fullmatch(name)


def test_same_settings_give_the_same_name_and_other_datasets_do_not():
    """같은 설정이면 같은 이름이라 중복 실험을 바로 알아챕니다.

    이름에 데이터셋이 들어가지 않으므로, 데이터셋 차이는 꼬리표가 맡습니다.
    """

    raw = {"epochs": 15, "batch_size": 4, "seed": 42}
    first = normalize_train_settings(dict(raw), data_uris())["run_id"]
    again = normalize_train_settings(dict(raw), data_uris())["run_id"]
    other_dataset = normalize_train_settings(dict(raw), data_uris("v4-9010"))["run_id"]
    other_seed = normalize_train_settings({**raw, "seed": 7}, data_uris())["run_id"]

    assert first == again
    assert first != other_dataset
    assert first != other_seed


def test_written_run_id_always_wins():
    settings = normalize_train_settings({"run_id": "my-own-name"}, data_uris())

    assert settings["run_id"] == "my-own-name"


def test_unknown_architecture_still_gets_a_name():
    """train에 모델이 늘어도 이름을 못 만들어 저장이 막히면 안 됩니다."""

    name = train_config.generate_settings_run_id(
        {
            "architecture": "some_new_detector_v9",
            "augmentation": "none",
            "epochs": 3,
            "batch_size": 1,
            "learning_rate": 0.001,
            "seed": 1,
        },
        {},
    )

    assert name.startswith("some-none-e3-b1-lr1e3-s1-")
    assert train_config.RUN_ID_PATTERN.fullmatch(name)


def test_resume_does_not_reuse_the_deterministic_name():
    """이어서 학습에 같은 이름을 주면 train이 시작을 거부합니다."""

    settings = normalize_train_settings({"epochs": 15, "seed": 42}, data_uris())
    config = build_runtime_config(settings, normalize_data_inputs(data_uris()))
    config["train"]["resume_from"] = "artifacts/experiments/completed/x/last_checkpoint.pt"

    resumed = train_config.build_resume_config(config)

    assert resumed["train"]["run_id"] != config["train"]["run_id"]
