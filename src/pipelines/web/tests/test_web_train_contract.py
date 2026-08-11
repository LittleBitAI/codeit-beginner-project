"""web이 복제해 둔 train 규칙이 train의 실제 source와 어긋나지 않는지 감시합니다.

web은 train을 import하지 않습니다(소유 경계). 그래서 검증 규칙과 모델 이름을 복제해
두는데, train이 값을 바꾸면 화면이 **실제와 다른 정보를 보여 주게 됩니다**. 실제로
모델 이름이 어긋나 화면에 잘못된 이름이 표시된 적이 있어서 이 test를 넣었습니다.

여기서는 train의 source file을 **글자로만** 읽습니다. import하거나 호출하지 않으므로
runtime 결합이 생기지 않고, 어긋나는 순간 test가 시끄럽게 실패합니다.
``src/pipelines/web/tests/test_web_pipeline.py``도 같은 방식(source 검사)을 씁니다.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.pipelines.web import train_config
from src.pipelines.web.api.routes_train import ARCHITECTURE
from src.pipelines.web.train_capabilities import (
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_AUGMENTATION,
    DEFAULT_INPUT_SIZE,
    MMDETECTION_ARCHITECTURES,
    MMDETECTION_REQUIRED,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_PRECISION,
    LEGACY_OPTIMIZER,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_AUGMENTATIONS,
    SUPPORTED_LR_SCHEDULERS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_PRECISIONS,
)
from src.pipelines.web.train_config import (
    DATA_ARTIFACT_KEYS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PREFIX,
    LR_SCHEDULER_DEFAULTS,
    LR_WARMUP_DEFAULTS,
    OPTIMIZER_PROFILES,
    RUN_ID_PATTERN,
    field_specs,
    normalize_train_settings,
)


TRAIN_DIR = Path(__file__).resolve().parents[2] / "train"


def read_source(name: str) -> str:
    path = TRAIN_DIR / name
    if not path.is_file():
        pytest.fail(
            f"train의 {name}을 찾지 못했습니다. train이 파일을 옮겼다면 이 test가 보는 "
            "위치도 함께 고쳐야 합니다."
        )
    return path.read_text(encoding="utf-8")


def module_constant(source: str, name: str) -> object:
    """모듈 최상위에 있는 상수 하나의 값을 읽습니다.

    ``re.compile("...")`` 처럼 감싸인 경우에는 안쪽 문자열을 꺼냅니다.
    """

    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != name:
                continue
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "compile"
                and value.args
            ):
                value = value.args[0]
            try:
                return ast.literal_eval(value)
            except ValueError:
                if isinstance(value, (ast.Tuple, ast.List)):
                    resolved: list[object] = []
                    for item in value.elts:
                        if isinstance(item, ast.Name):
                            resolved.append(module_constant(source, item.id))
                        elif isinstance(item, ast.Starred) and isinstance(
                            item.value, ast.Name
                        ):
                            # ``*OTHER`` 는 다른 상수를 펼쳐 담은 것입니다. train이
                            # 목록을 나눠 두면 이 모양이 됩니다.
                            resolved.extend(module_constant(source, item.value.id))
                        else:
                            resolved.append(ast.literal_eval(item))
                    return tuple(resolved) if isinstance(value, ast.Tuple) else resolved
                pytest.fail(f"train의 {name} 값을 읽지 못했습니다.")
    pytest.fail(f"train source에서 {name} 상수를 찾지 못했습니다.")


def call_defaults(source: str) -> dict[str, object]:
    """``_integer(raw, "epochs", 1, minimum=1)`` 같은 호출에서 기본값을 모읍니다."""

    found: dict[str, object] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"_integer", "_float"} or len(node.args) < 3:
            continue
        name_node, default_node = node.args[1], node.args[2]
        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
            try:
                found[name_node.value] = ast.literal_eval(default_node)
            except ValueError:
                continue
    return found


def get_defaults(source: str) -> dict[str, object]:
    """``raw.get("device", "cpu")`` 형태의 기본값을 모읍니다."""

    found: dict[str, object] = {}
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            try:
                found[node.args[0].value] = ast.literal_eval(node.args[1])
            except ValueError:
                continue
    return found


def function_node(source: str, name: str) -> ast.FunctionDef:
    """함수 하나를 이름으로 찾습니다. 그 안의 값만 보고 싶을 때 씁니다."""

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"train source에서 {name} 함수를 찾지 못했습니다.")


def test_architecture_matches_train_source():
    """화면에 보여 주는 모델 이름이 실제로 학습되는 모델과 같아야 합니다."""

    assert ARCHITECTURE == module_constant(read_source("model.py"), "ARCHITECTURE")


def test_mmdetection_8gb_combination_matches_train_source():
    """8GB 조합을 두 곳이 따로 들고 있으므로 값이 같은지 감시합니다.

    소유 경계 때문에 상수를 나눠 쓸 수 없습니다. 한쪽만 바꾸면 화면은 통과시키고
    train이 거부하거나, 반대로 화면이 막는데 train은 받는 상태가 됩니다. 둘 다
    사용자가 이유를 알 수 없는 실패로 보입니다.
    """

    assert MMDETECTION_REQUIRED == module_constant(
        read_source("pipeline.py"), "_MMDETECTION_REQUIRED"
    )


def test_the_form_is_told_which_models_use_input_size():
    """화면이 어느 모델에서 이 칸을 보일지 스스로 판단하지 않게 합니다.

    목록을 화면에 옮겨 적으면 여기서 architecture를 더해도 화면은 모르고, 그 값을
    쓰지 않는 모델에서 칸이 보인 채로 남습니다. 사용자는 값을 정할 수 있다고 믿는데
    보내면 거부됩니다.
    """

    spec = next(item for item in field_specs() if item["name"] == "input_size")

    assert spec["only_for_architectures"] == list(MMDETECTION_ARCHITECTURES)


def test_the_form_says_eight_for_models_that_default_to_eight():
    """비워 둔 칸에 화면이 안내하는 값이 실제로 쓰이는 값과 같아야 합니다.

    기본값 하나만 내려보내면 MMDetection을 고르고 비워 둔 사람에게 1이라고 안내하면서
    실제로는 8로 돕니다. 검토 화면의 diff도 그 값을 기본값과 다른 값으로 잘못 봅니다.
    """

    spec = next(
        item
        for item in field_specs()
        if item["name"] == "gradient_accumulation_steps"
    )

    assert spec["default"] == 1
    assert spec["defaults_by_architecture"] == {
        architecture: DEFAULT_ACCUMULATION_STEPS
        for architecture in MMDETECTION_ARCHITECTURES
    }


def test_mmdetection_numeric_defaults_match_train_source():
    """입력 크기와 모으는 수의 기본값도 train이 정한 값이어야 합니다."""

    adapter = read_source("mmdetection_adapter.py")

    assert DEFAULT_INPUT_SIZE == module_constant(adapter, "DEFAULT_INPUT_SIZE")
    assert DEFAULT_ACCUMULATION_STEPS == module_constant(
        adapter, "DEFAULT_ACCUMULATION_STEPS"
    )


def test_model_and_optimizer_choices_match_train_source():
    # train은 목록을 두 파일에 나눠 둡니다. torchvision 이름은 model.py에, MMDetection
    # 이름은 adapter에 있고 model.py가 그것을 펼쳐 담습니다. 한쪽만 읽으면 이 감시가
    # 반쪽이 되므로 두 source를 이어 붙여 읽습니다.
    train_source = (
        read_source("model.py") + chr(10) + read_source("mmdetection_adapter.py")
    )
    assert SUPPORTED_ARCHITECTURES == module_constant(
        train_source, "SUPPORTED_ARCHITECTURES"
    )
    assert MMDETECTION_ARCHITECTURES == module_constant(
        read_source("mmdetection_adapter.py"), "MMDETECTION_ARCHITECTURES"
    )
    assert SUPPORTED_OPTIMIZERS == module_constant(
        read_source("trainer.py"), "SUPPORTED_OPTIMIZERS"
    )


def test_augmentation_choices_match_train_source():
    """화면이 보여 주는 증강 preset이 train이 실제로 받는 이름과 같아야 합니다."""

    presets = module_constant(read_source("pipeline.py"), "AUGMENTATION_PRESETS")
    assert SUPPORTED_AUGMENTATIONS == tuple(presets)
    # train은 값이 없으면 none을 씁니다. 화면 기본값도 같아야 합니다.
    assert DEFAULT_AUGMENTATION in presets


def test_precision_choices_match_train_source():
    """화면이 보여 주는 정밀도가 train이 실제로 받는 이름과 같아야 합니다."""

    modes = module_constant(read_source("pipeline.py"), "PRECISION_MODES")
    assert SUPPORTED_PRECISIONS == tuple(modes)
    # train은 값이 없으면 fp32를 씁니다. 화면 기본값도 같아야 합니다.
    assert DEFAULT_PRECISION in modes


def test_unknown_precision_is_rejected_before_training_starts():
    with pytest.raises(Exception, match="precision"):
        normalize_train_settings({"precision": "fp8"})


@pytest.mark.parametrize("mode", ["amp", "fp16", "bf16"])
def test_half_precision_on_cpu_is_rejected_because_train_requires_cuda(mode):
    """train은 절반 정밀도에 device='cuda'를 요구합니다.

    subprocess까지 가서 실패할 이유가 없습니다. 거기까지 가면 어느 칸이 잘못됐는지
    화면에 남지 않습니다.
    """

    with pytest.raises(Exception, match="precision"):
        normalize_train_settings({"precision": mode, "device": "cpu"})


@pytest.mark.parametrize("mode", ["amp", "fp16"])
def test_modes_that_run_on_any_cuda_gpu_are_accepted(monkeypatch, mode):
    """amp와 fp16은 GPU 세대를 가리지 않습니다. T4에서도 그대로 됩니다."""

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    settings = normalize_train_settings({"precision": mode, "device": "cuda"})

    assert settings["precision"] == mode


def test_bf16_is_refused_on_a_computer_whose_gpu_cannot_do_it(monkeypatch):
    """T4에서 bf16을 고르면 저장하기 전에 막고 이유를 말해 줍니다."""

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)
    monkeypatch.setattr(train_config, "native_bf16_supported", lambda: False)

    with pytest.raises(Exception, match="precision"):
        normalize_train_settings({"precision": "bf16", "device": "cuda"})


def test_bf16_is_accepted_on_a_computer_whose_gpu_can_do_it(monkeypatch):
    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)
    monkeypatch.setattr(train_config, "native_bf16_supported", lambda: True)

    settings = normalize_train_settings({"precision": "bf16", "device": "cuda"})

    assert settings["precision"] == "bf16"


def test_bf16_is_left_to_train_when_this_computer_cannot_tell(monkeypatch):
    """확실하지 않으면 막지 않습니다.

    nvidia-smi가 없다고 bf16을 거부하면, 실제로는 되는 GPU에서 설정 저장 자체를
    못 합니다. 최종 판단은 GPU를 직접 보는 train이 합니다.
    """

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)
    monkeypatch.setattr(train_config, "native_bf16_supported", lambda: None)

    settings = normalize_train_settings({"precision": "bf16", "device": "cuda"})

    assert settings["precision"] == "bf16"


def test_lr_scheduler_choices_and_defaults_match_train_source():
    """화면이 보여 주는 schedule 이름과 기본값이 train이 실제로 받는 것과 같아야 합니다."""

    source = read_source("pipeline.py")
    schedules = module_constant(source, "LR_SCHEDULER_DEFAULTS")

    assert SUPPORTED_LR_SCHEDULERS == tuple(schedules)
    # train은 값이 없으면 상수 learning rate입니다. 화면 기본값도 같아야 합니다.
    assert DEFAULT_LR_SCHEDULER in schedules
    assert LR_SCHEDULER_DEFAULTS == schedules
    assert LR_WARMUP_DEFAULTS == module_constant(source, "LR_WARMUP_DEFAULTS")


@pytest.mark.parametrize("name", SUPPORTED_LR_SCHEDULERS)
def test_lr_scheduler_reaches_train_with_exactly_the_keys_it_uses(name):
    """train은 고른 schedule이 쓰지 않는 key가 하나만 있어도 object를 통째로 거부합니다."""

    schedules = module_constant(read_source("pipeline.py"), "LR_SCHEDULER_DEFAULTS")
    mirrored = normalize_train_settings({"lr_scheduler": name, "lr_warmup_steps": 1})

    assert set(mirrored["lr_scheduler"]) == {
        "name",
        "warmup_steps",
        "warmup_start_factor",
        *schedules[name],
    }


def test_optimizer_profiles_match_train_source():
    assert OPTIMIZER_PROFILES == module_constant(
        read_source("pipeline.py"), "OPTIMIZER_PROFILES"
    )


def test_fallback_optimizer_matches_train_source():
    """Capability이 없을 때 보여 주는 optimizer가 실제 고정 구현과 같아야 합니다."""

    source = read_source("trainer.py")
    assert f"torch.optim.{LEGACY_OPTIMIZER}(" in source


def test_numeric_defaults_match_train_source():
    """비워 둔 칸에 대해 화면이 알려 주는 기본값이 train의 기본값과 같아야 합니다."""

    train_defaults = call_defaults(read_source("pipeline.py"))
    mirrored = normalize_train_settings({})

    assert train_defaults, "train source에서 기본값 호출을 하나도 찾지 못했습니다."
    for name, expected in train_defaults.items():
        if name in {"learning_rate", "weight_decay", "momentum", "epsilon"}:
            continue
        assert mirrored[name] == expected, f"train.{name} 기본값이 어긋났습니다."


def test_train_source_supports_every_resume_setting_the_web_sends():
    """#110 없이 #111만 배포해 201 뒤 처음부터 학습하는 일을 막습니다."""

    source = read_source("pipeline.py")
    numeric_defaults = call_defaults(source)
    mirrored = normalize_train_settings({})
    settings = function_node(source, "_settings")
    read_keys = {
        node.args[0].value
        for node in ast.walk(settings)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
    }

    assert numeric_defaults["checkpoint_every"] == mirrored["checkpoint_every"]
    assert "resume_from" in read_keys


def test_string_and_boolean_defaults_match_train_source():
    train_defaults = get_defaults(read_source("pipeline.py"))

    assert train_defaults.get("device") == "cpu"
    assert train_defaults.get("pretrained") is False
    assert train_defaults.get("output_dir") == DEFAULT_OUTPUT_DIR
    assert train_defaults.get("output_prefix") == DEFAULT_OUTPUT_PREFIX


def test_augmentation_reaches_train_as_the_object_it_expects():
    """train은 preset key 하나만 든 object를 받고 다른 key가 있으면 거부합니다."""

    settings = normalize_train_settings({"augmentation": "pill_basic"})

    assert settings["augmentation"] == {"preset": "pill_basic"}


def test_augmentation_defaults_to_none_like_train():
    assert normalize_train_settings({})["augmentation"] == {"preset": DEFAULT_AUGMENTATION}


def test_unknown_augmentation_is_rejected_before_training_starts():
    with pytest.raises(Exception, match="augmentation"):
        normalize_train_settings({"augmentation": "무작위회전"})


def test_early_stopping_keys_match_train_source():
    """train은 모르는 key가 있으면 object를 통째로 거부합니다.

    화면이 만든 object의 key가 train이 허용하는 집합과 정확히 같아야 합니다.
    """

    function = function_node(read_source("pipeline.py"), "_early_stopping")
    allowed = [ast.literal_eval(node) for node in ast.walk(function) if isinstance(node, ast.Set)]

    assert len(allowed) == 1, "train의 _early_stopping에서 허용 key 집합을 하나만 찾아야 합니다."
    mirrored = normalize_train_settings(
        {"early_stopping": True, "early_stopping_patience": 5}
    )
    assert set(mirrored["early_stopping"]) == allowed[0]


def test_early_stopping_min_delta_default_matches_train_source():
    train_default = get_defaults(read_source("pipeline.py")).get("min_delta")
    mirrored = normalize_train_settings(
        {"early_stopping": True, "early_stopping_patience": 5}
    )

    assert train_default is not None, "train source에서 min_delta 기본값을 찾지 못했습니다."
    assert mirrored["early_stopping"]["min_delta"] == train_default


def test_run_id_pattern_matches_train_source():
    assert RUN_ID_PATTERN.pattern == module_constant(
        read_source("pipeline.py"), "RUN_ID_PATTERN"
    )


def test_required_data_artifact_keys_match_train_source():
    train_keys = module_constant(read_source("pipeline.py"), "DATA_ARTIFACT_KEYS")

    assert set(DATA_ARTIFACT_KEYS) == set(train_keys)


def test_gradient_accumulation_default_is_mirrored_before_train_reads_it():
    """train이 이 설정을 받기 시작해도 기본값이 어긋나지 않게 미리 맞춰 둡니다.

    `test_numeric_defaults_match_train_source`는 train의 기본값을 순회하며 web에
    같은 값이 있는지 봅니다. 그래서 train이 먼저 넣으면 그 순간 web이 깨지고,
    web이 먼저 넣으면 조용히 통과합니다. 순서가 web -> train인 이유입니다.
    """

    mirrored = normalize_train_settings({})

    assert mirrored["gradient_accumulation_steps"] == 1


def test_gradient_accumulation_above_one_is_accepted_now_that_train_reads_it():
    """train이 이 값을 읽기 시작했으므로 화면도 열립니다.

    읽지 않던 동안에는 기본값 말고 어떤 값도 받지 않았습니다. 받아 두면 config에는
    실리지만 학습은 그대로 돌아가고, 화면 기록에만 그 숫자가 남기 때문입니다.
    """

    settings = normalize_train_settings({"gradient_accumulation_steps": 4})

    assert settings["gradient_accumulation_steps"] == 4


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_gradient_accumulation_rejects_values_train_would_not_take(value):
    """train은 bool이 아닌 1 이상의 정수만 받습니다."""

    with pytest.raises(Exception, match="gradient_accumulation_steps"):
        normalize_train_settings({"gradient_accumulation_steps": value})


def test_gradient_accumulation_does_not_change_the_automatic_run_name():
    """생략과 1은 같은 동작이므로 자동 이름도 같아야 합니다.

    자동 이름의 꼬리표는 설정 지문입니다. 실제 학습이 이 변경 전과 똑같은데도 이름이
    달라지면, 같은 설정과 seed로 다시 돌렸을 때 예전 실행과 이름이 달라져 중복 실험을
    알아채지 못합니다. GPU 시간을 두 번 쓰게 됩니다.
    """

    settings = normalize_train_settings({})
    without = {
        name: value
        for name, value in settings.items()
        if name != "gradient_accumulation_steps"
    }

    assert train_config._settings_fingerprint(
        settings, None
    ) == train_config._settings_fingerprint(without, None)


def _mmdetection_request(**overrides):
    """8GB 제약을 갖춘 최소 요청입니다."""

    raw = {
        "architecture": "dino_r50_4scale",
        "device": "cuda",
        "precision": "amp",
        "optimizer": "AdamW",
        "batch_size": 1,
    }
    raw.update(overrides)
    return raw


def test_mmdetection_architectures_are_offered_on_the_new_experiment_form():
    choices = {spec["name"]: spec for spec in field_specs()}

    assert set(MMDETECTION_ARCHITECTURES) <= set(choices["architecture"]["choices"])
    assert "input_size" in choices
    assert "gradient_accumulation_steps" in choices


def test_input_size_defaults_to_the_value_train_uses(monkeypatch):
    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    settings = normalize_train_settings(_mmdetection_request())

    assert settings["input_size"] == 640


def test_input_size_is_refused_with_a_torchvision_architecture():
    """train이 거부하는 조합입니다. queue까지 가서 실패할 이유가 없습니다."""

    with pytest.raises(Exception, match="input_size"):
        normalize_train_settings(
            {"architecture": "retinanet_resnet50_fpn_v2", "input_size": 640}
        )


def test_a_torchvision_run_sends_no_input_size(monkeypatch):
    """train은 이 key가 오면 거부합니다. 기본값이라도 실어 보내면 안 됩니다."""

    settings = normalize_train_settings({})

    assert "input_size" not in settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cpu"),
        ("precision", "fp32"),
        ("optimizer", "SGD"),
        ("batch_size", 2),
    ],
)
def test_mmdetection_refuses_combinations_that_do_not_fit_8gb(
    monkeypatch, field, value
):
    """학습을 시작한 뒤 메모리로 터지면 그 밤을 통째로 버립니다."""

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    with pytest.raises(Exception, match=field):
        normalize_train_settings(_mmdetection_request(**{field: value}))
