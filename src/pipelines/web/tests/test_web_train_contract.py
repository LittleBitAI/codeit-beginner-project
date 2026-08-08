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
    DEFAULT_AUGMENTATION,
    DEFAULT_PRECISION,
    LEGACY_OPTIMIZER,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_AUGMENTATIONS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_PRECISIONS,
)
from src.pipelines.web.train_config import (
    DATA_ARTIFACT_KEYS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PREFIX,
    OPTIMIZER_PROFILES,
    RUN_ID_PATTERN,
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
                    resolved = [
                        module_constant(source, item.id)
                        if isinstance(item, ast.Name)
                        else ast.literal_eval(item)
                        for item in value.elts
                    ]
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


def test_model_and_optimizer_choices_match_train_source():
    assert SUPPORTED_ARCHITECTURES == module_constant(
        read_source("model.py"), "SUPPORTED_ARCHITECTURES"
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
        normalize_train_settings({"precision": "bf16"})


def test_amp_on_cpu_is_rejected_because_train_requires_cuda():
    """train은 amp에 device='cuda'를 요구합니다. subprocess까지 가서 실패할 이유가 없습니다."""

    with pytest.raises(Exception, match="precision"):
        normalize_train_settings({"precision": "amp", "device": "cpu"})


def test_amp_is_accepted_when_the_machine_has_cuda(monkeypatch):
    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    settings = normalize_train_settings({"precision": "amp", "device": "cuda"})

    assert settings["precision"] == "amp"


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
