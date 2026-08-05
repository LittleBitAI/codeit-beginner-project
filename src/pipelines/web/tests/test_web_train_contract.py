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

from src.pipelines.web.api.routes_train import ARCHITECTURE
from src.pipelines.web.train_capabilities import LEGACY_OPTIMIZER
from src.pipelines.web.train_config import (
    DATA_ARTIFACT_KEYS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PREFIX,
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


def test_architecture_matches_train_source():
    """화면에 보여 주는 모델 이름이 실제로 학습되는 모델과 같아야 합니다."""

    assert ARCHITECTURE == module_constant(read_source("model.py"), "ARCHITECTURE")


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
        assert mirrored[name] == expected, f"train.{name} 기본값이 어긋났습니다."


def test_string_and_boolean_defaults_match_train_source():
    train_defaults = get_defaults(read_source("pipeline.py"))

    assert train_defaults.get("device") == "cpu"
    assert train_defaults.get("pretrained") is False
    assert train_defaults.get("output_dir") == DEFAULT_OUTPUT_DIR
    assert train_defaults.get("output_prefix") == DEFAULT_OUTPUT_PREFIX


def test_run_id_pattern_matches_train_source():
    assert RUN_ID_PATTERN.pattern == module_constant(
        read_source("pipeline.py"), "RUN_ID_PATTERN"
    )


def test_required_data_artifact_keys_match_train_source():
    train_keys = module_constant(read_source("pipeline.py"), "DATA_ARTIFACT_KEYS")

    assert set(DATA_ARTIFACT_KEYS) == set(train_keys)
