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
import re
from pathlib import Path

import pytest

from src.common.train_contract import ARCHITECTURES, SETTING_DEFAULTS, SETTING_KEYS
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


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"optimizer": "SGD", "momentum": 0.9},
        {
            "optimizer": "AdamW",
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "augmentation": "pill_basic",
            "gradient_accumulation_steps": 2,
            "early_stopping": True,
            "early_stopping_patience": 2,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 10,
            "lr_min_factor": 0.1,
            "resume_from": "artifacts/experiments/completed/.old.partial/last_checkpoint.pt",
        },
        {
            "architecture": MMDETECTION_ARCHITECTURES[0],
            "device": "cuda",
            "precision": "amp",
            "optimizer": "AdamW",
            "batch_size": 1,
            "input_size": 640,
        },
    ],
    ids=["defaults", "sgd", "adamw-full", "mmdetection"],
)
def test_no_setting_leaves_here_under_a_name_train_does_not_read(raw):
    """train은 이 파일을 import할 수 없는 저쪽에서 같은 이름으로 값을 읽습니다.

    값이 같은지는 계약의 표들이 지키지만, 값을 담아 보내는 **이름**은 지금까지 아무도
    지키지 않았습니다. 여기서 이름을 하나 바꾸고 바로 옆 test까지 함께 고치면 web은
    전부 초록인 채로 그 값이 train에서 조용히 버려집니다. 계약의 목록만 보고,
    train을 부르지 않습니다.

    optimizer와 model마다 실려 가는 칸이 달라 네 조합을 함께 봅니다.
    """

    sent = set(normalize_train_settings(raw))

    assert sent <= set(SETTING_KEYS), f"계약에 없는 이름입니다: {sorted(sent - set(SETTING_KEYS))}"


def test_the_form_offers_every_architecture_the_contract_names():
    """고를 수 있는 모델은 계약이 정합니다. 화면이 그중 하나를 빠뜨리면 안 됩니다."""

    choices = {spec["name"]: spec for spec in field_specs()}

    assert choices["architecture"]["choices"] == list(ARCHITECTURES)
    assert "input_size" in choices
    assert "gradient_accumulation_steps" in choices


def test_numeric_and_flag_defaults_reach_train_as_the_contract_says():
    """비워 둔 칸에 화면이 안내하는 값이 train이 쓰는 기본값과 같아야 합니다.

    화면은 이 값들을 자기 표(`_INTEGER_FIELDS`)로 한 번 더 거쳐 보내므로, 계약을
    고쳐도 그 표를 함께 고치지 않으면 안내와 실제가 갈립니다.
    """

    mirrored = normalize_train_settings({})

    for name in ("seed", "epochs", "checkpoint_every", "batch_size", "device", "pretrained"):
        assert mirrored[name] == SETTING_DEFAULTS[name], f"{name} 기본값이 어긋났습니다."
    assert mirrored["gradient_accumulation_steps"] == SETTING_DEFAULTS[
        "gradient_accumulation_steps"
    ]
    assert train_config.DEFAULT_OUTPUT_DIR == SETTING_DEFAULTS["output_dir"]
    assert train_config.DEFAULT_OUTPUT_PREFIX == SETTING_DEFAULTS["output_prefix"]


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


def test_augmentation_reaches_train_as_the_object_it_expects():
    """train은 preset key 하나만 든 object를 받고 다른 key가 있으면 거부합니다."""

    settings = normalize_train_settings({"augmentation": "pill_basic"})

    assert settings["augmentation"] == {"preset": "pill_basic"}


def test_augmentation_defaults_to_none_like_train():
    assert normalize_train_settings({})["augmentation"] == {"preset": DEFAULT_AUGMENTATION}


def test_unknown_augmentation_is_rejected_before_training_starts():
    with pytest.raises(Exception, match="augmentation"):
        normalize_train_settings({"augmentation": "무작위회전"})


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


def test_every_offered_field_has_a_place_on_the_new_experiment_form():
    """서버가 주는 칸은 화면의 어느 표에든 자리가 있어야 합니다.

    화면은 `TABS`에 적힌 이름만 그립니다. 그래서 서버가 새 칸을 내려보내도 그 배열에
    없으면 **화면에 아예 나타나지 않고**, 사람은 그 값이 기본값으로 돈다는 것조차
    모릅니다. 실제로 `checkpoint_every`가 그렇게 오래 감춰져 있었습니다.

    train source를 읽는 위 test들과 같은 방식으로, 화면 source를 글자로만 읽습니다.
    """

    sheet = (
        Path(__file__).resolve().parents[1]
        / "frontend/src/screens/NewExperimentSheet.tsx"
    ).read_text(encoding="utf-8")
    start = sheet.index("const TABS")
    block = sheet[start : sheet.index("];", start)]
    # 주석은 먼저 덜어 냅니다. 주석 처리된 이름은 화면에 **없는** 것인데, 그대로 세면
    # 칸을 지워 놓고도 이 test가 조용합니다. 주석에서 칸 이름을 언급하는 경우도 같습니다.
    block = re.sub(r"/\*.*?\*/", "", block, flags=re.DOTALL)
    block = re.sub(r"//[^\n]*", "", block)
    # 표 자체의 key는 칸 이름이 아닙니다.
    placed = set(re.findall(r"'([a-z_0-9]+)'", block)) - {"basic", "hyper", "output"}

    offered = {spec["name"] for spec in field_specs()}

    assert offered - placed == set(), "화면에 자리가 없는 칸이 있습니다."
    assert placed - offered == set(), "서버가 주지 않는 칸이 화면에 적혀 있습니다."
