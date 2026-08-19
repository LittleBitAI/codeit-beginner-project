"""web이 train에 보내는 설정이 train의 계약을 지키는지 봅니다.

이름과 숫자는 이제 `src/common/train_contract.py` 한 곳에 있습니다. 예전에는 web이
그 값을 복제해 두고 이 파일이 train의 source를 `ast`로 읽어 두 벌을 대조했습니다.
값이 한 벌이 되면서 그 대조는 자기 자신과 자기를 비교하는 일이 되어 없앴습니다.

남은 것은 **값이 같다고 지켜지지 않는 것**들입니다. 화면이 그 값을 어떤 칸으로
보여 주는지, 그리고 보내는 object의 모양이 train이 받는 모양과 같은지입니다.
train은 자기가 쓰지 않는 key가 하나만 있어도 object를 통째로 거부하므로, 값이
맞아도 모양이 틀리면 GPU를 잡은 뒤에야 실패합니다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.common.train_contract import (
    ARCHITECTURES,
    AUGMENTATIONS,
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_AUGMENTATION,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_KEYS,
    LR_SCHEDULER_DEFAULTS,
    MMDETECTION_ARCHITECTURES,
    SETTING_DEFAULTS,
    SETTING_KEYS,
)
from src.pipelines.web import train_config
from src.pipelines.web.train_config import field_specs, normalize_train_settings


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
def test_no_setting_leaves_here_under_a_name_train_does_not_read(monkeypatch, raw):
    """train은 이 파일을 import할 수 없는 저쪽에서 같은 이름으로 값을 읽습니다.

    값이 같은지는 계약의 표들이 지키지만, 값을 담아 보내는 **이름**은 지금까지 아무도
    지키지 않았습니다. 여기서 이름을 하나 바꾸고 바로 옆 test까지 함께 고치면 web은
    전부 초록인 채로 그 값이 train에서 조용히 버려집니다. 계약의 목록만 보고,
    train을 부르지 않습니다.

    optimizer와 model마다 실려 가는 칸이 달라 네 조합을 함께 봅니다. MMDetection
    조합은 CUDA를 요구하므로 GPU 없는 CI에서도 돌도록 있는 척합니다.
    """

    monkeypatch.setattr(train_config, "cuda_is_available", lambda: True)

    sent = set(normalize_train_settings(raw))

    assert sent <= set(SETTING_KEYS), f"계약에 없는 이름입니다: {sorted(sent - set(SETTING_KEYS))}"


def test_the_form_is_told_which_models_use_input_size():
    """화면이 어느 모델에서 이 칸을 보일지 스스로 판단하지 않게 합니다.

    목록을 화면에 옮겨 적으면 계약에 architecture를 더해도 화면은 모르고, 그 값을
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
        item for item in field_specs() if item["name"] == "gradient_accumulation_steps"
    )

    assert spec["default"] == SETTING_DEFAULTS["gradient_accumulation_steps"]
    assert spec["defaults_by_architecture"] == {
        architecture: DEFAULT_ACCUMULATION_STEPS
        for architecture in MMDETECTION_ARCHITECTURES
    }


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


@pytest.mark.parametrize("name", tuple(LR_SCHEDULER_DEFAULTS))
def test_lr_scheduler_reaches_train_with_exactly_the_keys_it_uses(name):
    """train은 고른 schedule이 쓰지 않는 key가 하나만 있어도 object를 통째로 거부합니다."""

    mirrored = normalize_train_settings({"lr_scheduler": name, "lr_warmup_steps": 1})

    assert set(mirrored["lr_scheduler"]) == {
        "name",
        "warmup_steps",
        "warmup_start_factor",
        *LR_SCHEDULER_DEFAULTS[name],
    }


def test_augmentation_reaches_train_as_the_object_it_expects():
    """train은 preset key 하나만 든 object를 받고 다른 key가 있으면 거부합니다."""

    settings = normalize_train_settings({"augmentation": "pill_basic"})

    assert settings["augmentation"] == {"preset": "pill_basic"}


def test_augmentation_defaults_to_none_like_train():
    assert normalize_train_settings({})["augmentation"] == {"preset": DEFAULT_AUGMENTATION}
    assert DEFAULT_AUGMENTATION in AUGMENTATIONS


def test_unknown_augmentation_is_rejected_before_training_starts():
    with pytest.raises(Exception, match="augmentation"):
        normalize_train_settings({"augmentation": "무작위회전"})


def test_early_stopping_reaches_train_with_exactly_the_keys_it_expects():
    """train은 모르는 key가 있으면 object를 통째로 거부합니다."""

    mirrored = normalize_train_settings(
        {"early_stopping": True, "early_stopping_patience": 5}
    )

    assert set(mirrored["early_stopping"]) == set(EARLY_STOPPING_KEYS)
    assert mirrored["early_stopping"]["min_delta"] == DEFAULT_EARLY_STOPPING_MIN_DELTA


def test_gradient_accumulation_above_one_is_accepted_now_that_train_reads_it():
    """읽지 않던 동안에는 기본값 말고 어떤 값도 받지 않았습니다. 받아 두면 config에는
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
    """`MMDETECTION_REQUIRED`를 갖춘 최소 요청입니다."""

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


def test_a_torchvision_run_sends_no_input_size():
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
def test_mmdetection_refuses_unsupported_combinations(
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

    화면 source를 글자로만 읽습니다. import하지 않으므로 결합이 생기지 않습니다.
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
    # 반대쪽도 봅니다. 서버가 주지 않는 칸이 화면에 남아 있으면 사람은 그 값을 채우고
    # 시작 버튼을 누르는데, 그 값은 어디에도 실려 가지 않고 조용히 버려집니다.
    assert placed - offered == set(), "서버가 주지 않는 칸이 화면에 적혀 있습니다."
