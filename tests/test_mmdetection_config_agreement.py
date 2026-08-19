"""train이 학습한 model과 evaluate가 되살리는 model이 같은 모양인지 확인합니다.

pipeline은 서로를 import하지 않으므로 같은 detector 설정이 두 곳에 따로 있습니다.
어긋나면 checkpoint를 아예 못 싣거나(모양이 다를 때), **멈추지 않고 점수만
나빠집니다**(`window_size`처럼 모양은 같은데 뜻이 다른 값일 때). 두 pipeline을 함께
볼 수 있는 자리는 여기뿐입니다.
"""

from __future__ import annotations

import pytest

from src.common.train_contract import MMDETECTION_ARCHITECTURES
from src.pipelines.evaluate.mmdetection_backend import build_detector_config
from src.pipelines.train.mmdetection_adapter import build_mmdetection_config


@pytest.mark.parametrize("architecture", MMDETECTION_ARCHITECTURES)
def test_train_and_evaluate_build_the_same_detector(architecture: str):
    trained = build_mmdetection_config(architecture, foreground_classes=7)
    restored = build_detector_config(architecture, foreground_classes=7)

    # 학습에만 쓰는 key(`train_cfg`, `dn_cfg`)는 evaluate에 없어도 됩니다. 가중치가
    # 놓이는 자리를 정하는 나머지는 전부 같아야 합니다.
    shared = set(trained) & set(restored)
    assert "backbone" in shared and "neck" in shared
    differing = sorted(key for key in shared if trained[key] != restored[key])
    assert not differing, f"{architecture}: 두 pipeline의 설정이 갈립니다: {differing}"
