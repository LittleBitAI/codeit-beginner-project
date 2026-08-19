"""train이 학습한 model과 evaluate가 되살리는 model이 같은 모양인지 확인합니다.

pipeline은 서로를 import하지 않으므로 같은 detector 설정이 두 곳에 따로 있습니다.
어긋나면 checkpoint를 아예 못 싣거나(state의 모양이 달라질 때), **멈추지 않고 점수만
나빠집니다**(`test_cfg`의 `max_per_img`처럼 state에 자국을 남기지 않는 값일 때 —
300에서 MMDetection 기본값 100으로 줄어도 적재는 성공합니다). 두 pipeline을 함께 볼
수 있는 자리는 여기뿐입니다.
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

    # **교집합만 견주면 한쪽에서 key가 통째로 사라진 것을 놓칩니다.** 예를 들어
    # evaluate에서 `test_cfg`가 빠지면 DINO의 `max_per_img`가 300에서 MMDetection
    # 기본값 100으로 조용히 줄어듭니다. 그래서 key 집합부터 같은지 봅니다 — 지금
    # 두 벌은 `train_cfg`·`dn_cfg`까지 완전히 같습니다.
    assert set(trained) == set(restored), (
        f"{architecture}: key가 한쪽에만 있습니다 — "
        f"train만 {sorted(set(trained) - set(restored))}, "
        f"evaluate만 {sorted(set(restored) - set(trained))}"
    )
    differing = sorted(key for key in trained if trained[key] != restored[key])
    assert not differing, f"{architecture}: 두 pipeline의 설정이 갈립니다: {differing}"
