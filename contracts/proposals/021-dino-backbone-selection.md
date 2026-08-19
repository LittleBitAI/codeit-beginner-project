# 021. DINO의 backbone 선택 — evaluate와 web까지 같은 PR에 넣은 이유

## 무엇을 했는가

DINO의 backbone을 새 실험 화면에서 고를 수 있게 했습니다. 계약의 architecture 이름이
넷으로 갈립니다.

| 화면 backbone | architecture 이름 | scale |
| --- | --- | --- |
| `resnet50` (기본, 기존) | `dino_r50_4scale` | 4 |
| `swin_t` | `dino_swin_t_4scale` | 4 |
| `swin_b` | `dino_swin_b_4scale` | 4 |
| `swin_l` | `dino_swin_l_5scale` | 5 |

점수를 올린 축 가운데 필요한 크기(+0.014)의 전례가 있는 것은 아키텍처뿐이고
(frcnn → DINO가 +0.012), 현재 리더보드는 전부 ResNet-50 위에 서 있습니다.

**`src/pipelines/evaluate/`와 `src/pipelines/web/`도 같은 PR에서 고쳤습니다.**
소유자가 아닌 곳이므로, 왜 제안서만 남기지 않았는지 적습니다.

## 왜 쪼갤 수 없었는가

두 pipeline 모두 계약의 architecture 목록을 통째로 받습니다. 그래서 이름을 더하는
순간 **각자의 자기 test가 빨개집니다.**

```
FAILED src/pipelines/evaluate/tests/test_evaluate_mmdetection.py::
       test_this_pipeline_can_build_every_mmdetection_model_the_contract_names
FAILED src/pipelines/web/tests/test_web_train_contract.py::
       test_the_form_offers_every_architecture_the_contract_names
```

evaluate 쪽 test의 docstring이 이 상황을 그대로 적어 두었습니다 — *"계약에 세 번째
이름이 생기면 GUI는 그것을 내놓고 train은 학습까지 마치는데, 평가만 마지막에 거부합니다
— 밤새 학습한 뒤에."* **묶으라고 설계된 문**이라 train 쪽만 올리면 CI가 빨간 채 남습니다.
web도 마찬가지로, 화면이 새 이름을 내놓지 못하면 backbone을 고를 수단 자체가 없습니다.

## evaluate에서 바꾼 것

`_dino_config`가 architecture를 받아 backbone·neck·level 수를 정하도록 일반화했습니다.
train의 `_SWIN_VARIANTS`, `_swin_backbone()`, `_dino_in_channels()`, `_dino_levels()`를
그대로 옮긴 것입니다.

- 4scale 갈래는 backbone이 뒤 세 단계만 내보내고(`out_indices=(1,2,3)`) 나머지
  구조가 R50과 **완전히 같습니다.** `num_feature_levels`는 적지 않습니다 — 넷이 DINO의
  기본값이라, 지금까지 리더보드를 만든 4scale 설정을 한 글자도 바꾸지 않으려는 것입니다.
- 5scale(`swin_l`)만 `out_indices=(0,1,2,3)`, `neck.num_outs=5`,
  encoder `self_attn_cfg.num_levels=5`, decoder `cross_attn_cfg.num_levels=5`,
  `num_feature_levels=5`입니다. **한 자리만 넷으로 남으면 model이 만들어지지 않거나
  조용히 다른 것이 됩니다.**

## 두 벌이 갈라지지 않게 한 장치

`tests/test_mmdetection_config_agreement.py`가 계약의 모든 이름에 대해 train과
evaluate가 만든 설정의 공통 key를 견줍니다. pipeline은 서로를 import하지 않으므로 같은
설정이 두 곳에 따로 있고, `window_size`처럼 **모양은 같은데 뜻이 다른 값**이 어긋나면
지금까지는 멈추지 않고 점수만 나빠졌습니다. root `tests/`가 두 pipeline을 함께 볼 수
있는 유일한 자리입니다.

기존 `test_each_architecture_is_rebuilt_and_receives_the_checkpoint_state`의 docstring이
*"train과 값이 같은지는 확인하지 못합니다"*라고 적어 둔 자리를 메웁니다.

## web에서 바꾼 것 — 화면은 두 칸이지만 값은 하나입니다

새 실험 화면은 model 칸에 `dino` 하나만 보여 주고 backbone 칸을 따로 그립니다. 그러나
**보내고 저장하고 checkpoint에 남는 값은 여전히 architecture 이름 하나뿐입니다.** 두
칸은 그 이름을 나눠 보여 줄 뿐이고, 계약의 `ARCHITECTURE_BACKBONES` 표를 거쳐 다시
하나로 합쳐 올립니다.

- `train_config.py`의 `_architecture_choices()`가 갈래를 이름 하나로 접고,
  `architecture` spec에 `backbones`·`backbone_defaults`를 실어 보냅니다.
- `NewExperimentSheet.tsx`는 `backbones`가 있는 enum spec을 두 select로 그립니다.
  합치고 나누는 표 조회는 `lib/architectureBackbone.ts`에 있습니다.
- 같은 갈래를 다시 골랐을 때 backbone을 기본값으로 되돌리지 않습니다. 되돌리면 사람은
  swin_b를 골라 둔 채 resnet50을 학습합니다.

**설정 값을 하나로 유지한 것이 이 설계의 핵심입니다.** `backbone`을 별도 설정으로
두면 architecture와 어긋날 수 있고, 어긋난 쪽은 멈추지 않고 점수만 나빠집니다. 값이
하나라 evaluate·registry·`model_config`의 모양이 바뀌지 않았고 기존 checkpoint도
그대로 읽힙니다. `model_config.schema_version`은 1 그대로입니다.

## 소유자가 봐 주었으면 하는 것

값을 옮겨 적은 곳이라 오탈자가 그대로 점수가 됩니다.
`train/mmdetection_adapter.py`의 `_SWIN_VARIANTS`와 나란히 놓고 확인해 주십시오.
바꾸고 싶은 것이 있으면 두 벌을 함께 고쳐야 하고, 위 test가 그것을 강제합니다.
