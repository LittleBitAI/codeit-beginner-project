# 021. DINO의 backbone 갈래 — evaluate까지 같은 PR에 넣은 이유

## 무엇을 했는가

DINO를 backbone이 다른 갈래 넷으로 나눴습니다. 계약의 architecture 이름이 그만큼
늘어나고, 새 실험 화면의 model 목록에 그대로 나옵니다.

| architecture 이름 | backbone | scale |
| --- | --- | --- |
| `dino_r50_4scale` (기존) | ResNet-50 | 4 |
| `dino_swin_t_4scale` | Swin-T | 4 |
| `dino_swin_b_4scale` | Swin-B | 4 |
| `dino_swin_l_5scale` | Swin-L | 5 |

점수를 올린 축 가운데 필요한 크기(+0.014)의 전례가 있는 것은 아키텍처뿐이고
(frcnn → DINO가 +0.012), 현재 리더보드는 전부 ResNet-50 위에 서 있습니다.

**`src/pipelines/evaluate/`도 같은 PR에서 고쳤습니다.** 소유자가 아닌 곳이므로, 왜
제안서만 남기지 않았는지 적습니다.

## 왜 쪼갤 수 없었는가

evaluate는 계약의 architecture 목록을 통째로 `SUPPORTED_ARCHITECTURES`로 받습니다.
그래서 이름을 더하는 순간 **evaluate의 자기 test가 빨개집니다.**

```
FAILED src/pipelines/evaluate/tests/test_evaluate_mmdetection.py::
       test_this_pipeline_can_build_every_mmdetection_model_the_contract_names
```

그 test의 docstring이 이 상황을 그대로 적어 두었습니다 — *"계약에 이름이 하나 늘었는데
여기 config가 없으면 GUI는 그것을 내놓고 train은 학습까지 마치는데, 평가만 마지막에
거부합니다 — 밤새 학습한 뒤에."* **묶으라고 설계된 문**이라 train 쪽만 올리면 CI가
빨간 채 남습니다. (그 문장은 architecture가 둘이던 때 쓰여 "세 번째 이름"이라고 되어
있었습니다. 이 PR이 다섯으로 늘리므로 개수를 빼고 같이 고쳤습니다.)

web은 다릅니다. 화면은 계약 목록을 그대로 읽어 내놓으므로 이름이 늘어도 초록입니다.
그래서 **이 PR은 web의 판단 로직을 하나도 바꾸지 않습니다** — 거절하는 조합도, 거절하는
자리도 그대로입니다. 다만 **이 PR이 낡게 만든 설명은 고쳤습니다**:

| 파일 | 무엇 |
| --- | --- |
| `web/CLAUDE.md`, `web/AGENTS.md` | 지침서 문장. "MMDetection pair" → "ones", "the 8GB combination" → "the only combination"(각 2곳) |
| `web/train_config.py` | 개수를 못 박은 주석 둘, 그리고 **화면에 보이는 오류 메시지 한 줄** — "…는 8GB에서 돌리려면 X여야 합니다" → "…는 X로만 돌릴 수 있습니다" |
| `web/tests/test_web_train_contract.py` | test 이름 `..._do_not_fit_8gb` → `..._refuses_unsupported_combinations`, docstring 둘 |

**사용자에게 보이는 문자열이 하나 바뀌는 것**이 유일한 눈에 띄는 변화입니다. 그대로
두면 화면이 틀린 근거를 말합니다 — swin_l은 8GB에 안 들어가고, 애초에 이 조합은 메모리
때문이 아닙니다. 근거는 아래 "지원 범위" 절에 있습니다.

## 지원 범위 — `MMDETECTION_REQUIRED`의 근거를 바로잡았습니다

`{device: cuda, precision: amp, optimizer: AdamW, batch_size: 1}`을 저장소 곳곳에서
**"8GB에서 도는 조합"**이라고 설명해 왔습니다. 이 PR이 그것을 무너뜨립니다 —
`dino_swin_l_5scale`은 1280px·batch 1·amp로 **11.21 GiB**입니다.

근거를 다시 재 보니 애초에 메모리 이야기가 아니었습니다. 값마다 이유가 다릅니다:

- `batch_size=1`, `amp` — 메모리 때문
- `AdamW` — DETR 계열이 그것으로 수렴하기 때문. **메모리로는 오히려 SGD보다 optimizer
  state를 더 씁니다**
- `cuda` — `amp`가 CUDA에서만 되기 때문(`CUDA_ONLY_PRECISIONS`). CPU에서 model이 아예
  안 만들어지는 것은 아닙니다 — test가 CPU에서 다섯 detector의 loss까지 봅니다

그래서 **"이 저장소가 지원하는(=재 본) 하나뿐인 조합"**으로 바꿨습니다. 계약·train·
web의 주석과 화면 오류 메시지, test 이름까지 같은 표현으로 맞췄습니다. 동작은 그대로
입니다 — 거절하는 조합도, 거절하는 자리도 바뀌지 않았습니다.

## evaluate에서 바꾼 것

`_dino_config`가 architecture를 받아 backbone·neck·level 수를 정하도록 일반화했습니다.
train의 `_SWIN_VARIANTS`, `_swin_backbone()`, `_dino_in_channels()`, `_dino_levels()`를
그대로 옮긴 것입니다.

- 4scale 갈래는 backbone이 뒤 세 단계만 내보내고(`out_indices=(1,2,3)`),
  **backbone과 `neck.in_channels` 둘만 R50과 다릅니다.** 그 밖의 encoder·decoder·
  head·`neck.num_outs`는 완전히 같습니다. `num_feature_levels`는 적지 않습니다 —
  넷이 DINO의 기본값이라, 지금까지 리더보드를 만든 4scale 설정을 한 글자도 바꾸지
  않으려는 것입니다.
- 5scale(`swin_l`)만 `out_indices=(0,1,2,3)`, `neck.num_outs=5`,
  encoder `self_attn_cfg.num_levels=5`, decoder `cross_attn_cfg.num_levels=5`,
  `num_feature_levels=5`입니다. **한 자리만 넷으로 남으면 model이 만들어지지 않거나
  조용히 다른 것이 됩니다.**

## 두 벌이 갈라지지 않게 한 장치

`tests/test_mmdetection_config_agreement.py`가 계약의 모든 이름에 대해 train과
evaluate가 만든 설정의 **key 집합이 같은지, 그리고 모든 값이 같은지**를 견줍니다.
교집합만 견주면 한쪽에서 key가 통째로 사라진 것을 놓칩니다 — `test_cfg`가 빠지면
DINO의 `max_per_img`가 300에서 MMDetection 기본값 100으로 조용히 줄어듭니다.
pipeline은 서로를 import하지 않으므로 같은 설정이 두 곳에 따로 있습니다. backbone
값처럼 state의 모양을 바꾸는 것은 적재가 실패해 바로 드러나지만(`window_size`를 7에서
12로 바꾸면 `relative_position_bias_table`이 (169, heads)에서 (529, heads)가 됩니다),
**state에 자국을 남기지 않는 값**은 지금까지 멈추지 않고 점수만 나빠졌습니다. root
`tests/`가 두 pipeline을 함께 볼 수 있는 유일한 자리입니다.

기존 `test_each_architecture_is_rebuilt_and_receives_the_checkpoint_state`의 docstring이
*"train과 값이 같은지는 확인하지 못합니다"*라고 적어 둔 자리를 메웁니다.

## 화면은 이름 넷을 그대로 내놓습니다

새 실험 화면의 model 목록에 `dino_r50_4scale`·`dino_swin_t_4scale`·
`dino_swin_b_4scale`·`dino_swin_l_5scale`가 그대로 나옵니다. 화면 코드는 계약 목록을
그대로 읽으므로 **화면 동작을 고칠 것이 없습니다** — web에서 바꾼 것은 "왜 쪼갤 수
없었는가" 절의 표에 적은 설명들과 오류 메시지 한 줄뿐이고, 판단 로직은 그대로입니다.

목록을 model 한 칸과 backbone 한 칸으로 접는 화면은 이 PR에 넣지 않았습니다. CI를
초록으로 유지하는 데 필요하지 않고 `src/pipelines/web/`은 제 영역이 아니기 때문입니다.
별도 PR로 올립니다.

**설정 값이 하나라는 것이 이 설계의 핵심입니다.** `backbone`을 별도 설정으로 두면
architecture와 어긋날 수 있고, 어긋난 쪽은 멈추지 않고 점수만 나빠집니다. 값이 하나라
evaluate·registry·`model_config`의 모양이 바뀌지 않았고 기존 checkpoint도 그대로
읽힙니다. `model_config.schema_version`은 1 그대로입니다.

## 소유자가 봐 주었으면 하는 것

값을 옮겨 적은 곳이라 오탈자가 그대로 점수가 됩니다.
`train/mmdetection_adapter.py`의 `_SWIN_VARIANTS`와 나란히 놓고 확인해 주십시오.
바꾸고 싶은 것이 있으면 두 벌을 함께 고쳐야 하고, 위 test가 그것을 강제합니다.
