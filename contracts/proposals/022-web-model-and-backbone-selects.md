# 022. 새 실험 화면을 model 칸과 backbone 칸 둘로 나눈다

## 무엇을 했는가

#197이 DINO를 backbone이 다른 갈래 넷으로 나눴습니다. 지금 새 실험 화면의 model
목록에는 그 이름 넷이 그대로 서 있습니다:

```
fasterrcnn_mobilenet_v3_large_320_fpn   ← 기본
fasterrcnn_resnet50_fpn_v2
retinanet_resnet50_fpn_v2
dino_r50_4scale
dino_swin_t_4scale
dino_swin_b_4scale
dino_swin_l_5scale
cascade_rcnn_swin_t_fpn
```

이것을 **model 칸에 `dino` 하나**로 접고, 그 옆에 **backbone 칸**을 하나 더 그립니다
(`resnet50`(기본) / `swin_t` / `swin_b` / `swin_l`). DINO가 아닌 모델을 고르면 backbone
칸은 나오지 않습니다.

**#197에서 이 부분만 떼어 낸 것입니다.** 그때 이렇게 적었습니다 — *"CI를 초록으로
유지하는 데 필요하지 않고 `src/pipelines/web/`은 제 영역이 아니기 때문입니다. 별도
PR로 올립니다."* 이 PR이 그 별도 PR이고, 내용은 #197의 commit `a4e9685`를 되돌린
것입니다.

## 보내는 값은 여전히 이름 하나입니다

**이것이 이 변경의 핵심입니다.** 두 칸은 화면에만 있습니다. 어느 쪽을 바꾸든 계약의
`ARCHITECTURE_BACKBONES` 표를 거쳐 다시 `architecture` 이름 하나로 합쳐 보냅니다.

- 서버가 받는 값, 저장되는 값, checkpoint에 남는 값, registry가 읽는 값 — **전부 그대로**
- `model_config.schema_version`은 1 그대로. evaluate도 손댈 것이 없습니다
- 기존 기록을 다시 실행할 때도 `architecture` 하나만 읽으면 두 칸이 복원됩니다

`backbone`을 **별도 설정 값으로 두지 않은 이유**가 여기 있습니다. 값이 둘이면 서로
어긋날 수 있고, 어긋난 쪽은 오류 없이 점수만 나빠집니다. 이 저장소가 반복해서 당한
실패가 정확히 그 종류입니다.

## 무엇을 바꾸는가

| 파일 | 무엇 |
| --- | --- |
| `src/common/train_contract.py` | `ARCHITECTURE_BACKBONES`(갈래 → backbone → architecture 이름)와 `DEFAULT_ARCHITECTURE_BACKBONES`. 값이 아니라 **화면이 접었다 펴는 표**입니다 |
| `web/train_capabilities.py` | 위 둘을 그대로 재수출 |
| `web/train_config.py` | `architecture` spec에 `backbones`·`backbone_defaults`·label·hint를 실어 보냅니다. **`choices`는 계약 목록 그대로 둡니다** |
| `web/frontend/src/api/types.ts` | `FieldSpec`에 `backbones` 계열 optional 필드 |
| `web/frontend/src/screens/NewExperimentSheet.tsx` | `backbones`가 있는 enum spec을 select 둘로 그립니다. 2열 grid의 칸 두 개를 차지합니다 |
| `web/frontend/src/lib/architectureBackbone.ts` (+ test) | 목록을 접는 `displayChoices()`와 합치고 나누는 표 조회 |
| `web/tests/test_web_train_contract.py` | `choices`가 계약 목록 그대로인지, 갈래 이름이 계약 이름과 겹치지 않는지, 표가 가리키는 이름이 전부 계약에 있는지 |

## 조심한 자리

- **같은 갈래를 다시 골라도 backbone을 기본값으로 되돌리지 않습니다.** 되돌리면 사람은
  `swin_b`를 골라 둔 채 `resnet50`을 학습합니다. `architectureForFamily()`가 지금 값이
  이미 그 갈래면 그대로 둡니다. test가 이 경우를 따로 지킵니다.
- **접는 일은 화면 안에서만 일어납니다.** 서버가 내려 주는 `choices`는 계약의 진짜
  architecture 이름 그대로입니다. 거기에 `dino`를 실으면 그 목록을 그대로 보내는 다른
  소비자(generic form, API 이용자)가 서버에게 거절당합니다 —
  `normalize_train_settings({"architecture": "dino"})`는 실패합니다. 그래서 접기는
  `displayChoices()`가 화면에서만 합니다.
  `test_the_form_offers_every_architecture_the_contract_names`가 `choices`를 계약
  목록과 그대로 견주고, 갈래 이름이 계약 이름과 겹치지 않는지도 봅니다.
- **표에 없는 이름은 접히지 않고 그대로 나옵니다.** 계약에 architecture가 늘었는데
  표에 넣지 않아도 화면에서 사라지지는 않습니다 — 긴 이름 그대로 목록에 섭니다.
  표가 가리키는 이름이 전부 계약에 있는지는 test가 봅니다(없는 이름을 가리키면
  화면에는 고를 수 있게 보이는데 서버가 거절합니다).
- backbone 칸의 안내에 실측 GPU 사용량을 적었습니다(resnet50 3.0GB / swin_t 3.3 /
  swin_b 3.8 / **swin_l 11.2**). swin_l은 10GB 카드에 모자라며, **막지 않고 알리기만**
  합니다.

## 소유자가 봐 주었으면 하는 것

`src/pipelines/web/`은 제 영역이 아닙니다. 특히 두 가지를 봐 주십시오.

1. **select 둘을 한 spec으로 그리는 방식**이 이 화면의 결에 맞는지. `TrainField`가
   fragment로 `Field` 둘을 돌려주고 2열 grid가 각각을 한 칸으로 받습니다. 다른 칸을
   숨기는 `only_for_architectures` 기제는 그대로 두고 건드리지 않았습니다.
2. **표를 계약에 둔 것**이 맞는지. web에 두면 architecture 이름을 여기서 다시 적게
   되는데, `web/CLAUDE.md`가 *"never re-type them here — 한 번 복제했다가 이름이
   어긋난 적이 있다"*고 금지하는 자리입니다. 그래서 train이 소유한 계약에 두었습니다.
