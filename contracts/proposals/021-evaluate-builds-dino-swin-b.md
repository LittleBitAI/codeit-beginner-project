# 021. dino_swin_b_4scale — evaluate 쪽까지 같은 PR에 넣은 이유

## 무엇을 했는가

train에 MMDetection architecture가 하나 늘었습니다 — `dino_swin_b_4scale`.
`dino_r50_4scale`에서 backbone만 ResNet-50 → Swin-B로 바꾼 갈래입니다. 지금까지
점수를 올린 축 가운데 필요한 크기의 전례가 있는 것은 아키텍처뿐이고(frcnn → DINO가
+0.012), 현재 리더보드는 전부 R50 위에 서 있습니다.

**`src/pipelines/evaluate/mmdetection_backend.py`도 같은 PR에서 고쳤습니다.**
소유자가 아닌 곳이므로, 왜 제안서만 남기지 않았는지 적습니다.

## 왜 쪼갤 수 없었는가

이름은 `src/common/train_contract.py`의 `MMDETECTION_ARCHITECTURES`에 있고, evaluate는
그 tuple을 통째로 `SUPPORTED_ARCHITECTURES`로 받습니다. 그래서 이름을 더하는 순간
evaluate의 자기 test가 빨개집니다.

```
FAILED src/pipelines/evaluate/tests/test_evaluate_mmdetection.py::
       test_this_pipeline_can_build_every_mmdetection_model_the_contract_names
```

그 test의 docstring이 이 상황을 그대로 적어 두었습니다 — *"계약에 세 번째 이름이
생기면 GUI는 그것을 내놓고 train은 학습까지 마치는데, 평가만 마지막에 거부합니다 —
밤새 학습한 뒤에."* 즉 **계약·train·evaluate를 한 PR로 묶으라고 설계된 문**입니다.
train 쪽만 올리면 CI가 빨간 채로 남아 merge되지 않습니다.

## evaluate에서 바꾼 것

`_dino_config`에 `swin` 갈래를 더하고 `build_detector_config`에 분기 한 줄을
더했습니다. train의 `_swin_b_backbone()`을 그대로 옮긴 것이고, 나머지(encoder,
decoder, positional_encoding, bbox_head, `num_outs=4`)는 R50과 **완전히 같습니다** —
4scale을 그대로 둔 이유가 그것입니다.

| | `dino_r50_4scale` | `dino_swin_b_4scale` |
| --- | --- | --- |
| `backbone.type` | `ResNet` (depth 50) | `SwinTransformer` |
| backbone 설정 | — | `embed_dims=128`, `depths=[2,2,18,2]`, `num_heads=[4,8,16,32]`, `window_size=7`, `out_indices=(1,2,3)`, `with_cp=True`, `convert_weights=False`, `init_cfg=None` |
| `neck.in_channels` | `[512, 1024, 2048]` | `[256, 512, 1024]` |

## 두 벌이 갈라지지 않게 한 장치

`tests/test_mmdetection_config_agreement.py`를 새로 두었습니다. 계약의 모든 이름에
대해 train과 evaluate가 만든 설정의 공통 key를 견줍니다. pipeline은 서로를 import하지
않으므로 같은 설정이 두 곳에 따로 있고, `window_size`처럼 **모양은 같은데 뜻이 다른
값**이 어긋나면 지금까지는 멈추지 않고 점수만 나빠졌습니다. root `tests/`가 두
pipeline을 함께 볼 수 있는 유일한 자리입니다.

기존 `test_each_architecture_is_rebuilt_and_receives_the_checkpoint_state`의
docstring이 *"train과 값이 같은지는 확인하지 못합니다"*라고 적어 둔 자리를 메웁니다.

## 소유자가 봐 주었으면 하는 것

값을 옮겨 적은 곳이라 오탈자가 그대로 점수가 됩니다. 위 표와
`train/mmdetection_adapter.py`의 `_swin_b_backbone()`을 나란히 놓고 확인해 주십시오.
바꾸고 싶은 것이 있으면 두 벌을 함께 고쳐야 하고, 위 test가 그것을 강제합니다.
