# 012. MMDetection checkpoint 추론 계약

## 요청 이유

Train에 `dino_r50_4scale`와 `cascade_rcnn_swin_t_fpn`을 추가하려고 합니다. 현재
Evaluate는 checkpoint의 `architecture`를
`torchvision.models.detection`에서 찾아 모델을 다시 만듭니다. 두 모델은
MMDetection 모델이므로 지금 형식만으로는 학습 결과를 평가하거나 test submission에
사용할 수 없습니다.

## Train이 유지하는 것

- `run(config)` 반환 key와 artifact key는 바꾸지 않습니다.
- checkpoint의 `architecture`, `num_classes`, `model_state_dict`, `class_map`,
  `category_ids`를 유지합니다.
- `num_classes`는 지금처럼 background를 포함합니다.
- 기존 torchvision checkpoint에는 새 key가 없어도 됩니다.

## 추가하는 checkpoint 값

```json
{
  "backend": "mmdetection",
  "architecture": "dino_r50_4scale",
  "model_config": {
    "schema_version": 1,
    "input_size": 640,
    "resize": "longest_edge",
    "pad_multiple": 32
  }
}
```

Cascade 모델은 `architecture`만 `cascade_rcnn_swin_t_fpn`으로 달라집니다.
`model_config`는 JSON-safe 값만 가지며 credential, 로컬 절대 경로, weight download
URL을 넣지 않습니다.

## Evaluate에 요청하는 동작

1. `backend`가 없으면 기존과 같이 torchvision checkpoint로 읽습니다.
2. `backend="mmdetection"`이면 architecture allowlist에서 모델을 만듭니다. 임의의
   builder나 import 경로를 실행하지 않습니다.
3. MMDetection에는 `num_classes - 1`을 전달합니다. checkpoint의 model label 0은
   계속 background이고 실제 label은 1부터 시작합니다.
4. 긴 변이 `input_size`가 되도록 비율을 유지해 resize하고 32 배수로 padding합니다.
   예측 box는 원본 이미지 좌표로 되돌립니다.
5. 출력은 기존과 같은 `boxes`, `labels`, `scores`로 정규화하고 `category_ids`로 원래
   COCO category id를 복원합니다.
6. 알 수 없는 backend, architecture, `model_config.schema_version`, 잘못된 state shape는
   추측하거나 fallback하지 않고 기존 `PredictionError`로 보고합니다.

## 호환성과 검증

- 기존 torchvision checkpoint 추론 결과는 변하지 않아야 합니다.
- 두 신규 architecture의 작은 checkpoint fixture로 모델 재생성 및 state 적용을
  contract test로 확인합니다.
- resize 후 box를 원본 좌표로 되돌리는 test와 잘못된 backend/recipe/state 실패 test를
  둡니다.
- MMDetection dependency가 없을 때는 import 시점이 아니라 해당 backend를 선택했을 때
  안전한 오류를 냅니다.

Evaluate 담당자가 이 계약과 architecture 이름을 승인한 뒤 Train 구현을 merge합니다.
