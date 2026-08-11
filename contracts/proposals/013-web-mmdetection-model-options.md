# 013. Web의 MMDetection 학습 모델 선택

## 요청 이유

Train에 `dino_r50_4scale`와 `cascade_rcnn_swin_t_fpn`을 선택 가능한 architecture로
추가하려고 합니다. Web은 Train을 import하지 않고 architecture 목록, 기본값, 검증
규칙을 복사하므로 함께 바뀌지 않으면 GUI가 새 모델을 거부하거나 잘못된 config를
만듭니다.

## 유지하는 동작

- 기본 architecture는 계속 `fasterrcnn_mobilenet_v3_large_320_fpn`입니다.
- 기존 architecture와 optimizer 선택, legacy fallback은 바꾸지 않습니다.
- checkpoint, history, progress event 이름과 artifact 경로는 바꾸지 않습니다.

## Train에 추가할 설정

| 설정 | 신규 모델 기본값 | 의미 |
| --- | ---: | --- |
| `train.input_size` | `640` | 비율을 유지해 긴 변을 맞출 크기 |
| `train.gradient_accumulation_steps` | `8` | optimizer update 한 번에 모을 microbatch 수 |

두 값은 bool이 아닌 양의 정수여야 합니다. `input_size`는 MMDetection 모델에만 쓰므로
torchvision architecture와 함께 보내면 Train이 거부합니다.
`gradient_accumulation_steps`는 기존 모델에서 생략하면 `1`입니다.

## 신규 모델의 8GB 실행 제약

두 신규 architecture를 선택하면 Web도 아래 조합만 유효하게 처리해 사용자가 학습
시작 뒤에야 실패하지 않게 합니다.

- `device="cuda"`
- `precision="amp"`
- `optimizer="AdamW"`
- `batch_size=1`

`pretrained=true`는 COCO detector 가중치를 사용한다는 뜻이고 `false`도 허용합니다.
기존 모델에는 위 제약을 적용하지 않습니다.

## Web에 요청하는 변경

1. capability와 새 실험 화면의 architecture choices에 두 이름을 추가합니다.
2. `input_size`와 `gradient_accumulation_steps`를 설정 schema, payload, 검토 화면,
   실험 상세 기록에 포함합니다.
3. 잘못된 조합은 queue에 넣기 전에 각 필드와 필요한 값을 한국어로 안내합니다.
4. 이어서 학습할 때 두 값과 architecture를 원 실행에서 그대로 복사합니다.
5. Train source를 AST로 읽는 contract test에 architecture 목록, 두 기본값과 검증 규칙을
   추가합니다.

Web 담당자가 이 제안을 반영한 companion 변경을 준비한 뒤 Train의 architecture 공개
변경과 함께 merge합니다.
